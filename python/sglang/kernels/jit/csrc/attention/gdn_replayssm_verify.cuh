// GDN ReplaySSM speculative verify over a bf16 checkpoint and the circular compact (d, k, g) ring: the
// outputs of up to 4 draft tokens per request and this window's ring records, without committing.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace gdn_replayssm_verify {

constexpr int kD = 128;
constexpr int kS = 4;
constexpr int kL = 16;
constexpr int kCols = 16;
constexpr int kWarps = 4;
constexpr int kThreads = kWarps * 32;
constexpr int kEarly = 8;
constexpr unsigned kFull = 0xffffffffu;

constexpr uint32_t pack_pairs(const int (&v)[16]) {
  uint32_t r = 0;
  for (int i = 0; i < 16; ++i)
    r |= static_cast<uint32_t>(v[i]) << (2 * i);
  return r;
}
constexpr int kPairRow[16] = {1, 2, 2, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3};
constexpr int kPairCol[16] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 1, 2, 3, 2, 3, 3};
constexpr uint32_t kPairRows = pack_pairs(kPairRow);
constexpr uint32_t kPairCols = pack_pairs(kPairCol);

struct Params {
  const uint16_t* q;      // bf16 [tokens, H, 128]
  const uint16_t* k;      // bf16 [tokens, H, 128]
  const uint16_t* v;      // bf16 [tokens, HV, 128]
  const uint16_t* a;      // bf16 [tokens, HV]
  const uint16_t* b;      // bf16 [tokens, HV]
  const void* A_log;      // fp32 or bf16 [HV]
  const void* dt_bias;    // fp32 or bf16 [HV]
  uint16_t* o;            // bf16 [tokens, HV, 128]
  const uint16_t* h0;     // bf16 [slots, HV, 128, 128]
  uint16_t* d_hi;         // bf16 [rings, HV, 16, 128]
  uint16_t* d_lo;         // bf16 [rings, HV, 16, 128]
  uint16_t* k_hi;         // bf16 [rings, H, 16, 128]
  uint16_t* k_lo;         // bf16 [rings, H, 16, 128]
  float* g;               // fp32 [rings, HV, 16]
  const void* qsl;        // [rows + 1]
  const void* state_idx;  // [rows]
  const void* ring_idx;   // [rows]
  const int32_t* write_pos;
  const int32_t* cache_base;
  const int8_t* is_flush;
  int64_t s_q, s_k, s_v, s_a, s_b, s_o, s_h0, s_dhi, s_dlo, s_khi, s_klo, s_g;
  int64_t s_qsl, s_si, s_ri, s_wp, s_cb, s_fl;
  float scale;
  int32_t hv_per_h, null_block_id;
  bool qsl64, si64, ri64, gate16;
};

constexpr int kXStride = kD + 8;

struct Smem {
  uint16_t x[kCols][kXStride];
  float hw[kD][8];
  float sc_hi[kL][8];
  float sc_lo[kL][8];
  float sc[kL][8];
  float gram[kCols][kCols];
  float af[16];
  float ex[32];
  float gs[kS];
  float beta[kS];
};

SGL_DEVICE int64_t load_index(const void* p, const int64_t i, const bool is64) {
  return is64 ? static_cast<const int64_t*>(p)[i] : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

SGL_DEVICE float bf_lo(const uint32_t w) {
  return __uint_as_float(w << 16);
}

SGL_DEVICE float bf_hi(const uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

SGL_DEVICE uint32_t to_bf(const float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

SGL_DEVICE float load_gate(const void* p, const int i, const bool is16) {
  return is16 ? bf_lo(static_cast<const uint16_t*>(p)[i]) : static_cast<const float*>(p)[i];
}

SGL_DEVICE uint4 load_u4(const uint16_t* p) {
  return *reinterpret_cast<const uint4*>(p);
}

SGL_DEVICE void mma_bf16(
    float (&d)[4],
    const uint32_t a0,
    const uint32_t a1,
    const uint32_t a2,
    const uint32_t a3,
    const uint32_t b0,
    const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

SGL_DEVICE void tile_product(
    float (&acc)[2][4], const uint4 (&r0)[4], const uint4 (&r1)[4], const uint16_t* col0, const uint16_t* col1) {
#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const uint4 c0 = load_u4(col0 + 32 * s);
    const uint4 c1 = load_u4(col1 + 32 * s);
    mma_bf16(acc[0], r0[s].x, r1[s].x, r0[s].y, r1[s].y, c0.x, c0.y);
    mma_bf16(acc[1], r0[s].x, r1[s].x, r0[s].y, r1[s].y, c1.x, c1.y);
    mma_bf16(acc[0], r0[s].z, r1[s].z, r0[s].w, r1[s].w, c0.z, c0.w);
    mma_bf16(acc[1], r0[s].z, r1[s].z, r0[s].w, r1[s].w, c1.z, c1.w);
  }
}

SGL_DEVICE void load_rows(uint4 (&r)[4], const uint16_t* row, const bool ok) {
#pragma unroll
  for (int s = 0; s < 4; ++s)
    r[s] = ok ? load_u4(row + 32 * s) : make_uint4(0, 0, 0, 0);
}

SGL_DEVICE void store_rows(float (*dst)[8], const int row, const int t, const float (&acc)[2][4]) {
  *reinterpret_cast<float2*>(dst[row] + 2 * t) = make_float2(acc[0][0] + acc[1][0], acc[0][1] + acc[1][1]);
  *reinterpret_cast<float2*>(dst[row + 8] + 2 * t) = make_float2(acc[0][2] + acc[1][2], acc[0][3] + acc[1][3]);
}

SGL_DEVICE float pick(const float x0, const float x1, const float x2, const float x3, const int i) {
  return i == 0 ? x0 : i == 1 ? x1 : i == 2 ? x2 : x3;
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads) verify_kernel(const __grid_constant__ Params p) {
  __shared__ __align__(16) Smem sm;
  device::PDLWaitPrimary<kUsePDL>();

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int hv = blockIdx.x;
  const int n = blockIdx.y;
  const int h = hv / p.hv_per_h;

  const int64_t bos = load_index(p.qsl, n * p.s_qsl, p.qsl64);
  const int64_t eos = load_index(p.qsl, (n + 1) * p.s_qsl, p.qsl64);
  const int64_t slot = load_index(p.state_idx, n * p.s_si, p.si64);
  const int64_t ring = load_index(p.ring_idx, n * p.s_ri, p.ri64);
  const int S = static_cast<int>(min(eos - bos, static_cast<int64_t>(kS)));
  uint16_t* out = p.o + bos * p.s_o + hv * kD + tid;
  if (slot <= p.null_block_id) {
#pragma unroll
    for (int s = 0; s < kS; ++s) {
      if (s < S) out[s * p.s_o] = 0;
    }
    return;
  }

  const int8_t flush = p.is_flush[ring * p.s_fl];
  const int32_t write_pos = p.write_pos[ring * p.s_wp];
  const int32_t cache_base = p.cache_base[ring * p.s_cb];
  const int c = tid >> 4;
  const int ws = c & 3;
  const int chunk = tid & 15;
  const bool wlive = ws < S;
  const bool is_k = c >= kS;
  const int64_t wtoken = bos + (wlive ? ws : 0);
  const uint4 raw = load_u4((is_k ? p.k + wtoken * p.s_k : p.q + wtoken * p.s_q) + h * kD + chunk * 8);
  uint16_t v_raw[kS];
#pragma unroll
  for (int s = 0; s < kS; ++s)
    v_raw[s] = p.v[(bos + (s < S ? s : 0)) * p.s_v + hv * kD + tid];
  const uint16_t* state = p.h0 + slot * p.s_h0 + static_cast<int64_t>(hv) * kD * kD + (2 * warp * 16 + g) * kD + 8 * t;
  uint4 ra[2][4], rb[2][4];
  load_rows(ra[0], state, true);
  load_rows(ra[1], state + 8 * kD, true);

  if (flush != 0) return;
  const int r = min(max(write_pos, 0), kL);
  const int base = cache_base;
  const bool hist = warp < 2 && r > 0;
  if (hist) {
    const uint16_t* keys =
        (warp == 0 ? p.k_hi + ring * p.s_khi : p.k_lo + ring * p.s_klo) + static_cast<int64_t>(h) * kL * kD + 8 * t;
    load_rows(rb[0], keys + ((base + g) & (kL - 1)) * kD, g < r);
    load_rows(rb[1], keys + ((base + g + 8) & (kL - 1)) * kD, g + 8 < r);
  } else {
    load_rows(rb[0], state + 16 * kD, true);
    load_rows(rb[1], state + 24 * kD, true);
  }
  float* gates = p.g + ring * p.s_g + hv * kL;
  uint16_t* d_hi = p.d_hi + ring * p.s_dhi + static_cast<int64_t>(hv) * kL * kD + tid;
  uint16_t* d_lo = p.d_lo + ring * p.s_dlo + static_cast<int64_t>(hv) * kL * kD + tid;
  uint16_t dh_raw[kEarly], dl_raw[kEarly];
#pragma unroll
  for (int i = 0; i < kEarly; ++i) {
    const int phys = (base + i) & (kL - 1);
    dh_raw[i] = i < r ? d_hi[phys * kD] : 0;
    dl_raw[i] = i < r ? d_lo[phys * kD] : 0;
  }

  if (warp == 3) {
    const int s = lane & 3;
    const bool live = s < S;
    const int64_t token = bos + (live ? s : 0);
    const float x = bf_lo(p.a[token * p.s_a + hv]) + load_gate(p.dt_bias, hv, p.gate16);
    const float bv = bf_lo(p.b[token * p.s_b + hv]);
    const float rate = load_gate(p.A_log, hv, p.gate16);
    float gh[kL];
#pragma unroll
    for (int i = 0; i < kL; ++i)
      gh[i] = i < r ? gates[(base + i) & (kL - 1)] : 0.f;
    const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;
    const float gate = live ? -expf(rate) * sp : 0.f;
    const float beta = live ? 1.f / (1.f + expf(-bv)) : 0.f;
    const float G0 = __shfl_sync(kFull, gate, 0);
    const float G1 = G0 + __shfl_sync(kFull, gate, 1);
    const float G2 = G1 + __shfl_sync(kFull, gate, 2);
    const float G3 = G2 + __shfl_sync(kFull, gate, 3);
    float arg;
    if (lane < 4) {
      arg = pick(G0, G1, G2, G3, lane);
    } else if (lane < 10) {
      const int pi = (kPairRows >> (2 * (lane - 4))) & 3;
      const int pj = (kPairCols >> (2 * (lane - 4))) & 3;
      arg = pick(G0, G1, G2, G3, pi) - pick(G0, G1, G2, G3, pj);
    } else {
      const int j = lane - 11;
      arg = 0.f;
#pragma unroll
      for (int i = 0; i < kL; ++i) {
        if (i > j) arg += gh[i];
      }
    }
    sm.ex[lane] = expf(arg);
    if (lane < kS) {
      sm.gs[lane] = gate;
      sm.beta[lane] = beta;
    }
  }

  {
    const float x[8] = {
        bf_lo(raw.x), bf_hi(raw.x), bf_lo(raw.y), bf_hi(raw.y), bf_lo(raw.z), bf_hi(raw.z), bf_lo(raw.w), bf_hi(raw.w)};
    float ss = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e)
      ss += x[e] * x[e];
    ss += __shfl_xor_sync(kFull, ss, 8);
    ss += __shfl_xor_sync(kFull, ss, 4);
    ss += __shfl_xor_sync(kFull, ss, 2);
    ss += __shfl_xor_sync(kFull, ss, 1);
    float rn = wlive ? 1.f / sqrtf(ss + 1e-6f) : 0.f;
    if (!is_k) rn *= p.scale;
    uint32_t hi[4], lo[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const float y0 = x[2 * e] * rn;
      const float y1 = x[2 * e + 1] * rn;
      const uint32_t h0 = to_bf(y0);
      const uint32_t h1 = to_bf(y1);
      hi[e] = h0 | (h1 << 16);
      lo[e] = to_bf(y0 - bf_lo(h0)) | (to_bf(y1 - bf_lo(h1)) << 16);
    }
    const uint4 hi4 = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    const uint4 lo4 = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    *reinterpret_cast<uint4*>(sm.x[c] + chunk * 8) = hi4;
    *reinterpret_cast<uint4*>(sm.x[8 + c] + chunk * 8) = lo4;
    if (is_k && wlive && hv % p.hv_per_h == 0 && r + ws < kL) {
      const int64_t off = (static_cast<int64_t>(h) * kL + ((base + r + ws) & (kL - 1))) * kD + chunk * 8;
      *reinterpret_cast<uint4*>(p.k_hi + ring * p.s_khi + off) = hi4;
      *reinterpret_cast<uint4*>(p.k_lo + ring * p.s_klo + off) = lo4;
    }
  }
  __syncthreads();

  const uint16_t* c0 = sm.x[g] + 8 * t;
  const uint16_t* c1 = sm.x[8 + g] + 8 * t;
  {
    float acc[2][4] = {};
    tile_product(acc, ra[0], ra[1], c0, c1);
    store_rows(sm.hw, 2 * warp * 16 + g, t, acc);
  }
  if (hist) {
    load_rows(ra[0], state + 16 * kD, true);
    load_rows(ra[1], state + 24 * kD, true);
  }
  {
    float acc[2][4] = {};
    tile_product(acc, rb[0], rb[1], c0, c1);
    if (hist) {
      store_rows(warp == 0 ? sm.sc_hi : sm.sc_lo, g, t, acc);
    } else {
      store_rows(sm.hw, (2 * warp + 1) * 16 + g, t, acc);
    }
  }
  if (hist) {
    float acc[2][4] = {};
    tile_product(acc, ra[0], ra[1], c0, c1);
    store_rows(sm.hw, (2 * warp + 1) * 16 + g, t, acc);
  }
  if (warp == 2) {
    load_rows(ra[0], c0, true);
    load_rows(ra[1], c1, true);
    float acc[2][4] = {};
    tile_product(acc, ra[0], ra[1], c0, c1);
    *reinterpret_cast<float2*>(sm.gram[g] + 2 * t) = make_float2(acc[0][0], acc[0][1]);
    *reinterpret_cast<float2*>(sm.gram[g] + 8 + 2 * t) = make_float2(acc[1][0], acc[1][1]);
    *reinterpret_cast<float2*>(sm.gram[g + 8] + 2 * t) = make_float2(acc[0][2], acc[0][3]);
    *reinterpret_cast<float2*>(sm.gram[g + 8] + 8 + 2 * t) = make_float2(acc[1][2], acc[1][3]);
  }
  __syncthreads();

  {
    const int j = tid >> 3;
    if (j < r) sm.sc[j][tid & 7] = (sm.sc_hi[j][tid & 7] + sm.sc_lo[j][tid & 7]) * sm.ex[11 + j];
    if (tid < 16) {
      const int pi = (kPairRows >> (2 * tid)) & 3;
      const int pj = (kPairCols >> (2 * tid)) & 3;
      const int ra = 4 + pi;
      const int cb = (tid < 6 ? 4 : 0) + pj;
      const float dot = sm.gram[ra][cb] + sm.gram[ra][cb + 8] + sm.gram[ra + 8][cb] + sm.gram[ra + 8][cb + 8];
      float w;
      if (tid < 6) {
        w = sm.beta[pi] * sm.ex[4 + tid];
      } else {
        w = pi == pj ? 1.f : sm.ex[4 + pj * (pj - 1) / 2 + pi];
      }
      sm.af[tid] = w * dot;
    }
  }
  __syncthreads();

  const float4 a0 = *reinterpret_cast<const float4*>(sm.af);
  const float4 a1 = *reinterpret_cast<const float4*>(sm.af + 4);
  const float4 a2 = *reinterpret_cast<const float4*>(sm.af + 8);
  const float4 a3 = *reinterpret_cast<const float4*>(sm.af + 12);
  const float a10 = a0.x, a20 = a0.y, a21 = a0.z, a30 = a0.w, a31 = a1.x, a32 = a1.y;
  const float f00 = a1.z, f01 = a1.w, f02 = a2.x, f03 = a2.y, f11 = a2.z, f12 = a2.w;
  const float f13 = a3.x, f22 = a3.y, f23 = a3.z, f33 = a3.w;
  const float t10 = -a10;
  const float t21 = -a21;
  const float t32 = -a32;
  const float t20 = -a20 - a21 * t10;
  const float t31 = -a31 - a32 * t21;
  const float t30 = -a30 - a31 * t10 - a32 * t20;
  const float4 eG4 = *reinterpret_cast<const float4*>(sm.ex);
  const float4 beta4 = *reinterpret_cast<const float4*>(sm.beta);
  const float eG[kS] = {eG4.x, eG4.y, eG4.z, eG4.w};
  const float beta[kS] = {beta4.x, beta4.y, beta4.z, beta4.w};

  const float4 hq4 = *reinterpret_cast<const float4*>(sm.hw[tid]);
  const float4 hk4 = *reinterpret_cast<const float4*>(sm.hw[tid] + 4);
  float hq[kS] = {hq4.x, hq4.y, hq4.z, hq4.w};
  float hk[kS] = {hk4.x, hk4.y, hk4.z, hk4.w};
  if (r > 0) {
    const float decay = sm.ex[10];
#pragma unroll
    for (int s = 0; s < kS; ++s) {
      hq[s] *= decay;
      hk[s] *= decay;
    }
    for (int i = 0; i < r; ++i) {
      float d;
      if (i < kEarly) {
        d = bf_lo(dh_raw[i]) + bf_lo(dl_raw[i]);
      } else {
        const int phys = (base + i) & (kL - 1);
        d = bf_lo(d_hi[phys * kD]) + bf_lo(d_lo[phys * kD]);
      }
      const float4 sq = *reinterpret_cast<const float4*>(sm.sc[i]);
      const float4 sk = *reinterpret_cast<const float4*>(sm.sc[i] + 4);
      hq[0] = fmaf(d, sq.x, hq[0]);
      hq[1] = fmaf(d, sq.y, hq[1]);
      hq[2] = fmaf(d, sq.z, hq[2]);
      hq[3] = fmaf(d, sq.w, hq[3]);
      hk[0] = fmaf(d, sk.x, hk[0]);
      hk[1] = fmaf(d, sk.y, hk[1]);
      hk[2] = fmaf(d, sk.z, hk[2]);
      hk[3] = fmaf(d, sk.w, hk[3]);
    }
  }

  float R[kS];
#pragma unroll
  for (int s = 0; s < kS; ++s)
    R[s] = beta[s] * (bf_lo(v_raw[s]) - eG[s] * hk[s]);
  const float D[kS] = {
      R[0],
      R[0] * t10 + R[1],
      R[0] * t20 + R[1] * t21 + R[2],
      R[0] * t30 + R[1] * t31 + R[2] * t32 + R[3],
  };
  const float O[kS] = {
      eG[0] * hq[0] + D[0] * f00,
      eG[1] * hq[1] + D[0] * f01 + D[1] * f11,
      eG[2] * hq[2] + D[0] * f02 + D[1] * f12 + D[2] * f22,
      eG[3] * hq[3] + D[0] * f03 + D[1] * f13 + D[2] * f23 + D[3] * f33,
  };
#pragma unroll
  for (int s = 0; s < kS; ++s) {
    if (s < S) {
      out[s * p.s_o] = static_cast<uint16_t>(to_bf(O[s]));
      if (r + s < kL) {
        const int phys = (base + r + s) & (kL - 1);
        const uint32_t dh = to_bf(D[s]);
        d_hi[phys * kD] = static_cast<uint16_t>(dh);
        d_lo[phys * kD] = static_cast<uint16_t>(to_bf(D[s] - bf_lo(dh)));
        if (tid == 0) gates[phys] = sm.gs[s];
      }
    }
  }
}

}  // namespace gdn_replayssm_verify

template <bool kUsePDL>
struct GdnReplaySSMVerify {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView a,
      const tvm::ffi::TensorView b,
      const tvm::ffi::TensorView A_log,
      const tvm::ffi::TensorView dt_bias,
      const tvm::ffi::TensorView o,
      const tvm::ffi::TensorView h0,
      const tvm::ffi::TensorView d_hi,
      const tvm::ffi::TensorView d_lo,
      const tvm::ffi::TensorView k_hi,
      const tvm::ffi::TensorView k_lo,
      const tvm::ffi::TensorView g,
      const tvm::ffi::TensorView qsl,
      const tvm::ffi::TensorView state_idx,
      const tvm::ffi::TensorView ring_idx,
      const tvm::ffi::TensorView write_pos,
      const tvm::ffi::TensorView cache_base,
      const tvm::ffi::TensorView is_flush,
      double scale,
      int64_t null_block_id) {
    using namespace host;
    using namespace gdn_replayssm_verify;
    auto T_ = SymbolicSize{"tokens"};
    auto H_ = SymbolicSize{"key_heads"};
    auto HV_ = SymbolicSize{"value_heads"};
    auto Slots_ = SymbolicSize{"state_slots"};
    auto Rings_ = SymbolicSize{"ring_rows"};
    auto B_ = SymbolicSize{"rows"};
    auto B1_ = SymbolicSize{"rows_plus_one"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({T_, H_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, kD, 1}).verify(q);
    TensorMatcher({T_, H_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, kD, 1}).verify(k);
    TensorMatcher({T_, HV_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, kD, 1}).verify(v);
    TensorMatcher({T_, HV_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(a);
    TensorMatcher({T_, HV_}).with_dtype<bf16_t>().with_device(device).with_strides({-1, 1}).verify(b);
    TensorMatcher({HV_}).with_dtype<fp32_t, bf16_t>().with_device(device).with_strides({1}).verify(A_log);
    TensorMatcher({HV_}).with_dtype<fp32_t, bf16_t>().with_device(device).with_strides({1}).verify(dt_bias);
    TensorMatcher({T_, HV_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, kD, 1}).verify(o);
    TensorMatcher({Slots_, HV_, kD, kD})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, kD * kD, kD, 1})
        .verify(h0);
    TensorMatcher({Rings_, HV_, kL, kD})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, kL * kD, kD, 1})
        .verify(d_hi)
        .verify(d_lo);
    TensorMatcher({Rings_, H_, kL, kD})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, kL * kD, kD, 1})
        .verify(k_hi)
        .verify(k_lo);
    TensorMatcher({Rings_, HV_, kL}).with_dtype<fp32_t>().with_device(device).with_strides({-1, kL, 1}).verify(g);
    TensorMatcher({B1_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({-1}).verify(qsl);
    TensorMatcher({B_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({-1}).verify(state_idx);
    TensorMatcher({B_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({-1}).verify(ring_idx);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).with_strides({-1}).verify(write_pos);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).with_strides({-1}).verify(cache_base);
    TensorMatcher({-1}).with_dtype<int8_t>().with_device(device).with_strides({-1}).verify(is_flush);
    const int64_t rows = B_.unwrap();
    const int64_t num_h = H_.unwrap();
    const int64_t num_hv = HV_.unwrap();
    RuntimeCheck(B1_.unwrap() == rows + 1, "query_start_loc needs rows + 1 entries");
    RuntimeCheck(num_h > 0 && num_hv % num_h == 0, "value heads must be a multiple of key heads");
    RuntimeCheck(A_log.dtype().bits == dt_bias.dtype().bits, "A_log and dt_bias must share a dtype");
    const auto aligned = [](const tvm::ffi::TensorView t) {
      return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && t.stride(0) % 8 == 0;
    };
    RuntimeCheck(
        aligned(q) && aligned(k) && aligned(h0) && aligned(k_hi) && aligned(k_lo),
        "q, k, the checkpoint and the key rings must be 16-byte aligned per row");
    if (rows == 0) return;
    const auto params = Params{
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .k = static_cast<const uint16_t*>(k.data_ptr()),
        .v = static_cast<const uint16_t*>(v.data_ptr()),
        .a = static_cast<const uint16_t*>(a.data_ptr()),
        .b = static_cast<const uint16_t*>(b.data_ptr()),
        .A_log = A_log.data_ptr(),
        .dt_bias = dt_bias.data_ptr(),
        .o = static_cast<uint16_t*>(o.data_ptr()),
        .h0 = static_cast<const uint16_t*>(h0.data_ptr()),
        .d_hi = static_cast<uint16_t*>(d_hi.data_ptr()),
        .d_lo = static_cast<uint16_t*>(d_lo.data_ptr()),
        .k_hi = static_cast<uint16_t*>(k_hi.data_ptr()),
        .k_lo = static_cast<uint16_t*>(k_lo.data_ptr()),
        .g = static_cast<float*>(g.data_ptr()),
        .qsl = qsl.data_ptr(),
        .state_idx = state_idx.data_ptr(),
        .ring_idx = ring_idx.data_ptr(),
        .write_pos = static_cast<const int32_t*>(write_pos.data_ptr()),
        .cache_base = static_cast<const int32_t*>(cache_base.data_ptr()),
        .is_flush = static_cast<const int8_t*>(is_flush.data_ptr()),
        .s_q = q.stride(0),
        .s_k = k.stride(0),
        .s_v = v.stride(0),
        .s_a = a.stride(0),
        .s_b = b.stride(0),
        .s_o = o.stride(0),
        .s_h0 = h0.stride(0),
        .s_dhi = d_hi.stride(0),
        .s_dlo = d_lo.stride(0),
        .s_khi = k_hi.stride(0),
        .s_klo = k_lo.stride(0),
        .s_g = g.stride(0),
        .s_qsl = qsl.stride(0),
        .s_si = state_idx.stride(0),
        .s_ri = ring_idx.stride(0),
        .s_wp = write_pos.stride(0),
        .s_cb = cache_base.stride(0),
        .s_fl = is_flush.stride(0),
        .scale = static_cast<float>(scale),
        .hv_per_h = static_cast<int32_t>(num_hv / num_h),
        .null_block_id = static_cast<int32_t>(null_block_id),
        .qsl64 = qsl.dtype().bits == 64,
        .si64 = state_idx.dtype().bits == 64,
        .ri64 = ring_idx.dtype().bits == 64,
        .gate16 = A_log.dtype().bits == 16,
    };
    const dim3 grid(static_cast<uint32_t>(num_hv), static_cast<uint32_t>(rows));
    LaunchKernel(grid, kThreads, device.unwrap()).enable_pdl(kUsePDL)(verify_kernel<kUsePDL>, params);
  }
};

}  // namespace sglang
