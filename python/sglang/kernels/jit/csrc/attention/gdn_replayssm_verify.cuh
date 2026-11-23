// GDN ReplaySSM speculative verify on SM100 over a bf16 checkpoint and the compact (d, k, g) ring: outputs of up
// to 4 draft tokens per request and this window's ring records. One CTA per (request, key head), TMA to tcgen05.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace gdn_replayssm_verify {

using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kD = 128;
constexpr int kS = 4;
constexpr int kL = 16;
constexpr int kCols = 16;                // operand columns: q hi, k hi, q lo, k lo of the 4 tokens
constexpr int kWorkers = 128;            // thread = checkpoint row = TMEM lane
constexpr int kThreads = kWorkers + 32;  // and the warp that issues the copies and the MMAs
constexpr int kEarly = 8;
constexpr int kPlane = kD * 128;        // 128 checkpoint rows x 64 K
constexpr int kColPlane = kCols * 128;  // 16 operand columns x 64 K
constexpr int kTmemCols = 4 * kCols;    // checkpoint, key history hi, lo, Gram
constexpr int kSmemPerSM = 233472;
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
  CUtensorMap h0_map;     // bf16 [slots, HV * 128, 128] in tiles of 128 rows x 64 K, 128 B swizzle
  const uint16_t* q;      // bf16 [tokens, H, 128]
  const uint16_t* k;      // bf16 [tokens, H, 128]
  const uint16_t* v;      // bf16 [tokens, HV, 128]
  const uint16_t* a;      // bf16 [tokens, HV]
  const uint16_t* b;      // bf16 [tokens, HV]
  const void* A_log;      // fp32 or bf16 [HV]
  const void* dt_bias;    // fp32 or bf16 [HV]
  uint16_t* o;            // bf16 [tokens, HV, 128]
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
  int64_t s_q, s_k, s_v, s_a, s_b, s_o, s_dhi, s_dlo, s_khi, s_klo, s_g;
  int64_t s_qsl, s_si, s_ri, s_wp, s_cb, s_fl;
  float scale;
  int32_t hv_per_h, null_block_id;
  bool qsl64, si64, ri64, alog16, dtb16;
};

struct Layout {
  static constexpr int aux = 0;                  // key history hi, lo [2][2 K halves][16][128 B]; then the tables
  static constexpr int sc = aux;                 // f32 [16][8] decayed key-history scores
  static constexpr int coef = sc + kL * 8 * 4;   // f32 [4][8] rows of A and F
  static constexpr int ex = coef + kS * 8 * 4;   // f32 [4] decays, [4] betas, total decay, [4] gates
  static constexpr int tables = 1024;            // after the key products: two table buffers in aux
  static constexpr int b = aux + 4 * kColPlane;  // operand columns [2 K halves][16][128 B]
  static constexpr int a = b + 2 * kColPlane;    // checkpoint [2 K halves][128 rows][128 B]
  static constexpr int bar = a + 2 * kPlane;     // mbarriers: checkpoint in, product done, small products done
  static constexpr int tmem = bar + 24;
  static constexpr int bytes = tmem + 8;
  static constexpr int per_sm = kSmemPerSM / (bytes + 1024);
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

SGL_DEVICE float gate_value(const uint32_t raw, const bool is16) {
  return is16 ? bf_lo(raw) : __uint_as_float(raw);
}

SGL_DEVICE uint4 load_u4(const uint16_t* p) {
  return *reinterpret_cast<const uint4*>(p);
}

SGL_DEVICE float pick(const float x0, const float x1, const float x2, const float x3, const int i) {
  return i == 0 ? x0 : i == 1 ? x1 : i == 2 ? x2 : x3;
}

SGL_DEVICE bool mbar_test_parity(uint64_t* bar, const uint32_t parity) {
  uint32_t done;
  asm volatile("{\n\t.reg .pred p;\n\tmbarrier.test_wait.parity.shared.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
               : "=r"(done)
               : "r"(device::ptx::to_shared(bar)), "r"(parity)
               : "memory");
  return done != 0;
}

// D[m rows of A][16 operand columns] over K = 128, into TMEM columns t_cols
SGL_DEVICE void issue_product(
    const uint32_t t_cols,
    const tcgen05::SmemDesc adesc,
    const uint32_t a_plane,
    const uint32_t m,
    const tcgen05::SmemDesc bdesc) {
  const uint32_t idesc = tcgen05::make_instr_desc_f16(m, kCols, false, false) | (1u << 7) | (1u << 10);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    tcgen05::mma_f16_ss(
        t_cols,
        adesc.advance_bytes((i >> 2) * a_plane + (i & 3) * 32).bits,
        bdesc.advance_bytes((i >> 2) * kColPlane + (i & 3) * 32).bits,
        idesc,
        i > 0);
  }
}

// the two K halves of value head hv of checkpoint `slot` into the operand tile at s_a, completing on bar
SGL_DEVICE void load_checkpoint(const Params& p, uint64_t* bar, const uint32_t s_a, const int hv, const int64_t slot) {
  device::ptx::mbar_arrive_expect_tx(bar, 2 * kPlane);
  device::ptx::cp_async_bulk_tensor_3d(s_a, &p.h0_map, 0, hv * kD, static_cast<int32_t>(slot), bar);
  device::ptx::cp_async_bulk_tensor_3d(s_a + kPlane, &p.h0_map, kD / 2, hv * kD, static_cast<int32_t>(slot), bar);
}

// this thread's v of the window and its first kEarly ring d records of value head hv, as loaded
SGL_DEVICE void load_head(
    const Params& p,
    const int64_t ring,
    const int64_t bos,
    const int S,
    const int r,
    const int base,
    const int hv,
    const bool on,
    uint16_t (&v)[kS],
    uint16_t (&d_hi)[kEarly],
    uint16_t (&d_lo)[kEarly]) {
  const int tid = threadIdx.x;
  const int64_t hv_off = static_cast<int64_t>(hv) * kL * kD + tid;
  const uint16_t* hi = p.d_hi + ring * p.s_dhi + hv_off;
  const uint16_t* lo = p.d_lo + ring * p.s_dlo + hv_off;
#pragma unroll
  for (int s = 0; s < kS; ++s) {
    if (on) v[s] = p.v[(bos + (s < S ? s : 0)) * p.s_v + hv * kD + tid];
  }
#pragma unroll
  for (int i = 0; i < kEarly; ++i) {
    const int phys = (base + i) & (kL - 1);
    if (on && i < r) {
      d_hi[i] = hi[phys * kD];
      d_lo[i] = lo[phys * kD];
    }
  }
}

// lane's gate inputs of value head hv as loaded: a and b of token lane & 3, the head's rates, ring gate `lane`
struct GateInputs {
  uint16_t a, b;
  uint32_t rate, bias;
  float g;
};

SGL_DEVICE GateInputs load_gates(
    const Params& p,
    const int64_t ring,
    const int64_t token,
    const int r,
    const int base,
    const int hv,
    const bool on) {
  const int lane = threadIdx.x & 31;
  GateInputs in{};
  if (on) {
    in.a = p.a[token * p.s_a + hv];
    in.b = p.b[token * p.s_b + hv];
    in.rate = p.alog16 ? static_cast<const uint16_t*>(p.A_log)[hv] : static_cast<const uint32_t*>(p.A_log)[hv];
    in.bias = p.dtb16 ? static_cast<const uint16_t*>(p.dt_bias)[hv] : static_cast<const uint32_t*>(p.dt_bias)[hv];
    if (lane < r) in.g = p.g[ring * p.s_g + hv * kL + ((base + lane) & (kL - 1))];
  }
  return in;
}

SGL_DEVICE float sum8(float x) {
  x += __shfl_xor_sync(kFull, x, 4);
  x += __shfl_xor_sync(kFull, x, 2);
  x += __shfl_xor_sync(kFull, x, 1);
  return x;
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, Layout::per_sm) verify_kernel(const __grid_constant__ Params p) {
  using Lay = Layout;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const bool issuer = warp == kWorkers / 32;
  const int h = blockIdx.x;
  const int n = blockIdx.y;
  const int heads = p.hv_per_h;
  const int hv0 = h * heads;
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  uint32_t* const s_tmem = reinterpret_cast<uint32_t*>(smem + Lay::tmem);

  if (issuer) {
    tcgen05::alloc(s_tmem, kTmemCols);
    tcgen05::relinquish_alloc_permit();
  }
  device::PDLWaitPrimary<kUsePDL>();

  const int64_t slot = load_index(p.state_idx, n * p.s_si, p.si64);
  const int64_t bos = load_index(p.qsl, n * p.s_qsl, p.qsl64);
  const int64_t eos = load_index(p.qsl, (n + 1) * p.s_qsl, p.qsl64);
  const int64_t ring = load_index(p.ring_idx, n * p.s_ri, p.ri64);
  const int S = static_cast<int>(min(eos - bos, static_cast<int64_t>(kS)));
  if (slot <= p.null_block_id) {
    if (issuer) {
      tcgen05::dealloc(*s_tmem, kTmemCols);
    } else {
      for (int j = 0; j < heads; ++j) {
#pragma unroll
        for (int s = 0; s < kS; ++s) {
          if (s < S) p.o[(bos + s) * p.s_o + (hv0 + j) * kD + tid] = 0;
        }
      }
    }
    return;
  }
  const uint32_t s_a = device::ptx::to_shared(smem + Lay::a);
  if (issuer && lane == 0) {
    device::ptx::mbar_init(s_bar, 1);
    device::ptx::mbar_init(s_bar + 1, 1);
    device::ptx::mbar_init(s_bar + 2, 1);
    device::ptx::fence_barrier_init();
    load_checkpoint(p, s_bar, s_a, hv0, slot);
  }
  const int8_t flush = p.is_flush[ring * p.s_fl];
  const int r = min(max(p.write_pos[ring * p.s_wp], 0), kL);
  const int base = p.cache_base[ring * p.s_cb];
  if (flush != 0) {
    if (issuer) {
      if (lane == 0) device::ptx::mbar_wait_parity(s_bar, 0);
      tcgen05::dealloc(*s_tmem, kTmemCols);
    }
    return;
  }

  if (issuer) {
    const int s = lane & 3;
    const bool live = s < S;
    const int64_t token = bos + (live ? s : 0);
    const auto adesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::a);
    const auto bdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::b);
    float kx[2 * kS], sc[2 * kS];
    GateInputs next = load_gates(p, ring, token, r, base, hv0, true);
#pragma unroll 1
    for (int j = 0; j < heads; ++j) {
      const int hv = hv0 + j;
      const GateInputs in = next;
      next = load_gates(p, ring, token, r, base, hv + 1, j + 1 < heads);
      if (j == 0) {
        tcgen05::fence_before_thread_sync();
        device::ptx::bar_sync(1, kThreads);
        tcgen05::fence_after_thread_sync();
      } else if (j >= 2) {
        if (j & 1) {
          device::ptx::bar_sync(4, kThreads);
        } else {
          device::ptx::bar_sync(3, kThreads);
        }
      }
      float* const s_sc = reinterpret_cast<float*>(smem + Lay::aux + (j & 1) * Lay::tables);
      float* const s_coef = s_sc + kL * 8;
      float* const s_ex = s_coef + kS * 8;
      const uint32_t t_d = *s_tmem;
      const float x = bf_lo(in.a) + gate_value(in.bias, p.dtb16);
      const float sp = x <= 20.f ? logf(1.f + expf(x)) : x;
      const float gate = live ? -expf(gate_value(in.rate, p.alog16)) * sp : 0.f;
      const float beta = live ? 1.f / (1.f + expf(-bf_lo(in.b))) : 0.f;
      const float G0 = __shfl_sync(kFull, gate, 0);
      const float G1 = G0 + __shfl_sync(kFull, gate, 1);
      const float G2 = G1 + __shfl_sync(kFull, gate, 2);
      const float G3 = G2 + __shfl_sync(kFull, gate, 3);
      float before = in.g;
#pragma unroll
      for (int step = 1; step < kL; step *= 2) {
        const float up = __shfl_up_sync(kFull, before, step);
        if (lane >= step) before += up;
      }
      const float total = __shfl_sync(kFull, before, kL - 1);
      float arg = total;
      if (lane < kL) {
        arg = total - before;
      } else if (lane < kL + 4) {
        arg = pick(G0, G1, G2, G3, lane - kL);
      } else if (lane < kL + 10) {
        const int pi = (kPairRows >> (2 * (lane - kL - 4))) & 3;
        const int pj = (kPairCols >> (2 * (lane - kL - 4))) & 3;
        arg = pick(G0, G1, G2, G3, pi) - pick(G0, G1, G2, G3, pj);
      }
      const float e = expf(arg);
      const float eG = __shfl_sync(kFull, e, kL + s);
      const float decay = __shfl_sync(kFull, e, kL + 10);
      float pe[kS];
#pragma unroll
      for (int m = 0; m < kS; ++m) {
        const int hi = max(s, m);
        const int lo = min(s, m);
        pe[m] = __shfl_sync(kFull, e, kL + 4 + hi * (hi - 1) / 2 + lo);
      }
      if (j == 0) {
        device::ptx::mbar_wait_parity(s_bar + 2, 0);
        tcgen05::fence_after_thread_sync();
#pragma unroll
        for (int block = 3; block > 0; --block) {
          float t[kCols];
          if (block == 3 || r > 0) tcgen05::ld_32x32b<kCols>(t_d + block * kCols, t);
          tcgen05::wait_ld();
#pragma unroll
          for (int m = 0; m < 2 * kS; ++m) {
            const float sum = t[m] + t[2 * kS + m];
            if (block == 3) {
              kx[m] = sum + __shfl_xor_sync(kFull, sum, 8);
            } else if (block == 2) {
              sc[m] = sum;
            } else {
              sc[m] = r > 0 ? sc[m] + sum : 0.f;
            }
          }
        }
      }
      if (r > 0 && lane < kL) {
        *reinterpret_cast<float4*>(s_sc + lane * 8) = make_float4(sc[0] * e, sc[1] * e, sc[2] * e, sc[3] * e);
        *reinterpret_cast<float4*>(s_sc + lane * 8 + 4) = make_float4(sc[4] * e, sc[5] * e, sc[6] * e, sc[7] * e);
      }
      if (lane >= kS && lane < 2 * kS) {
        float af[2 * kS];
#pragma unroll
        for (int m = 0; m < kS; ++m) {
          af[m] = m < s ? beta * pe[m] * kx[kS + m] : 0.f;
          af[kS + m] = m < s ? 0.f : m == s ? kx[m] : pe[m] * kx[m];
        }
        *reinterpret_cast<float4*>(s_coef + s * 8) = make_float4(af[0], af[1], af[2], af[3]);
        *reinterpret_cast<float4*>(s_coef + s * 8 + 4) = make_float4(af[4], af[5], af[6], af[7]);
      }
      if (lane < kS) {
        s_ex[lane] = eG;
        s_ex[kS + lane] = beta;
        s_ex[2 * kS + lane] = decay;
        s_ex[3 * kS + lane] = gate;
      }
      if (j >= 1) {
        device::ptx::bar_sync(5, kThreads);
        tcgen05::fence_after_thread_sync();
      }
      if (lane == 0) {
        device::ptx::mbar_wait_parity(s_bar, j & 1);
        issue_product(t_d, adesc, kPlane, kD, bdesc);
        tcgen05::commit(s_bar + 1);
      }
      __syncwarp();
      device::ptx::bar_arrive(2, kThreads);
      if (lane == 0 && j + 1 < heads) {
        device::ptx::mbar_wait_parity(s_bar + 1, j & 1);
        load_checkpoint(p, s_bar, s_a, hv + 1, slot);
      }
    }
    device::ptx::bar_sync(5, kThreads);
    device::ptx::bar_sync(3, kThreads);
    if (heads >= 2) device::ptx::bar_sync(4, kThreads);
    tcgen05::fence_after_thread_sync();
    tcgen05::dealloc(*s_tmem, kTmemCols);
    return;
  }

  const int c = tid >> 4;
  const int ws = c & 3;
  const int chunk = tid & 15;
  const int row = tid >> 3;
  const int part = tid & 7;
  const bool wlive = ws < S;
  const bool is_k = c >= kS;
  const int64_t wtoken = bos + (wlive ? ws : 0);
  const uint4 raw = load_u4((is_k ? p.k + wtoken * p.s_k : p.q + wtoken * p.s_q) + h * kD + chunk * 8);
  uint16_t v_raw[kS] = {}, dh_raw[kEarly] = {}, dl_raw[kEarly] = {};
  load_head(p, ring, bos, S, r, base, hv0, true, v_raw, dh_raw, dl_raw);
  uint4 key_hi[2], key_lo[2];
  {
    const int64_t off = (static_cast<int64_t>(h) * kL + ((base + row) & (kL - 1))) * kD + part * 16;
    const uint16_t* hi = p.k_hi + ring * p.s_khi + off;
    const uint16_t* lo = p.k_lo + ring * p.s_klo + off;
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      key_hi[e] = row < r ? load_u4(hi + 8 * e) : make_uint4(0, 0, 0, 0);
      key_lo[e] = row < r ? load_u4(lo + 8 * e) : make_uint4(0, 0, 0, 0);
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
    ss = sum8(ss);
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
    uint8_t* const col = smem + Lay::b + (chunk >> 3) * kColPlane;
    *reinterpret_cast<uint4*>(col + tcgen05::sw128_off(c, chunk & 7)) = hi4;
    *reinterpret_cast<uint4*>(col + tcgen05::sw128_off(2 * kS + c, chunk & 7)) = lo4;
    if (is_k && wlive && r + ws < kL) {
      const int64_t off = (static_cast<int64_t>(h) * kL + ((base + r + ws) & (kL - 1))) * kD + chunk * 8;
      *reinterpret_cast<uint4*>(p.k_hi + ring * p.s_khi + off) = hi4;
      *reinterpret_cast<uint4*>(p.k_lo + ring * p.s_klo + off) = lo4;
    }
  }
  if (r > 0) {
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      const int c16 = 2 * part + e;
      uint8_t* const dst = smem + Lay::aux + (c16 >> 3) * kColPlane + tcgen05::sw128_off(row, c16 & 7);
      *reinterpret_cast<uint4*>(dst) = key_hi[e];
      *reinterpret_cast<uint4*>(dst + 2 * kColPlane) = key_lo[e];
    }
  }
  device::ptx::fence_proxy_async_shared();
  device::ptx::bar_sync(1, kThreads);
  tcgen05::fence_after_thread_sync();
  const uint32_t t_d = *s_tmem + (static_cast<uint32_t>(warp * 32) << 16);
  if (tid == 0) {
    const auto bdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::b);
    issue_product(t_d + 3 * kCols, bdesc, kColPlane, 64, bdesc);
    if (r > 0) {
      const auto kdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::aux);
      issue_product(t_d + kCols, kdesc, kColPlane, 64, bdesc);
      issue_product(t_d + 2 * kCols, kdesc.advance_bytes(2 * kColPlane), kColPlane, 64, bdesc);
    }
    tcgen05::commit(s_bar + 2);
  }

#pragma unroll 1
  for (int j = 0; j < heads; ++j) {
    const int hv = hv0 + j;
    device::ptx::bar_sync(2, kThreads);
    tcgen05::fence_after_thread_sync();
    const float* const s_sc = reinterpret_cast<const float*>(smem + Lay::aux + (j & 1) * Lay::tables);
    const float* const s_coef = s_sc + kL * 8;
    const float* const s_ex = s_coef + kS * 8;
    const float4 c0 = *reinterpret_cast<const float4*>(s_coef + 4);
    const float4 a1 = *reinterpret_cast<const float4*>(s_coef + 8);
    const float4 c1 = *reinterpret_cast<const float4*>(s_coef + 12);
    const float4 a2 = *reinterpret_cast<const float4*>(s_coef + 16);
    const float4 c2 = *reinterpret_cast<const float4*>(s_coef + 20);
    const float4 a3 = *reinterpret_cast<const float4*>(s_coef + 24);
    const float4 c3 = *reinterpret_cast<const float4*>(s_coef + 28);
    const float4 eG4 = *reinterpret_cast<const float4*>(s_ex);
    const float4 beta4 = *reinterpret_cast<const float4*>(s_ex + kS);
    const float eG[kS] = {eG4.x, eG4.y, eG4.z, eG4.w};
    const float beta[kS] = {beta4.x, beta4.y, beta4.z, beta4.w};
    const float a10 = a1.x, a20 = a2.x, a21 = a2.y, a30 = a3.x, a31 = a3.y, a32 = a3.z;
    const float f00 = c0.x, f01 = c0.y, f02 = c0.z, f03 = c0.w, f11 = c1.y;
    const float f12 = c1.z, f13 = c1.w, f22 = c2.z, f23 = c2.w, f33 = c3.w;
    const float t10 = -a10;
    const float t21 = -a21;
    const float t32 = -a32;
    const float t20 = -a20 - a21 * t10;
    const float t31 = -a31 - a32 * t21;
    const float t30 = -a30 - a31 * t10 - a32 * t20;

    device::ptx::mbar_wait_parity(s_bar + 1, j & 1);
    tcgen05::fence_after_thread_sync();
    float hw[kCols];
    tcgen05::ld_32x32b<kCols>(t_d, hw);
    tcgen05::wait_ld();
    tcgen05::fence_before_thread_sync();
    device::ptx::bar_arrive(5, kThreads);
    float hq[kS], hk[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s) {
      hq[s] = hw[s] + hw[2 * kS + s];
      hk[s] = hw[kS + s] + hw[3 * kS + s];
    }
    const int64_t hv_off = static_cast<int64_t>(hv) * kL * kD + tid;
    uint16_t* const d_hi = p.d_hi + ring * p.s_dhi + hv_off;
    uint16_t* const d_lo = p.d_lo + ring * p.s_dlo + hv_off;
    if (r > 0) {
      const float decay = s_ex[2 * kS];
#pragma unroll
      for (int s = 0; s < kS; ++s) {
        hq[s] *= decay;
        hk[s] *= decay;
      }
      for (int i = 0; i < r; ++i) {
        float d;
        if (i < kEarly) {
          uint32_t hi = dh_raw[0];
          uint32_t lo = dl_raw[0];
#pragma unroll
          for (int e = 1; e < kEarly; ++e) {
            if (i == e) {
              hi = dh_raw[e];
              lo = dl_raw[e];
            }
          }
          d = bf_lo(hi) + bf_lo(lo);
        } else {
          const int phys = (base + i) & (kL - 1);
          d = bf_lo(d_hi[phys * kD]) + bf_lo(d_lo[phys * kD]);
        }
        const float4 sq = *reinterpret_cast<const float4*>(s_sc + i * 8);
        const float4 sk = *reinterpret_cast<const float4*>(s_sc + i * 8 + 4);
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
    load_head(p, ring, bos, S, r, base, hv + 1, j + 1 < heads, v_raw, dh_raw, dl_raw);
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
    uint16_t* const out = p.o + bos * p.s_o + hv * kD + tid;
#pragma unroll
    for (int s = 0; s < kS; ++s) {
      if (s < S) {
        out[s * p.s_o] = static_cast<uint16_t>(to_bf(O[s]));
        if (r + s < kL) {
          const int phys = (base + r + s) & (kL - 1);
          const uint32_t dh = to_bf(D[s]);
          d_hi[phys * kD] = static_cast<uint16_t>(dh);
          d_lo[phys * kD] = static_cast<uint16_t>(to_bf(D[s] - bf_lo(dh)));
          if (tid == 0) p.g[ring * p.s_g + hv * kL + phys] = s_ex[3 * kS + s];
        }
      }
    }
    if (j & 1) {
      device::ptx::bar_arrive(4, kThreads);
    } else {
      device::ptx::bar_arrive(3, kThreads);
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
    const int64_t slots = Slots_.unwrap();
    RuntimeCheck(B1_.unwrap() == rows + 1, "query_start_loc needs rows + 1 entries");
    RuntimeCheck(num_h > 0 && num_hv % num_h == 0, "value heads must be a multiple of key heads");
    const auto aligned = [](const tvm::ffi::TensorView t) {
      return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0 && t.stride(0) % 8 == 0;
    };
    RuntimeCheck(
        aligned(q) && aligned(k) && aligned(h0) && aligned(k_hi) && aligned(k_lo),
        "q, k, the checkpoint and the key rings must be 16-byte aligned per row");
    if (rows == 0 || slots == 0) return;
    const auto params = Params{
        .h0_map = tma::make_map(
            h0.data_ptr(),
            CU_TENSOR_MAP_DATA_TYPE_UINT16,
            {kD, static_cast<cuuint64_t>(num_hv * kD), static_cast<cuuint64_t>(slots)},
            {kD * 2, static_cast<cuuint64_t>(h0.stride(0) * 2)},
            {kD / 2, kD, 1},
            {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B}),
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .k = static_cast<const uint16_t*>(k.data_ptr()),
        .v = static_cast<const uint16_t*>(v.data_ptr()),
        .a = static_cast<const uint16_t*>(a.data_ptr()),
        .b = static_cast<const uint16_t*>(b.data_ptr()),
        .A_log = A_log.data_ptr(),
        .dt_bias = dt_bias.data_ptr(),
        .o = static_cast<uint16_t*>(o.data_ptr()),
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
        .alog16 = A_log.dtype().bits == 16,
        .dtb16 = dt_bias.dtype().bits == 16,
    };
    const DLDevice dev = device.unwrap();
    constexpr int smem = Layout::bytes;
    runtime::set_max_dynamic_smem_per_device<verify_kernel<kUsePDL>>(dev, smem);
    const dim3 grid(static_cast<uint32_t>(num_h), static_cast<uint32_t>(rows));
    LaunchKernel(grid, kThreads, dev, smem).enable_pdl(kUsePDL)(verify_kernel<kUsePDL>, params);
  }
};

}  // namespace sglang
