// QSA prefill sparse attention on SM100 (tcgen05) over packed fp8 K and bf16 V: a union kernel merges the top-k rows
// of kRows consecutive query rows into one list with a row mask, an attention kernel attends it once per KV head.
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include "qsa_sparse_decode_sm100.cuh"

namespace sglang {

namespace qsa_sparse_prefill_sm100 {

using qsa_sparse_decode::cp_async16;
using qsa_sparse_decode::cp_async_commit;
using qsa_sparse_decode::cp_async_wait;
using qsa_sparse_decode::ex2_approx;
using qsa_sparse_decode::kD;
using qsa_sparse_decode_sm100::bf16x2_e4m3_terms;
using qsa_sparse_decode_sm100::kMaxSmem;
using qsa_sparse_decode_sm100::kPlane;
using qsa_sparse_decode_sm100::kTile;
using qsa_sparse_decode_sm100::kWorkers;
using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kBlock = kWorkers + 64;  // the workers, the warp that issues the QK MMAs and the one of the PV MMAs
constexpr int kSync = kWorkers + 32;   // a named barrier joins the workers and one issuing warp
constexpr int kRing = 8;               // tiles of entries in shared memory
constexpr float kMove = 40.f;          // log2 distance of a score above its head's reference that moves the reference
constexpr int kUnionThreads = 512;
constexpr int kMeta = 4;  // per group: slots, first row, rows, sequence

struct UnionParams {
  const int32_t* idx;     // [R, TOPK] logical positions, -1 = hole
  const int32_t* cu_q;    // [S + 1] first row of each sequence
  const int32_t* cu_k;    // [S + 1] first packed K / V row of each sequence
  const int32_t* kv_len;  // [S] sequence length with its rows
  int32_t* slots;         // [G, cap] packed K / V rows
  uint8_t* masks;         // [G, cap]
  int32_t* meta;          // [G, kMeta]
  int64_t stride_idx;
  int32_t topk, cap, nseq, map_words;
};

struct Params {
  const uint16_t* q;  // bf16 [R, HQ, D]
  const uint8_t* k;   // fp8 e4m3 [rows, HKV, D]
  const uint8_t* v;   // bf16 [rows, HKV, D]
  const int32_t* slots;
  const uint8_t* masks;
  const int32_t* meta;
  uint16_t* out;  // bf16 [R, HQ, D]
  int64_t stride_qr, stride_qh, stride_krow, stride_kh, stride_vrow, stride_vh, stride_or, stride_oh;
  float qk_scale;
  int32_t cap, HKV;
};

SGL_DEVICE uint32_t f32x2_bf16x2(const float lo, const float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// true on every thread of the warpgroup that waits on barrier `id` when `pred` holds on one of them
SGL_DEVICE bool any_in_warpgroup(const int id, const bool pred) {
  uint32_t r;
  asm volatile(
      "{\n .reg .pred p, q;\n setp.ne.u32 q, %1, 0;\n bar.red.or.pred p, %2, 128, q;\n selp.u32 %0, 1, 0, p;\n}"
      : "=r"(r)
      : "r"(static_cast<uint32_t>(pred)), "r"(id)
      : "memory");
  return r != 0;
}

// bit 8 i set when byte i of `w` is not zero
SGL_DEVICE uint32_t used_bytes(const uint32_t w) {
  return ((((w & 0x7F7F7F7Fu) + 0x7F7F7F7Fu) | w) & 0x80808080u) >> 7;
}

// Shared memory, in words: [map_words] a row mask byte per position; [40] scratch.
template <int kRows>
__global__ void __launch_bounds__(kUnionThreads) union_kernel(const __grid_constant__ UnionParams p) {
  extern __shared__ __align__(16) uint32_t smem_union[];
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = blockIdx.x;
  uint32_t* const s_map = smem_union;
  uint32_t* const s_aux = s_map + p.map_words;  // [32] warp sums, twice, count, sequence, its first group

  if (tid == 0) {
    int s = 0;
    int g0 = 0;
    while (s + 1 < p.nseq) {
      const int next = g0 + (p.cu_q[s + 1] - p.cu_q[s] + kRows - 1) / kRows;
      if (group < next) break;
      g0 = next;
      ++s;
    }
    s_aux[32] = 0;
    s_aux[33] = 0;
    s_aux[34] = static_cast<uint32_t>(s);
    s_aux[35] = static_cast<uint32_t>(g0);
  }
  __syncthreads();
  const int s = static_cast<int>(s_aux[34]);
  const int q0 = p.cu_q[s];
  const int q1 = p.cu_q[s + 1];
  const int row0 = q0 + (group - static_cast<int>(s_aux[35])) * kRows;
  const int n = min(kRows, q1 - row0);
  const int vis0 = p.kv_len[s] - q1 + row0 + 1;  // positions row0 may attend
  const int words = (vis0 + n + 2) >> 2;
  const int base = p.cu_k[s];
  int32_t* const out_slot = p.slots + static_cast<int64_t>(group) * p.cap;
  uint8_t* const out_mask = p.masks + static_cast<int64_t>(group) * p.cap;

  bool merged = words <= p.map_words;
  if (merged) {
    for (int w = tid; w < words; w += threads)
      s_map[w] = 0;
    __syncthreads();
    bool twice = false;
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      if (j < n) {
        const int32_t* row = p.idx + (row0 + j) * p.stride_idx;
        for (int c = tid; c < p.topk; c += threads) {
          const int pos = row[c];
          if (pos >= 0 && pos < vis0 + j) {
            const uint32_t bit = 1u << (8 * (pos & 3) + j);
            twice |= (atomicOr(s_map + (pos >> 2), bit) & bit) != 0;
          }
        }
      }
    }
    if (twice) s_aux[32] = 1;
    __syncthreads();
    merged = s_aux[32] == 0;
  }
  if (merged) {
    const int per = (words + threads - 1) / threads;
    const int w0 = min(tid * per, words);
    const int w1 = min(w0 + per, words);
    int cnt = 0;
    for (int w = w0; w < w1; ++w)
      cnt += __popc(used_bytes(s_map[w]));
    const int scan = device::warp::inclusive_sum(cnt);
    if (lane == 31) s_aux[warp] = scan;
    __syncthreads();
    if (warp == 0) {
      const int total = lane < threads / 32 ? static_cast<int>(s_aux[lane]) : 0;
      const int before = device::warp::inclusive_sum(total);
      s_aux[lane] = before - total;
      if (lane == 31) s_aux[33] = before;
    }
    __syncthreads();
    int o = static_cast<int>(s_aux[warp]) + scan - cnt;
    for (int w = w0; w < w1; ++w) {
      const uint32_t word = s_map[w];
      uint32_t used = used_bytes(word);
      while (used != 0) {
        const int b = __ffs(used) - 1;
        used &= used - 1;
        out_slot[o] = base + 4 * w + (b >> 3);
        out_mask[o] = static_cast<uint8_t>(word >> b);
        ++o;
      }
    }
  } else {
    int last = 0;
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      if (j < n) {
        const int32_t* row = p.idx + (row0 + j) * p.stride_idx;
        for (int c = tid; c < p.topk; c += threads) {
          const int pos = row[c];
          const bool ok = pos >= 0 && pos < vis0 + j;
          out_slot[j * p.topk + c] = ok ? base + pos : 0;
          out_mask[j * p.topk + c] = ok ? static_cast<uint8_t>(1u << j) : 0;
          if (ok) last = j * p.topk + c + 1;
        }
      }
    }
    last = device::warp::reduce_max(last);
    if (lane == 0) atomicMax(s_aux + 33, static_cast<uint32_t>(last));
  }
  __syncthreads();
  const int count = static_cast<int>(s_aux[33]);
  for (int i = count + tid; i < ((count + kTile - 1) & -kTile); i += threads) {
    out_slot[i] = 0;
    out_mask[i] = 0;
  }
  if (tid == 0) {
    int32_t* meta = p.meta + static_cast<int64_t>(group) * kMeta;
    meta[0] = count;
    meta[1] = row0;
    meta[2] = n;
    meta[3] = s;
  }
}

template <int kRows, int kHeads>
struct Layout {
  static constexpr int kReal = kRows * kHeads;  // head columns, query row major
  static constexpr int kNQ = 2 * kReal;         // QK N: hi terms, lo terms
  static constexpr int kPer = kReal / 4;        // heads of one softmax warpgroup: those of kRows / 4 query rows
  // P row and O lane of head n: 32 (n / kPer) + n % kPer
  static constexpr int kQPlane = kNQ * 128;
  static constexpr int kTmemCols = 512;           // S [kNQ], O [kD]
  static constexpr int op_v = 0;                  // bf16 V tile, 4 planes [slot][64 d]
  static constexpr int op_k = op_v + 4 * kPlane;  // fp8 K tiles of even / odd t, 2 planes [slot][128 d] each
  static constexpr int q = op_k + 4 * kPlane;     // e4m3 Q terms, 2 planes [column][128 d]
  static constexpr int pr = q + 2 * kQPlane;      // bf16 P, 2 planes [slot][64 heads]
  static constexpr int ent = pr + 2 * kPlane;     // packed rows of kRing tiles
  static constexpr int msk = ent + kRing * kTile * 4;
  static constexpr int red = msk + kRing * kTile;  // [128][4 warps] f32
  static constexpr int fac = red + 128 * 16;       // [128] f32
  static constexpr int ref = fac + 128 * 4;        // [128] f32: -reference of each head, log2 units
  static constexpr int has = ref + 128 * 4;        // [128] u32: the head has a reference
  static constexpr int off = has + 128 * 4;        // [16] f32: -inf
  static constexpr int flag = off + 64;            // [2][16 warps] u8, [4 warpgroups] u8
  static constexpr int bar = flag + 40;            // mbarriers: QK, PV
  static constexpr int tmem = bar + 16;
  static constexpr int bytes = tmem + 8;
};

// Worker: the group's q heads as e4m3 hi / lo terms in the 128 B-swizzled QK operand, zero past `rows`; `after_scan`
// runs once the workers' pending copies are in. Returns the score scale in log2 units.
template <int kRows, int kHeads, typename AfterScan>
SGL_DEVICE float split_q(
    const Params& p,
    const int row0,
    const int rows,
    const int kvh,
    uint8_t* const s_q,
    float* const s_red,
    const AfterScan after_scan) {
  using Lay = Layout<kRows, kHeads>;
  constexpr int kReal = Lay::kReal;
  const int tid = threadIdx.x;
  const int real = rows * kHeads;
  const auto head = [&](const int n) {
    return p.q + (row0 + n / kHeads) * p.stride_qr + (kvh * kHeads + n % kHeads) * p.stride_qh;
  };
  uint32_t qtop = 0;
  for (int i = tid; i < real * 32; i += kWorkers) {
    const uint4 raw = *reinterpret_cast<const uint4*>(head(i >> 5) + (i & 31) * 8);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const uint32_t w = e < 2 ? raw.x : e < 4 ? raw.y : e < 6 ? raw.z : raw.w;
      const uint32_t mag = (w >> (16 * (e & 1))) & 0x7FFFu;
      qtop = max(qtop, mag < 0x7F80u ? mag : 0u);
    }
  }
  qtop = device::warp::reduce_max(qtop);
  if ((tid & 31) == 0) s_red[tid >> 5] = __uint_as_float(qtop);
  cp_async_wait<0>();
  device::ptx::bar_sync(3, kWorkers);
  after_scan();
#pragma unroll
  for (int w = 0; w < kWorkers / 32; ++w)
    qtop = max(qtop, __float_as_uint(s_red[w]));
  // q * 2^shift has its largest finite magnitude in [128, 256), under the e4m3 maximum of 448
  const int qexp = static_cast<int>(qtop >> 7);
  const int shift = qexp == 0 ? 0 : min(max(134 - qexp, -126), 126);
  const float qmul = __int_as_float((127 + shift) << 23);
  for (int i = real * 16 + tid; i < kReal * 16; i += kWorkers) {
    uint8_t* dst = s_q + ((i & 15) >> 3) * Lay::kQPlane;
    *reinterpret_cast<uint4*>(dst + tcgen05::sw128_off(i >> 4, i & 7)) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(dst + tcgen05::sw128_off(kReal + (i >> 4), i & 7)) = make_uint4(0, 0, 0, 0);
  }
  for (int i = tid; i < real * 16; i += kWorkers) {
    const int n = i >> 4;
    const int c = i & 15;
    const uint4 raw0 = *reinterpret_cast<const uint4*>(head(n) + c * 16);
    const uint4 raw1 = *reinterpret_cast<const uint4*>(head(n) + c * 16 + 8);
    uint32_t hi[4];
    uint32_t lo[4];
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      uint32_t h0, l0, h1, l1;
      const uint4 raw = e < 2 ? raw0 : raw1;
      bf16x2_e4m3_terms((e & 1) == 0 ? raw.x : raw.z, qmul, h0, l0);
      bf16x2_e4m3_terms((e & 1) == 0 ? raw.y : raw.w, qmul, h1, l1);
      hi[e] = h0 | (h1 << 16);
      lo[e] = l0 | (l1 << 16);
    }
    uint8_t* dst = s_q + (c >> 3) * Lay::kQPlane;
    *reinterpret_cast<uint4*>(dst + tcgen05::sw128_off(n, c & 7)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
    *reinterpret_cast<uint4*>(dst + tcgen05::sw128_off(kReal + n, c & 7)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
  }
  return p.qk_scale * __int_as_float(0x3FB8AA3B) * __int_as_float((127 - shift) << 23);
}

// Issuing warps: QK waits on barrier 1 (next K tile in, S read), PV on barrier 2 (P and V of this tile written).
// QK: S [slot][column] = K [slot][d] Q [column][d]; PV: O [head][d] += P [head][slot] V [d][slot].
template <int kRows, int kHeads>
SGL_DEVICE void issue_mmas(const bool pv_warp, const int tiles, const uint32_t t_s, uint8_t* const smem) {
  using Lay = Layout<kRows, kHeads>;
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  const bool issuer = (threadIdx.x & 31) == 0;
  const auto kdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::op_k);
  const auto qdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::q);
  auto vdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::op_v);
  vdesc.leading_byte_offset = kPlane >> 4;
  auto pdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::pr);
  pdesc.leading_byte_offset = kPlane >> 4;
  const uint32_t idesc_qk = tcgen05::make_instr_desc_f8<kTile, Lay::kNQ>().bits;
  const uint32_t idesc_pv = tcgen05::make_instr_desc_f16(128, kD, true, true) | (1u << 7) | (1u << 10);  // bf16
  const auto qk = [&](const int t) {
    if (issuer) {
      tcgen05::fence_after_thread_sync();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        tcgen05::mma_f8_ss(
            t_s,
            kdesc.advance_bytes(((t & 1) * 2 + (i >> 2)) * kPlane + (i & 3) * 32).bits,
            qdesc.advance_bytes((i >> 2) * Lay::kQPlane + (i & 3) * 32).bits,
            idesc_qk,
            i > 0);
      }
      tcgen05::commit(s_bar);
    }
  };
  if (pv_warp) {
    for (int t = 0; t < tiles; ++t) {
      device::ptx::bar_sync(2, kSync);
      if (issuer) {
        tcgen05::fence_after_thread_sync();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          tcgen05::mma_f16_ss(
              t_s + Lay::kNQ,
              pdesc.advance_bytes(i * 2048).bits,
              vdesc.advance_bytes(i * 2048).bits,
              idesc_pv,
              t > 0 || i > 0);
        }
        tcgen05::commit(s_bar + 1);
      }
    }
    return;
  }
  device::ptx::bar_sync(1, kSync);
  if (tiles > 0) qk(0);
  for (int t = 0; t < tiles; ++t) {
    device::ptx::bar_sync(1, kSync);
    if (t + 1 < tiles) qk(t + 1);
  }
}

template <int kRows, int kHeads>
__global__ void __launch_bounds__(kBlock, 1) attn_kernel(const __grid_constant__ Params p) {
  using Lay = Layout<kRows, kHeads>;
  constexpr int kReal = Lay::kReal;
  constexpr int kPer = Lay::kPer;
  constexpr int kSegs = kRows / 4;  // query rows of a softmax warpgroup, kHeads columns each
  static_assert(Lay::bytes <= kMaxSmem && kReal <= 128 && Lay::kNQ + kD <= Lay::kTmemCols);
  static_assert(kRows % 4 == 0 && kHeads % 4 == 0 && kHeads <= 16 && kPer <= 32);
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wg = warp >> 2;
  const int tl = tid & 127;
  const int group = blockIdx.x / p.HKV;
  const int kvh = blockIdx.x % p.HKV;
  uint8_t* const s_opv = smem + Lay::op_v;
  uint8_t* const s_opk = smem + Lay::op_k;
  uint8_t* const s_p = smem + Lay::pr;
  uint32_t* const s_ent = reinterpret_cast<uint32_t*>(smem + Lay::ent);
  uint8_t* const s_msk = smem + Lay::msk;
  float* const s_red = reinterpret_cast<float*>(smem + Lay::red);
  float* const s_fac = reinterpret_cast<float*>(smem + Lay::fac);
  float* const s_ref = reinterpret_cast<float*>(smem + Lay::ref);
  uint32_t* const s_has = reinterpret_cast<uint32_t*>(smem + Lay::has);
  float* const s_off = reinterpret_cast<float*>(smem + Lay::off);
  uint8_t* const s_moved = smem + Lay::flag + 32;  // per warpgroup: 1 a reference moved, 2 a head has none
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  uint32_t* const s_tmem = reinterpret_cast<uint32_t*>(smem + Lay::tmem);
  const float ninf = __int_as_float(0xFF800000);
  const bool worker = warp < kWorkers / 32;
  const bool pv_warp = warp == kWorkers / 32 + 1;
  const int32_t* const meta = p.meta + static_cast<int64_t>(group) * kMeta;
  const int tiles = (meta[0] + kTile - 1) / kTile;
  const int row0 = meta[1];
  const int rows = meta[2];

  if (tid == 0) {
    device::ptx::mbar_init(s_bar, 1);
    device::ptx::mbar_init(s_bar + 1, 1);
    device::ptx::fence_barrier_init();
  }
  const int32_t* const slots = p.slots + static_cast<int64_t>(group) * p.cap;
  const uint8_t* const masks = p.masks + static_cast<int64_t>(group) * p.cap;
  // a worker moves 16-byte chunk gc of the K rows of slots gk + 32 i of a tile
  const int gk = tid >> 4;
  const int gc = tid & 15;
  const int vk = tid >> 3;
  const int vc = tid & 7;
  const int first = wg * kPer;  // this thread's heads: first + i, those of query rows j0 + r
  const int j0 = wg * kSegs;
  uint32_t n_qk = 0;
  uint32_t n_pv = 0;

  const auto load_entries = [&](const int t) {
    if (t < tiles && tid < 32) {
      cp_async16(device::ptx::to_shared(s_ent + (t % kRing) * kTile + tid * 4), slots + t * kTile + tid * 4, 16);
    }
    if (t < tiles && tid < 8) {
      cp_async16(device::ptx::to_shared(s_msk + (t % kRing) * kTile + tid * 16), masks + t * kTile + tid * 16, 16);
    }
  };
  const auto gather_k = [&](const int t) {
    if (t < tiles) {
      const uint8_t* base = p.k + kvh * p.stride_kh + gc * 16;
      const uint32_t dst = device::ptx::to_shared(s_opk + ((t & 1) * 2 + (gc >> 3)) * kPlane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int e = (t % kRing) * kTile + gk + 32 * i;
        cp_async16(
            dst + tcgen05::sw128_off(gk + 32 * i, gc & 7),
            base + static_cast<int64_t>(s_ent[e]) * p.stride_krow,
            s_msk[e] != 0 ? 16 : 0);
      }
    }
  };
  // this thread's V copies of a tile: chunks vc + 8 j of the rows of slots vk and vk + 64
  const uint8_t* const v_base = p.v + kvh * p.stride_vh + vc * 16;
  const uint32_t v_dst[2] = {
      device::ptx::to_shared(s_opv) + tcgen05::sw128_off(vk, vc),
      device::ptx::to_shared(s_opv) + tcgen05::sw128_off(vk + 64, vc)};
  // and its P stores: bf16 pairs of kStore heads each
  constexpr int kStore = kPer % 8 == 0 ? 8 : 4;
  uint8_t* p_dst[kPer / kStore];
#pragma unroll
  for (int u = 0; u < kPer / kStore; ++u) {
    const int n = 32 * wg + kStore * u;
    p_dst[u] = s_p + (n >> 6) * kPlane + tl * 128 + ((((n >> 3) & 7) ^ (tl & 7)) << 4) + ((2 * n) & 15);
  }
  const auto wait_pv = [&]() {
    if (n_pv > 0) device::ptx::mbar_wait_parity(s_bar + 1, (n_pv - 1) & 1);
  };

  float sc = 0.f;
  if (worker) {
    // the first entries land under the scan of q, the first two K tiles under its conversion
#pragma unroll
    for (int t = 0; t < 5; ++t)
      load_entries(t);
    cp_async_commit();
    sc = split_q<kRows, kHeads>(p, row0, rows, kvh, smem + Lay::q, s_red, [&]() {
      gather_k(0);
      gather_k(1);
      cp_async_commit();
    });
    // P rows of no head stay zero
    for (int i = tid; i < 16 * kTile; i += kWorkers) {
      const int c = i / kTile;
      *reinterpret_cast<uint4*>(s_p + (c >> 3) * kPlane + tcgen05::sw128_off(i % kTile, c & 7)) =
          make_uint4(0, 0, 0, 0);
    }
    if (tid < 128) {
      s_ref[tid] = 0.f;
      s_has[tid] = 0;
    }
    if (tid < 16) s_off[tid] = ninf;
    cp_async_wait<0>();
    device::ptx::fence_proxy_async_shared();
  } else if (!pv_warp) {
    tcgen05::alloc(s_tmem, Lay::kTmemCols);
    tcgen05::relinquish_alloc_permit();
    tcgen05::fence_before_thread_sync();
  }
  __syncthreads();
  tcgen05::fence_after_thread_sync();

  const uint32_t t_s = *s_tmem + (worker ? static_cast<uint32_t>((warp & 3) * 32) << 16 : 0u);
  const uint32_t t_o = t_s + Lay::kNQ;
  if (!worker) {
    issue_mmas<kRows, kHeads>(pv_warp, tiles, t_s, smem);
    if (!pv_warp) {
      device::ptx::bar_sync(1, kSync);
      tcgen05::dealloc(*s_tmem, Lay::kTmemCols);
    }
    return;
  }
  device::ptx::bar_arrive(1, kSync);

  float ls[kPer];  // sum of p over this thread's slots
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    ls[i] = 0.f;
  uint32_t lack = 1;  // a head of the group has no reference yet

#pragma unroll 1
  for (int t = 0; t < tiles; ++t) {
    device::ptx::mbar_wait_parity(s_bar, n_qk & 1);
    ++n_qk;
    tcgen05::fence_after_thread_sync();
    // scores of this thread's slot in log2 units over the references: (hi + lo) * sc - reference, -inf for the
    // rows that did not select the slot
    const uint32_t mine = s_msk[(t % kRing) * kTile + tl];
    float x[kPer];
#pragma unroll
    for (int r = 0; r < kSegs; ++r) {
      const float* bias = ((mine >> (j0 + r)) & 1u) != 0 ? s_ref + first + r * kHeads : s_off;
      float term[16];
      tcgen05::ld_32x32b<16>(t_s + kReal + first + r * kHeads, term);
      tcgen05::wait_ld();
#pragma unroll
      for (int i = 0; i < kHeads; i += 4) {
        const float4 b = *reinterpret_cast<const float4*>(bias + i);
        x[r * kHeads + i] = fmaf(term[i], sc, b.x);
        x[r * kHeads + i + 1] = fmaf(term[i + 1], sc, b.y);
        x[r * kHeads + i + 2] = fmaf(term[i + 2], sc, b.z);
        x[r * kHeads + i + 3] = fmaf(term[i + 3], sc, b.w);
      }
      tcgen05::ld_32x32b<16>(t_s + first + r * kHeads, term);
      tcgen05::wait_ld();
#pragma unroll
      for (int i = 0; i < kHeads; ++i)
        x[r * kHeads + i] = fmaf(term[i], sc, x[r * kHeads + i]);
    }
    // S of this tile is read and K(t + 1) is in: QK of tile t + 1 may go
    tcgen05::fence_before_thread_sync();
    cp_async_wait<0>();
    device::ptx::fence_proxy_async_shared();
    device::ptx::bar_arrive(1, kSync);
    gather_k(t + 2);
    load_entries(t + 5);
    cp_async_commit();
    float top = ninf;
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      top = fmaxf(top, x[i]);
    if (any_in_warpgroup(4 + wg, top > kMove || lack != 0)) {
      // a score is kMove over its head's reference, or the head has none: the tile's maximum becomes the reference
      // and the head's sums and O are rescaled
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        x[i] -= s_ref[first + i];
        const float m = device::warp::reduce_max(x[i]);
        if (lane == 0) s_red[(first + i) * 4 + (warp & 3)] = m;
      }
      device::ptx::bar_sync(4 + wg, 128);
      if (lane == 0 && (warp & 3) == 0) {
        uint32_t state = 0;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          const int n = first + i;
          const float4 r4 = *reinterpret_cast<const float4*>(s_red + n * 4);
          const float m = fmaxf(fmaxf(r4.x, r4.y), fmaxf(r4.z, r4.w));
          const float nm = s_ref[n];
          const bool had = s_has[n] != 0;
          const bool move = had ? m > kMove - nm : m > ninf;
          s_fac[n] = move && had ? ex2_approx(-nm - m) : 1.f;
          s_ref[n] = move ? -m : nm;
          s_has[n] = had || move ? 1u : 0u;
          state |= (move && had ? 1u : 0u) | (had || move ? 0u : 2u);
        }
        s_moved[wg] = static_cast<uint8_t>(state);
      }
      device::ptx::bar_sync(4 + wg, 128);
      const uint32_t state = s_moved[wg];
      lack = state & 2u;
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        ls[i] *= s_fac[first + i];
        x[i] += s_ref[first + i];
      }
      if ((state & 1u) != 0 && t > 0 && (warp & 3) == wg) {
        wait_pv();
        tcgen05::fence_after_thread_sync();
        const float f = lane < kPer ? s_fac[first + lane] : 1.f;
#pragma unroll 4
        for (int b = 0; b < kD / 16; ++b) {
          float o[16];
          tcgen05::ld_32x32b<16>(t_o + 16 * b, o);
          tcgen05::wait_ld();
#pragma unroll
          for (int e = 0; e < 16; ++e)
            o[e] *= f;
          tcgen05::st_32x32b<16>(t_o + 16 * b, o);
        }
        tcgen05::wait_st();
      }
    }
    uint32_t pp[kPer / 2];
#pragma unroll
    for (int i = 0; i < kPer; i += 2) {
      const float p0 = ex2_approx(x[i]);
      const float p1 = ex2_approx(x[i + 1]);
      ls[i] += p0;
      ls[i + 1] += p1;
      pp[i / 2] = f32x2_bf16x2(p0, p1);
    }
    const uint8_t* v_row[2];
    int v_bytes[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int e = (t % kRing) * kTile + vk + 64 * i;
      v_row[i] = v_base + static_cast<int64_t>(s_ent[e]) * p.stride_vrow;
      v_bytes[i] = s_msk[e] != 0 ? 16 : 0;
    }
    // the P and V operands are free once the PV of tile t - 1 is in: only copies and stores from here to PV(t)
    wait_pv();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j)
        cp_async16(v_dst[i] + j * kPlane, v_row[i] + j * 128, v_bytes[i]);
    }
    cp_async_commit();
#pragma unroll
    for (int u = 0; u < kPer / kStore; ++u) {
      if constexpr (kStore == 8) {
        *reinterpret_cast<uint4*>(p_dst[u]) = make_uint4(pp[4 * u], pp[4 * u + 1], pp[4 * u + 2], pp[4 * u + 3]);
      } else {
        *reinterpret_cast<uint2*>(p_dst[u]) = make_uint2(pp[2 * u], pp[2 * u + 1]);
      }
    }
    cp_async_wait<0>();
    device::ptx::fence_proxy_async_shared();
    ++n_pv;
    device::ptx::bar_arrive(2, kSync);
  }
  cp_async_wait<0>();

  // out = O / sum of p
  wait_pv();
  tcgen05::fence_after_thread_sync();
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const float sum = device::warp::reduce_sum(ls[i]);
    if (lane == 0) s_red[(first + i) * 4 + (warp & 3)] = sum;
  }
  device::ptx::bar_sync(3, kWorkers);
  if (lane == 0 && (warp & 3) == 0) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const float4 r4 = *reinterpret_cast<const float4*>(s_red + (first + i) * 4);
      const float sum = (r4.x + r4.y) + (r4.z + r4.w);
      s_fac[first + i] = sum > 0.f ? 1.f / sum : 0.f;
    }
  }
  device::ptx::bar_sync(3, kWorkers);
  // lane tl of O is head (tl / 32) kPer + tl % 32; this thread writes its d [64 wg, 64 wg + 64)
  const int head = (tl >> 5) * kPer + (tl & 31);
  const bool live = (tl & 31) < kPer && head < rows * kHeads;
  const float inv = live ? s_fac[head] : 0.f;
  uint16_t* const dst =
      p.out + (row0 + head / kHeads) * p.stride_or + (kvh * kHeads + head % kHeads) * p.stride_oh + 64 * wg;
#pragma unroll
  for (int b = 0; b < 4; ++b) {
    float o[16];
#pragma unroll
    for (int e = 0; e < 16; ++e)
      o[e] = 0.f;
    if (tiles > 0) {
      tcgen05::ld_32x32b<16>(t_o + 64 * wg + 16 * b, o);
      tcgen05::wait_ld();
    }
    if (live) {
      uint32_t w[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        w[e] = static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(o[2 * e] * inv))) |
               (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(o[2 * e + 1] * inv))) << 16);
      }
      *reinterpret_cast<uint4*>(dst + 16 * b) = make_uint4(w[0], w[1], w[2], w[3]);
      *reinterpret_cast<uint4*>(dst + 16 * b + 8) = make_uint4(w[4], w[5], w[6], w[7]);
    }
  }
  tcgen05::fence_before_thread_sync();
  device::ptx::bar_arrive(1, kSync);
}

}  // namespace qsa_sparse_prefill_sm100

template <int kRows, int kHeads>
struct QsaSparsePrefillSm100 {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView idx,
      const tvm::ffi::TensorView cu_q,
      const tvm::ffi::TensorView cu_k,
      const tvm::ffi::TensorView kv_len,
      const tvm::ffi::TensorView slots,
      const tvm::ffi::TensorView masks,
      const tvm::ffi::TensorView meta,
      const tvm::ffi::TensorView out,
      double qk_scale,
      int64_t max_kv_len) {
    using namespace host;
    using namespace qsa_sparse_prefill_sm100;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto S_ = SymbolicSize{"kv_rows"};
    auto HKV_ = SymbolicSize{"kv_heads"};
    auto TOPK_ = SymbolicSize{"topk"};
    auto NS_ = SymbolicSize{"seqs"};
    auto NS1_ = SymbolicSize{"seqs_plus_1"};
    auto NG_ = SymbolicSize{"groups"};
    auto CAP_ = SymbolicSize{"cap"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(q);
    // fp8 e4m3 K and bf16 V passed as uint8 views
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(k);
    TensorMatcher({S_, HKV_, 2 * kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(v);
    TensorMatcher({R_, TOPK_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(idx);
    TensorMatcher({NS1_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(cu_q);
    TensorMatcher({NS1_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(cu_k);
    TensorMatcher({NS_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(kv_len);
    TensorMatcher({NG_, CAP_}).with_dtype<int32_t>().with_device(device).with_strides({CAP_, 1}).verify(slots);
    TensorMatcher({NG_, CAP_}).with_dtype<uint8_t>().with_device(device).with_strides({CAP_, 1}).verify(masks);
    TensorMatcher({NG_, kMeta}).with_dtype<int32_t>().with_device(device).with_strides({kMeta, 1}).verify(meta);
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(out);
    const int64_t hkv = HKV_.unwrap();
    const int64_t topk = TOPK_.unwrap();
    const int64_t groups = NG_.unwrap();
    const int64_t cap = CAP_.unwrap();
    RuntimeCheck(HQ_.unwrap() == hkv * kHeads, "q heads per kv head must match the kernel");
    RuntimeCheck(NS1_.unwrap() == NS_.unwrap() + 1, "cu_q and cu_k hold one more entry than there are sequences");
    RuntimeCheck(topk > 0 && cap % kTile == 0 && cap >= kRows * topk, "slots must hold every column in whole tiles");
    const auto a16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(
        a16(k.data_ptr()) && a16(v.data_ptr()) && k.stride(0) % 16 == 0 && k.stride(1) % 16 == 0 &&
            v.stride(0) % 16 == 0 && v.stride(1) % 16 == 0,
        "K / V rows must be 16-byte aligned");
    RuntimeCheck(a16(q.data_ptr()) && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0, "q rows must be 16-byte aligned");
    RuntimeCheck(
        a16(out.data_ptr()) && out.stride(0) % 8 == 0 && out.stride(1) % 8 == 0, "out rows must be 16-byte aligned");
    RuntimeCheck(a16(slots.data_ptr()) && a16(masks.data_ptr()), "slots and masks must be 16-byte aligned");
    if (groups == 0) return;
    const DLDevice dev = device.unwrap();
    // without room for a mask byte per position of the longest sequence the rows of a group are not merged
    const int64_t map_words = (max_kv_len + 3) / 4 + 4;
    const bool mapped = (map_words + 40) * 4 <= kMaxSmem;
    const auto union_params = UnionParams{
        .idx = static_cast<const int32_t*>(idx.data_ptr()),
        .cu_q = static_cast<const int32_t*>(cu_q.data_ptr()),
        .cu_k = static_cast<const int32_t*>(cu_k.data_ptr()),
        .kv_len = static_cast<const int32_t*>(kv_len.data_ptr()),
        .slots = static_cast<int32_t*>(slots.data_ptr()),
        .masks = static_cast<uint8_t*>(masks.data_ptr()),
        .meta = static_cast<int32_t*>(meta.data_ptr()),
        .stride_idx = idx.stride(0),
        .topk = static_cast<int32_t>(topk),
        .cap = static_cast<int32_t>(cap),
        .nseq = static_cast<int32_t>(NS_.unwrap()),
        .map_words = static_cast<int32_t>(mapped ? map_words : 0),
    };
    const auto params = Params{
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .v = static_cast<const uint8_t*>(v.data_ptr()),
        .slots = static_cast<const int32_t*>(slots.data_ptr()),
        .masks = static_cast<const uint8_t*>(masks.data_ptr()),
        .meta = static_cast<const int32_t*>(meta.data_ptr()),
        .out = static_cast<uint16_t*>(out.data_ptr()),
        .stride_qr = q.stride(0),
        .stride_qh = q.stride(1),
        .stride_krow = k.stride(0),
        .stride_kh = k.stride(1),
        .stride_vrow = v.stride(0),
        .stride_vh = v.stride(1),
        .stride_or = out.stride(0),
        .stride_oh = out.stride(1),
        .qk_scale = static_cast<float>(qk_scale),
        .cap = static_cast<int32_t>(cap),
        .HKV = static_cast<int32_t>(hkv),
    };
    const std::size_t union_smem = static_cast<std::size_t>(union_params.map_words + 40) * 4;
    runtime::set_max_dynamic_smem_per_device<union_kernel<kRows>>(dev, kMaxSmem);
    LaunchKernel(static_cast<uint32_t>(groups), kUnionThreads, dev, union_smem)(union_kernel<kRows>, union_params);
    constexpr int smem = Layout<kRows, kHeads>::bytes;
    runtime::set_max_dynamic_smem_per_device<attn_kernel<kRows, kHeads>>(dev, smem);
    LaunchKernel(static_cast<uint32_t>(groups * hkv), kBlock, dev, smem)(attn_kernel<kRows, kHeads>, params);
  }
};

}  // namespace sglang
