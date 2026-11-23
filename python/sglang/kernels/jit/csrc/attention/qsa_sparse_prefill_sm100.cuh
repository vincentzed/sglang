// QSA prefill sparse attention over an FP8 E4M3 KV pool on SM100 (tcgen05): a union kernel merges the top-k rows of
// kRows consecutive query rows of a sequence into one ascending slot list with a row mask per slot, an attention
// kernel attends that list once per KV head.
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
using qsa_sparse_decode_sm100::e4m3x8_f16x8;
using qsa_sparse_decode_sm100::f32x2_f16x2;
using qsa_sparse_decode_sm100::kMaxSmem;
using qsa_sparse_decode_sm100::kPlane;
using qsa_sparse_decode_sm100::kRescale;
using qsa_sparse_decode_sm100::kThreads;
using qsa_sparse_decode_sm100::kTile;
using qsa_sparse_decode_sm100::kWorkers;
using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kStages = 2;  // V items (128 slots x 128 d) in shared memory
constexpr int kRing = 8;    // tiles of entries in shared memory
constexpr int kUnionThreads = 512;
constexpr int kMeta = 4;  // per group: slots, first row, rows, sequence

struct UnionParams {
  const int32_t* idx;           // [R, TOPK] logical positions, -1 = hole
  const int32_t* cu_q;          // [S + 1] first row of each sequence
  const int32_t* kv_len;        // [S] sequence length with its rows
  const int32_t* req;           // [S] row of req_to_token
  const int32_t* req_to_token;  // [requests, context]
  int32_t* slots;               // [G, cap]
  uint8_t* masks;               // [G, cap]
  int32_t* meta;                // [G, kMeta]
  int64_t stride_idx, stride_r2t;
  int32_t topk, cap, nseq, map_words;
};

struct Params {
  const uint16_t* q;  // bf16 [R, HQ, D]
  const uint8_t* k;   // fp8 e4m3 [slots, HKV, D]
  const uint8_t* v;
  const int32_t* slots;
  const uint8_t* masks;
  const int32_t* meta;
  uint16_t* out;  // bf16 [R, HQ, D]
  int64_t stride_qr, stride_qh, stride_kslot, stride_kh, stride_or, stride_oh;
  float qk_scale;
  int32_t cap, HKV;
};

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
  const int32_t* const r2t = p.req_to_token + static_cast<int64_t>(p.req[s]) * p.stride_r2t;
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
        out_slot[o] = r2t[4 * w + (b >> 3)];
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
          out_slot[j * p.topk + c] = ok ? r2t[pos] : 0;
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
  static constexpr int kReal = kRows * kHeads;                                       // head columns, query row major
  static constexpr int kNQ = (2 * kReal + 7) & -8;                                   // QK N: hi terms, lo terms, zeros
  static constexpr int kPer = ((kReal + 3) / 4 + 1) & -2;                            // heads of one softmax warpgroup
  static constexpr int kWin = kPer <= 4 ? 4 : kPer <= 8 ? 8 : kPer <= 16 ? 16 : 32;  // TMEM columns per term
  static constexpr int kQPlane = kNQ * 128;
  static constexpr int kTmemCols = 512;             // S [kNQ], O [kD]
  static constexpr int op_v = 0;                    // f16 V tile, 4 planes [slot][64 d]
  static constexpr int op_k = op_v + 4 * kPlane;    // fp8 K tile, 2 planes [slot][128 d]
  static constexpr int stg = op_k + 2 * kPlane;     // fp8 V items
  static constexpr int q = stg + kStages * kPlane;  // e4m3 Q terms, 2 planes [column][128 d]
  static constexpr int pr = q + 2 * kQPlane;        // f16 P, 2 planes [slot][64 heads]
  static constexpr int ent = pr + 2 * kPlane;       // slots of kRing tiles
  static constexpr int msk = ent + kRing * kTile * 4;
  static constexpr int red = msk + kRing * kTile;  // [128][4 warps] f32
  static constexpr int fac = red + 128 * 16;       // [128] f32
  static constexpr int ref = fac + 128 * 4;        // [128] f32: -reference of each head, log2 units
  static constexpr int refn = ref + 128 * 4;
  static constexpr int flag = refn + 128 * 4;  // [16 warps] u8, u32
  static constexpr int bar = flag + 24;        // mbarriers: QK, PV
  static constexpr int tmem = bar + 16;
  static constexpr int bytes = tmem + 8;
};

// Worker: the group's q heads as e4m3 terms in the 128 B-swizzled QK operand, rows [0, kReal) hi and
// [kReal, 2 kReal) lo; rows past `rows` query rows are zero. Returns the score scale in log2 units.
template <int kRows, int kHeads>
SGL_DEVICE float
split_q(const Params& p, const int row0, const int rows, const int kvh, uint8_t* const s_q, float* const s_red) {
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
  device::ptx::bar_sync(3, kWorkers);
#pragma unroll
  for (int w = 0; w < kWorkers / 32; ++w)
    qtop = max(qtop, __float_as_uint(s_red[w]));
  // q * 2^shift has its largest finite magnitude in [128, 256), under the e4m3 maximum of 448
  const int qexp = static_cast<int>(qtop >> 7);
  const int shift = qexp == 0 ? 0 : min(max(134 - qexp, -126), 126);
  const float qmul = __int_as_float((127 + shift) << 23);
  for (int i = tid; i < Lay::kNQ * 16; i += kWorkers) {
    *reinterpret_cast<uint4*>(s_q + ((i & 15) >> 3) * Lay::kQPlane + tcgen05::sw128_off(i >> 4, i & 7)) =
        make_uint4(0, 0, 0, 0);
  }
  device::ptx::bar_sync(3, kWorkers);
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

// The MMA warp: barrier 1 = the K tile of the next QK is in, barrier 2 = P and V of this tile are written.
// QK: S [slot][column] = K [slot][d] Q [column][d]; PV: O [head][d] += P [head][slot] V [d][slot].
template <int kRows, int kHeads>
SGL_DEVICE void issue_mmas(const int tiles, const uint32_t t_s, uint8_t* const smem) {
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
  const uint32_t idesc_pv = tcgen05::make_instr_desc_f16(128, kD, true, true);
  const auto qk = [&]() {
    if (issuer) {
      tcgen05::fence_after_thread_sync();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        tcgen05::mma_f8_ss(
            t_s,
            kdesc.advance_bytes((i >> 2) * kPlane + (i & 3) * 32).bits,
            qdesc.advance_bytes((i >> 2) * Lay::kQPlane + (i & 3) * 32).bits,
            idesc_qk,
            i > 0);
      }
      tcgen05::commit(s_bar);
    }
  };
  device::ptx::bar_sync(1, kThreads);
  if (tiles > 0) qk();
  for (int t = 0; t < tiles; ++t) {
    device::ptx::bar_sync(1, kThreads);
    if (t + 1 < tiles) qk();
    device::ptx::bar_sync(2, kThreads);
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
}

template <int kRows, int kHeads>
__global__ void __launch_bounds__(kThreads, 1) attn_kernel(const __grid_constant__ Params p) {
  using Lay = Layout<kRows, kHeads>;
  constexpr int kReal = Lay::kReal;
  constexpr int kPer = Lay::kPer;
  constexpr int kWin = Lay::kWin;
  static_assert(Lay::bytes <= kMaxSmem && kReal <= 128 && Lay::kNQ + kD <= Lay::kTmemCols);
  static_assert(kPer <= kHeads || kPer == 2 * kHeads, "a softmax warpgroup spans at most two query rows");
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
  uint8_t* const s_stg = smem + Lay::stg;
  uint8_t* const s_p = smem + Lay::pr;
  uint32_t* const s_ent = reinterpret_cast<uint32_t*>(smem + Lay::ent);
  uint8_t* const s_msk = smem + Lay::msk;
  float* const s_red = reinterpret_cast<float*>(smem + Lay::red);
  float* const s_fac = reinterpret_cast<float*>(smem + Lay::fac);
  float* const s_ref = reinterpret_cast<float*>(smem + Lay::ref);
  float* const s_refn = reinterpret_cast<float*>(smem + Lay::refn);
  uint8_t* const s_flag = smem + Lay::flag;
  uint32_t* const s_any = reinterpret_cast<uint32_t*>(smem + Lay::flag + 16);
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  uint32_t* const s_tmem = reinterpret_cast<uint32_t*>(smem + Lay::tmem);
  const float ninf = __int_as_float(0xFF800000);
  const bool worker = warp < kWorkers / 32;
  const int32_t* const meta = p.meta + static_cast<int64_t>(group) * kMeta;
  const int tiles = (meta[0] + kTile - 1) / kTile;
  const int row0 = meta[1];
  const int rows = meta[2];

  if (tid == 0) {
    device::ptx::mbar_init(s_bar, 1);
    device::ptx::mbar_init(s_bar + 1, 1);
    device::ptx::fence_barrier_init();
  }
  float sc = 0.f;
  if (worker) {
    sc = split_q<kRows, kHeads>(p, row0, rows, kvh, smem + Lay::q, s_red);
    for (int i = tid; i < 2 * kPlane / 16; i += kWorkers)
      *reinterpret_cast<uint4*>(s_p + i * 16) = make_uint4(0, 0, 0, 0);
    if (tid < 128) s_ref[tid] = 1e30f;
    device::ptx::fence_proxy_async_shared();
  } else {
    tcgen05::alloc(s_tmem, Lay::kTmemCols);
    tcgen05::relinquish_alloc_permit();
    tcgen05::fence_before_thread_sync();
  }
  __syncthreads();
  tcgen05::fence_after_thread_sync();

  const uint32_t t_s = *s_tmem + (static_cast<uint32_t>((warp & 3) * 32) << 16);
  const uint32_t t_o = t_s + Lay::kNQ;
  if (!worker) {
    issue_mmas<kRows, kHeads>(tiles, t_s, smem);
    device::ptx::bar_sync(1, kThreads);
    tcgen05::dealloc(*s_tmem, Lay::kTmemCols);
    return;
  }

  const int32_t* const slots = p.slots + static_cast<int64_t>(group) * p.cap;
  const uint8_t* const masks = p.masks + static_cast<int64_t>(group) * p.cap;
  // a worker moves 16-byte chunk gc of the K rows of slots gk + 32 i of a tile, and chunk vc of the V rows of
  // slots vk and vk + 64 of an item: one cache line per copy
  const int gk = tid >> 4;
  const int gc = tid & 15;
  const int vk = tid >> 3;
  const int vc = tid & 7;
  const int first = wg * kPer;  // this thread's heads: first + i, i < heads
  const int heads = min(max(kReal - first, 0), kPer);
  const int j0 = first / kHeads;  // their query rows: j0, then j0 + 1 from head i = split
  const int split = (j0 + 1) * kHeads - first;
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
      const uint32_t dst = device::ptx::to_shared(s_opk + (gc >> 3) * kPlane);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int e = (t % kRing) * kTile + gk + 32 * i;
        cp_async16(
            dst + tcgen05::sw128_off(gk + 32 * i, gc & 7),
            base + static_cast<int64_t>(s_ent[e]) * p.stride_kslot,
            s_msk[e] != 0 ? 16 : 0);
      }
    }
  };
  // V item 2 t + h: d [128 h, 128 h + 128) of the V rows of tile t
  const auto gather_v = [&](const int item, const uint32_t stage) {
    if (item < 2 * tiles) {
      const uint8_t* base = p.v + kvh * p.stride_kh + (item & 1) * 128 + vc * 16;
      const uint32_t dst = device::ptx::to_shared(s_stg + stage * kPlane);
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int e = ((item >> 1) % kRing) * kTile + vk + 64 * i;
        cp_async16(
            dst + tcgen05::sw128_off(vk + 64 * i, vc),
            base + static_cast<int64_t>(s_ent[e]) * p.stride_kslot,
            s_msk[e] != 0 ? 16 : 0);
      }
    }
  };
  const auto convert_v = [&](const int item, const uint32_t stage) {
    const uint8_t* src = s_stg + stage * kPlane;
    uint8_t* dst = s_opv + ((item & 1) * 2 + (vc >> 2)) * kPlane;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int slot = vk + 64 * i;
      const uint4 raw = *reinterpret_cast<const uint4*>(src + tcgen05::sw128_off(slot, vc));
      *reinterpret_cast<uint4*>(dst + tcgen05::sw128_off(slot, 2 * (vc & 3))) = e4m3x8_f16x8(raw.x, raw.y);
      *reinterpret_cast<uint4*>(dst + tcgen05::sw128_off(slot, 2 * (vc & 3) + 1)) = e4m3x8_f16x8(raw.z, raw.w);
    }
  };
  // P of this thread's slot and heads, in pairs
  const auto store_p = [&](const float (&pv)[kPer]) {
#pragma unroll
    for (int i = 0; i < kPer; i += 2) {
      const int n = first + i;
      if (i < heads) {
        uint8_t* row = s_p + (n >> 6) * kPlane + tl * 128;
        *reinterpret_cast<uint32_t*>(row + ((((n >> 3) & 7) ^ (tl & 7)) << 4) + ((2 * n) & 15)) =
            f32x2_f16x2(pv[i], i + 1 < heads ? pv[i + 1] : 0.f);
      }
    }
  };
  // scores of this thread's slot and heads in log2 units: (hi + lo) * sc + bias, -inf where not selected
  const auto load_scores = [&](float (&x)[kPer], const float* bias, const uint32_t mine) {
    const bool on0 = ((mine >> j0) & 1u) != 0;
    const bool on1 = ((mine >> (j0 + 1)) & 1u) != 0;
    float term[kWin];
    tcgen05::ld_32x32b<kWin>(t_s + kReal + first, term);
    tcgen05::wait_ld();
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      x[i] = bias != nullptr && i < heads ? fmaf(term[i], sc, bias[first + i]) : term[i] * sc;
    tcgen05::ld_32x32b<kWin>(t_s + first, term);
    tcgen05::wait_ld();
    tcgen05::fence_before_thread_sync();
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      x[i] = (i < split ? on0 : on1) && i < heads ? fmaf(term[i], sc, x[i]) : ninf;
  };
  const auto wait_pv = [&]() {
    if (n_pv > 0) device::ptx::mbar_wait_parity(s_bar + 1, (n_pv - 1) & 1);
  };

#pragma unroll
  for (int t = 0; t < 5; ++t)
    load_entries(t);
  cp_async_commit();
  cp_async_wait<0>();
  device::ptx::bar_sync(3, kWorkers);
  gather_k(0);
  gather_v(0, 0);
  gather_v(1, 1);
  cp_async_commit();
  cp_async_wait<0>();
  device::ptx::fence_proxy_async_shared();
  device::ptx::bar_arrive(1, kThreads);

  float ls[kPer];  // sum of p over this thread's slots
#pragma unroll
  for (int i = 0; i < kPer; ++i)
    ls[i] = 0.f;

#pragma unroll 1
  for (int t = 0; t < tiles; ++t) {
    device::ptx::mbar_wait_parity(s_bar, n_qk & 1);
    ++n_qk;
    tcgen05::fence_after_thread_sync();
    gather_k(t + 1);
    load_entries(t + 5);
    cp_async_commit();
    const uint32_t mine = s_msk[(t % kRing) * kTile + tl];
    float pv[kPer];
    load_scores(pv, s_ref, mine);
    float top = ninf;
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      top = fmaxf(top, pv[i]);
      pv[i] = ex2_approx(pv[i]);
    }
    top = device::warp::reduce_max(top);
    if (lane == 0) s_flag[warp] = top > kRescale ? 1 : 0;
    device::ptx::bar_sync(3, kWorkers);

    const uint2 flags = *reinterpret_cast<const uint2*>(s_flag);
    const uint2 flags2 = *reinterpret_cast<const uint2*>(s_flag + 8);
    if ((flags.x | flags.y | flags2.x | flags2.y) != 0) {
      // some head is more than kRescale above its reference (or has none yet): move those references to the
      // tile maxima, rescale their sums and O, and redo this tile's P
      tcgen05::fence_after_thread_sync();
      load_scores(pv, nullptr, mine);
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        const float m = device::warp::reduce_max(pv[i]);
        if (lane == 0 && i < heads) s_red[(first + i) * 4 + (warp & 3)] = m;
      }
      if (tid == 0) *s_any = 0;
      device::ptx::bar_sync(3, kWorkers);
      bool moved = false;
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        if (i < heads) {
          const float4 r4 = *reinterpret_cast<const float4*>(s_red + (first + i) * 4);
          const float m = fmaxf(fmaxf(r4.x, r4.y), fmaxf(r4.z, r4.w));
          const float nm = s_ref[first + i];
          const float ref = -nm;
          const bool move = m > ref + kRescale;
          const float f = move ? ex2_approx(ref - m) : 1.f;
          const float next = move ? -m : nm;
          ls[i] *= f;
          pv[i] = ex2_approx(pv[i] + next);
          moved |= move && ref > -1e29f;
          if (lane == 0 && (warp & 3) == 0) {
            s_fac[first + i] = f;
            s_refn[first + i] = next;
          }
        }
      }
      if (moved && lane == 0 && (warp & 3) == 0) *s_any = 1;
      device::ptx::bar_sync(3, kWorkers);
      if (lane == 0 && (warp & 3) == 0) {
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          if (i < heads) s_ref[first + i] = s_refn[first + i];
        }
      }
      if (*s_any != 0 && t > 0) {
        wait_pv();
        tcgen05::fence_after_thread_sync();
        const float f = tl < kReal ? s_fac[tl] : 1.f;
#pragma unroll
        for (int b = 0; b < 4; ++b) {
          float o[16];
          tcgen05::ld_32x32b<16>(t_o + 64 * wg + 16 * b, o);
          tcgen05::wait_ld();
#pragma unroll
          for (int e = 0; e < 16; ++e)
            o[e] *= f;
          tcgen05::st_32x32b<16>(t_o + 64 * wg + 16 * b, o);
        }
        tcgen05::wait_st();
        tcgen05::fence_before_thread_sync();
      }
    }
#pragma unroll
    for (int i = 0; i < kPer; ++i)
      ls[i] += pv[i];
    // K(t + 1) and V(t) are in and S of this tile is read: QK of tile t + 1 may go
    cp_async_wait<0>();
    device::ptx::fence_proxy_async_shared();
    device::ptx::bar_arrive(1, kThreads);

    // the P and V operands are free once the PV of tile t - 1 is in
    wait_pv();
    store_p(pv);
    convert_v(2 * t, 0);
    convert_v(2 * t + 1, 1);
    device::ptx::fence_proxy_async_shared();
    ++n_pv;
    device::ptx::bar_arrive(2, kThreads);
    gather_v(2 * t + 2, 0);
    gather_v(2 * t + 3, 1);
    cp_async_commit();
  }
  cp_async_wait<0>();

  // out = O / sum of p
  wait_pv();
  tcgen05::fence_after_thread_sync();
#pragma unroll
  for (int i = 0; i < kPer; ++i) {
    const float sum = device::warp::reduce_sum(ls[i]);
    if (lane == 0 && i < heads) s_red[(first + i) * 4 + (warp & 3)] = sum;
  }
  device::ptx::bar_sync(3, kWorkers);
  if (lane == 0 && (warp & 3) == 0) {
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      if (i < heads) {
        const float4 r4 = *reinterpret_cast<const float4*>(s_red + (first + i) * 4);
        const float sum = (r4.x + r4.y) + (r4.z + r4.w);
        s_fac[first + i] = sum > 0.f ? 1.f / sum : 0.f;
      }
    }
  }
  device::ptx::bar_sync(3, kWorkers);
  // lane tl of O is head column tl; this thread writes its d [64 wg, 64 wg + 64)
  const bool live = tl < rows * kHeads;
  const float inv = live ? s_fac[tl] : 0.f;
  uint16_t* const dst =
      p.out + (row0 + tl / kHeads) * p.stride_or + (kvh * kHeads + tl % kHeads) * p.stride_oh + 64 * wg;
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
  device::ptx::bar_arrive(1, kThreads);
}

}  // namespace qsa_sparse_prefill_sm100

template <int kRows, int kHeads>
struct QsaSparsePrefillSm100 {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView req,
      const tvm::ffi::TensorView idx,
      const tvm::ffi::TensorView cu_q,
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
    auto S_ = SymbolicSize{"slots"};
    auto HKV_ = SymbolicSize{"kv_heads"};
    auto NR_ = SymbolicSize{"reqs"};
    auto CTX_ = SymbolicSize{"ctx"};
    auto TOPK_ = SymbolicSize{"topk"};
    auto NS_ = SymbolicSize{"seqs"};
    auto NS1_ = SymbolicSize{"seqs_plus_1"};
    auto NG_ = SymbolicSize{"groups"};
    auto CAP_ = SymbolicSize{"cap"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(q);
    // fp8 e4m3 pools passed as uint8 views
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(k);
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(v);
    TensorMatcher({NR_, CTX_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(req_to_token);
    TensorMatcher({NS_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(req);
    TensorMatcher({R_, TOPK_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(idx);
    TensorMatcher({NS1_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(cu_q);
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
    RuntimeCheck(NS1_.unwrap() == NS_.unwrap() + 1, "cu_q holds one more entry than there are sequences");
    RuntimeCheck(topk > 0 && cap % kTile == 0 && cap >= kRows * topk, "slots must hold every column in whole tiles");
    RuntimeCheck(k.stride(0) == v.stride(0) && k.stride(1) == v.stride(1), "K and V pools must share strides");
    const auto a16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(
        a16(k.data_ptr()) && a16(v.data_ptr()) && k.stride(0) % 16 == 0 && k.stride(1) % 16 == 0,
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
        .kv_len = static_cast<const int32_t*>(kv_len.data_ptr()),
        .req = static_cast<const int32_t*>(req.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .slots = static_cast<int32_t*>(slots.data_ptr()),
        .masks = static_cast<uint8_t*>(masks.data_ptr()),
        .meta = static_cast<int32_t*>(meta.data_ptr()),
        .stride_idx = idx.stride(0),
        .stride_r2t = req_to_token.stride(0),
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
        .stride_kslot = k.stride(0),
        .stride_kh = k.stride(1),
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
    LaunchKernel(static_cast<uint32_t>(groups * hkv), kThreads, dev, smem)(attn_kernel<kRows, kHeads>, params);
  }
};

}  // namespace sglang
