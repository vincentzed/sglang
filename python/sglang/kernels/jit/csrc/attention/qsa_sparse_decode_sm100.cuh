// QSA decode / target-verify attention over an FP8 E4M3 KV pool on SM100 (tcgen05): a union kernel merges the top-k
// rows of kRows query rows of a request into one slot list, an attention kernel attends that list once per KV head.
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include "qsa_sparse_decode.cuh"

namespace sglang {

namespace qsa_sparse_decode_sm100 {

using qsa_sparse_decode::cp_async16;
using qsa_sparse_decode::cp_async_commit;
using qsa_sparse_decode::cp_async_wait;
using qsa_sparse_decode::ex2_approx;
using qsa_sparse_decode::kD;
using qsa_sparse_decode::load_row_req;
using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kTile = 128;               // slots per tile: the M of both MMAs
constexpr int kWorkers = 512;            // four warpgroups; thread & 127 is its TMEM lane
constexpr int kThreads = kWorkers + 32;  // and the warp that issues the MMAs
constexpr int kPlane = kTile * 128;      // 128 d of an fp8 K tile or V item, or 64 d of an f16 V tile
constexpr int kStages = 3;               // V items (128 slots x 128 d) in shared memory
constexpr int kSlotBits = 27;            // entry = (row mask << 27) | slot
constexpr uint32_t kSlotMask = (1u << kSlotBits) - 1;
constexpr int kUnionThreads = 1024;
constexpr int kMaxSmem = 232448;
constexpr float kRescale = 12.f;  // log2 distance of a score above its head's reference that moves the reference

struct UnionParams {
  const int32_t* idx;           // [R, TOPK] logical positions, -1 = hole
  const int32_t* seqlen;        // [R]
  const int32_t* req_to_token;  // [requests, context]
  const void* row_req;          // int32 or int64 [R]
  int32_t* entries;             // [R / kRows, cap]
  int32_t* counts;              // [R / kRows]
  int64_t stride_idx, stride_r2t;
  int32_t topk, cap, map_words;
};

struct Params {
  const uint16_t* q;  // bf16 [R, HQ, D]
  const uint8_t* k;   // fp8 e4m3 [slots, HKV, D]
  const uint8_t* v;
  const int32_t* entries;  // [R / kRows, cap]
  const int32_t* counts;
  uint16_t* out;  // bf16 [R, HQ, D]
  int64_t stride_qr, stride_qh, stride_kslot, stride_kh, stride_or, stride_oh;
  float qk_scale;
  int32_t cap, HKV;
  int32_t groups;
};

SGL_DEVICE void cp_async4(uint32_t dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(dst), "l"(src) : "memory");
}

// bytes 0 and 1 of `two` (e4m3) -> f16x2
SGL_DEVICE uint32_t e4m3x2_f16x2(const uint32_t two) {
  uint32_t r;
  asm("{\n.reg .b16 c;\ncvt.u16.u32 c, %1;\ncvt.rn.f16x2.e4m3x2 %0, c;\n}" : "=r"(r) : "r"(two));
  return r;
}

SGL_DEVICE uint4 e4m3x8_f16x8(const uint32_t a, const uint32_t b) {
  return make_uint4(e4m3x2_f16x2(a), e4m3x2_f16x2(a >> 16), e4m3x2_f16x2(b), e4m3x2_f16x2(b >> 16));
}

SGL_DEVICE uint32_t f32x2_f16x2(const float lo, const float hi) {
  uint32_t r;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// {lo, hi} -> e4m3 codes in bytes 0 and 1, round to nearest, clamped to +-448
SGL_DEVICE uint32_t f32x2_e4m3x2(const float lo, const float hi) {
  uint32_t r;
  asm("{\n.reg .b16 c;\ncvt.rn.satfinite.e4m3x2.f32 c, %1, %2;\ncvt.u32.u16 %0, c;\n}" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// bf16 pair `w` times `mul` as two e4m3 terms each (x = hi + lo), codes in bytes 0 and 1; inf / NaN -> the NaN code
SGL_DEVICE void bf16x2_e4m3_terms(const uint32_t w, const float mul, uint32_t& hi, uint32_t& lo) {
  const float x0 = __uint_as_float(w << 16) * mul;
  const float x1 = __uint_as_float(w & 0xFFFF0000u) * mul;
  hi = f32x2_e4m3x2(x0, x1);
  const uint32_t back = e4m3x2_f16x2(hi);
  const float2 top = __half22float2(*reinterpret_cast<const __half2*>(&back));
  lo = f32x2_e4m3x2(x0 - top.x, x1 - top.y);
  if ((w & 0x7F80u) == 0x7F80u) {
    hi |= 0x7Fu;
    lo &= 0xFF00u;
  }
  if ((w & 0x7F800000u) == 0x7F800000u) {
    hi |= 0x7F00u;
    lo &= 0x00FFu;
  }
}

// values[i] through a select chain: a run-time index would put the register array on the stack
template <int N>
SGL_DEVICE float select_at(const float (&values)[N], const int i) {
  float v = values[0];
#pragma unroll
  for (int j = 1; j < N; ++j) {
    if (i == j) v = values[j];
  }
  return v;
}

// bit 4 i set when nibble i of `w` is not zero
SGL_DEVICE uint32_t used_nibbles(const uint32_t w) {
  return (w | (w >> 1) | (w >> 2) | (w >> 3)) & 0x11111111u;
}

// Words of the union kernel's staging area: slots and row masks of an unmerged list, or the offsets of a merged one.
constexpr int64_t union_stage_words(const int64_t cap, const int64_t map_words, const int64_t threads) {
  return cap + cap / 4 > map_words / 4 + threads ? cap + cap / 4 : map_words / 4 + threads;
}

// Shared memory, in words: [map_words] a 4-bit row mask per position; [kRows][row_words] the rows' top-k, each at
// the 16-byte phase of its global row; [union_stage_words] staging; [48] scratch.
template <int kRows, bool kRowReq64, bool kUsePDL>
__global__ void __launch_bounds__(kUnionThreads) union_kernel(const __grid_constant__ UnionParams p) {
  extern __shared__ __align__(16) uint32_t smem_union[];
  const int tid = threadIdx.x;
  const int threads = blockDim.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int group = blockIdx.x;
  const int row0 = group * kRows;
  const int row_words = (p.topk + 7) & -4;
  uint32_t* const s_map = smem_union;
  uint32_t* const s_idx = s_map + p.map_words;
  uint32_t* const s_slot = s_idx + kRows * row_words;
  uint8_t* const s_mask = reinterpret_cast<uint8_t*>(s_slot + p.cap);
  // [32] warp sums, twice, count, [6] seq lens, [8] requests
  uint32_t* const s_aux = s_slot + union_stage_words(p.cap, p.map_words, threads);

  device::PDLWaitPrimary<kUsePDL>();
  device::PDLTriggerSecondary<kUsePDL>();

  const uint32_t* s_row[kRows];
#pragma unroll
  for (int j = 0; j < kRows; ++j) {
    const int32_t* row = p.idx + (row0 + j) * p.stride_idx;
    const int phase = static_cast<int>((reinterpret_cast<uintptr_t>(row) >> 2) & 3);
    s_row[j] = s_idx + j * row_words + phase;
    for (int u = tid; 4 * u < phase + p.topk; u += threads) {
      const int c = 4 * u - phase;
      if (c >= 0 && c + 4 <= p.topk) {
        cp_async16(device::ptx::to_shared(s_row[j] + c), row + c, 16);
      } else {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (c + e >= 0 && c + e < p.topk) cp_async4(device::ptx::to_shared(s_row[j] + c + e), row + c + e);
        }
      }
    }
  }
  if (tid < kRows) {
    const int32_t* req = static_cast<const int32_t*>(p.row_req) + (kRowReq64 ? 2 : 1) * (row0 + tid);
    cp_async4(device::ptx::to_shared(s_aux + 34 + tid), p.seqlen + row0 + tid);
    cp_async4(device::ptx::to_shared(s_aux + 40 + 2 * tid), req);
    if constexpr (kRowReq64) cp_async4(device::ptx::to_shared(s_aux + 41 + 2 * tid), req + 1);
  }
  cp_async_commit();
  cp_async_wait<0>();
  if (tid == 0) {
    s_aux[32] = 0;
    s_aux[33] = 0;
  }
  __syncthreads();
  int seqlen[kRows];
  int maxlen = 0;
#pragma unroll
  for (int j = 0; j < kRows; ++j) {
    seqlen[j] = static_cast<int>(s_aux[34 + j]);
    maxlen = max(maxlen, seqlen[j]);
  }
  const auto request = [&](const int j) {
    const int64_t low = kRowReq64 ? static_cast<int64_t>(s_aux[40 + 2 * j]) : static_cast<int32_t>(s_aux[40 + 2 * j]);
    return kRowReq64 ? low | (static_cast<int64_t>(s_aux[41 + 2 * j]) << 32) : low;
  };
  const int words = (((maxlen + 7) >> 3) + 1) & -2;
  bool merged = kRows > 1 && words <= p.map_words;
#pragma unroll
  for (int j = 1; j < kRows; ++j)
    merged &= request(j) == request(0);
  if (merged) {
    for (int w = tid; w < words; w += threads)
      s_map[w] = 0;
    __syncthreads();
    bool twice = false;
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      for (int c = 4 * tid; c < p.topk; c += 4 * threads) {
        int pos[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          pos[e] = c + e < p.topk ? static_cast<int>(s_row[j][c + e]) : -1;
        if (pos[0] >= 0 && (pos[0] & 3) == 0 && pos[1] == pos[0] + 1 && pos[2] == pos[0] + 2 && pos[3] == pos[0] + 3 &&
            pos[3] < seqlen[j]) {
          const uint32_t bits = 0x1111u << (4 * (pos[0] & 7) + j);
          twice |= (atomicOr(s_map + (pos[0] >> 3), bits) & bits) != 0;
        } else {
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            if (pos[e] >= 0 && pos[e] < seqlen[j]) {
              const uint32_t bit = 1u << (4 * (pos[e] & 7) + j);
              twice |= (atomicOr(s_map + (pos[e] >> 3), bit) & bit) != 0;
            }
          }
        }
      }
    }
    if (twice) s_aux[32] = 1;
    __syncthreads();
    merged = s_aux[32] == 0;
  }
  if (merged) {
    // a thread counts 2^shift pairs of words: s_pre[q] entries precede pair q among them, s_first[t] precede thread t's
    const int32_t* r2t = p.req_to_token + request(0) * p.stride_r2t;
    uint16_t* const s_pre = reinterpret_cast<uint16_t*>(s_slot);
    uint32_t* const s_first = s_slot + p.map_words / 4;
    const int pairs = words >> 1;
    int shift = 0;
    while ((threads << shift) < pairs)
      ++shift;
    int n = 0;
    for (int q = tid << shift; q < min((tid + 1) << shift, pairs); ++q) {
      s_pre[q] = static_cast<uint16_t>(n);
      n += __popc(used_nibbles(s_map[2 * q])) + __popc(used_nibbles(s_map[2 * q + 1]));
    }
    const int scan = device::warp::inclusive_sum(n);
    if (lane == 31) s_aux[warp] = scan;
    __syncthreads();
    if (warp == 0) {
      const int total = lane < threads / 32 ? static_cast<int>(s_aux[lane]) : 0;
      const int before = device::warp::inclusive_sum(total);
      s_aux[lane] = before - total;
      if (lane == 31) s_aux[33] = before;
    }
    __syncthreads();
    s_first[tid] = s_aux[warp] + static_cast<uint32_t>(scan - n);
    __syncthreads();
    const int count = static_cast<int>(s_aux[33]);
    int32_t* const out = p.entries + static_cast<int64_t>(group) * p.cap;
    // entries before position `pos`, and the row masks of the 8 positions of its word
    const auto locate = [&](const int pos, uint32_t& word) {
      const int w = pos >> 3;
      word = s_map[w];
      int o = static_cast<int>(s_first[w >> (shift + 1)]) + s_pre[w >> 1];
      if ((w & 1) != 0) o += __popc(used_nibbles(s_map[w - 1]));
      return o + __popc(used_nibbles(word) & ((1u << (4 * (pos & 7))) - 1u));
    };
    // the lowest row that selected a position writes its entry: the slots of a thread's blocks are read together
    for (int c = 4 * tid; c < p.topk; c += 4 * threads) {
      int at[kRows][4];
      int slot[kRows][4];
      uint32_t mask[kRows][4];
#pragma unroll
      for (int j = 0; j < kRows; ++j) {
        int pos[4];
#pragma unroll
        for (int e = 0; e < 4; ++e)
          pos[e] = c + e < p.topk ? static_cast<int>(s_row[j][c + e]) : -1;
        const bool block = pos[0] >= 0 && (pos[0] & 3) == 0 && pos[1] == pos[0] + 1 && pos[2] == pos[0] + 2 &&
                           pos[3] == pos[0] + 3 && pos[3] < seqlen[j];
        uint32_t word = 0;
        const int first = block ? locate(pos[0], word) : 0;
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const bool valid = pos[e] >= 0 && pos[e] < seqlen[j];
          at[j][e] = first + e;
          if (!block && valid) at[j][e] = locate(pos[e], word);
          mask[j][e] = valid ? (word >> (4 * (pos[e] & 7))) & 15u : 0u;
          const bool mine = __ffs(mask[j][e]) - 1 == j;
          slot[j][e] = mine ? r2t[pos[e]] : -1;
          mask[j][e] = mine ? mask[j][e] : 0u;
        }
      }
#pragma unroll
      for (int j = 0; j < kRows; ++j) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          if (mask[j][e] != 0) {
            out[at[j][e]] = slot[j][e] >= 0 ? static_cast<int32_t>((mask[j][e] << kSlotBits) | slot[j][e]) : 0;
          }
        }
      }
    }
    for (int i = count + tid; i < ((count + kTile - 1) & -kTile); i += threads)
      out[i] = 0;
    if (tid == 0) p.counts[group] = count;
    return;
  } else {
    int last = 0;
#pragma unroll
    for (int j = 0; j < kRows; ++j) {
      const int32_t* r2t = p.req_to_token + request(j) * p.stride_r2t;
      for (int c = tid; c < p.topk; c += threads) {
        const int pos = static_cast<int>(s_row[j][c]);
        const bool ok = pos >= 0 && pos < seqlen[j];
        if (ok) {
          cp_async4(device::ptx::to_shared(s_slot + j * p.topk + c), r2t + pos);
          last = j * p.topk + c + 1;
        }
        s_mask[j * p.topk + c] = ok ? static_cast<uint8_t>(1u << j) : 0;
      }
    }
    last = device::warp::reduce_max(last);
    if (lane == 0) atomicMax(s_aux + 33, static_cast<uint32_t>(last));
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  const int count = static_cast<int>(s_aux[33]);
  int32_t* const out = p.entries + static_cast<int64_t>(group) * p.cap;
  for (int i = tid; i < ((count + kTile - 1) & -kTile); i += threads) {
    const int slot = static_cast<int>(s_slot[i]);
    const uint32_t mask = i < count ? s_mask[i] : 0u;
    out[i] = mask != 0 && slot >= 0 ? static_cast<int32_t>((mask << kSlotBits) | slot) : 0;
  }
  if (tid == 0) p.counts[group] = count;
}

template <int kRows, int kHeads>
struct Layout {
  static constexpr int kReal = kRows * kHeads;                        // head columns, query row major
  static constexpr int kNQ = (2 * kReal + 7) & -8;                    // QK N: hi terms, lo terms, zeros
  static constexpr int kNP = (kReal + (kRows > 1 ? 1 : 0) + 7) & -8;  // PV N: heads, sum(V) column, zeros
  static constexpr int kPer = ((kReal + 3) / 4 + 1) & -2;             // heads of one softmax warpgroup
  static constexpr int kWin = kPer <= 4 ? 4 : kPer <= 8 ? 8 : 16;     // TMEM columns it loads per term
  static constexpr int kQPlane = kNQ * 128;
  static constexpr int kCols = kNQ + 2 * kNP + kWin;  // S [kNQ], O [2][kNP]
  static constexpr int kTmemCols = kCols <= 64 ? 64 : kCols <= 128 ? 128 : 256;
  static constexpr int op_v = 0;                    // f16 V tile, 4 planes [slot][64 d]
  static constexpr int op_k = op_v + 4 * kPlane;    // fp8 K tiles of even / odd t, 2 planes [slot][128 d] each
  static constexpr int stg = op_k + 4 * kPlane;     // fp8 V items
  static constexpr int q = stg + kStages * kPlane;  // e4m3 Q terms, 2 planes [column][128 d]
  static constexpr int pr = q + 2 * kQPlane;        // f16 P [slot][128 B]
  static constexpr int ent = pr + kTile * 128;      // entries of 8 tiles
  static constexpr int red = ent + 8 * kTile * 4;   // [kNP][4 warps] f32
  static constexpr int fac = red + kNP * 16;        // [kNP] f32
  static constexpr int flag = fac + kNP * 4;        // [16 warps] u8, 2 x u32
  static constexpr int bar = flag + 24;             // mbarriers: QK, PV
  static constexpr int tmem = bar + 16;
  static constexpr int bytes = tmem + 8;
};

// Worker: the CTA's q heads as e4m3 terms in the 128 B-swizzled QK operand, rows [0, kReal) hi and
// [kReal, 2 kReal) lo. Returns the score scale in log2 units.
template <int kRows, int kHeads>
SGL_DEVICE float split_q(const Params& p, const int row0, const int kvh, uint8_t* const s_q, float* const s_red) {
  using Lay = Layout<kRows, kHeads>;
  constexpr int kReal = Lay::kReal;
  const int tid = threadIdx.x;
  const auto head = [&](const int n) {
    return p.q + (row0 + n / kHeads) * p.stride_qr + (kvh * kHeads + n % kHeads) * p.stride_qh;
  };
  uint32_t qtop = 0;
  for (int i = tid; i < kReal * 32; i += kWorkers) {
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
  for (int i = tid; i < kReal * 16; i += kWorkers) {
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
  for (int i = 2 * kReal * 16 + tid; i < Lay::kNQ * 16; i += kWorkers) {
    *reinterpret_cast<uint4*>(s_q + ((i & 15) >> 3) * Lay::kQPlane + tcgen05::sw128_off(i >> 4, i & 7)) =
        make_uint4(0, 0, 0, 0);
  }
  return p.qk_scale * __int_as_float(0x3FB8AA3B) * __int_as_float((127 - shift) << 23);
}

// The MMA warp: barrier 1 = the K tile of the next QK is in, barrier 2 = P and V of this tile are written.
template <int kRows, int kHeads>
SGL_DEVICE void issue_mmas(const int tiles, const uint32_t t_s, uint8_t* const smem) {
  using Lay = Layout<kRows, kHeads>;
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  const uint32_t* const s_more = reinterpret_cast<const uint32_t*>(smem + Lay::flag + 20);
  const bool issuer = (threadIdx.x & 31) == 0;
  const auto kdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::op_k);
  const auto qdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::q);
  auto vdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::op_v);
  vdesc.leading_byte_offset = kPlane >> 4;
  auto pdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::pr);
  pdesc.leading_byte_offset = 128 >> 4;
  const uint32_t idesc_qk = tcgen05::make_instr_desc_f8<kTile, Lay::kNQ>().bits;
  const uint32_t idesc_pv = tcgen05::make_instr_desc_f16(kTile, Lay::kNP, true, true);
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
  do {
    device::ptx::bar_sync(1, kThreads);
    if (tiles > 0) qk(0);
    for (int t = 0; t < tiles; ++t) {
      device::ptx::bar_sync(1, kThreads);
      if (t + 1 < tiles) qk(t + 1);
      device::ptx::bar_sync(2, kThreads);
      if (issuer) {
        tcgen05::fence_after_thread_sync();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          tcgen05::mma_f16_ss(
              t_s + Lay::kNQ + (i >> 3) * Lay::kNP,
              vdesc.advance_bytes((i >> 3) * 2 * kPlane + (i & 7) * 2048).bits,
              pdesc.advance_bytes((i & 7) * 2048).bits,
              idesc_pv,
              t > 0 || (i & 7) > 0);
        }
        tcgen05::commit(s_bar + 1);
      }
    }
    device::ptx::bar_sync(1, kThreads);
  } while (*s_more != 0);
}

// Every thread of the CTA: the group with the `rank`-th most slots, ties in group order. s_count holds p.groups + 1
// words.
SGL_DEVICE int group_of_rank(const Params& p, const int rank, uint32_t* const s_count) {
  const int tid = threadIdx.x;
  for (int g = tid; g < p.groups; g += kThreads)
    cp_async4(device::ptx::to_shared(s_count + g), p.counts + g);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  for (int g = tid; g < p.groups; g += kThreads) {
    const uint32_t mine = s_count[g];
    int before = 0;
    for (int h = 0; h < p.groups; ++h) {
      const uint32_t other = s_count[h];
      before += other > mine || (other == mine && h < g) ? 1 : 0;
    }
    if (before == rank) s_count[p.groups] = static_cast<uint32_t>(g);
  }
  __syncthreads();
  return static_cast<int>(s_count[p.groups]);
}

// kSorted: block b attends the group with the b / HKV-th most slots instead of group b / HKV, so that with more
// CTAs than SMs the short lists are the ones that wait for an SM.
template <int kRows, int kHeads, bool kUsePDL, bool kSorted>
__global__ void __launch_bounds__(kThreads, 1) attn_kernel(const __grid_constant__ Params p) {
  using Lay = Layout<kRows, kHeads>;
  static_assert(Lay::bytes <= kMaxSmem && Lay::kPer <= 16);
  constexpr int kReal = Lay::kReal;
  constexpr int kNP = Lay::kNP;
  constexpr int kPer = Lay::kPer;
  constexpr int kWin = Lay::kWin;
  constexpr int kBlocks = kNP / 8;             // 8-column blocks of O
  constexpr int kBlocks0 = (kBlocks + 1) / 2;  // those of warpgroups 0 and 1; 2 and 3 have the rest
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wg = warp >> 2;
  const int tl = tid & 127;
  int group = blockIdx.x / p.HKV;
  const int kvh = blockIdx.x % p.HKV;
  int row0 = group * kRows;
  uint8_t* const s_opv = smem + Lay::op_v;
  uint8_t* const s_opk = smem + Lay::op_k;
  uint8_t* const s_stg = smem + Lay::stg;
  uint8_t* const s_p = smem + Lay::pr;
  uint32_t* const s_ent = reinterpret_cast<uint32_t*>(smem + Lay::ent);
  float* const s_red = reinterpret_cast<float*>(smem + Lay::red);
  float* const s_fac = reinterpret_cast<float*>(smem + Lay::fac);
  uint8_t* const s_flag = smem + Lay::flag;
  uint32_t* const s_any = reinterpret_cast<uint32_t*>(smem + Lay::flag + 16);
  uint32_t* const s_more = s_any + 1;
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  uint32_t* const s_tmem = reinterpret_cast<uint32_t*>(smem + Lay::tmem);
  const float ninf = __int_as_float(0xFF800000);
  const bool worker = warp < kWorkers / 32;

  if (tid == 0) {
    device::ptx::mbar_init(s_bar, 1);
    device::ptx::mbar_init(s_bar + 1, 1);
    device::ptx::fence_barrier_init();
  }
  float sc = 0.f;
  if (worker) {
    if constexpr (!kSorted) {
      sc = split_q<kRows, kHeads>(p, row0, kvh, smem + Lay::q, s_red);
      device::ptx::fence_proxy_async_shared();
    }
  } else {
    tcgen05::alloc(s_tmem, Lay::kTmemCols);
    tcgen05::relinquish_alloc_permit();
    tcgen05::fence_before_thread_sync();
  }
  __syncthreads();
  tcgen05::fence_after_thread_sync();

  const uint32_t t_s = *s_tmem + (static_cast<uint32_t>((warp & 3) * 32) << 16);
  device::PDLWaitPrimary<kUsePDL>();
  if constexpr (kSorted) {
    group = group_of_rank(p, group, reinterpret_cast<uint32_t*>(s_p));
    row0 = group * kRows;
    if (worker) {
      sc = split_q<kRows, kHeads>(p, row0, kvh, smem + Lay::q, s_red);
      device::ptx::fence_proxy_async_shared();
    }
  }
  const int32_t* entries = p.entries + static_cast<int64_t>(group) * p.cap;
  const int tiles = (p.counts[group] + kTile - 1) / kTile;

  if (!worker) {
    issue_mmas<kRows, kHeads>(tiles, t_s, smem);
    tcgen05::dealloc(*s_tmem, Lay::kTmemCols);
    return;
  }

  // a worker copies 16-byte chunk gc of the K rows of slots gk + 32 i and chunk vc of the V rows of slots vk, vk + 64;
  // lanes 8 j..8 j + 7 take V rows j and j ^ 5, so a warp's 8 copies of one f16 plane fill the 32 banks once
  const int gk = tid >> 4;
  const int gc = tid & 15;
  const int vk = ((tid >> 6) << 3) | (((tid >> 3) & 3) ^ (((tid >> 2) & 1) * 5));
  const int vc = (tid & 3) | (((tid >> 5) & 1) << 2);
  const int first = wg * kPer;  // this thread's heads: first + i, i < heads
  const int heads = min(max(kReal - first, 0), kPer);
  const int j0 = first / kHeads;  // their query rows: j0, then j0 + 1 from head i = split
  const int split = (j0 + 1) * kHeads - first;
  const int ob = (wg >> 1) * kBlocks0;  // this thread's blocks of O, of d half wg & 1: ob + b, b < blocks
  const int blocks = (wg >> 1) == 0 ? kBlocks0 : kBlocks - kBlocks0;
  const uint32_t t_om = t_s + Lay::kNQ + (wg & 1) * kNP + 8 * ob;
  uint32_t n_qk = 0;
  uint32_t n_pv = 0;
  bool repeat = false;
#pragma unroll 1
  for (int pass = 0; pass < (repeat ? 1 + kRows : 1); ++pass) {
    // pass 0 takes every row; after a NaN in V, pass 1 + j takes row j alone
    const uint32_t rowmask = pass == 0 ? (1u << kRows) - 1 : 1u << (pass - 1);

    const auto load_entries = [&](const int t) {
      if (t < tiles && tid < 32) {
        cp_async16(device::ptx::to_shared(s_ent + (t & 7) * kTile + tid * 4), entries + t * kTile + tid * 4, 16);
      }
    };
    const auto gather_k = [&](const int t) {
      if (t < tiles) {
        const uint8_t* base = p.k + kvh * p.stride_kh + gc * 16;
        const uint32_t dst = device::ptx::to_shared(s_opk + ((t & 1) * 2 + (gc >> 3)) * kPlane);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const uint32_t entry = s_ent[(t & 7) * kTile + gk + 32 * i];
          cp_async16(
              dst + tcgen05::sw128_off(gk + 32 * i, gc & 7),
              base + static_cast<int64_t>(entry & kSlotMask) * p.stride_kslot,
              ((entry >> kSlotBits) & rowmask) != 0 ? 16 : 0);
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
          const uint32_t entry = s_ent[((item >> 1) & 7) * kTile + vk + 64 * i];
          cp_async16(
              dst + tcgen05::sw128_off(vk + 64 * i, vc),
              base + static_cast<int64_t>(entry & kSlotMask) * p.stride_kslot,
              ((entry >> kSlotBits) & rowmask) != 0 ? 16 : 0);
        }
      }
      cp_async_commit();
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
    // P of this thread's slot and heads in pairs, then the sum(V) column at kReal
    const auto store_p = [&](const float (&pv)[kPer], const uint32_t mine) {
      uint8_t* row = s_p + tl * 128;
#pragma unroll
      for (int i = 0; i < kPer; i += 2) {
        const int n = first + i;
        if (i < heads) {
          *reinterpret_cast<uint32_t*>(row + (((n >> 3) ^ (tl & 7)) << 4) + ((2 * n) & 15)) =
              f32x2_f16x2(pv[i], i + 1 < kPer ? pv[i + 1] : 0.f);
        }
      }
      if (kRows > 1 && first <= kReal - 1 && kReal - 1 < first + kPer) {
        const float seen = mine != 0 ? 1.f : 0.f;
        constexpr int n = kReal & -2;
        uint32_t pair = f32x2_f16x2(seen, 0.f);
        if constexpr ((kReal & 1) != 0) pair = f32x2_f16x2(select_at(pv, kReal - 1 - first), seen);
        *reinterpret_cast<uint32_t*>(row + (((n >> 3) ^ (tl & 7)) << 4) + ((2 * n) & 15)) = pair;
      }
    };
    // scores of this thread's slot and heads in log2 units: (hi + lo) * sc + bias, -inf where not selected
    const auto load_scores = [&](float (&x)[kPer], const float (&bias)[kPer], const uint32_t mine) {
      const bool on0 = ((mine >> j0) & 1u) != 0;
      const bool on1 = ((mine >> (j0 + 1)) & 1u) != 0;
      float term[kWin];
      if (heads > 0) tcgen05::ld_32x32b<kWin>(t_s + kReal + first, term);
      tcgen05::wait_ld();
#pragma unroll
      for (int i = 0; i < kPer; ++i)
        x[i] = fmaf(term[i], sc, bias[i]);
      if (heads > 0) tcgen05::ld_32x32b<kWin>(t_s + first, term);
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
    // copy groups in flight at the top of tile t, oldest first: V(t, 0), K(t + 1), V(t, 1), V(t + 1, 0)
    gather_k(0);
    cp_async_commit();
    gather_v(0, 0);
    gather_k(1);
    cp_async_commit();
    gather_v(1, 1);
    gather_v(2, 2);
    cp_async_wait<4>();
    device::ptx::fence_proxy_async_shared();
    device::ptx::bar_arrive(1, kThreads);

    float nm[kPer];  // -reference of the head, log2 units
    float ls[kPer];  // sum of p over this thread's slots
    float zero[kPer];
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      nm[i] = 1e30f;
      ls[i] = 0.f;
      zero[i] = 0.f;
    }

    uint32_t stage = 0;  // (2 t) % kStages
#pragma unroll 1
    for (int t = 0; t < tiles; ++t) {
      device::ptx::mbar_wait_parity(s_bar, n_qk & 1);
      ++n_qk;
      tcgen05::fence_after_thread_sync();
      // even warps issue their K copies before the softmax and odd warps after it
      if ((warp & 1) == 0) {
        gather_k(t + 2);
        load_entries(t + 5);
        cp_async_commit();
      }
      const uint32_t mine = (s_ent[(t & 7) * kTile + tl] >> kSlotBits) & rowmask;
      float pv[kPer];
      load_scores(pv, nm, mine);
      float top = ninf;
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        top = fmaxf(top, pv[i]);
        pv[i] = ex2_approx(pv[i]);
      }
      top = device::warp::reduce_max(top);
      if (lane == 0) s_flag[warp] = top > kRescale ? 1 : 0;
      device::ptx::bar_sync(3, kWorkers);
      if ((warp & 1) != 0) {
        gather_k(t + 2);
        cp_async_commit();
      }

      const uint2 flags = *reinterpret_cast<const uint2*>(s_flag);
      const uint2 flags2 = *reinterpret_cast<const uint2*>(s_flag + 8);
      if ((flags.x | flags.y | flags2.x | flags2.y) != 0) {
        // some head is more than kRescale above its reference (or has none yet): move those references to the
        // tile maxima, rescale their sums and O, and redo this tile's P
        tcgen05::fence_after_thread_sync();
        load_scores(pv, zero, mine);
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
            const float ref = -nm[i];
            const bool move = m > ref + kRescale;
            const float f = move ? ex2_approx(ref - m) : 1.f;
            nm[i] = move ? -m : nm[i];
            ls[i] *= f;
            pv[i] = ex2_approx(pv[i] + nm[i]);
            moved |= move && ref > -1e29f;
            if (lane == 0 && (warp & 3) == 0) s_fac[first + i] = f;
          }
        }
        if (moved && lane == 0 && (warp & 3) == 0) *s_any = 1;
        device::ptx::bar_sync(3, kWorkers);
        if (*s_any != 0 && t > 0) {
          wait_pv();
          tcgen05::fence_after_thread_sync();
#pragma unroll
          for (int b = 0; b < kBlocks0; ++b) {
            if (b < blocks) {
              float o[8];
              tcgen05::ld_32x32b<8>(t_om + 8 * b, o);
              tcgen05::wait_ld();
#pragma unroll
              for (int e = 0; e < 8; ++e) {
                if (8 * (ob + b) + e < kReal) o[e] *= s_fac[8 * (ob + b) + e];
              }
              tcgen05::st_32x32b<8>(t_om + 8 * b, o);
            }
          }
          tcgen05::wait_st();
          tcgen05::fence_before_thread_sync();
        }
      }
#pragma unroll
      for (int i = 0; i < kPer; ++i)
        ls[i] += pv[i];
      // K(t + 1) is in and S of this tile is read: QK of tile t + 1 may go
      cp_async_wait<3>();
      device::ptx::fence_proxy_async_shared();
      device::ptx::bar_arrive(1, kThreads);

      // the P and V operands are free once the PV of tile t - 1 is in
      wait_pv();
      store_p(pv, mine);
      const uint32_t stage1 = (stage + 1) % 3u;
      cp_async_wait<2>();
      convert_v(2 * t, stage);
      convert_v(2 * t + 1, stage1);
      device::ptx::fence_proxy_async_shared();
      ++n_pv;
      device::ptx::bar_arrive(2, kThreads);
      gather_v(2 * t + 3, stage);
      gather_v(2 * t + 4, stage1);
      stage = (stage + 2) % 3u;
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
    if (tid == 0) *s_any = 0;
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
    uint16_t* const s_out = reinterpret_cast<uint16_t*>(s_opv);  // [kReal][256] bf16
    const int d = (wg & 1) * 128 + tl;
    bool poisoned = false;
#pragma unroll
    for (int b = 0; b < kBlocks0; ++b) {
      if (b < blocks) {
        float o[8];
#pragma unroll
        for (int e = 0; e < 8; ++e)
          o[e] = 0.f;
        if (tiles > 0) {
          tcgen05::ld_32x32b<8>(t_om + 8 * b, o);
          tcgen05::wait_ld();
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int n = 8 * (ob + b) + e;
          if (n < kReal) {
            s_out[n * kD + d] = __bfloat16_as_ushort(__float2bfloat16_rn(o[e] * s_fac[n]));
          } else if (kRows > 1 && n == kReal && pass == 0) {
            poisoned = o[e] != o[e];
          }
        }
      }
    }
    if (poisoned) *s_any = 1;
    tcgen05::fence_before_thread_sync();
    device::ptx::bar_sync(3, kWorkers);
    if (kRows > 1 && pass == 0) repeat = *s_any != 0;
    for (int i = tid; i < kReal * 32; i += kWorkers) {
      const int n = i >> 5;
      const int j = n / kHeads;
      if (((rowmask >> j) & 1u) != 0) {
        uint16_t* dst = p.out + (row0 + j) * p.stride_or + (kvh * kHeads + n % kHeads) * p.stride_oh + (i & 31) * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(s_out + n * kD + (i & 31) * 8);
      }
    }
    if (tid == 0) *s_more = pass + 1 < (repeat ? 1 + kRows : 1) ? 1 : 0;
    device::ptx::bar_sync(3, kWorkers);
    device::ptx::bar_arrive(1, kThreads);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_sparse_decode_sm100

template <int kRows, int kHeads, bool kUsePDL>
struct QsaSparseDecodeSm100 {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView v,
      const tvm::ffi::TensorView req_to_token,
      const tvm::ffi::TensorView row_req,
      const tvm::ffi::TensorView idx,
      const tvm::ffi::TensorView seqlen,
      const tvm::ffi::TensorView entries,
      const tvm::ffi::TensorView counts,
      const tvm::ffi::TensorView out,
      double qk_scale) {
    using namespace host;
    using namespace qsa_sparse_decode_sm100;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto S_ = SymbolicSize{"slots"};
    auto HKV_ = SymbolicSize{"kv_heads"};
    auto NR_ = SymbolicSize{"reqs"};
    auto CTX_ = SymbolicSize{"ctx"};
    auto TOPK_ = SymbolicSize{"topk"};
    auto NG_ = SymbolicSize{"groups"};
    auto CAP_ = SymbolicSize{"cap"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(q);
    // fp8 e4m3 pools passed as uint8 views
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(k);
    TensorMatcher({S_, HKV_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, -1, 1}).verify(v);
    TensorMatcher({NR_, CTX_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(req_to_token);
    TensorMatcher({R_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({1}).verify(row_req);
    TensorMatcher({R_, TOPK_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(idx);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(seqlen);
    TensorMatcher({NG_, CAP_}).with_dtype<int32_t>().with_device(device).with_strides({CAP_, 1}).verify(entries);
    TensorMatcher({NG_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(counts);
    TensorMatcher({R_, HQ_, kD}).with_dtype<bf16_t>().with_device(device).with_strides({-1, -1, 1}).verify(out);
    const int64_t rows = R_.unwrap();
    const int64_t hkv = HKV_.unwrap();
    const int64_t topk = TOPK_.unwrap();
    const int64_t groups = NG_.unwrap();
    const int64_t cap = CAP_.unwrap();
    RuntimeCheck(HQ_.unwrap() == hkv * kHeads, "q heads per kv head must match the kernel");
    RuntimeCheck(rows == groups * kRows, "rows must be a multiple of the rows per request");
    RuntimeCheck(topk > 0 && cap % kTile == 0 && cap >= kRows * topk, "entries must hold every column in whole tiles");
    RuntimeCheck(S_.unwrap() <= (int64_t{1} << kSlotBits), "KV pool too large for the packed slot");
    RuntimeCheck(k.stride(0) == v.stride(0) && k.stride(1) == v.stride(1), "K and V pools must share strides");
    const auto a16 = [](const void* ptr) { return reinterpret_cast<uintptr_t>(ptr) % 16 == 0; };
    RuntimeCheck(
        a16(k.data_ptr()) && a16(v.data_ptr()) && k.stride(0) % 16 == 0 && k.stride(1) % 16 == 0,
        "K / V rows must be 16-byte aligned");
    RuntimeCheck(a16(q.data_ptr()) && q.stride(0) % 8 == 0 && q.stride(1) % 8 == 0, "q rows must be 16-byte aligned");
    RuntimeCheck(
        a16(out.data_ptr()) && out.stride(0) % 8 == 0 && out.stride(1) % 8 == 0, "out rows must be 16-byte aligned");
    RuntimeCheck(a16(entries.data_ptr()), "entries must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(idx.data_ptr()) % 4 == 0, "indices must be 4-byte aligned");
    if (rows == 0) return;
    const DLDevice dev = device.unwrap();
    // without room for 4 bits per position of the context the rows of a request are not merged
    const int64_t map_words = kRows > 1 ? ((CTX_.unwrap() + 7) / 8 + 3) & -4 : 0;
    const int union_threads = kRows > 1 ? kUnionThreads : 256;
    const int64_t row_words = kRows * ((topk + 7) & -4) + 48;
    const bool mapped =
        (map_words + row_words + union_stage_words(cap, map_words, union_threads)) * 4 <= kMaxSmem && cap < 65536;
    const auto union_params = UnionParams{
        .idx = static_cast<const int32_t*>(idx.data_ptr()),
        .seqlen = static_cast<const int32_t*>(seqlen.data_ptr()),
        .req_to_token = static_cast<const int32_t*>(req_to_token.data_ptr()),
        .row_req = row_req.data_ptr(),
        .entries = static_cast<int32_t*>(entries.data_ptr()),
        .counts = static_cast<int32_t*>(counts.data_ptr()),
        .stride_idx = idx.stride(0),
        .stride_r2t = req_to_token.stride(0),
        .topk = static_cast<int32_t>(topk),
        .cap = static_cast<int32_t>(cap),
        .map_words = static_cast<int32_t>(mapped ? map_words : 0),
    };
    const auto params = Params{
        .q = static_cast<const uint16_t*>(q.data_ptr()),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .v = static_cast<const uint8_t*>(v.data_ptr()),
        .entries = static_cast<const int32_t*>(entries.data_ptr()),
        .counts = static_cast<const int32_t*>(counts.data_ptr()),
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
        .groups = static_cast<int32_t>(groups),
    };
    const std::size_t union_smem =
        static_cast<std::size_t>(
            union_params.map_words + row_words + union_stage_words(cap, union_params.map_words, union_threads)) *
        4;
    if (is_type<int64_t>(row_req.dtype())) {
      launch_union<true>(groups, union_threads, union_smem, union_params, dev);
    } else {
      launch_union<false>(groups, union_threads, union_smem, union_params, dev);
    }
    // the rows of a request make lists of different lengths; single rows all have their top-k
    if (kRows > 1 && groups * hkv > runtime::get_sm_count(dev.device_id) && groups < kTile * 32) {
      launch_attn<true>(groups * hkv, params, dev);
    } else {
      launch_attn<false>(groups * hkv, params, dev);
    }
  }

 private:
  template <bool kSorted>
  static void launch_attn(const int64_t blocks, const qsa_sparse_decode_sm100::Params& params, const DLDevice dev) {
    using namespace host;
    using namespace qsa_sparse_decode_sm100;
    constexpr int smem = Layout<kRows, kHeads>::bytes;
    runtime::set_max_dynamic_smem_per_device<attn_kernel<kRows, kHeads, kUsePDL, kSorted>>(dev, smem);
    LaunchKernel(static_cast<uint32_t>(blocks), kThreads, dev, smem)
        .enable_pdl(kUsePDL)(attn_kernel<kRows, kHeads, kUsePDL, kSorted>, params);
  }

  template <bool kRowReq64>
  static void launch_union(
      const int64_t groups,
      const int threads,
      const std::size_t smem,
      const qsa_sparse_decode_sm100::UnionParams& params,
      const DLDevice dev) {
    using namespace host;
    using qsa_sparse_decode_sm100::kMaxSmem;
    using qsa_sparse_decode_sm100::union_kernel;
    runtime::set_max_dynamic_smem_per_device<union_kernel<kRows, kRowReq64, kUsePDL>>(dev, kMaxSmem);
    LaunchKernel(static_cast<uint32_t>(groups), threads, dev, smem)
        .enable_pdl(kUsePDL)(union_kernel<kRows, kRowReq64, kUsePDL>, params);
  }
};

}  // namespace sglang
