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
  float* part;         // [splits][HKV][2] partials of split lists
  int32_t* split_cnt;  // [splits][HKV] halves done, zero between launches
  int32_t splits;
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

// Every thread of the CTA: how many groups have more slots than `group` (ties in group order); groups <= kThreads.
SGL_DEVICE int rank_of_group(const Params& p, const int group, uint32_t* const s_count) {
  const int tid = threadIdx.x;
  for (int g = tid; g < p.groups; g += kThreads)
    cp_async4(device::ptx::to_shared(s_count + g), p.counts + g);
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();
  const uint32_t mine = s_count[group];
  const uint32_t other = tid < p.groups ? s_count[tid] : 0u;
  return __syncthreads_count(tid < p.groups && (other > mine || (other == mine && tid < group)));
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

  // a worker moves 16-byte chunk gc of the K rows of slots gk + 32 i of a tile, and chunk vc of the V rows of
  // slots vk and vk + 64 of an item: one cache line per copy. A warp has 4 chunks of 8 V rows, all of one f16
  // plane, and its lanes 8 j to 8 j + 7 have rows j and j ^ 5: their 8 copies fill the 32 banks once, staged or
  // converted
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

// Block-scaled FP8 PV: V is the MMA operand as gathered, P is two e4m3 terms with a UE8M0 scale per head and 32 slots.
template <int kRows, int kHeads>
struct LayoutPv8 {
  static constexpr int kReal = kRows * kHeads;
  static constexpr int kNQ = (2 * kReal + 7) & -8;
  static constexpr int kPer = ((kReal + 3) / 4 + 1) & -2;
  static constexpr int kWin = kPer <= 4 ? 4 : kPer <= 8 ? 8 : 16;
  static constexpr int kSeg = kWin;
  static constexpr int kNP = 8 * kSeg;
  static constexpr int kBlk = kSeg < 8 ? kSeg : 8;
  static constexpr int kQPlane = kNQ * 128;
  static constexpr int kOCol = 2 * kNQ;
  static constexpr int kSfaCol = (kOCol + 2 * kNP + 3) & -4;
  static constexpr int kSfbCol = kSfaCol + 4;
  static constexpr int kCols = kSfbCol + 8;
  static constexpr int kTmemCols = kCols <= 128 ? 128 : kCols <= 256 ? 256 : 512;
  static constexpr int k = 0;
  static constexpr int v = k + 4 * kPlane;
  static constexpr int q = v + 6 * kPlane;
  static constexpr int pr = q + 2 * kQPlane;
  static constexpr int sf = pr + 2 * kTile * 128;
  static constexpr int ent = sf + 3 * 512;
  static constexpr int red = ent + 8 * kTile * 4;
  static constexpr int fac = red + kNP * 16;
  static constexpr int nmr = fac + kNP * 4;
  static constexpr int fo = nmr + kNP * 4;
  static constexpr int flag = fo + kNP * 4;
  static constexpr int bar = flag + 56;
  static constexpr int tmem = bar + 11 * 8;
  static constexpr int bytes = tmem + 8;
  static constexpr int kRec = kReal * (kD + 2) + 4;  // O [kReal][kD], sum [kReal], reference [kReal], NaN flag
};

SGL_DEVICE bool bar_red_or(const uint32_t id, const uint32_t threads, const bool pred) {
  uint32_t any;
  asm volatile(
      "{\n.reg .pred p, q;\nsetp.ne.u32 q, %1, 0;\nbar.red.or.pred p, %2, %3, q;\nselp.u32 %0, 1, 0, p;\n}"
      : "=r"(any)
      : "r"(static_cast<uint32_t>(pred)), "r"(id), "r"(threads)
      : "memory");
  return any != 0;
}

SGL_DEVICE void cp_async_arrive_noinc(const uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}

SGL_DEVICE float ex2_ftz(const float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

// The MMA warp. Barrier 1 brackets a pass; mbarrier 9 + g % 2: P, scales and O of tile g are written, S of g is read.
// Tile g of the CTA's sequence: V in slot g % 3, K and S and P in slot g % 2.
template <int kRows, int kHeads>
SGL_DEVICE void issue_mmas_pv8(const uint32_t t_base, uint8_t* const smem) {
  using Lay = LayoutPv8<kRows, kHeads>;
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  uint64_t* const qk_bar = s_bar;
  uint64_t* const pv_bar = s_bar + 2;
  uint64_t* const k_full = s_bar + 4;
  uint64_t* const v_full = s_bar + 6;
  const uint32_t* const s_more = reinterpret_cast<const uint32_t*>(smem + Lay::flag + 20);
  const bool issuer = (threadIdx.x & 31) == 0;
  const auto kdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::k);
  const auto qdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::q);
  const auto vdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::v);
  const auto pdesc = tcgen05::make_desc_kmajor_sw128(smem + Lay::pr);
  const uint32_t idesc_qk = tcgen05::make_instr_desc_f8<kTile, Lay::kNQ>().bits;
  auto idesc_pv = tcgen05::make_instr_desc_mxf8<kTile, Lay::kNP>();
  idesc_pv.bits |= (1u << 15) | (1u << 16);
  if (issuer) {
    tcgen05::cp_32x128b_warpx4(t_base + Lay::kSfaCol, tcgen05::make_desc_sf_chunk(smem + Lay::sf).bits);
  }
  int g0 = 0;
  const auto qk = [&](const int g) {
    if (issuer) {
      device::ptx::mbar_wait_parity(k_full + (g & 1), (g >> 1) & 1);
      device::ptx::fence_proxy_async_shared();
      tcgen05::fence_after_thread_sync();
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        tcgen05::mma_f8_ss(
            t_base + (g & 1) * Lay::kNQ,
            kdesc.advance_bytes(((g & 1) * 2 + (i >> 2)) * kPlane + (i & 3) * 32).bits,
            qdesc.advance_bytes((i >> 2) * Lay::kQPlane + (i & 3) * 32).bits,
            idesc_qk,
            i > 0);
      }
      tcgen05::commit(qk_bar + (g & 1));
    }
  };
  do {
    device::ptx::bar_sync(1, kThreads);
    const int tiles = static_cast<int>(*(s_more + 1));
    if (tiles > 0) qk(g0);
    if (tiles > 1) qk(g0 + 1);
    for (int t = 0; t < tiles; ++t) {
      const int g = g0 + t;
      if (issuer) {
        device::ptx::mbar_wait_parity(s_bar + 9 + (g & 1), (g >> 1) & 1);
        device::ptx::mbar_wait_parity(v_full + g % 3, (g / 3) & 1);
        device::ptx::fence_proxy_async_shared();
        tcgen05::fence_after_thread_sync();
        tcgen05::cp_32x128b_warpx4(
            t_base + Lay::kSfbCol + 4 * (g & 1), tcgen05::make_desc_sf_chunk(smem + Lay::sf + 512 * (1 + (g & 1))).bits);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int h = i >> 2;
          const int kk = i & 3;
          tcgen05::mma_mxf8_ss(
              t_base + Lay::kOCol + h * Lay::kNP,
              vdesc.advance_bytes(((g % 3) * 2 + h) * kPlane + kk * 4096).bits,
              pdesc.advance_bytes((g & 1) * kTile * 128 + kk * 4096).bits,
              idesc_pv.with_sf_id(kk).bits,
              t_base + Lay::kSfaCol,
              t_base + Lay::kSfbCol + 4 * (g & 1),
              t > 0 || kk > 0);
        }
        tcgen05::commit(pv_bar + (g & 1));
      }
      if (t + 2 < tiles) qk(g + 2);
    }
    g0 += tiles;
    device::ptx::bar_sync(1, kThreads);
  } while (*s_more != 0);
}

// Every thread: the list half this CTA attends when the longest lists are split over spare SMs. s_split gets
// {half (-1 = whole list), split index, group}.
__noinline__ __device__ void split_assign(const Params& p, const bool spare, const int group, int32_t* const s_split,
                                          uint32_t* const s_count) {
  int seg = -1;
  int sidx = 0;
  int g = group;
  if (spare) {
    sidx = (static_cast<int>(blockIdx.x) - p.groups * p.HKV) / p.HKV;
    g = group_of_rank(p, sidx, s_count);
    seg = 1;
  } else {
    sidx = rank_of_group(p, group, s_count);
    seg = sidx < p.splits ? 0 : -1;
  }
  if (threadIdx.x == 0) {
    s_split[0] = seg;
    s_split[1] = sidx;
    s_split[2] = g;
  }
  __syncthreads();
}

// Workers of a CTA that attended one half of a split list: its unnormalized O, sums and references go to its partial,
// and the second half to finish gets the factors of both halves in s_fac (first half) and s_fo (second half).
template <int kRows, int kHeads>
__noinline__ __device__ void split_partial(const Params& p, uint8_t* const smem, const int kvh, const int tiles) {
  using Lay = LayoutPv8<kRows, kHeads>;
  constexpr int kReal = Lay::kReal;
  constexpr int kNP = Lay::kNP;
  constexpr int kSeg = Lay::kSeg;
  constexpr int kBlk = Lay::kBlk;
  constexpr int kPer = Lay::kPer;
  const int tid = threadIdx.x;
  const int wg = tid >> 7;
  const int tl = tid & 127;
  const int first = wg * kPer;
  const int heads = kReal == 4 * kPer ? kPer : min(max(kReal - first, 0), kPer);
  float* const s_red = reinterpret_cast<float*>(smem + Lay::red);
  float* const s_fac = reinterpret_cast<float*>(smem + Lay::fac);
  float* const s_nm = reinterpret_cast<float*>(smem + Lay::nmr);
  float* const s_fo = reinterpret_cast<float*>(smem + Lay::fo);
  uint32_t* const s_any = reinterpret_cast<uint32_t*>(smem + Lay::flag + 16);
  uint32_t* const s_last = s_any + 3;
  const int32_t* const s_split = reinterpret_cast<const int32_t*>(s_any + 4);
  const int seg = s_split[0];
  const int sidx = s_split[1];
  float* const rec = p.part + static_cast<int64_t>((sidx * p.HKV + kvh) * 2 + (seg & 1)) * Lay::kRec;
  const float* const other = p.part + static_cast<int64_t>((sidx * p.HKV + kvh) * 2 + ((seg & 1) ^ 1)) * Lay::kRec;
  const uint32_t t_o0 = *reinterpret_cast<const uint32_t*>(smem + Lay::tmem) +
                        (static_cast<uint32_t>(((tid >> 5) & 3) * 32) << 16) + Lay::kOCol;
  if (tl < heads) {
    const float4 r4 = *reinterpret_cast<const float4*>(s_red + (first + tl) * 4);
    rec[kReal * kD + first + tl] = (r4.x + r4.y) + (r4.z + r4.w);
    rec[kReal * kD + kReal + first + tl] = -s_nm[first + tl];
  }
  bool poisoned = false;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int dd = h * 128 + tl;
#pragma unroll
    for (int b = 0; b < kSeg; b += kBlk) {
      float o[kBlk];
      float o2[kBlk];
#pragma unroll
      for (int e = 0; e < kBlk; ++e) {
        o[e] = 0.f;
        o2[e] = 0.f;
      }
      if (tiles > 0) {
        tcgen05::ld_32x32b<kBlk>(t_o0 + h * kNP + wg * kSeg + b, o);
        tcgen05::ld_32x32b<kBlk>(t_o0 + h * kNP + 4 * kSeg + wg * kSeg + b, o2);
        tcgen05::wait_ld();
      }
#pragma unroll
      for (int e = 0; e < kBlk; ++e) {
        if (b + e < heads) {
          rec[(first + b + e) * kD + dd] = o[e] + o2[e];
        } else if (kRows > 1 && wg == 3 && b + e == kSeg - 1) {
          poisoned |= o[e] != o[e];
        }
      }
    }
  }
  if (poisoned) *s_any = 1;
  device::ptx::bar_sync(3, kWorkers);
  if (tid == 0) {
    rec[kReal * (kD + 2)] = *s_any != 0 ? 1.f : 0.f;
    __threadfence();
    const int done = atomicAdd(p.split_cnt + sidx * p.HKV + kvh, 1);
    if (done != 0) p.split_cnt[sidx * p.HKV + kvh] = 0;
    *s_last = static_cast<uint32_t>(done);
    __threadfence();
  }
  device::ptx::bar_sync(3, kWorkers);
  if (*s_last != 0) {
    if (tid == 0 && other[kReal * (kD + 2)] != 0.f) *s_any = 1;
    if (tl < heads) {
      const int n = first + tl;
      const bool first_half = seg == 0;
      const float ls0 = first_half ? rec[kReal * kD + n] : other[kReal * kD + n];
      const float ref0 = first_half ? rec[kReal * kD + kReal + n] : other[kReal * kD + kReal + n];
      const float ls1 = first_half ? other[kReal * kD + n] : rec[kReal * kD + n];
      const float ref1 = first_half ? other[kReal * kD + kReal + n] : rec[kReal * kD + kReal + n];
      const float m = fmaxf(ref0, ref1);
      const float f0 = ex2_ftz(ref0 - m);
      const float f1 = ex2_ftz(ref1 - m);
      const float l = fmaf(ls0, f0, ls1 * f1);
      const float inv = l > 0.f ? 1.f / l : 0.f;
      s_fac[n] = f0 * inv;
      s_fo[n] = f1 * inv;
    }
  }
}

template <int kRows, int kHeads, bool kUsePDL, bool kSorted>
__global__ void __launch_bounds__(kThreads, 1) attn_pv8_kernel(const __grid_constant__ Params p) {
  using Lay = LayoutPv8<kRows, kHeads>;
  static_assert(Lay::bytes <= kMaxSmem && Lay::kPer <= 16 && Lay::kNP <= 256);
  static_assert(kRows == 1 || Lay::kReal - 3 * Lay::kPer < Lay::kSeg);
  constexpr int kReal = Lay::kReal;
  constexpr int kNP = Lay::kNP;
  constexpr int kSeg = Lay::kSeg;
  constexpr int kBlk = Lay::kBlk;
  constexpr int kPer = Lay::kPer;
  extern __shared__ __align__(1024) uint8_t smem[];
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wg = warp >> 2;
  const int tl = tid & 127;
  const bool spare = !kSorted && static_cast<int>(blockIdx.x) >= p.groups * p.HKV;
  int group = spare ? 0 : blockIdx.x / p.HKV;
  const int kvh = blockIdx.x % p.HKV;
  int row0 = group * kRows;
  uint8_t* const s_k = smem + Lay::k;
  uint8_t* const s_v = smem + Lay::v;
  uint8_t* const s_p = smem + Lay::pr;
  uint8_t* const s_sf = smem + Lay::sf;
  uint32_t* const s_ent = reinterpret_cast<uint32_t*>(smem + Lay::ent);
  float* const s_red = reinterpret_cast<float*>(smem + Lay::red);
  float* const s_fac = reinterpret_cast<float*>(smem + Lay::fac);
  float* const s_nm = reinterpret_cast<float*>(smem + Lay::nmr);
  float* const s_fo = reinterpret_cast<float*>(smem + Lay::fo);
  uint32_t* const s_anyw = reinterpret_cast<uint32_t*>(smem + Lay::flag);
  uint32_t* const s_any = reinterpret_cast<uint32_t*>(smem + Lay::flag + 16);
  uint32_t* const s_more = s_any + 1;
  uint32_t* const s_tiles = s_any + 2;
  uint32_t* const s_last = s_any + 3;
  int32_t* const s_split = reinterpret_cast<int32_t*>(s_any + 4);
  uint64_t* const s_bar = reinterpret_cast<uint64_t*>(smem + Lay::bar);
  uint64_t* const qk_bar = s_bar;
  uint64_t* const pv_bar = s_bar + 2;
  uint32_t* const s_tmem = reinterpret_cast<uint32_t*>(smem + Lay::tmem);
  const float ninf = __int_as_float(0xFF800000);
  const bool worker = warp < kWorkers / 32;

  if (tid == 0) {
    for (int i = 0; i < 4; ++i)
      device::ptx::mbar_init(s_bar + i, 1);
    for (int i = 4; i < 9; ++i)
      device::ptx::mbar_init(s_bar + i, kWorkers);
    for (int i = 9; i < 11; ++i)
      device::ptx::mbar_init(s_bar + i, kWorkers / 32);
    device::ptx::fence_barrier_init();
  }
  float sc = 0.f;
  if (worker) {
    for (int i = tid; i < 3 * 128; i += kWorkers)
      reinterpret_cast<uint32_t*>(s_sf)[i] = 0x7F7F7F7Fu;
    if (!kSorted && !spare) sc = split_q<kRows, kHeads>(p, row0, kvh, smem + Lay::q, s_red);
    device::ptx::fence_proxy_async_shared();
  } else {
    tcgen05::alloc(s_tmem, Lay::kTmemCols);
    tcgen05::relinquish_alloc_permit();
    tcgen05::fence_before_thread_sync();
  }
  __syncthreads();
  tcgen05::fence_after_thread_sync();

  const uint32_t t_base = *s_tmem;
  const uint32_t t_lane = t_base + (static_cast<uint32_t>((warp & 3) * 32) << 16);
  device::PDLWaitPrimary<kUsePDL>();
  if constexpr (kSorted) {
    group = group_of_rank(p, group, reinterpret_cast<uint32_t*>(s_p));
    row0 = group * kRows;
  } else {
    if (tid == 0) s_split[0] = -1;
    if (p.splits > 0) split_assign(p, spare, group, s_split, reinterpret_cast<uint32_t*>(s_p));
    __syncthreads();
    group = s_split[0] >= 0 ? s_split[2] : group;
    row0 = group * kRows;
  }
  if (tid == 0) {
    const int seg = kSorted ? -1 : s_split[0];
    const int all_tiles = (p.counts[group] + kTile - 1) / kTile;
    const int half = (all_tiles + 1) / 2;
    if constexpr (kSorted) s_split[0] = -1;
    s_split[3] = all_tiles;
    s_split[4] = seg < 0 ? all_tiles : seg == 0 ? half : all_tiles - half;
    s_split[5] = seg == 1 ? half : 0;
  }
  __syncthreads();

  if (!worker) {
    issue_mmas_pv8<kRows, kHeads>(t_base, smem);
    tcgen05::dealloc(t_base, Lay::kTmemCols);
    return;
  }

  const int gk = tid >> 4;
  const int gc = tid & 15;
  const int vk = ((tid >> 6) << 3) | (((tid >> 3) & 3) ^ (((tid >> 2) & 1) * 5));
  const int vc = (tid & 3) | (((tid >> 5) & 1) << 2);
  const int first = wg * kPer;
  const int heads = kReal == 4 * kPer ? kPer : min(max(kReal - first, 0), kPer);
  const int j0 = first / kHeads;
  const int split = kPer == kHeads ? kPer : (j0 + 1) * kHeads - first;
  const uint32_t t_o0 = t_lane + Lay::kOCol;
  const uint8_t* const k_src = p.k + kvh * p.stride_kh + gc * 16;
  const uint8_t* const v_src = p.v + kvh * p.stride_kh + vc * 16;
  const uint32_t k_dst = device::ptx::to_shared(s_k) + (gc >> 3) * kPlane;
  const uint32_t v_dst = device::ptx::to_shared(s_v);
  const uint32_t bar_u32 = device::ptx::to_shared(s_bar);
  int g0 = 0;
  bool repeat = false;
  float ls[kPer];
#pragma unroll 1
  for (int pass = 0; pass < (repeat ? 1 + kRows : 1); ++pass) {
    const uint32_t rowmask = pass == 0 ? (1u << kRows) - 1 : 1u << (pass - 1);
    const int tiles = s_split[4];
    const int32_t* const ents = p.entries + static_cast<int64_t>(row0 / kRows) * p.cap + s_split[5] * kTile;

    const auto gather_k = [&](const int t) {
      const int g = g0 + t;
      const uint32_t dst = k_dst + (g & 1) * 2 * kPlane;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t entry = s_ent[(t & 7) * kTile + gk + 32 * i];
        cp_async16(
            dst + tcgen05::sw128_off(gk + 32 * i, gc & 7),
            k_src + static_cast<int64_t>(entry & kSlotMask) * p.stride_kslot,
            ((entry >> kSlotBits) & rowmask) != 0 ? 16 : 0);
      }
      cp_async_arrive_noinc(bar_u32 + 8 * (4 + (g & 1)));
    };
    const auto gather_v = [&](const int t) {
      const int g = g0 + t;
      const int slot = g % 3;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const uint8_t* base = v_src + h * 128;
        const uint32_t dst = v_dst + (slot * 2 + h) * kPlane;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
          const uint32_t entry = s_ent[(t & 7) * kTile + vk + 64 * i];
          cp_async16(
              dst + tcgen05::sw128_off(vk + 64 * i, vc),
              base + static_cast<int64_t>(entry & kSlotMask) * p.stride_kslot,
              ((entry >> kSlotBits) & rowmask) != 0 ? 16 : 0);
        }
      }
      cp_async_arrive_noinc(bar_u32 + 8 * (6 + slot));
    };
    const auto wait_pv = [&](const int g) { device::ptx::mbar_wait_parity(pv_bar + (g & 1), (g >> 1) & 1); };
    // P columns wg kSeg + i, 4 kSeg + wg kSeg + i: hi and lo e4m3 terms of p * 2^(134 - E) for head first + i, E the
    // exponent of its largest p in this warp's 32 slots (the UE8M0 scale E - 7); column 4 kSeg - 1: sum(V)
    const auto store_p = [&](const float (&pv)[kPer], const uint32_t mine, const int g) {
      uint8_t* row = s_p + (g & 1) * kTile * 128 + tl * 128;
      uint8_t* sfb = s_sf + 512 * (1 + (g & 1)) + (warp & 3);
      uint32_t hi[kSeg / 4];
      uint32_t lo[kSeg / 4];
#pragma unroll
      for (int j = 0; j < kSeg / 4; ++j) {
        hi[j] = 0;
        lo[j] = 0;
      }
      uint32_t code = 127;
#pragma unroll
      for (int i = 0; i < kPer; i += 2) {
        float x[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
          const uint32_t top = i + e < heads ? device::warp::reduce_max(__float_as_uint(pv[i + e])) : 0u;
          const uint32_t c = static_cast<uint32_t>(max(static_cast<int>(top >> 23) - 7, 1));
          if (lane == i + e) code = c;
          x[e] = pv[i + e] * __uint_as_float((254u - c) << 23);
        }
        const uint32_t hp = f32x2_e4m3x2(x[0], x[1]);
        const uint32_t back = e4m3x2_f16x2(hp);
        const float2 b2 = __half22float2(*reinterpret_cast<const __half2*>(&back));
        hi[i / 4] |= hp << (16 * ((i / 2) & 1));
        lo[i / 4] |= f32x2_e4m3x2(x[0] - b2.x, x[1] - b2.y) << (16 * ((i / 2) & 1));
      }
      if (kRows > 1 && wg == 3 && mine != 0) hi[kSeg / 4 - 1] |= 0x38u << 24;
      if (lane < heads) {
        const int c = wg * kSeg + lane;
        sfb[(c & 31) * 16 + (c >> 5) * 4] = static_cast<uint8_t>(code);
        sfb[((4 * kSeg + c) & 31) * 16 + ((4 * kSeg + c) >> 5) * 4] = static_cast<uint8_t>(code);
      }
      const auto at = [&](const int b) { return row + ((((b >> 4) ^ (tl & 7)) << 4) | (b & 15)); };
      if constexpr (kSeg == 16) {
        *reinterpret_cast<uint4*>(at(wg * kSeg)) = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        *reinterpret_cast<uint4*>(at(4 * kSeg + wg * kSeg)) = make_uint4(lo[0], lo[1], lo[2], lo[3]);
      } else if constexpr (kSeg == 8) {
        *reinterpret_cast<uint2*>(at(wg * kSeg)) = make_uint2(hi[0], hi[1]);
        *reinterpret_cast<uint2*>(at(4 * kSeg + wg * kSeg)) = make_uint2(lo[0], lo[1]);
      } else {
        *reinterpret_cast<uint32_t*>(at(wg * kSeg)) = hi[0];
        *reinterpret_cast<uint32_t*>(at(4 * kSeg + wg * kSeg)) = lo[0];
      }
    };
    const auto ld_heads = [&](const uint32_t col, float (&dst)[kPer]) {
      constexpr int kA = kPer >= 8 ? 8 : 4;
      tcgen05::ld_32x32b<kA>(col, dst);
      if constexpr (kPer > kA) tcgen05::ld_32x32b<kPer - kA>(col + kA, dst + kA);
    };
    const auto load_scores = [&](float (&x)[kPer], const float (&nm)[kPer], const bool biased, const uint32_t mine,
                                 const int g) {
      const bool on0 = ((mine >> j0) & 1u) != 0;
      const bool on1 = ((mine >> (j0 + 1)) & 1u) != 0;
      const uint32_t t_s = t_lane + (g & 1) * Lay::kNQ;
      float term[kPer];
      if (heads > 0) ld_heads(t_s + kReal + first, term);
      tcgen05::wait_ld();
#pragma unroll
      for (int i = 0; i < kPer; ++i)
        x[i] = biased ? fmaf(term[i], sc, nm[i]) : term[i] * sc;
      if (heads > 0) ld_heads(t_s + first, term);
      tcgen05::wait_ld();
      tcgen05::fence_before_thread_sync();
#pragma unroll
      for (int i = 0; i < kPer; ++i)
        x[i] = (i < split ? on0 : on1) && i < heads ? fmaf(term[i], sc, x[i]) : ninf;
    };

    for (int t = 0; t < 5; ++t) {
      if (warp == t && t < tiles) {
        *reinterpret_cast<uint4*>(s_ent + (t & 7) * kTile + lane * 4) =
            *reinterpret_cast<const uint4*>(ents + t * kTile + lane * 4);
      }
    }
    if (tid < kReal) s_nm[tid] = 1e30f;
    if (tid == 0) {
      *s_any = 0;
      *s_tiles = static_cast<uint32_t>(tiles);
    }
    device::ptx::bar_sync(3, kWorkers);
    for (int t = 0; t < 2; ++t) {
      if (t < tiles) gather_k(t);
    }
    if (tiles > 0) gather_v(0);
    if ((kSorted || spare) && pass == 0) {
      sc = split_q<kRows, kHeads>(p, row0, kvh, smem + Lay::q, s_red);
      device::ptx::fence_proxy_async_shared();
    }
    device::ptx::bar_arrive(1, kThreads);

#pragma unroll
    for (int i = 0; i < kPer; ++i)
      ls[i] = 0.f;

#pragma unroll 1
    for (int t = 0; t < tiles; ++t) {
      const int g = g0 + t;
      uint4 ent_next = make_uint4(0, 0, 0, 0);
      if (warp == 0 && t + 5 < tiles) ent_next = *reinterpret_cast<const uint4*>(ents + (t + 5) * kTile + lane * 4);
      device::ptx::mbar_wait_parity(qk_bar + (g & 1), (g >> 1) & 1);
      tcgen05::fence_after_thread_sync();
      if (t + 2 < tiles) gather_k(t + 2);
      const uint32_t mine = (s_ent[(t & 7) * kTile + tl] >> kSlotBits) & rowmask;
      float pv[kPer];
      float nm[kPer];
#pragma unroll
      for (int i = 0; i < kPer; ++i)
        nm[i] = s_nm[first + i];
      load_scores(pv, nm, true, mine, g);
      float top = ninf;
#pragma unroll
      for (int i = 0; i < kPer; ++i) {
        top = fmaxf(top, pv[i]);
        pv[i] = ex2_ftz(pv[i]);
      }
      const bool rescale = bar_red_or(4 + wg, 128, top > kRescale);
      if (t + 1 < tiles) gather_v(t + 1);
      if (rescale) {
        tcgen05::fence_after_thread_sync();
        load_scores(pv, nm, false, mine, g);
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          const float m = device::warp::reduce_max(pv[i]);
          if (lane == 0 && i < heads) s_red[(first + i) * 4 + (warp & 3)] = m;
        }
        if (tl == 0) s_anyw[wg] = 0;
        device::ptx::bar_sync(4 + wg, 128);
        bool moved = false;
#pragma unroll
        for (int i = 0; i < kPer; ++i) {
          if (i < heads) {
            const float4 r4 = *reinterpret_cast<const float4*>(s_red + (first + i) * 4);
            const float m = fmaxf(fmaxf(r4.x, r4.y), fmaxf(r4.z, r4.w));
            const float ref = -nm[i];
            const bool move = m > ref + kRescale;
            const float f = move ? ex2_ftz(ref - m) : 1.f;
            nm[i] = move ? -m : nm[i];
            ls[i] *= f;
            pv[i] = ex2_ftz(pv[i] + nm[i]);
            moved |= move && ref > -1e29f;
            if (lane == 0 && (warp & 3) == 0) {
              s_fac[first + i] = f;
              s_nm[first + i] = nm[i];
            }
          }
        }
        if (moved && tl == 0) s_anyw[wg] = 1;
        device::ptx::bar_sync(4 + wg, 128);
        if (s_anyw[wg] != 0 && t > 0) {
          wait_pv(g - 1);
          tcgen05::fence_after_thread_sync();
#pragma unroll
          for (int h = 0; h < 2; ++h) {
#pragma unroll
            for (int b = 0; b < kSeg; b += kBlk) {
#pragma unroll
              for (int part = 0; part < 2; ++part) {
                float o[kBlk];
                const uint32_t col = t_o0 + h * kNP + part * 4 * kSeg + wg * kSeg + b;
                tcgen05::ld_32x32b<kBlk>(col, o);
                tcgen05::wait_ld();
#pragma unroll
                for (int e = 0; e < kBlk; ++e) {
                  if (b + e < heads) o[e] *= s_fac[first + b + e];
                }
                tcgen05::st_32x32b<kBlk>(col, o);
              }
            }
          }
          tcgen05::wait_st();
          tcgen05::fence_before_thread_sync();
        }
      }
#pragma unroll
      for (int i = 0; i < kPer; ++i)
        ls[i] += pv[i];
      store_p(pv, mine, g);
      __syncwarp();
      if (lane == 0) device::ptx::mbar_arrive(s_bar + 9 + (g & 1));
      if (warp == 0 && t + 5 < tiles) *reinterpret_cast<uint4*>(s_ent + ((t + 5) & 7) * kTile + lane * 4) = ent_next;
    }

    if (tiles > 0) wait_pv(g0 + tiles - 1);
    tcgen05::fence_after_thread_sync();
#pragma unroll
    for (int i = 0; i < kPer; ++i) {
      const float sum = device::warp::reduce_sum(ls[i]);
      if (lane == 0 && i < heads) s_red[(first + i) * 4 + (warp & 3)] = sum;
    }
    device::ptx::bar_sync(4 + wg, 128);
    const int seg_e = s_split[0];
    const int sidx_e = s_split[1];
    const bool part = seg_e >= 0;
    const float* const other = p.part + static_cast<int64_t>((sidx_e * p.HKV + kvh) * 2 + ((seg_e & 1) ^ 1)) * Lay::kRec;
    if (part) {
      split_partial<kRows, kHeads>(p, smem, kvh, tiles);
    } else if (tl < heads) {
      const float4 r4 = *reinterpret_cast<const float4*>(s_red + (first + tl) * 4);
      const float sum = (r4.x + r4.y) + (r4.z + r4.w);
      s_fac[first + tl] = sum > 0.f ? 1.f / sum : 0.f;
    }
    device::ptx::bar_sync(4 + wg, 128);
    const bool merge = part && *s_last != 0;
    uint16_t* const s_out = reinterpret_cast<uint16_t*>(s_k);
    bool poisoned = false;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int d = h * 128 + tl;
#pragma unroll
      for (int b = 0; b < kSeg; b += kBlk) {
        float o[kBlk];
        float o2[kBlk];
#pragma unroll
        for (int e = 0; e < kBlk; ++e) {
          o[e] = 0.f;
          o2[e] = 0.f;
        }
        if (tiles > 0 && (!part || merge)) {
          tcgen05::ld_32x32b<kBlk>(t_o0 + h * kNP + wg * kSeg + b, o);
          tcgen05::ld_32x32b<kBlk>(t_o0 + h * kNP + 4 * kSeg + wg * kSeg + b, o2);
          tcgen05::wait_ld();
        }
#pragma unroll
        for (int e = 0; e < kBlk; ++e) {
          const int n = first + b + e;
          if (b + e < heads) {
            const float own = o[e] + o2[e];
            const float oth = merge ? other[n * kD + d] : 0.f;
            const float y = merge ? (seg_e == 0 ? fmaf(own, s_fac[n], oth * s_fo[n]) : fmaf(oth, s_fac[n], own * s_fo[n]))
                                  : own * s_fac[n];
            s_out[n * kD + d] = __bfloat16_as_ushort(__float2bfloat16_rn(y));
          } else if (kRows > 1 && wg == 3 && b + e == kSeg - 1 && pass == 0 && !part) {
            poisoned |= o[e] != o[e];
          }
        }
      }
    }
    if (poisoned) *s_any = 1;
    tcgen05::fence_before_thread_sync();
    device::ptx::bar_sync(3, kWorkers);
    const bool write = !part || *s_last != 0;
    if (kRows > 1 && pass == 0 && write) repeat = *s_any != 0;
    for (int i = tid; i < kReal * 32; i += kWorkers) {
      const int n = i >> 5;
      const int j = n / kHeads;
      if (write && ((rowmask >> j) & 1u) != 0) {
        uint16_t* dst = p.out + (row0 + j) * p.stride_or + (kvh * kHeads + n % kHeads) * p.stride_oh + (i & 31) * 8;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(s_out + n * kD + (i & 31) * 8);
      }
    }
    if (tid == 0) *s_more = pass + 1 < (repeat ? 1 + kRows : 1) ? 1 : 0;
    g0 += tiles;
    if (part && repeat && tid == 0) {
      s_split[0] = -1;
      s_split[4] = s_split[3];
      s_split[5] = 0;
    }
    device::ptx::bar_sync(3, kWorkers);
    device::ptx::bar_arrive(1, kThreads);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}


}  // namespace qsa_sparse_decode_sm100

template <int kRows, int kHeads, bool kUsePDL, bool kPv8>
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
      const tvm::ffi::TensorView part,
      const tvm::ffi::TensorView split_cnt,
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
        .part = static_cast<float*>(part.data_ptr()),
        .split_cnt = static_cast<int32_t*>(split_cnt.data_ptr()),
        .splits = 0,
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
    const int64_t sms = runtime::get_sm_count(dev.device_id);
    if (kRows > 1 && groups * hkv > sms && groups < kTile * 32) {
      launch_attn<true>(groups * hkv, params, dev);
    } else {
      auto split = params;
      if constexpr (kPv8 && kRows > 1) {
        split.splits = static_cast<int32_t>(std::max<int64_t>(0, std::min<int64_t>((sms - groups * hkv) / hkv, groups)));
        const int64_t rec = LayoutPv8<kRows, kHeads>::kRec;
        RuntimeCheck(
            part.numel() >= split.splits * hkv * 2 * rec && split_cnt.numel() >= split.splits * hkv,
            "split scratch too small");
      }
      launch_attn<false>(groups * hkv + split.splits * hkv, split, dev);
    }
  }

 private:
  template <bool kSorted>
  static void launch_attn(const int64_t blocks, const qsa_sparse_decode_sm100::Params& params, const DLDevice dev) {
    using namespace host;
    using namespace qsa_sparse_decode_sm100;
    if constexpr (kPv8) {
      constexpr int smem = LayoutPv8<kRows, kHeads>::bytes;
      runtime::set_max_dynamic_smem_per_device<attn_pv8_kernel<kRows, kHeads, kUsePDL, kSorted>>(dev, smem);
      LaunchKernel(static_cast<uint32_t>(blocks), kThreads, dev, smem)
          .enable_pdl(kUsePDL)(attn_pv8_kernel<kRows, kHeads, kUsePDL, kSorted>, params);
    } else {
      constexpr int smem = Layout<kRows, kHeads>::bytes;
      runtime::set_max_dynamic_smem_per_device<attn_kernel<kRows, kHeads, kUsePDL, kSorted>>(dev, smem);
      LaunchKernel(static_cast<uint32_t>(blocks), kThreads, dev, smem)
          .enable_pdl(kUsePDL)(attn_kernel<kRows, kHeads, kUsePDL, kSorted>, params);
    }
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
