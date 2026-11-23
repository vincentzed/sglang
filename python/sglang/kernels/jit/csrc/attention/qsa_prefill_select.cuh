// QSA (Qwen4-Exp) prefill block selection over packed FP8 E4M3 compressed keys: per row, the 512 keys of
// [starts[row], ends[row]) with the largest sum over the 4 index heads of relu(q[row, head] . key).
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace qsa_prefill_select {

constexpr int kD = 128;
constexpr int kHeads = 4;
constexpr int kTopK = 512;
constexpr int kWarps = 16;
constexpr int kThreads = kWarps * 32;
constexpr int kChains = 2;
constexpr int kWarpRows = 2 * kChains;
constexpr int kRows = kWarps * kWarpRows;
constexpr int kTileKeys = 128;
constexpr int kSubTiles = kTileKeys / 16;
constexpr int kKeyStride = kD + 16;
constexpr int kTileBytes = kTileKeys * kKeyStride;
constexpr int kBins = 256;
constexpr int kLoads = kTileKeys * 8 / kThreads;
constexpr int kCapacity = 1664;
constexpr int kSampleRank = 112;
constexpr int kCollect = 255;
constexpr uint32_t kCoarseLo = 0x3e800000u;
constexpr uint32_t kFineLo = 0x40800000u;
constexpr uint32_t kFineHi = 0x43800000u;
constexpr int kCoarseShift = 20;
constexpr int kFineShift = 18;
constexpr int kCoarseBins = 32;
constexpr int kFineBins = 192;
constexpr uint32_t kKeyMax = 0x7fffffffu;
constexpr unsigned kFull = 0xffffffffu;
constexpr std::size_t kHistBytes = kRows * kBins * sizeof(uint32_t);
constexpr std::size_t kSmemBytes = kWarps * kCapacity * sizeof(uint32_t);
static_assert(kSmemBytes >= kHistBytes + 2 * kTileBytes);
static_assert(kChains == 2 && kLoads == 2);

enum ScanMode { kSample, kFast, kWindow, kRefine };

struct Params {
  const uint8_t* q;       // fp8 e4m3 [rows, q_heads, 128]
  const uint8_t* k;       // fp8 e4m3 [keys, 128]
  const int32_t* starts;  // [rows]
  const int32_t* ends;    // [rows]
  int32_t* out;           // [rows, 512]
  uint64_t* candidates;   // [rows, capacity]
  int64_t stride_q;
  int32_t rows, keys, capacity;
};

struct RowState {
  int32_t start, len, pending, scan_len, need, emitted, collected, shift;
  uint32_t lo, hi, hi_prev;
};

SGL_DEVICE uint32_t load_u32(const uint8_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

SGL_DEVICE uint32_t window_bin(const uint32_t key) {
  if (key < kFineLo) return (max(key, kCoarseLo) - kCoarseLo) >> kCoarseShift;
  if (key < kFineHi) return kCoarseBins + ((key - kFineLo) >> kFineShift);
  return min(kCoarseBins + kFineBins + ((key - kFineHi) >> kCoarseShift), uint32_t(kBins - 1));
}

SGL_DEVICE uint32_t window_edge(const int bin) {
  if (bin == 0) return 0;
  if (bin < kCoarseBins) return kCoarseLo + (static_cast<uint32_t>(bin) << kCoarseShift);
  if (bin < kCoarseBins + kFineBins) return kFineLo + (static_cast<uint32_t>(bin - kCoarseBins) << kFineShift);
  return kFineHi + (static_cast<uint32_t>(bin - kCoarseBins - kFineBins) << kCoarseShift);
}

SGL_DEVICE void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

SGL_DEVICE void refine(
    const Params& p,
    RowState* state,
    uint32_t* hist,
    const int row0,
    const int row,
    const int pos,
    const int len,
    const uint32_t key,
    const uint32_t lo,
    const uint32_t hi,
    const uint32_t hi_prev,
    const int shift) {
  if (pos >= len || key > hi_prev || key < lo) return;
  if (key > hi) {
    const int e = atomicAdd(&state[row].emitted, 1);
    if (e < kTopK) p.out[static_cast<int64_t>(row0 + row) * kTopK + e] = pos;
  } else if (shift == kCollect) {
    const int e = atomicAdd(&state[row].collected, 1);
    if (e < p.capacity) {
      p.candidates[static_cast<int64_t>(row0 + row) * p.capacity + e] =
          (static_cast<uint64_t>(key) << 32) | static_cast<uint32_t>(pos);
    }
  } else {
    atomicAdd(hist + row * kBins + ((key - lo) >> shift), 1u);
  }
}

SGL_DEVICE void flush_stage(
    const Params& p,
    uint32_t* hist,
    const int64_t out_row,
    const int r,
    const int count,
    const int staged,
    const int sent) {
  const int lane = threadIdx.x & 31;
  uint64_t* stage = reinterpret_cast<uint64_t*>(hist + r * kBins);
  const uint64_t v = stage[lane];
  const int slot = sent + lane;
  if (lane < count && slot < p.capacity) {
    p.candidates[out_row * p.capacity + slot] = v;
  }
  const uint64_t rest = stage[32 + (lane & 15)];
  if (lane < staged - count) stage[lane] = rest;
}

template <ScanMode kMode>
SGL_DEVICE void scan(
    const Params& p,
    RowState* state,
    uint32_t* hist,
    uint8_t* tiles,
    const uint32_t (&q_frag)[kChains][4][2],
    const int row0,
    const int seg_start,
    const int seg_len) {
  constexpr int kTileSpan = kMode == kSample ? kTileKeys * kSubTiles : kTileKeys;
  constexpr int kSubSpan = kMode == kSample ? kTileKeys + 16 : 16;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int side = t >> 1;
  const int key_in_sub = g + 8 * (t & 1);
  const unsigned row_lanes = side ? 0xccccccccu : 0x33333333u;
  const unsigned lanes_below = (1u << lane) - 1;
  const int row_a = warp * kWarpRows + side;
  const int row_b = row_a + 2;
  const RowState& state_a = state[row_a];
  const RowState& state_b = state[row_b];
  const int len_a = state_a.scan_len, len_b = state_b.scan_len;
  const int shift_a = state_a.shift, shift_b = state_b.shift;
  const uint32_t lo_a = state_a.lo, lo_b = state_b.lo;
  const uint32_t hi_a = state_a.hi, hi_b = state_b.hi;
  const uint32_t hi_prev_a = state_a.hi_prev, hi_prev_b = state_b.hi_prev;
  int staged_a0 = 0, staged_a1 = 0, staged_b0 = 0, staged_b1 = 0;
  int sent_a0 = 0, sent_a1 = 0, sent_b0 = 0, sent_b1 = 0;

  const int load_byte = (threadIdx.x & 7) * 16;
  const int slot0 = threadIdx.x >> 3;
  const int slot1 = slot0 + kThreads / 8;
  const int key0 = seg_start + (slot0 >> 4) * kSubSpan + (slot0 & 15);
  const int key1 = seg_start + (slot1 >> 4) * kSubSpan + (slot1 & 15);
  uint8_t* dst0 = tiles + slot0 * kKeyStride + load_byte;
  uint8_t* dst1 = tiles + slot1 * kKeyStride + load_byte;
  const int num_tiles = (seg_len + kTileSpan - 1) / kTileSpan;
  uint4 next0 = make_uint4(0, 0, 0, 0);
  uint4 next1 = make_uint4(0, 0, 0, 0);
  if (key0 < p.keys) next0 = *reinterpret_cast<const uint4*>(p.k + static_cast<int64_t>(key0) * kD + load_byte);
  if (key1 < p.keys) next1 = *reinterpret_cast<const uint4*>(p.k + static_cast<int64_t>(key1) * kD + load_byte);
  *reinterpret_cast<uint4*>(dst0) = next0;
  *reinterpret_cast<uint4*>(dst1) = next1;
  __syncthreads();

  for (int tile = 0; tile < num_tiles; ++tile) {
    const bool has_next = tile + 1 < num_tiles;
    const int ahead = (tile + 1) * kTileSpan;
    if (has_next && key0 + ahead < p.keys) {
      next0 = *reinterpret_cast<const uint4*>(p.k + static_cast<int64_t>(key0 + ahead) * kD + load_byte);
    }
    if (has_next && key1 + ahead < p.keys) {
      next1 = *reinterpret_cast<const uint4*>(p.k + static_cast<int64_t>(key1 + ahead) * kD + load_byte);
    }
    const uint8_t* tile_keys = tiles + (tile & 1) * kTileBytes;
#pragma unroll
    for (int u = 0; u < kSubTiles; ++u) {
      const int sub_pos = tile * kTileSpan + u * kSubSpan;
      const uint8_t* ka = tile_keys + (u * 16 + g) * kKeyStride + 4 * t;
      const uint8_t* kb = ka + 8 * kKeyStride;
      float acc_a[4] = {};
      float acc_b[4] = {};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t a[4] = {
            load_u32(ka + 32 * c), load_u32(kb + 32 * c), load_u32(ka + 32 * c + 16), load_u32(kb + 32 * c + 16)};
        mma_e4m3(acc_a, a, q_frag[0][c][0], q_frag[0][c][1]);
        mma_e4m3(acc_b, a, q_frag[1][c][0], q_frag[1][c][1]);
      }
      const int pos = sub_pos + key_in_sub;
      float a0 = fmaxf(acc_a[0], 0.f) + fmaxf(acc_a[1], 0.f);
      float a1 = fmaxf(acc_a[2], 0.f) + fmaxf(acc_a[3], 0.f);
      float b0 = fmaxf(acc_b[0], 0.f) + fmaxf(acc_b[1], 0.f);
      float b1 = fmaxf(acc_b[2], 0.f) + fmaxf(acc_b[3], 0.f);
      a0 += __shfl_xor_sync(kFull, a0, 1);
      a1 += __shfl_xor_sync(kFull, a1, 1);
      b0 += __shfl_xor_sync(kFull, b0, 1);
      b1 += __shfl_xor_sync(kFull, b1, 1);
      const uint32_t key_a = __float_as_uint((t & 1) ? a1 : a0) & kKeyMax;
      const uint32_t key_b = __float_as_uint((t & 1) ? b1 : b0) & kKeyMax;
      if (kMode == kFast) {
        const bool hit_a = pos < len_a && key_a >= lo_a;
        const bool hit_b = pos < len_b && key_b >= lo_b;
        const unsigned hits_a = __ballot_sync(kFull, hit_a);
        const unsigned hits_b = __ballot_sync(kFull, hit_b);
        if (hit_a) {
          reinterpret_cast<uint64_t*>(
              hist + row_a * kBins)[(side ? staged_a1 : staged_a0) + __popc(hits_a & row_lanes & lanes_below)] =
              (static_cast<uint64_t>(key_a) << 32) | static_cast<uint32_t>(pos);
        }
        if (hit_b) {
          reinterpret_cast<uint64_t*>(
              hist + row_b * kBins)[(side ? staged_b1 : staged_b0) + __popc(hits_b & row_lanes & lanes_below)] =
              (static_cast<uint64_t>(key_b) << 32) | static_cast<uint32_t>(pos);
        }
        staged_a0 += __popc(hits_a & 0x33333333u);
        staged_a1 += __popc(hits_a & 0xccccccccu);
        staged_b0 += __popc(hits_b & 0x33333333u);
        staged_b1 += __popc(hits_b & 0xccccccccu);
        if (max(max(staged_a0, staged_a1), max(staged_b0, staged_b1)) >= 32) {
          const int r = warp * kWarpRows;
          if (staged_a0 >= 32) {
            flush_stage(p, hist, row0 + r, r, 32, staged_a0, sent_a0);
            staged_a0 -= 32;
            sent_a0 += 32;
          }
          if (staged_a1 >= 32) {
            flush_stage(p, hist, row0 + r + 1, r + 1, 32, staged_a1, sent_a1);
            staged_a1 -= 32;
            sent_a1 += 32;
          }
          if (staged_b0 >= 32) {
            flush_stage(p, hist, row0 + r + 2, r + 2, 32, staged_b0, sent_b0);
            staged_b0 -= 32;
            sent_b0 += 32;
          }
          if (staged_b1 >= 32) {
            flush_stage(p, hist, row0 + r + 3, r + 3, 32, staged_b1, sent_b1);
            staged_b1 -= 32;
            sent_b1 += 32;
          }
        }
      } else if (kMode != kRefine) {
        if (pos < len_a) atomicAdd(hist + row_a * kBins + window_bin(key_a), 1u);
        if (pos < len_b) atomicAdd(hist + row_b * kBins + window_bin(key_b), 1u);
      } else {
        refine(p, state, hist, row0, row_a, pos, len_a, key_a, lo_a, hi_a, hi_prev_a, shift_a);
        refine(p, state, hist, row0, row_b, pos, len_b, key_b, lo_b, hi_b, hi_prev_b, shift_b);
      }
    }
    if (has_next) {
      *reinterpret_cast<uint4*>(dst0 + ((tile + 1) & 1) * kTileBytes) = next0;
      *reinterpret_cast<uint4*>(dst1 + ((tile + 1) & 1) * kTileBytes) = next1;
    }
    __syncthreads();
  }
  if (kMode == kFast) {
    const int r = warp * kWarpRows;
    flush_stage(p, hist, row0 + r, r, staged_a0, staged_a0, sent_a0);
    flush_stage(p, hist, row0 + r + 1, r + 1, staged_a1, staged_a1, sent_a1);
    flush_stage(p, hist, row0 + r + 2, r + 2, staged_b0, staged_b0, sent_b0);
    flush_stage(p, hist, row0 + r + 3, r + 3, staged_b1, staged_b1, sent_b1);
    state[row_a].collected = side ? sent_a1 + staged_a1 : sent_a0 + staged_a0;
    state[row_b].collected = side ? sent_b1 + staged_b1 : sent_b0 + staged_b0;
    __syncthreads();
  }
}

SGL_DEVICE void seed(RowState& s, const uint32_t* hist) {
  int above = 0;
  int bin = kBins - 1;
  while (bin > 0 && above + static_cast<int>(hist[bin]) < kSampleRank) {
    above += hist[bin];
    --bin;
  }
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    s.lo = window_edge(bin);
  }
  __syncwarp();
}

SGL_DEVICE void restart(RowState& s) {
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    s.need = kTopK;
    s.emitted = 0;
    s.collected = 0;
    s.shift = 0;
    s.lo = 0;
    s.hi = kKeyMax;
    s.hi_prev = kKeyMax;
  }
  __syncwarp();
}

SGL_DEVICE void narrow(const Params& p, RowState& s, const uint32_t* hist, const bool first) {
  int above = 0;
  int bin = kBins - 1;
  while (bin > 0 && above + static_cast<int>(hist[bin]) < s.need) {
    above += hist[bin];
    --bin;
  }
  const int count = hist[bin];
  const uint32_t edge = first ? window_edge(bin) : s.lo + (static_cast<uint32_t>(bin) << s.shift);
  const uint32_t next = first ? window_edge(bin + 1) : s.lo + (static_cast<uint32_t>(bin + 1) << s.shift);
  const uint32_t lo = bin == 0 ? s.lo : edge;
  const uint32_t hi = bin == kBins - 1 ? s.hi : min(s.hi, next - 1);
  const uint32_t hi_prev = s.hi;
  const int need = s.need - above;
  __syncwarp();
  if ((threadIdx.x & 31) == 0) {
    s.need = need;
    s.lo = lo;
    s.hi = hi;
    s.hi_prev = hi_prev;
    s.collected = 0;
    s.shift = (count <= p.capacity || hi == lo) ? kCollect : max(0, 24 - __clz(hi - lo));
  }
  __syncwarp();
}

SGL_DEVICE int count_at_least(const uint32_t* keys, const int padded, const uint32_t threshold) {
  int n = 0;
  for (int i = threadIdx.x & 31; i < padded; i += 32) {
    n += keys[i] >= threshold;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    n += __shfl_xor_sync(kFull, n, o);
  }
  return n;
}

SGL_DEVICE void pick(const Params& p, RowState& s, const int row, uint32_t* keys) {
  const int lane = threadIdx.x & 31;
  const uint64_t* candidates = p.candidates + static_cast<int64_t>(row) * p.capacity;
  int32_t* out = p.out + static_cast<int64_t>(row) * kTopK;
  const int count = min(s.collected, p.capacity);
  const int padded = (count + 31) / 32 * 32;
  const int need = s.need;
  uint32_t top = 0;
  for (int i = lane; i < padded; i += 32) {
    const uint32_t key = i < count ? static_cast<uint32_t>(candidates[i] >> 32) : 0u;
    keys[i] = key;
    top = max(top, key);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    top = max(top, __shfl_xor_sync(kFull, top, o));
  }
  __syncwarp();
  uint32_t lo = s.lo;
  uint32_t hi = min(s.hi, top);
  while (lo < hi) {
    const uint32_t mid = lo + (hi - lo + 1) / 2;
    if (count_at_least(keys, padded, mid) >= need) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int ties = need - count_at_least(keys, padded, lo + 1);
  int written = s.emitted;
  int ties_seen = 0;
  for (int i = lane; i < padded; i += 32) {
    const uint32_t key = keys[i];
    const bool tie = i < count && key == lo;
    const unsigned tie_mask = __ballot_sync(kFull, tie);
    const bool take = i < count && (key > lo || (tie && ties_seen + __popc(tie_mask & ((1u << lane) - 1)) < ties));
    const unsigned take_mask = __ballot_sync(kFull, take);
    const int slot = written + __popc(take_mask & ((1u << lane) - 1));
    if (take && slot < kTopK) out[slot] = static_cast<int32_t>(static_cast<uint32_t>(candidates[i]));
    written += __popc(take_mask);
    ties_seen += __popc(tie_mask);
  }
  __syncwarp();
  if (lane == 0) {
    s.scan_len = 0;
    s.pending = 0;
  }
  __syncwarp();
}

SGL_DEVICE bool any_scanning(const RowState* state) {
  bool any = false;
  for (int i = 0; i < kRows; ++i) {
    any |= state[i].scan_len > 0;
  }
  return any;
}

SGL_DEVICE void clear(uint32_t* hist) {
  for (int i = threadIdx.x; i < kRows * kBins / 4; i += kThreads) {
    reinterpret_cast<uint4*>(hist)[i] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads) select_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ RowState state[kRows];
  device::PDLWaitPrimary<kUsePDL>();

  uint32_t* hist = reinterpret_cast<uint32_t*>(smem);
  uint8_t* tiles = smem + kHistBytes;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row0 = blockIdx.x * kRows;
  RowState* own = state + warp * kWarpRows;
  uint32_t* own_hist = hist + warp * kWarpRows * kBins;
  uint32_t* scratch = hist + warp * kCapacity;

  if (tid < kRows) {
    const int r = row0 + tid;
    const bool ok = r < p.rows;
    const int start = ok ? p.starts[r] : 0;
    const int len = ok ? min(p.ends[r], p.keys) - start : 0;
    state[tid].start = start;
    state[tid].len = len;
    state[tid].pending = len > kTopK;
    state[tid].scan_len = 0;
  }
  __syncthreads();
  int first_start = -1;
  for (int i = kRows - 1; i >= 0; --i) {
    if (state[i].pending) first_start = state[i].start;
  }
  __syncthreads();
  if (tid < kRows) {
    state[tid].pending &= (state[tid].start == first_start) == (blockIdx.y == 0);
  }
  __syncthreads();
  bool any_pending = false;
  for (int i = 0; i < kRows; ++i) {
    any_pending |= state[i].pending != 0;
  }
  for (int i = 0; blockIdx.y == 0 && i < kRows; ++i) {
    const int len = state[i].len;
    if (row0 + i < p.rows && len <= kTopK) {
      for (int c = tid; c < kTopK; c += kThreads) {
        p.out[static_cast<int64_t>(row0 + i) * kTopK + c] = c < len ? c : -1;
      }
    }
  }

  uint32_t q_frag[kChains][4][2];
#pragma unroll
  for (int j = 0; any_pending && j < kChains; ++j) {
    const int r = row0 + warp * kWarpRows + 2 * j + (g >> 2);
    const bool ok = r < p.rows;
    const uint8_t* src = p.q + (ok ? r : 0) * p.stride_q + (g & 3) * kD + 4 * t;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      q_frag[j][c][0] = ok ? load_u32(src + 32 * c) : 0u;
      q_frag[j][c][1] = ok ? load_u32(src + 32 * c + 16) : 0u;
    }
  }

  while (true) {
    int head = -1;
    for (int i = kRows - 1; i >= 0; --i) {
      if (state[i].pending) head = i;
    }
    if (head < 0) break;
    const int seg_start = state[head].start;
    int seg_len = 0;
    for (int i = 0; i < kRows; ++i) {
      if (state[i].pending && state[i].start == seg_start) seg_len = max(seg_len, state[i].len);
    }
    if (tid < kRows) {
      RowState& s = state[tid];
      s.scan_len = s.pending && s.start == seg_start ? s.len : 0;
      s.need = kTopK;
      s.emitted = 0;
      s.hi = kKeyMax;
    }
    clear(hist);

    scan<kSample>(p, state, hist, tiles, q_frag, row0, seg_start, seg_len);
    for (int i = 0; i < kWarpRows; ++i) {
      if (own[i].scan_len > 0) seed(own[i], own_hist + i * kBins);
    }
    __syncthreads();
    scan<kFast>(p, state, hist, tiles, q_frag, row0, seg_start, seg_len);
    for (int i = 0; i < kWarpRows; ++i) {
      if (own[i].scan_len == 0) continue;
      if (own[i].collected >= kTopK && own[i].collected <= p.capacity) {
        pick(p, own[i], row0 + warp * kWarpRows + i, scratch);
      } else {
        restart(own[i]);
      }
    }
    __syncthreads();

    for (bool first = true; any_scanning(state); first = false) {
      clear(hist);
      if (first) {
        scan<kWindow>(p, state, hist, tiles, q_frag, row0, seg_start, seg_len);
      } else {
        scan<kRefine>(p, state, hist, tiles, q_frag, row0, seg_start, seg_len);
      }
      unsigned collecting = 0;
      for (int i = 0; i < kWarpRows; ++i) {
        if (own[i].scan_len == 0) continue;
        if (own[i].shift == kCollect) {
          collecting |= 1u << i;
        } else {
          narrow(p, own[i], own_hist + i * kBins, first);
        }
      }
      __syncthreads();
      for (int i = 0; i < kWarpRows; ++i) {
        if (collecting >> i & 1) pick(p, own[i], row0 + warp * kWarpRows + i, scratch);
      }
      __syncthreads();
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_prefill_select

template <bool kUsePDL>
struct QsaPrefillSelect {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView starts,
      const tvm::ffi::TensorView ends,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView candidates) {
    using namespace host;
    using namespace qsa_prefill_select;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto N_ = SymbolicSize{"keys"};
    auto C_ = SymbolicSize{"capacity"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, kD, 1}).verify(q);
    TensorMatcher({N_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({kD, 1}).verify(k);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(starts);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(ends);
    TensorMatcher({R_, kTopK}).with_dtype<int32_t>().with_device(device).with_strides({kTopK, 1}).verify(out);
    TensorMatcher({R_, C_}).with_dtype<int64_t>().with_device(device).with_strides({C_, 1}).verify(candidates);
    const int64_t rows = R_.unwrap();
    RuntimeCheck(HQ_.unwrap() >= kHeads, "q needs at least 4 heads");
    RuntimeCheck(C_.unwrap() >= kTopK && C_.unwrap() <= kCapacity, "candidate capacity must be in [512, 1664]");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0 && q.stride(0) % 4 == 0, "q rows must be 4-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0, "keys must be 16-byte aligned");
    if (rows == 0) return;
    const auto params = Params{
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .starts = static_cast<const int32_t*>(starts.data_ptr()),
        .ends = static_cast<const int32_t*>(ends.data_ptr()),
        .out = static_cast<int32_t*>(out.data_ptr()),
        .candidates = static_cast<uint64_t*>(candidates.data_ptr()),
        .stride_q = q.stride(0),
        .rows = static_cast<int32_t>(rows),
        .keys = static_cast<int32_t>(N_.unwrap()),
        .capacity = static_cast<int32_t>(C_.unwrap()),
    };
    const auto kernel = select_kernel<kUsePDL>;
    RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(kSmemBytes)));
    const dim3 grid(static_cast<uint32_t>((rows + kRows - 1) / kRows), 2);
    LaunchKernel(grid, kThreads, device.unwrap(), kSmemBytes).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
