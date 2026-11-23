// QSA (Qwen4-Exp) prefill block selection over packed FP8 E4M3 compressed keys: per row, the 512 keys of
// [starts[row], ends[row]) with the largest sum over the 4 index heads of relu(q[row, head] . key), ties at the
// cut broken by the lower key index. Rows with at most 512 keys get 0..len-1, then -1.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace sglang {

namespace qsa_prefill_topk {

using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kD = 128;
constexpr int kHeads = 4;
constexpr int kTopK = 512;
constexpr int kRows = 64;
constexpr int kN = kRows * kHeads;
constexpr int kTileKeys = 128;
constexpr int kEpiWarps = 16;
constexpr int kEpiThreads = kEpiWarps * 32;
constexpr int kThreads = kEpiThreads + 32;
constexpr int kWarpRows = kRows * 4 / kEpiWarps;
constexpr int kQBytes = kN * kD;
constexpr int kTileBytes = kTileKeys * kD;
constexpr int kStages = 3;
constexpr int kOpBytes = kQBytes + kStages * kTileBytes;
constexpr int kTmemCols = 512;
constexpr int kBins = 256;
constexpr int kHistStride = kBins + 1;
constexpr int kHistBytes = kRows * kHistStride * 4;
constexpr int kSampleEvery = 8;
constexpr int kSampleRank = 112;
constexpr int kCapacity = 1664;
constexpr int kTieLimit = kCapacity - kTopK - kTileKeys;
constexpr int kPickWarps = 4;
constexpr int kExactPickWarps = 8;
constexpr int kPickBytes = kCapacity * 8 + kBins * 4;
constexpr int kMaxSplits = 64;
constexpr int kWorkStride = kBins + 2 + kMaxSplits;
constexpr uint32_t kCoarseLo = 0x3e800000u;
constexpr uint32_t kFineLo = 0x40800000u;
constexpr uint32_t kFineHi = 0x43800000u;
constexpr int kCoarseShift = 20;
constexpr int kFineShift = 18;
constexpr int kCoarseBins = 32;
constexpr int kFineBins = 192;
constexpr uint32_t kKeyMax = 0x7fffffffu;
constexpr unsigned kFull = 0xffffffffu;
constexpr int kExactScratch = kHistBytes > kExactPickWarps * kPickBytes ? kHistBytes : kExactPickWarps * kPickBytes;
constexpr int kSampleSmem = kOpBytes + kHistBytes + 1024;
constexpr int kCollectSmem = kOpBytes + 1024;
constexpr int kExactSmem = kOpBytes + kExactScratch + 1024;
static_assert(kWarpRows == 16 && kRows % kExactPickWarps == 0);

enum Mode { kSample, kCollect, kCollectScores, kHist, kCollectLimited };

struct Params {
  const uint8_t* q;     // fp8 e4m3 [rows, q_heads, 128]
  const uint8_t* k;     // fp8 e4m3 [keys, 128]
  const int32_t* starts;
  const int32_t* ends;
  int32_t* out;         // [rows, 512]
  uint64_t* cand;       // [rows, capacity]: score bits << 32 | key
  uint32_t* work;       // [rows, 258]: sample histogram, candidate count, fallback flag (zeroed)
  float* scores;        // optional [rows, score_stride]: raw sums of the collect pass
  int64_t stride_q, score_stride, cand_stride;
  int32_t rows, keys, splits, split_cap;
};

struct RowInfo {
  int32_t start, len, act;
  uint32_t lo, hi;
  int32_t shift, limit, mode;
};

struct Shared {
  uint64_t bars[2];
  uint32_t tmem;
  uint32_t mma_count;
  RowInfo info[kRows];
  uint32_t counts[kRows];
  int32_t owned[kRows];
};

SGL_DEVICE uint8_t* align_smem(uint8_t* raw) {
  return raw + ((1024 - (device::ptx::to_shared(raw) & 1023)) & 1023);
}

SGL_DEVICE uint32_t* count_of(const Params& p, const int row) {
  return p.work + static_cast<int64_t>(row) * kWorkStride + kBins;
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

SGL_DEVICE uint32_t hist_bin(const uint32_t key, const uint32_t lo, const int shift) {
  return shift < 0 ? window_bin(key) : (key - lo) >> shift;
}

SGL_DEVICE void load_rows(const Params& p, RowInfo* info, const int row0, const bool fallback) {
  const int tid = threadIdx.x;
  if (tid < kRows) {
    const int r = row0 + tid;
    const bool ok = r < p.rows;
    const int start = ok ? p.starts[r] : 0;
    const int len = ok ? min(p.ends[r], p.keys) - start : 0;
    info[tid].start = start;
    info[tid].len = len;
    info[tid].mode = len > kTopK && (!fallback || p.work[static_cast<int64_t>(r) * kWorkStride + kBins + 1] != 0);
    info[tid].act = 0;
    info[tid].lo = 0;
    info[tid].hi = 0xffffffffu;
    info[tid].shift = 0;
    info[tid].limit = INT_MAX;
  }
  __syncthreads();
}

SGL_DEVICE int first_segment(const RowInfo* info) {
  for (int i = 0; i < kRows; ++i) {
    if (info[i].mode) return info[i].start;
  }
  return -1;
}

SGL_DEVICE bool next_segment(const RowInfo* info, int& seg_start, int& seg_len, const int first) {
  seg_start = -1;
  for (int i = 0; i < kRows; ++i) {
    if (info[i].mode && (first < 0 || (blockIdx.z == 0) == (info[i].start == first))) {
      seg_start = info[i].start;
      break;
    }
  }
  if (seg_start < 0) return false;
  seg_len = 0;
  for (int i = 0; i < kRows; ++i) {
    if (info[i].mode && info[i].start == seg_start) seg_len = max(seg_len, info[i].len);
  }
  return true;
}


SGL_DEVICE float relu_bits(const uint32_t x) {
  return __int_as_float(max(static_cast<int>(x), 0));
}

SGL_DEVICE uint32_t score_key(const uint32_t a0, const uint32_t a1, const uint32_t b0, const uint32_t b1) {
  return __float_as_uint((relu_bits(a0) + relu_bits(a1)) + (relu_bits(b0) + relu_bits(b1)));
}

SGL_DEVICE void cp_async_zfill(const uint32_t dst, const void* src, const bool valid) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(valid ? 16 : 0));
}

SGL_DEVICE void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int kPending>
SGL_DEVICE void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(kPending));
}

SGL_DEVICE void load_q(const Params& p, uint8_t* s_q, const int row0) {
  for (int c = threadIdx.x; c < kN * 8; c += kEpiThreads) {
    const int n = c >> 3;
    const int r = row0 + (n >> 2);
    uint4 v = make_uint4(0, 0, 0, 0);
    if (r < p.rows) {
      v = *reinterpret_cast<const uint4*>(p.q + static_cast<int64_t>(r) * p.stride_q + (n & 3) * kD + (c & 7) * 16);
    }
    *reinterpret_cast<uint4*>(s_q + tcgen05::sw128_off(n, c & 7)) = v;
  }
  device::ptx::fence_proxy_async_shared();
}

SGL_DEVICE void load_keys(const Params& p, uint8_t* stage, const int key0, const int key_step) {
#pragma unroll
  for (int w = 0; w < kTileKeys * 8 / kEpiThreads; ++w) {
    const int c = threadIdx.x + w * kEpiThreads;
    const int key = key0 + (c >> 3) * key_step;
    const bool valid = key < p.keys;
    cp_async_zfill(
        device::ptx::to_shared(stage + tcgen05::sw128_off(c >> 3, c & 7)),
        p.k + static_cast<int64_t>(valid ? key : 0) * kD + (c & 7) * 16,
        valid);
  }
}

template <Mode kMode>
SGL_DEVICE void epilogue(
    const Params& p,
    Shared& sh,
    uint32_t* hist,
    const uint32_t g,
    const int row0,
    const int pos,
    const int act,
    const uint32_t lo,
    const uint32_t hi,
    const int shift,
    const bool open,
    const int pos_max,
    const uint32_t (&los)[kWarpRows]) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int rbase = (warp >> 2) * kWarpRows;
  const unsigned below = (1u << lane) - 1;
  device::ptx::mbar_wait_parity(sh.bars + (g & 1), (g >> 1) & 1);
  tcgen05::fence_after_thread_sync();
  const uint32_t taddr = sh.tmem + (static_cast<uint32_t>((warp & 3) * 32) << 16) + (g & 1) * kN + rbase * kHeads;
  uint32_t s[kWarpRows];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    uint32_t v[32];
    tcgen05::ld_32x32b<32>(taddr + half * 32, v);
    tcgen05::wait_ld();
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      s[half * 8 + i] = score_key(v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
    }
  }
  tcgen05::fence_before_thread_sync();
  if constexpr (kMode == kSample) {
#pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      const int act_r = __shfl_sync(kFull, act, r);
      if (pos < act_r) atomicAdd(hist + (rbase + r) * kHistStride + window_bin(s[r]), 1u);
    }
  } else if constexpr (kMode == kHist) {
#pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      const uint32_t lo_r = __shfl_sync(kFull, lo, r);
      const uint32_t hi_r = __shfl_sync(kFull, hi, r);
      const int shift_r = __shfl_sync(kFull, shift, r);
      const int act_r = __shfl_sync(kFull, act, r);
      if (pos < act_r && s[r] >= lo_r) {
        atomicAdd(hist + (rbase + r) * kHistStride + (s[r] > hi_r ? kBins : hist_bin(s[r], lo_r, shift_r)), 1u);
      }
    }
  } else {
    unsigned m[kWarpRows];
    int count = 0;
    const bool interior = __all_sync(kFull, lane >= kWarpRows || act == 0 || pos_max < act);
#pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      bool take = s[r] >= los[r];
      if (!interior || kMode != kCollect) {
        const bool valid = pos < __shfl_sync(kFull, act, r);
        take = take && valid;
        if constexpr (kMode == kCollectScores) {
          if (valid) p.scores[static_cast<int64_t>(row0 + rbase + r) * p.score_stride + pos] = __uint_as_float(s[r]);
        }
      }
      if constexpr (kMode == kCollectLimited) {
        const uint32_t hi_r = __shfl_sync(kFull, hi, r);
        const bool open_r = __shfl_sync(kFull, open, r);
        take = take && (s[r] > hi_r || open_r);
      }
      m[r] = __ballot_sync(kFull, take);
      if (lane == r) count = __popc(m[r]);
    }
    constexpr bool kSplitRegion = kMode != kCollectLimited;
    const int cap = kSplitRegion ? p.split_cap : kCapacity;
    int base = 0;
    if (count > 0) {
      base = static_cast<int>(
          kSplitRegion ? atomicAdd(&sh.counts[rbase + lane], static_cast<uint32_t>(count))
                       : atomicAdd(count_of(p, row0 + rbase + lane), static_cast<uint32_t>(count)));
    }
    uint64_t* const rows_out =
        p.cand + static_cast<int64_t>(row0 + rbase) * p.cand_stride + (kSplitRegion ? blockIdx.y * p.split_cap : 0);
    const uint64_t pos_bits = static_cast<uint32_t>(pos);
#pragma unroll
    for (int r = 0; r < kWarpRows; ++r) {
      const int slot = __shfl_sync(kFull, base, r) + __popc(m[r] & below);
      if ((m[r] >> lane & 1) && slot < cap) {
        rows_out[r * p.cand_stride + slot] = (static_cast<uint64_t>(s[r]) << 32) | pos_bits;
      }
    }
  }
}

template <Mode kMode>
SGL_DEVICE void scan(
    const Params& p,
    Shared& sh,
    uint8_t* smem,
    uint32_t* hist,
    const int row0,
    const int seg_start,
    const int seg_len,
    const int first,
    const int step) {
  constexpr int kKeyStep = kMode == kSample ? kSampleEvery : 1;
  constexpr int kTileSpan = kTileKeys * kKeyStep;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool issuer = warp == kEpiWarps;
  const int num_tiles = (seg_len + kTileSpan - 1) / kTileSpan;
  const int mine = first < num_tiles ? (num_tiles - first + step - 1) / step : 0;
  const uint32_t g0 = sh.mma_count;
  uint8_t* s_k = smem + kQBytes;
  int act = 0, shift = 0, limit = INT_MAX;
  uint32_t lo = 0, hi = 0xffffffffu;
  int grow = 0;
  if (!issuer && lane < kWarpRows) {
    const int lrow = (warp >> 2) * kWarpRows + lane;
    const RowInfo& r = sh.info[lrow];
    act = r.act;
    lo = r.act > 0 ? r.lo : 0xffffffffu;
    hi = r.hi;
    shift = r.shift;
    limit = r.limit;
    grow = row0 + lrow;
  }
  const int key_in_tile = (warp & 3) * 32 + lane;
  uint32_t los[kWarpRows];
#pragma unroll
  for (int r = 0; r < kWarpRows; ++r) los[r] = __shfl_sync(kFull, lo, r);
  if (!issuer) {
    if (mine > 0) load_keys(p, s_k, seg_start + first * kTileSpan, kKeyStep);
    cp_async_commit();
    if (mine > 1) load_keys(p, s_k + kTileBytes, seg_start + (first + step) * kTileSpan, kKeyStep);
    cp_async_commit();
  }
  const auto qdesc = tcgen05::make_desc_kmajor_sw128(smem);
  const auto kdesc = tcgen05::make_desc_kmajor_sw128(s_k);
  const uint32_t idesc = tcgen05::make_instr_desc_f8<kTileKeys, kN>().bits;
  for (int n = 0; n <= mine; ++n) {
    const uint32_t g = g0 + n;
    if (!issuer && n < mine) {
      cp_async_wait<1>();
      device::ptx::fence_proxy_async_shared();
    }
    device::ptx::bar_sync(1, kThreads);
    if (issuer) {
      if (n < mine && lane == 0) {
        tcgen05::fence_after_thread_sync();
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          tcgen05::mma_f8_ss(
              sh.tmem + (g & 1) * kN,
              kdesc.advance_bytes((n % kStages) * kTileBytes + i * 32).bits,
              qdesc.advance_bytes(i * 32).bits,
              idesc,
              i > 0);
        }
        tcgen05::commit(sh.bars + (g & 1));
      }
      __syncwarp();
      continue;
    }
    if (n >= 1) {
      bool open = true;
      if constexpr (kMode == kCollectLimited) {
        open = lane < kWarpRows && act > 0 && *reinterpret_cast<volatile uint32_t*>(count_of(p, grow)) < limit;
        device::ptx::bar_sync(2, kEpiThreads);
      }
      const int tile_pos = (first + (n - 1) * step) * kTileKeys;
      const int pos = (tile_pos + key_in_tile) * kKeyStep;
      epilogue<kMode>(
          p, sh, hist, g - 1, row0, pos, act, lo, hi, shift, open, (tile_pos + kTileKeys - 1) * kKeyStep, los);
    }
    if (n + 2 < mine) {
      load_keys(p, s_k + ((n + 2) % kStages) * kTileBytes, seg_start + (first + (n + 2) * step) * kTileSpan, kKeyStep);
    }
    cp_async_commit();
  }
  if (!issuer) cp_async_wait<0>();
  __syncthreads();
  if (threadIdx.x == 0) sh.mma_count = g0 + mine;
  __syncthreads();
}

SGL_DEVICE void cta_setup(const Params& p, Shared& sh, uint8_t* smem, const int row0, const bool init) {
  const int warp = threadIdx.x >> 5;
  if (init && threadIdx.x == 0) {
    device::ptx::mbar_init(sh.bars, 1);
    device::ptx::mbar_init(sh.bars + 1, 1);
    device::ptx::fence_barrier_init();
    sh.mma_count = 0;
  }
  if (warp == kEpiWarps) {
    if (init) {
      tcgen05::alloc(&sh.tmem, kTmemCols);
      tcgen05::relinquish_alloc_permit();
      tcgen05::fence_before_thread_sync();
    }
  } else {
    load_q(p, smem, row0);
  }
  __syncthreads();
  tcgen05::fence_after_thread_sync();
}

SGL_DEVICE void cta_teardown(Shared& sh) {
  __syncthreads();
  if ((threadIdx.x >> 5) == kEpiWarps) {
    tcgen05::fence_after_thread_sync();
    tcgen05::dealloc(sh.tmem, kTmemCols);
  }
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) sample_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ Shared sh;
  uint8_t* smem = align_smem(smem_raw);
  device::PDLWaitPrimary<kUsePDL>();
  RowInfo* info = sh.info;
  uint32_t* hist = reinterpret_cast<uint32_t*>(smem + kOpBytes);
  const int row0 = blockIdx.x * kRows;
  load_rows(p, info, row0, false);
  const int first = first_segment(info);
  int seg_start, seg_len;
  if (next_segment(info, seg_start, seg_len, first)) {
    for (int i = threadIdx.x; i < kRows * kHistStride; i += kThreads) hist[i] = 0;
    cta_setup(p, sh, smem, row0, true);
    do {
      __syncthreads();
      if (threadIdx.x < kRows) {
        RowInfo& r = info[threadIdx.x];
        r.act = r.mode && r.start == seg_start ? r.len : 0;
      }
      __syncthreads();
      scan<kSample>(p, sh, smem, hist, row0, seg_start, seg_len, blockIdx.y, p.splits);
      if (threadIdx.x < kRows && info[threadIdx.x].act) info[threadIdx.x].mode = 0;
      __syncthreads();
    } while (next_segment(info, seg_start, seg_len, first));
    for (int i = threadIdx.x; i < kRows * kBins; i += kThreads) {
      const int r = i / kBins;
      const uint32_t v = hist[r * kHistStride + i % kBins];
      if (v) atomicAdd(p.work + static_cast<int64_t>(row0 + r) * kWorkStride + i % kBins, v);
    }
    cta_teardown(sh);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

SGL_DEVICE uint32_t sample_bound(const uint32_t* row_hist) {
  const int lane = threadIdx.x & 31;
  uint32_t h[8];
  uint32_t sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    h[i] = row_hist[kBins - 1 - 8 * lane - i];
    sum += h[i];
  }
  uint32_t incl = sum;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(kFull, incl, o);
    if (lane >= o) incl += v;
  }
  const unsigned hit = __ballot_sync(kFull, incl >= kSampleRank);
  if (hit == 0) return 0;
  const int src = __ffs(hit) - 1;
  int bin = 0;
  if (lane == src) {
    uint32_t cum = incl - sum;
    int i = 0;
    while (cum + h[i] < kSampleRank) cum += h[i++];
    bin = kBins - 1 - 8 * lane - i;
  }
  return window_edge(__shfl_sync(kFull, bin, src));
}


template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) collect_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ Shared sh;
  uint8_t* smem = align_smem(smem_raw);
  device::PDLWaitPrimary<kUsePDL>();
  RowInfo* info = sh.info;
  const int row0 = blockIdx.x * kRows;
  const int warp = threadIdx.x >> 5;
  load_rows(p, info, row0, false);
  const int first = first_segment(info);
  int seg_start, seg_len;
  if (next_segment(info, seg_start, seg_len, first)) {
    for (int i = warp; i < kRows; i += kThreads / 32) {
      if (!info[i].mode) continue;
      const uint32_t lo = sample_bound(p.work + static_cast<int64_t>(row0 + i) * kWorkStride);
      if ((threadIdx.x & 31) == 0) info[i].lo = lo;
    }
    if (threadIdx.x < kRows) {
      sh.counts[threadIdx.x] = 0;
      sh.owned[threadIdx.x] = 0;
    }
    cta_setup(p, sh, smem, row0, true);
    do {
      __syncthreads();
      if (threadIdx.x < kRows) {
        RowInfo& r = info[threadIdx.x];
        r.act = r.mode && r.start == seg_start ? r.len : 0;
        sh.owned[threadIdx.x] |= r.act > 0;
      }
      __syncthreads();
      if (p.scores != nullptr) {
        scan<kCollectScores>(p, sh, smem, nullptr, row0, seg_start, seg_len, blockIdx.y, p.splits);
      } else {
        scan<kCollect>(p, sh, smem, nullptr, row0, seg_start, seg_len, blockIdx.y, p.splits);
      }
      if (threadIdx.x < kRows && info[threadIdx.x].act) info[threadIdx.x].mode = 0;
      __syncthreads();
    } while (next_segment(info, seg_start, seg_len, first));
    if (threadIdx.x < kRows && sh.owned[threadIdx.x]) {
      p.work[static_cast<int64_t>(row0 + threadIdx.x) * kWorkStride + kBins + 2 + blockIdx.y] = sh.counts[threadIdx.x];
    }
    cta_teardown(sh);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

SGL_DEVICE void pick_row(
    const Params& p,
    const int row,
    uint32_t* keys,
    uint32_t* idx,
    uint32_t* bins,
    const int segments,
    const int seg_cap,
    const uint32_t* seg_counts) {
  const int lane = threadIdx.x & 31;
  const unsigned below = (1u << lane) - 1;
  uint32_t k_or = 0, k_and = 0xffffffffu;
  int count = 0;
  for (int seg = 0; seg < segments; ++seg) {
    const int n = static_cast<int>(seg_counts[seg]);
    const uint64_t* cand = p.cand + static_cast<int64_t>(row) * p.cand_stride + seg * seg_cap;
    for (int i0 = 0; i0 < n; i0 += 8 * 32) {
      uint64_t v[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = i0 + j * 32 + lane;
        v[j] = i < n ? cand[i] : 0;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const int i = i0 + j * 32 + lane;
        if (i < n) {
          const uint32_t key = static_cast<uint32_t>(v[j] >> 32);
          keys[count + i] = key;
          idx[count + i] = static_cast<uint32_t>(v[j]);
          k_or |= key;
          k_and &= key;
        }
      }
    }
    count += n;
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    k_or |= __shfl_xor_sync(kFull, k_or, o);
    k_and &= __shfl_xor_sync(kFull, k_and, o);
  }
  __syncwarp();
  int need = kTopK;
  uint32_t prefix = k_and;
  const uint32_t diff = k_or ^ k_and;
  if (diff != 0) {
    int top = 31 - __clz(diff);
    uint32_t mask = ~((2u << top) - 1);
    prefix = k_and & mask;
    while (true) {
      const int shift = max(top - 7, 0);
      const uint32_t digit = (2u << (top - shift)) - 1;
      for (int i = lane; i < kBins; i += 32) bins[i] = 0;
      __syncwarp();
      for (int i = lane; i < count; i += 32) {
        if ((keys[i] & mask) == prefix) atomicAdd(bins + ((keys[i] >> shift) & digit), 1u);
      }
      __syncwarp();
      uint32_t h[8];
      uint32_t sum = 0;
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        h[i] = bins[kBins - 1 - 8 * lane - i];
        sum += h[i];
      }
      uint32_t incl = sum;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        const uint32_t v = __shfl_up_sync(kFull, incl, o);
        if (lane >= o) incl += v;
      }
      const int src = __ffs(__ballot_sync(kFull, incl >= static_cast<uint32_t>(need))) - 1;
      int bin = 0, above = 0;
      if (lane == src) {
        uint32_t cum = incl - sum;
        int i = 0;
        while (cum + h[i] < static_cast<uint32_t>(need)) cum += h[i++];
        bin = kBins - 1 - 8 * lane - i;
        above = static_cast<int>(cum);
      }
      bin = __shfl_sync(kFull, bin, src);
      need -= __shfl_sync(kFull, above, src);
      prefix |= static_cast<uint32_t>(bin) << shift;
      mask |= digit << shift;
      __syncwarp();
      if (shift == 0) break;
      top = shift - 1;
    }
  }
  int ties = 0;
  for (int i = lane; i < count; i += 32) ties += keys[i] == prefix;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ties += __shfl_xor_sync(kFull, ties, o);
  uint32_t cut = 0xffffffffu;
  if (ties > need) {
    uint32_t a = 0, b = 0xffffffffu;
    while (a < b) {
      const uint32_t mid = a + (b - a) / 2;
      int n = 0;
      for (int i = lane; i < count; i += 32) n += keys[i] == prefix && idx[i] <= mid;
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) n += __shfl_xor_sync(kFull, n, o);
      if (n >= need) {
        b = mid;
      } else {
        a = mid + 1;
      }
    }
    cut = a;
  }
  int32_t* out = p.out + static_cast<int64_t>(row) * kTopK;
  int written = 0;
  for (int i0 = 0; i0 < count; i0 += 32) {
    const int i = i0 + lane;
    const uint32_t key = i < count ? keys[i] : 0u;
    const bool tie = i < count && key == prefix;
    const bool take = i < count && (key > prefix || (tie && idx[i] <= cut));
    const unsigned m = __ballot_sync(kFull, take);
    if (take) out[written + __popc(m & below)] = static_cast<int32_t>(idx[i]);
    written += __popc(m);
  }
  __syncwarp();
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kPickWarps * 32) pick_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem[];
  device::PDLWaitPrimary<kUsePDL>();
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * kPickWarps + warp;
  if (row < p.rows) {
    const int start = p.starts[row];
    const int len = min(p.ends[row], p.keys) - start;
    int32_t* out = p.out + static_cast<int64_t>(row) * kTopK;
    if (len <= kTopK) {
      for (int c = lane; c < kTopK; c += 32) out[c] = c < len ? c : -1;
    } else {
      const uint32_t* seg_counts = p.work + static_cast<int64_t>(row) * kWorkStride + kBins + 2;
      uint32_t total = 0, worst = 0;
      for (int seg = 0; seg < p.splits; ++seg) {
        total += seg_counts[seg];
        worst = max(worst, seg_counts[seg]);
      }
      if (total < kTopK || total > kCapacity || worst > static_cast<uint32_t>(p.split_cap)) {
        if (lane == 0) count_of(p, row)[1] = total < kTopK ? 1 : 2;
      } else {
        uint8_t* base = smem + warp * kPickBytes;
        uint32_t* keys = reinterpret_cast<uint32_t*>(base);
        pick_row(p, row, keys, keys + kCapacity, keys + 2 * kCapacity, p.splits, p.split_cap, seg_counts);
      }
    }
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

SGL_DEVICE void narrow(RowInfo& r, const uint32_t* row_hist) {
  const int lane = threadIdx.x & 31;
  uint32_t h[8];
  uint32_t sum = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    h[i] = row_hist[kBins - 1 - 8 * lane - i];
    sum += h[i];
  }
  uint32_t incl = sum + (lane == 0 ? row_hist[kBins] : 0u);
  const uint32_t own = incl;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    const uint32_t v = __shfl_up_sync(kFull, incl, o);
    if (lane >= o) incl += v;
  }
  const int src = __ffs(__ballot_sync(kFull, incl >= kTopK)) - 1;
  int bin = 0;
  uint32_t total = 0;
  if (lane == src) {
    uint32_t cum = incl - own + (lane == 0 ? row_hist[kBins] : 0u);
    int i = 0;
    while (cum + h[i] < kTopK) cum += h[i++];
    bin = kBins - 1 - 8 * lane - i;
    total = cum + h[i];
  }
  bin = __shfl_sync(kFull, bin, src);
  total = __shfl_sync(kFull, total, src);
  const uint32_t new_lo = r.shift < 0 ? window_edge(bin) : r.lo + (static_cast<uint32_t>(bin) << r.shift);
  const uint32_t next = r.shift < 0 ? (bin == kBins - 1 ? r.hi : window_edge(bin + 1) - 1)
                                    : r.lo + ((static_cast<uint32_t>(bin) + 1) << r.shift) - 1;
  const uint32_t new_hi = min(r.hi, next);
  __syncwarp();
  if (lane == 0) {
    if (total <= kCapacity) {
      r.mode = 2;
      r.lo = new_lo;
      r.hi = 0xffffffffu;
      r.limit = INT_MAX;
    } else if (new_lo == new_hi) {
      r.mode = 2;
      r.lo = new_lo;
      r.hi = new_lo;
      r.limit = kTieLimit;
    } else {
      r.lo = new_lo;
      r.hi = new_hi;
      r.shift = max(0, 24 - __clz(new_hi - new_lo));
    }
  }
  __syncwarp();
}


template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) exact_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  __shared__ Shared sh;
  uint8_t* smem = align_smem(smem_raw);
  __shared__ int segment_of[kRows];
  device::PDLWaitPrimary<kUsePDL>();
  RowInfo* info = sh.info;
  uint32_t* hist = reinterpret_cast<uint32_t*>(smem + kOpBytes);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  constexpr int kPickRows = kRows / kExactPickWarps;
  const int tiles = (p.rows + kRows - 1) / kRows;
  bool ready = false;
  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int row0 = tile * kRows;
    __syncthreads();
    load_rows(p, info, row0, true);
    int seg_start, seg_len;
    if (!next_segment(info, seg_start, seg_len, -1)) continue;
    cta_setup(p, sh, smem, row0, !ready);
    ready = true;
    do {
      __syncthreads();
      if (threadIdx.x < kRows) {
        RowInfo& r = info[threadIdx.x];
        segment_of[threadIdx.x] = r.mode && r.start == seg_start;
        if (segment_of[threadIdx.x]) {
          r.mode = 1;
          r.lo = 0;
          r.hi = kKeyMax;
          r.shift = -1;
        }
      }
      __syncthreads();
      while (true) {
        bool any_hist = false, any_collect = false;
        for (int i = 0; i < kRows; ++i) {
          any_hist |= segment_of[i] && info[i].mode == 1;
          any_collect |= segment_of[i] && info[i].mode == 2;
        }
        if (!any_hist && !any_collect) break;
        __syncthreads();
        if (any_hist) {
          for (int i = threadIdx.x; i < kRows * kHistStride; i += kThreads) hist[i] = 0;
          if (threadIdx.x < kRows) {
            info[threadIdx.x].act = segment_of[threadIdx.x] && info[threadIdx.x].mode == 1 ? info[threadIdx.x].len : 0;
          }
          __syncthreads();
          scan<kHist>(p, sh, smem, hist, row0, seg_start, seg_len, 0, 1);
          for (int i = warp; i < kRows; i += kThreads / 32) {
            if (info[i].act) narrow(info[i], hist + i * kHistStride);
          }
          __syncthreads();
        }
        bool any_two = false;
        for (int i = 0; i < kRows; ++i) any_two |= segment_of[i] && info[i].mode == 2;
        if (any_two) {
          if (threadIdx.x < kRows) {
            const bool on = segment_of[threadIdx.x] && info[threadIdx.x].mode == 2;
            info[threadIdx.x].act = on ? info[threadIdx.x].len : 0;
            if (on) *count_of(p, row0 + threadIdx.x) = 0;
          }
          __syncthreads();
          scan<kCollectLimited>(p, sh, smem, nullptr, row0, seg_start, seg_len, 0, 1);
          if (warp < kExactPickWarps) {
            uint32_t* keys = hist + warp * (kPickBytes / 4);
            for (int i = 0; i < kPickRows; ++i) {
              const int lrow = warp * kPickRows + i;
              if (!info[lrow].act) continue;
              pick_row(p, row0 + lrow, keys, keys + kCapacity, keys + 2 * kCapacity, 1, 0, count_of(p, row0 + lrow));
              if (lane == 0) info[lrow].mode = 3;
              __syncwarp();
            }
          }
          __syncthreads();
        }
      }
      if (threadIdx.x < kRows && segment_of[threadIdx.x]) info[threadIdx.x].mode = 0;
      __syncthreads();
    } while (next_segment(info, seg_start, seg_len, -1));
  }
  if (ready) cta_teardown(sh);
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_prefill_topk

template <bool kUsePDL>
struct QsaPrefillTopk {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView starts,
      const tvm::ffi::TensorView ends,
      const tvm::ffi::TensorView out,
      const tvm::ffi::TensorView cand,
      const tvm::ffi::TensorView work,
      const tvm::ffi::TensorView scores,
      const int64_t splits) {
    using namespace host;
    using namespace qsa_prefill_topk;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto N_ = SymbolicSize{"keys"};
    auto W_ = SymbolicSize{"score_width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, kD, 1}).verify(q);
    TensorMatcher({N_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({kD, 1}).verify(k);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(starts);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(ends);
    TensorMatcher({R_, kTopK}).with_dtype<int32_t>().with_device(device).with_strides({kTopK, 1}).verify(out);
    auto CS_ = SymbolicSize{"cand_stride"};
    TensorMatcher({R_, CS_}).with_dtype<int64_t>().with_device(device).with_strides({CS_, 1}).verify(cand);
    TensorMatcher({R_, kWorkStride}).with_dtype<int32_t>().with_device(device).with_strides({kWorkStride, 1}).verify(work);
    TensorMatcher({R_, W_}).with_dtype<float>().with_device(device).with_strides({-1, 1}).verify(scores);
    const int64_t rows = R_.unwrap();
    RuntimeCheck(HQ_.unwrap() >= kHeads, "q needs at least 4 heads");
    RuntimeCheck(splits >= 1 && splits <= kMaxSplits, "splits must be in [1, 64]");
    const int64_t split_cap = splits == 1 ? kCapacity : CS_.unwrap() / splits;
    RuntimeCheck(CS_.unwrap() >= kCapacity && split_cap >= 32, "candidate rows must hold the top-k capacity");
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
        .cand = static_cast<uint64_t*>(cand.data_ptr()),
        .work = static_cast<uint32_t*>(work.data_ptr()),
        .scores = W_.unwrap() > 0 ? static_cast<float*>(scores.data_ptr()) : nullptr,
        .stride_q = q.stride(0),
        .score_stride = W_.unwrap() > 0 ? scores.stride(0) : 0,
        .cand_stride = CS_.unwrap(),
        .rows = static_cast<int32_t>(rows),
        .keys = static_cast<int32_t>(N_.unwrap()),
        .splits = static_cast<int32_t>(splits),
        .split_cap = static_cast<int32_t>(split_cap),
    };
    const uint32_t tiles = static_cast<uint32_t>((rows + kRows - 1) / kRows);
    const dim3 split_grid(tiles, static_cast<uint32_t>(splits), 2);
    const auto sample = sample_kernel<kUsePDL>;
    const auto collect = collect_kernel<kUsePDL>;
    const auto pick = pick_kernel<kUsePDL>;
    const auto exact = exact_kernel<kUsePDL>;
    RuntimeDeviceCheck(cudaFuncSetAttribute(sample, cudaFuncAttributeMaxDynamicSharedMemorySize, kSampleSmem));
    RuntimeDeviceCheck(cudaFuncSetAttribute(collect, cudaFuncAttributeMaxDynamicSharedMemorySize, kCollectSmem));
    RuntimeDeviceCheck(
        cudaFuncSetAttribute(pick, cudaFuncAttributeMaxDynamicSharedMemorySize, kPickWarps * kPickBytes));
    RuntimeDeviceCheck(cudaFuncSetAttribute(exact, cudaFuncAttributeMaxDynamicSharedMemorySize, kExactSmem));
    LaunchKernel(split_grid, kThreads, device.unwrap(), kSampleSmem).enable_pdl(kUsePDL)(sample, params);
    LaunchKernel(split_grid, kThreads, device.unwrap(), kCollectSmem).enable_pdl(kUsePDL)(collect, params);
    LaunchKernel(
        static_cast<uint32_t>((rows + kPickWarps - 1) / kPickWarps),
        kPickWarps * 32,
        device.unwrap(),
        kPickWarps * kPickBytes)
        .enable_pdl(kUsePDL)(pick, params);
    LaunchKernel(min(tiles, static_cast<uint32_t>(runtime::get_sm_count(device.unwrap().device_id))), kThreads, device.unwrap(), kExactSmem).enable_pdl(kUsePDL)(exact, params);
  }
};

}  // namespace sglang
