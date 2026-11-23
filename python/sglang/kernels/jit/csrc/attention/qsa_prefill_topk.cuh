// QSA (Qwen4-Exp) prefill block selection over packed FP8 E4M3 compressed keys: per row, the 512 keys of
// [starts[row], ends[row]) with the largest sum over the 4 index heads of relu(q[row, head] . key), ties at the
// cut broken by the lower key index. Rows with at most 512 keys get 0..len-1, then -1.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <climits>
#include <cstdint>

namespace sglang {

namespace qsa_prefill_topk {

constexpr int kD = 128;
constexpr int kHeads = 4;
constexpr int kTopK = 512;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kRowGroups = 2;
constexpr int kWarpRows = 4 * kRowGroups;
constexpr int kRows = kWarps * kWarpRows;
constexpr int kTileKeys = 64;
constexpr int kSubTiles = kTileKeys / 16;
constexpr int kKeyStride = 2 * kD + 16;
constexpr int kTileBytes = kTileKeys * kKeyStride;
constexpr int kStages = 2;
constexpr int kTilesBytes = kStages * kTileBytes;
constexpr int kBins = 256;
constexpr int kHistStride = kBins + 1;
constexpr int kHistBytes = kRows * kHistStride * 4;
constexpr int kStageEntries = 48;
constexpr int kStagingBytes = kRows * kStageEntries * 8;
constexpr int kSampleEvery = 8;
constexpr int kSampleRank = 112;
constexpr int kCapacity = 1664;
constexpr int kTieLimit = kCapacity - kTopK - 32;
constexpr int kPickWarps = 4;
constexpr int kPickBytes = kCapacity * 8 + kBins * 4;
constexpr int kWorkStride = kBins + 2;
constexpr uint32_t kCoarseLo = 0x3e800000u;
constexpr uint32_t kFineLo = 0x40800000u;
constexpr uint32_t kFineHi = 0x43800000u;
constexpr int kCoarseShift = 20;
constexpr int kFineShift = 18;
constexpr int kCoarseBins = 32;
constexpr int kFineBins = 192;
constexpr uint32_t kKeyMax = 0x7fffffffu;
constexpr unsigned kFull = 0xffffffffu;
constexpr int kSampleSmem = kTilesBytes + kHistBytes;
constexpr int kCollectSmem = kTilesBytes + kStagingBytes;
constexpr int kExactScratch = kHistBytes > kWarps * kPickBytes ? kHistBytes : kWarps * kPickBytes;
constexpr int kExactSmem = kTilesBytes + kStagingBytes + kExactScratch;

enum Mode { kSample, kCollect, kHist, kCollectLimited };

struct Params {
  const uint8_t* q;     // fp8 e4m3 [rows, q_heads, 128]
  const uint8_t* k;     // fp8 e4m3 [keys, 128]
  const int32_t* starts;
  const int32_t* ends;
  int32_t* out;         // [rows, 512]
  uint64_t* cand;       // [rows, capacity]: score bits << 32 | key
  uint32_t* work;       // [rows, 258]: sample histogram, candidate count, fallback flag (zeroed)
  int64_t stride_q;
  int32_t rows, keys, splits;
};

struct RowInfo {
  int32_t start, len, act;
  uint32_t lo, hi;
  int32_t shift, limit, mode;
};

SGL_DEVICE uint32_t load_u32(const uint8_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
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

SGL_DEVICE void mma_f16(float (&d)[4], const uint32_t* a, const uint32_t b0, const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

SGL_DEVICE uint32_t e4m3x2_to_f16x2(const uint32_t x) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<uint16_t>(x)));
  return r;
}

SGL_DEVICE void mma_e4m3_k32(
    float (&acc)[4], const uint32_t (&a_lo)[4], const uint32_t (&a_hi)[4], const uint32_t (&b)[4], const bool first) {
  float part[4] = {};
  mma_f16(part, a_lo, b[0], b[1]);
  mma_f16(part, a_hi, b[2], b[3]);
#pragma unroll
  for (int i = 0; i < 4; ++i) acc[i] = first ? part[i] : acc[i] + part[i];
}

SGL_DEVICE void ldmatrix_x4(uint32_t (&a)[4], const uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}

SGL_DEVICE uint32_t score_key(const float a0, const float a1, const float b0, const float b1) {
  const float s = (fmaxf(a0, 0.f) + fmaxf(a1, 0.f)) + (fmaxf(b0, 0.f) + fmaxf(b1, 0.f));
  return __float_as_uint(s) & kKeyMax;
}

SGL_DEVICE void load_q(const Params& p, const int row0, uint32_t (&qf)[kRowGroups][2][4][4]) {
  const int lane = threadIdx.x & 31;
  const int g = lane >> 2;
  const int t = lane & 3;
#pragma unroll
  for (int rg = 0; rg < kRowGroups; ++rg) {
    const int r = row0 + (threadIdx.x >> 5) * kWarpRows + rg * 4 + (g >> 1);
    const bool ok = r < p.rows;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const uint8_t* src = p.q + static_cast<int64_t>(ok ? r : 0) * p.stride_q + (2 * j + (g & 1)) * kD + 4 * t;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        const uint32_t b0 = ok ? load_u32(src + 32 * c) : 0u;
        const uint32_t b1 = ok ? load_u32(src + 32 * c + 16) : 0u;
        qf[rg][j][c][0] = e4m3x2_to_f16x2(b0);
        qf[rg][j][c][1] = e4m3x2_to_f16x2(b1);
        qf[rg][j][c][2] = e4m3x2_to_f16x2(b0 >> 16);
        qf[rg][j][c][3] = e4m3x2_to_f16x2(b1 >> 16);
      }
    }
  }
}

constexpr int kLoads = kTileKeys * 8 / kThreads;

SGL_DEVICE void fetch_tile(const Params& p, uint4 (&regs)[kLoads], const int key0, const int key_step) {
#pragma unroll
  for (int w = 0; w < kLoads; ++w) {
    const int chunk = threadIdx.x + w * kThreads;
    const int key = key0 + (chunk >> 3) * key_step;
    regs[w] = key < p.keys ? *reinterpret_cast<const uint4*>(p.k + static_cast<int64_t>(key) * kD + (chunk & 7) * 16)
                           : make_uint4(0, 0, 0, 0);
  }
}

SGL_DEVICE void store_tile(uint8_t* stage, const uint4 (&regs)[kLoads]) {
#pragma unroll
  for (int w = 0; w < kLoads; ++w) {
    const int chunk = threadIdx.x + w * kThreads;
    const int part = chunk & 7;
    uint8_t* row = stage + (chunk >> 3) * kKeyStride + (part >> 1) * 64 + (part & 1) * 16;
    const uint32_t v[4] = {regs[w].x, regs[w].y, regs[w].z, regs[w].w};
    uint4 lo, hi;
    lo.x = e4m3x2_to_f16x2(v[0]);
    lo.y = e4m3x2_to_f16x2(v[1]);
    lo.z = e4m3x2_to_f16x2(v[2]);
    lo.w = e4m3x2_to_f16x2(v[3]);
    hi.x = e4m3x2_to_f16x2(v[0] >> 16);
    hi.y = e4m3x2_to_f16x2(v[1] >> 16);
    hi.z = e4m3x2_to_f16x2(v[2] >> 16);
    hi.w = e4m3x2_to_f16x2(v[3] >> 16);
    *reinterpret_cast<uint4*>(row) = lo;
    *reinterpret_cast<uint4*>(row + 32) = hi;
  }
}

SGL_DEVICE void flush_row(
    const Params& p, uint64_t* stage, const int grow, const int n, const int keep, const int reserved_base) {
  const int lane = threadIdx.x & 31;
  int base = reserved_base;
  if (base < 0) {
    if (lane == 0) base = static_cast<int>(atomicAdd(count_of(p, grow), static_cast<uint32_t>(n)));
    base = __shfl_sync(kFull, base, 0);
  }
  if (lane < n && base + lane < kCapacity) p.cand[static_cast<int64_t>(grow) * kCapacity + base + lane] = stage[lane];
  const uint64_t rest = lane < keep ? stage[n + lane] : 0;
  __syncwarp();
  if (lane < keep) stage[lane] = rest;
  __syncwarp();
}

template <Mode kMode>
SGL_DEVICE void scan(
    const Params& p,
    const RowInfo* info,
    uint8_t* tiles,
    uint32_t* hist,
    uint64_t* staging,
    const uint32_t (&qf)[kRowGroups][2][4][4],
    const int row0,
    const int seg_start,
    const int seg_len,
    const int first,
    const int step) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const unsigned row_lanes = 0x11111111u << t;
  const unsigned below = (1u << lane) - 1;
  int act[kRowGroups];
  uint32_t lo[kRowGroups], hi[kRowGroups];
  int shift[kRowGroups], limit[kRowGroups], staged[kRowGroups], sent[kRowGroups];
#pragma unroll
  for (int rg = 0; rg < kRowGroups; ++rg) {
    const RowInfo& r = info[warp * kWarpRows + rg * 4 + t];
    act[rg] = r.act;
    lo[rg] = r.lo;
    hi[rg] = r.hi;
    shift[rg] = r.shift;
    limit[rg] = r.limit;
    staged[rg] = 0;
    sent[rg] = 0;
  }
  constexpr int kKeyStep = kMode == kSample ? kSampleEvery : 1;
  constexpr int kTileSpan = kTileKeys * kKeyStep;
  const int num_tiles = (seg_len + kTileSpan - 1) / kTileSpan;
  const int mine = first < num_tiles ? (num_tiles - first + step - 1) / step : 0;
  const uint32_t tiles_addr = static_cast<uint32_t>(__cvta_generic_to_shared(tiles));
  const uint32_t lm_off = ((lane & 7) + ((lane >> 3) & 1) * 8) * kKeyStride + (lane >> 4) * 16;

  uint4 regs[kLoads];
  if (mine > 0) {
    fetch_tile(p, regs, seg_start + first * kTileSpan, kKeyStep);
    store_tile(tiles, regs);
  }
  if (mine > 1) fetch_tile(p, regs, seg_start + (first + step) * kTileSpan, kKeyStep);
  for (int n = 0; n < mine; ++n) {
    __syncthreads();
    if (n + 1 < mine) {
      store_tile(tiles + ((n + 1) & 1) * kTileBytes, regs);
      if (n + 2 < mine) fetch_tile(p, regs, seg_start + (first + (n + 2) * step) * kTileSpan, kKeyStep);
    }
    const uint32_t stage_addr = tiles_addr + (n & 1) * kTileBytes + lm_off;
    const int tile_pos = (first + n * step) * kTileSpan;
#pragma unroll 1
    for (int u = 0; u < kSubTiles; ++u) {
      float acc[kRowGroups][2][4] = {};
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        uint32_t a_lo[4], a_hi[4];
        ldmatrix_x4(a_lo, stage_addr + u * 16 * kKeyStride + 64 * c);
        ldmatrix_x4(a_hi, stage_addr + u * 16 * kKeyStride + 64 * c + 32);
#pragma unroll
        for (int rg = 0; rg < kRowGroups; ++rg) {
          mma_e4m3_k32(acc[rg][0], a_lo, a_hi, qf[rg][0][c], c == 0);
          mma_e4m3_k32(acc[rg][1], a_lo, a_hi, qf[rg][1][c], c == 0);
        }
      }
      const int pos0 = tile_pos + (u * 16 + g) * kKeyStep;
      const int pos1 = pos0 + 8 * kKeyStep;
#pragma unroll
      for (int rg = 0; rg < kRowGroups; ++rg) {
        const int lrow = warp * kWarpRows + rg * 4 + t;
        const uint32_t key0 = score_key(acc[rg][0][0], acc[rg][0][1], acc[rg][1][0], acc[rg][1][1]);
        const uint32_t key1 = score_key(acc[rg][0][2], acc[rg][0][3], acc[rg][1][2], acc[rg][1][3]);
        if constexpr (kMode == kSample) {
          if (pos0 < act[rg]) atomicAdd(hist + lrow * kHistStride + window_bin(key0), 1u);
          if (pos1 < act[rg]) atomicAdd(hist + lrow * kHistStride + window_bin(key1), 1u);
        } else if constexpr (kMode == kHist) {
          if (pos0 < act[rg] && key0 >= lo[rg]) {
            atomicAdd(hist + lrow * kHistStride + (key0 > hi[rg] ? kBins : hist_bin(key0, lo[rg], shift[rg])), 1u);
          }
          if (pos1 < act[rg] && key1 >= lo[rg]) {
            atomicAdd(hist + lrow * kHistStride + (key1 > hi[rg] ? kBins : hist_bin(key1, lo[rg], shift[rg])), 1u);
          }
        } else {
          bool p0 = pos0 < act[rg] && key0 >= lo[rg];
          bool p1 = pos1 < act[rg] && key1 >= lo[rg];
          if constexpr (kMode == kCollectLimited) {
            const bool open = sent[rg] + staged[rg] < limit[rg];
            p0 = p0 && (key0 > hi[rg] || open);
            p1 = p1 && (key1 > hi[rg] || open);
          }
          const unsigned m0 = __ballot_sync(kFull, p0);
          const unsigned m1 = __ballot_sync(kFull, p1);
          const int c0 = __popc(m0 & row_lanes);
          uint64_t* row_stage = staging + lrow * kStageEntries + staged[rg];
          if (p0) row_stage[__popc(m0 & row_lanes & below)] = (static_cast<uint64_t>(key0) << 32) | pos0;
          if (p1) row_stage[c0 + __popc(m1 & row_lanes & below)] = (static_cast<uint64_t>(key1) << 32) | pos1;
          staged[rg] += c0 + __popc(m1 & row_lanes);
        }
      }
      if constexpr (kMode == kCollect || kMode == kCollectLimited) {
#pragma unroll
        for (int rg = 0; rg < kRowGroups; ++rg) {
          const unsigned full = __ballot_sync(kFull, staged[rg] >= 32) & 0xfu;
          if (full == 0) continue;
          __syncwarp();
          for (int r = 0; r < 4; ++r) {
            if (!(full >> r & 1)) continue;
            const int lrow = warp * kWarpRows + rg * 4 + r;
            const int n_staged = __shfl_sync(kFull, staged[rg], r);
            flush_row(p, staging + lrow * kStageEntries, row0 + lrow, 32, n_staged - 32, -1);
          }
          if (full >> t & 1) {
            staged[rg] -= 32;
            sent[rg] += 32;
          }
        }
      }
    }
  }
  if constexpr (kMode == kCollect || kMode == kCollectLimited) {
    __syncwarp();
#pragma unroll
    for (int rg = 0; rg < kRowGroups; ++rg) {
      for (int r = 0; r < 4; ++r) {
        const int n_staged = __shfl_sync(kFull, staged[rg], r);
        if (n_staged == 0) continue;
        const int lrow = warp * kWarpRows + rg * 4 + r;
        flush_row(p, staging + lrow * kStageEntries, row0 + lrow, n_staged, 0, -1);
      }
    }
  }
  __syncthreads();
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

SGL_DEVICE bool next_segment(const RowInfo* info, const int first, int& seg_start, int& seg_len) {
  seg_start = -1;
  for (int i = 0; i < kRows; ++i) {
    if (info[i].mode && (blockIdx.z == 0) == (info[i].start == first)) {
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

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 2) sample_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ RowInfo info[kRows];
  device::PDLWaitPrimary<kUsePDL>();
  uint8_t* tiles = smem;
  uint32_t* hist = reinterpret_cast<uint32_t*>(smem + kTilesBytes);
  const int row0 = blockIdx.x * kRows;
  load_rows(p, info, row0, false);
  const int first = first_segment(info);
  int seg_start, seg_len;
  if (first >= 0 && next_segment(info, first, seg_start, seg_len)) {
    for (int i = threadIdx.x; i < kRows * kHistStride; i += kThreads) hist[i] = 0;
    uint32_t qf[kRowGroups][2][4][4];
    load_q(p, row0, qf);
    do {
      __syncthreads();
      if (threadIdx.x < kRows) {
        RowInfo& r = info[threadIdx.x];
        r.act = r.mode && r.start == seg_start ? r.len : 0;
      }
      __syncthreads();
      scan<kSample>(
          p, info, tiles, hist, nullptr, qf, row0, seg_start, seg_len, blockIdx.y, p.splits);
      if (threadIdx.x < kRows && info[threadIdx.x].act) info[threadIdx.x].mode = 0;
      __syncthreads();
    } while (next_segment(info, first, seg_start, seg_len));
    for (int i = threadIdx.x; i < kRows * kBins; i += kThreads) {
      const int r = i / kBins;
      const uint32_t v = hist[r * kHistStride + i % kBins];
      if (v) atomicAdd(p.work + static_cast<int64_t>(row0 + r) * kWorkStride + i % kBins, v);
    }
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
__global__ void __launch_bounds__(kThreads, 2) collect_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ RowInfo info[kRows];
  device::PDLWaitPrimary<kUsePDL>();
  uint8_t* tiles = smem;
  uint64_t* staging = reinterpret_cast<uint64_t*>(smem + kTilesBytes);
  const int row0 = blockIdx.x * kRows;
  const int warp = threadIdx.x >> 5;
  load_rows(p, info, row0, false);
  const int first = first_segment(info);
  int seg_start, seg_len;
  if (first >= 0 && next_segment(info, first, seg_start, seg_len)) {
    for (int i = 0; i < kWarpRows; ++i) {
      const int lrow = warp * kWarpRows + i;
      if (!info[lrow].mode) continue;
      const uint32_t lo = sample_bound(p.work + static_cast<int64_t>(row0 + lrow) * kWorkStride);
      if ((threadIdx.x & 31) == 0) info[lrow].lo = lo;
    }
    uint32_t qf[kRowGroups][2][4][4];
    load_q(p, row0, qf);
    do {
      __syncthreads();
      if (threadIdx.x < kRows) {
        RowInfo& r = info[threadIdx.x];
        r.act = r.mode && r.start == seg_start ? r.len : 0;
      }
      __syncthreads();
      scan<kCollect>(p, info, tiles, nullptr, staging, qf, row0, seg_start, seg_len, blockIdx.y, p.splits);
      if (threadIdx.x < kRows && info[threadIdx.x].act) info[threadIdx.x].mode = 0;
      __syncthreads();
    } while (next_segment(info, first, seg_start, seg_len));
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

SGL_DEVICE void pick_row(const Params& p, const int row, uint32_t* keys, uint32_t* idx, uint32_t* bins) {
  const int lane = threadIdx.x & 31;
  const unsigned below = (1u << lane) - 1;
  const int count = min(static_cast<int>(*count_of(p, row)), kCapacity);
  const uint64_t* cand = p.cand + static_cast<int64_t>(row) * kCapacity;
  uint32_t k_or = 0, k_and = 0xffffffffu;
  for (int i = lane; i < count; i += 32) {
    const uint64_t v = cand[i];
    keys[i] = static_cast<uint32_t>(v >> 32);
    idx[i] = static_cast<uint32_t>(v);
    k_or |= keys[i];
    k_and &= keys[i];
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
    const bool take = i < count && (keys[i] > prefix || (keys[i] == prefix && idx[i] <= cut));
    const unsigned m = __ballot_sync(kFull, take);
    if (take) out[written + __popc(m & below)] = static_cast<int32_t>(idx[i]);
    written += __popc(m);
  }
  __syncwarp();
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kPickWarps * 32) pick_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
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
      const uint32_t count = *count_of(p, row);
      if (count < kTopK || count > kCapacity) {
        if (lane == 0) count_of(p, row)[1] = count < kTopK ? 1 : 2;
      } else {
        uint8_t* base = smem + warp * kPickBytes;
        uint32_t* keys = reinterpret_cast<uint32_t*>(base);
        pick_row(p, row, keys, keys + kCapacity, keys + 2 * kCapacity);
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

SGL_DEVICE bool any_mode(const RowInfo* info, const int mode) {
  bool any = false;
  for (int i = 0; i < kRows; ++i) any |= info[i].mode == mode;
  return any;
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) exact_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  __shared__ RowInfo info[kRows];
  __shared__ int segment_of[kRows];
  device::PDLWaitPrimary<kUsePDL>();
  uint8_t* tiles = smem;
  uint64_t* staging = reinterpret_cast<uint64_t*>(smem + kTilesBytes);
  uint32_t* hist = reinterpret_cast<uint32_t*>(smem + kTilesBytes + kStagingBytes);
  const int row0 = blockIdx.x * kRows;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  load_rows(p, info, row0, true);
  const int first = first_segment(info);
  int seg_start, seg_len;
  if (first >= 0 && next_segment(info, first, seg_start, seg_len)) {
    uint32_t qf[kRowGroups][2][4][4];
    load_q(p, row0, qf);
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
          scan<kHist>(p, info, tiles, hist, nullptr, qf, row0, seg_start, seg_len, 0, 1);
          for (int i = 0; i < kWarpRows; ++i) {
            const int lrow = warp * kWarpRows + i;
            if (info[lrow].act) narrow(info[lrow], hist + lrow * kHistStride);
          }
          __syncthreads();
        }
        if (any_mode(info, 2)) {
          if (threadIdx.x < kRows) {
            const bool on = segment_of[threadIdx.x] && info[threadIdx.x].mode == 2;
            info[threadIdx.x].act = on ? info[threadIdx.x].len : 0;
            if (on) *count_of(p, row0 + threadIdx.x) = 0;
          }
          __syncthreads();
          scan<kCollectLimited>(p, info, tiles, nullptr, staging, qf, row0, seg_start, seg_len, 0, 1);
          uint32_t* keys = hist + warp * (kPickBytes / 4);
          for (int i = 0; i < kWarpRows; ++i) {
            const int lrow = warp * kWarpRows + i;
            if (!info[lrow].act) continue;
            pick_row(p, row0 + lrow, keys, keys + kCapacity, keys + 2 * kCapacity);
            if (lane == 0) info[lrow].mode = 3;
            __syncwarp();
          }
          __syncthreads();
        }
      }
      if (threadIdx.x < kRows && segment_of[threadIdx.x]) info[threadIdx.x].mode = 0;
      __syncthreads();
    } while (next_segment(info, first, seg_start, seg_len));
  }
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
      const int64_t splits) {
    using namespace host;
    using namespace qsa_prefill_topk;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto N_ = SymbolicSize{"keys"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, kD, 1}).verify(q);
    TensorMatcher({N_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({kD, 1}).verify(k);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(starts);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(ends);
    TensorMatcher({R_, kTopK}).with_dtype<int32_t>().with_device(device).with_strides({kTopK, 1}).verify(out);
    TensorMatcher({R_, kCapacity}).with_dtype<int64_t>().with_device(device).with_strides({kCapacity, 1}).verify(cand);
    TensorMatcher({R_, kWorkStride}).with_dtype<int32_t>().with_device(device).with_strides({kWorkStride, 1}).verify(work);
    const int64_t rows = R_.unwrap();
    RuntimeCheck(HQ_.unwrap() >= kHeads, "q needs at least 4 heads");
    RuntimeCheck(splits >= 1 && splits <= 1024, "splits must be in [1, 1024]");
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
        .stride_q = q.stride(0),
        .rows = static_cast<int32_t>(rows),
        .keys = static_cast<int32_t>(N_.unwrap()),
        .splits = static_cast<int32_t>(splits),
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
    LaunchKernel(dim3(tiles, 1, 2), kThreads, device.unwrap(), kExactSmem).enable_pdl(kUsePDL)(exact, params);
  }
};

}  // namespace sglang
