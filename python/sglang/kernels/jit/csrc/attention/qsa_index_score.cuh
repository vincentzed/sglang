// QSA (Qwen4-Exp) decode / target-verify index scoring over the paged FP8 E4M3 compressed-key cache:
// logits[row, n] = (sum over the 4 index heads of relu(q[row, head] . key[n])) / scale for n < lens[row].
// The decode top-512 picks each row's 512 largest logits, ties at the cut to the lower key index.
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

namespace qsa_index_score {

constexpr int kD = 128;
constexpr int kHeads = 4;
constexpr int kPageKeys = 16;
constexpr int kPageBytes = kPageKeys * kD;
constexpr int kMaxGroupRows = 4;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kKeyStride = kD + 16;
constexpr int kTileBytes = kPageKeys * kKeyStride;
constexpr unsigned kFull = 0xffffffffu;

struct Params {
  const uint8_t* q;           // fp8 e4m3 [rows, q_heads, 128]
  const uint8_t* k;           // fp8 e4m3 [cache_pages, 16 * 128]
  const int32_t* page_table;  // [rows, max_pages]
  const int32_t* lens;        // [rows]
  float* logits;              // [rows, width]
  int64_t stride_q, stride_pt, stride_logits;
  float scale;
  int32_t rows, group_rows, cache_pages, max_pages, width, min_scored_len;
};

SGL_DEVICE uint32_t load_u32(const uint8_t* p) {
  return *reinterpret_cast<const uint32_t*>(p);
}

SGL_DEVICE void mma_e4m3(float (&d)[4], const uint32_t (&a)[4], const uint32_t b0, const uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads) decode_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  device::PDLWaitPrimary<kUsePDL>();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;
  const int t = lane & 3;
  const int row0 = blockIdx.x * p.group_rows;
  const int table_row = min(row0 + p.group_rows, p.rows) - 1;
  const int ctx = p.lens[table_row];
  if (ctx > p.min_scored_len) {
    uint32_t q_frag[2][4][2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int group_row = (j * 8 + g) / kHeads;
      const int head = (j * 8 + g) % kHeads;
      const bool ok = group_row < p.group_rows && row0 + group_row < p.rows;
      const uint8_t* src = p.q + (row0 + (ok ? group_row : 0)) * p.stride_q + head * kD + 4 * t;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        q_frag[j][c][0] = ok ? load_u32(src + 32 * c) : 0u;
        q_frag[j][c][1] = ok ? load_u32(src + 32 * c + 16) : 0u;
      }
    }
    int len[2];
    float* out[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int group_row = 2 * j + (t >> 1);
      const bool ok = group_row < p.group_rows && row0 + group_row < p.rows;
      const int n = ok ? p.lens[row0 + group_row] : 0;
      len[j] = n > p.min_scored_len ? min(n, p.width) : 0;
      out[j] = p.logits + (row0 + (ok ? group_row : 0)) * p.stride_logits;
    }

    const int num_pages = min((ctx + kPageKeys - 1) / kPageKeys, p.max_pages);
    const int streams = gridDim.y * kWarps;
    const int chunk = (num_pages + streams - 1) / streams;
    const int begin = (blockIdx.y * kWarps + warp) * chunk;
    const int end = min(begin + chunk, num_pages);
    const int32_t* pt = p.page_table + table_row * p.stride_pt;
    uint8_t* tile = smem + warp * kTileBytes;
    uint8_t* stage = tile + (lane >> 3) * kKeyStride + (lane & 7) * 16;
    const uint8_t* ka = tile + g * kKeyStride + 4 * t;
    const uint8_t* kb = ka + 8 * kKeyStride;
    const bool writer = (t & 1) == 0;
    const float inv_scale = __fdividef(1.f, p.scale);

    for (int base = begin; base < end; base += 32) {
      const int pids = base + lane < end ? pt[base + lane] : 0;
      const int n = min(32, end - base);
      for (int i = 0; i < n; ++i) {
        const int pid = min(max(__shfl_sync(kFull, pids, i), 0), p.cache_pages - 1);
        const uint8_t* src = p.k + static_cast<int64_t>(pid) * kPageBytes + lane * 16;
        uint4 raw[4];
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          raw[x] = *reinterpret_cast<const uint4*>(src + x * 512);
        }
#pragma unroll
        for (int x = 0; x < 4; ++x) {
          *reinterpret_cast<uint4*>(stage + x * 4 * kKeyStride) = raw[x];
        }
        float acc[2][4] = {};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint32_t a[4] = {
              load_u32(ka + 32 * c), load_u32(kb + 32 * c), load_u32(ka + 32 * c + 16), load_u32(kb + 32 * c + 16)};
          mma_e4m3(acc[0], a, q_frag[0][c][0], q_frag[0][c][1]);
          mma_e4m3(acc[1], a, q_frag[1][c][0], q_frag[1][c][1]);
        }
        const int pos = (base + i) * kPageKeys + g;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float lo = fmaxf(acc[j][0], 0.f) + fmaxf(acc[j][1], 0.f);
          float hi = fmaxf(acc[j][2], 0.f) + fmaxf(acc[j][3], 0.f);
          lo += __shfl_xor_sync(kFull, lo, 1);
          hi += __shfl_xor_sync(kFull, hi, 1);
          if (writer && pos < len[j]) out[j][pos] = lo * inv_scale;
          if (writer && pos + 8 < len[j]) out[j][pos + 8] = hi * inv_scale;
        }
      }
    }
  }

  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_index_score

namespace qsa_index_score_sm100 {

using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kD = 128;
constexpr int kPageKeys = 16;
constexpr int kPageBytes = kPageKeys * kD;
constexpr int kTilePages = 8;
constexpr int kTileKeys = kTilePages * kPageKeys;
constexpr int kTileBytes = kTilePages * kPageBytes;
constexpr int kN = 16;
constexpr int kBBytes = kN * kD * 2;
constexpr int kKBlockBytes = kN * 128;
constexpr int kGroupCols = 128;
constexpr unsigned kFull = 0xffffffffu;

constexpr int kGroups = 4;
constexpr int kStages = 8;
constexpr int kCpWarps = 3;
constexpr int kTmaWarp = kGroups * 4;
constexpr int kCpWarp = kTmaWarp + 1;
constexpr int kThreads = (kCpWarp + kCpWarps) * 32;
constexpr int kTmaPages = kTilePages - kCpWarps;
constexpr int kTmemCols = kGroups * kGroupCols;
constexpr int kOpBytes = kStages * kTileBytes + kGroups * kBBytes;
static_assert(kStages % kGroups == 0 && kTmemCols <= 512);

struct Params {
  CUtensorMap k_map;          // fp8 e4m3 keys as [cache_pages * 16, 128] bytes, 128 B swizzle, 16-row box
  const uint8_t* k;           // fp8 e4m3 [cache_pages, 16 * 128]
  const uint8_t* q;           // fp8 e4m3 [rows, q_heads, 128]
  const int32_t* page_table;  // [rows, max_pages]
  const int32_t* lens;        // [rows]
  float* logits;              // [rows, width]
  int64_t stride_q, stride_pt, stride_logits;
  float scale;
  int32_t rows, group_rows, groups, cache_pages, max_pages, width, min_scored_len;
};

struct Shared {
  uint64_t full[kStages];
  uint64_t empty[kStages];
  uint64_t d_full[kGroups];
  uint32_t tmem;
};

SGL_DEVICE uint32_t f16x2_of_e4m3x2(const uint32_t x) {
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<uint16_t>(x)));
  return r;
}

SGL_DEVICE int find_group(const int32_t* prefix, const int groups, const int t) {
  int lo = 0, hi = groups - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (prefix[mid] <= t) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  return lo;
}

SGL_DEVICE int table_row_of(const Params& p, const int g) {
  return min((g + 1) * p.group_rows, p.rows) - 1;
}

SGL_DEVICE uint4 load_q_chunk(const Params& p, const int g, const int n, const int chunk) {
  const int row = g * p.group_rows + (n >> 2);
  if (g >= p.groups || (n >> 2) >= p.group_rows || row >= p.rows) return make_uint4(0, 0, 0, 0);
  return *reinterpret_cast<const uint4*>(p.q + row * p.stride_q + (n & 3) * kD + chunk * 16);
}

SGL_DEVICE void store_b_chunk(uint8_t* s_b, const uint4 v, const int n, const int chunk) {
  const uint32_t w[4] = {v.x, v.y, v.z, v.w};
  uint4 lo, hi;
  lo.x = f16x2_of_e4m3x2(w[0]);
  lo.y = f16x2_of_e4m3x2(w[1]);
  lo.z = f16x2_of_e4m3x2(w[2]);
  lo.w = f16x2_of_e4m3x2(w[3]);
  hi.x = f16x2_of_e4m3x2(w[0] >> 16);
  hi.y = f16x2_of_e4m3x2(w[1] >> 16);
  hi.z = f16x2_of_e4m3x2(w[2] >> 16);
  hi.w = f16x2_of_e4m3x2(w[3] >> 16);
  const int k_lo = 32 * (chunk >> 1) + 8 * (chunk & 1);
  const int k_hi = k_lo + 16;
  *reinterpret_cast<uint4*>(s_b + (k_lo >> 6) * kKBlockBytes + tcgen05::sw128_off(n, (k_lo & 63) >> 3)) = lo;
  *reinterpret_cast<uint4*>(s_b + (k_hi >> 6) * kKBlockBytes + tcgen05::sw128_off(n, (k_hi & 63) >> 3)) = hi;
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 1) decode_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024 - (device::ptx::to_shared(smem_raw) & 1023)) & 1023);
  uint8_t* ring = smem;
  uint8_t* s_b_all = smem + kStages * kTileBytes;
  auto& sh = *reinterpret_cast<Shared*>(smem + kOpBytes);
  int32_t* prefix = reinterpret_cast<int32_t*>(smem + kOpBytes + sizeof(Shared));
  int32_t* npages = prefix + p.groups + 1;
  int32_t* lim = npages + p.groups;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (warp == kTmaWarp) {
    if (lane == 0) {
      device::ptx::prefetch_tensormap(&p.k_map);
      for (int s = 0; s < kStages; ++s) {
        device::ptx::mbar_init(&sh.full[s], 1 + 32 * kCpWarps);
        device::ptx::mbar_init(&sh.empty[s], 4);
      }
      for (int w = 0; w < kGroups; ++w)
        device::ptx::mbar_init(&sh.d_full[w], 1);
      device::ptx::fence_barrier_init();
    }
    __syncwarp();
    tcgen05::alloc(&sh.tmem, kTmemCols);
    tcgen05::relinquish_alloc_permit();
  }
  device::PDLWaitPrimary<kUsePDL>();

  for (int g = tid; g < p.groups; g += kThreads) {
    const int ctx = p.lens[table_row_of(p, g)];
    const int n = ctx > p.min_scored_len ? min((ctx + kPageKeys - 1) / kPageKeys, p.max_pages) : 0;
    npages[g] = n;
    prefix[g + 1] = (n + kTilePages - 1) / kTilePages;
  }
  __syncthreads();
  for (int r = tid; r < p.rows; r += kThreads) {
    const int n = p.lens[r];
    lim[r] = min(n > p.min_scored_len ? min(n, p.width) : 0, npages[r / p.group_rows] * kPageKeys);
  }
  if (warp == 0) {
    int carry = 0;
    for (int base = 0; base < p.groups; base += 32) {
      int v = base + lane < p.groups ? prefix[base + lane + 1] : 0;
#pragma unroll
      for (int o = 1; o < 32; o *= 2) {
        const int up = __shfl_up_sync(kFull, v, o);
        if (lane >= o) v += up;
      }
      if (base + lane < p.groups) prefix[base + lane + 1] = v + carry;
      carry += __shfl_sync(kFull, v, 31);
    }
    if (lane == 0) prefix[0] = 0;
  }
  tcgen05::fence_before_thread_sync();
  __syncthreads();
  tcgen05::fence_after_thread_sync();

  const int total = prefix[p.groups];
  const int t_begin = static_cast<int>(static_cast<int64_t>(total) * blockIdx.x / gridDim.x);
  const int t_end = static_cast<int>(static_cast<int64_t>(total) * (blockIdx.x + 1) / gridDim.x);
  const int n_tiles = t_end - t_begin;

  if (warp == kTmaWarp) {
    int g = n_tiles > 0 ? find_group(prefix, p.groups, t_begin) : 0;
    auto fetch = [&](const int b, bool& ok) -> int {
      const int t = t_begin + 4 * b + (lane >> 3);
      ok = false;
      if (t >= t_end || (lane & 7) >= kTmaPages) return 0;
      while (prefix[g + 1] <= t)
        ++g;
      const int page = (t - prefix[g]) * kTilePages + (lane & 7);
      if (page >= npages[g]) return 0;
      ok = true;
      return p.page_table[static_cast<int64_t>(table_row_of(p, g)) * p.stride_pt + page];
    };
    bool next_ok;
    int next = fetch(0, next_ok);
    for (int b = 0; 4 * b < n_tiles; ++b) {
      const int cur = min(max(next, 0), p.cache_pages - 1);
      const bool cur_ok = next_ok;
      next = fetch(b + 1, next_ok);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int i = 4 * b + j;
        if (i >= n_tiles) break;
        const int s = i % kStages;
        const int pid = __shfl_sync(kFull, cur, 8 * j + (lane & 7));
        const bool ok = __shfl_sync(kFull, cur_ok, 8 * j + (lane & 7)) && lane < 8;
        const unsigned valid = __ballot_sync(kFull, ok);
        device::ptx::mbar_wait_parity(&sh.empty[s], ((i / kStages) & 1) ^ 1);
        if (lane == 0) device::ptx::mbar_arrive_expect_tx(&sh.full[s], __popc(valid) * kPageBytes);
        __syncwarp();
        if (ok) {
          device::ptx::cp_async_bulk_tensor_2d(
              device::ptx::to_shared(ring + s * kTileBytes + lane * kPageBytes),
              &p.k_map,
              0,
              pid * kPageKeys,
              &sh.full[s]);
        }
      }
    }
  } else if (warp >= kCpWarp) {
    const int page_of_warp = kTmaPages + warp - kCpWarp;
    int g = n_tiles > 0 ? find_group(prefix, p.groups, t_begin) : 0;
    auto fetch = [&](const int i0, bool& ok) -> int {
      const int t = t_begin + i0 + lane;
      ok = false;
      if (t >= t_end) return 0;
      while (prefix[g + 1] <= t)
        ++g;
      const int page = (t - prefix[g]) * kTilePages + page_of_warp;
      if (page >= npages[g]) return 0;
      ok = true;
      return p.page_table[static_cast<int64_t>(table_row_of(p, g)) * p.stride_pt + page];
    };
    bool next_ok;
    int next = fetch(0, next_ok);
    const int row_lo = lane >> 3;
    const int chunk = lane & 7;
    for (int i0 = 0; i0 < n_tiles; i0 += 32) {
      const int cur = min(max(next, 0), p.cache_pages - 1);
      const bool cur_ok = next_ok;
      next = fetch(i0 + 32, next_ok);
      for (int j = 0; j < 32 && i0 + j < n_tiles; ++j) {
        const int i = i0 + j;
        const int s = i % kStages;
        const int pid = __shfl_sync(kFull, cur, j);
        const bool ok = __shfl_sync(kFull, cur_ok, j);
        device::ptx::mbar_wait_parity(&sh.empty[s], ((i / kStages) & 1) ^ 1);
        if (ok) {
          const uint8_t* src = p.k + static_cast<int64_t>(pid) * kPageBytes + lane * 16;
          uint8_t* dst = ring + s * kTileBytes + page_of_warp * kPageBytes;
#pragma unroll
          for (int x = 0; x < 4; ++x) {
            const int row = 4 * x + row_lo;
            device::ptx::cp_async_16B(
                device::ptx::to_shared(dst + row * 128 + ((chunk ^ (row & 7)) << 4)), src + x * 512);
          }
        }
        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(device::ptx::to_shared(&sh.full[s]))
                     : "memory");
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
  } else {
    const int w = warp >> 2;
    const int m = (warp & 3) * 32 + lane;
    const bool issuer = (warp & 3) == (w & 3);
    const uint32_t tmem_a = sh.tmem + w * kGroupCols;
    const uint32_t a_tmem = tmem_a + (static_cast<uint32_t>((warp & 3) * 32) << 16);
    const uint32_t d_tmem = a_tmem + 64;
    uint8_t* s_b = s_b_all + w * kBBytes;
    const uint64_t bdesc0 = tcgen05::make_desc_kmajor_sw128(s_b).bits;
    const uint32_t idesc = tcgen05::make_instr_desc_f16(128, kN, false, false);
    const int qn = m >> 3;
    const int qchunk = m & 7;
    const float inv_scale = __fdividef(1.f, p.scale);
    const int key_row = m & 15;
    const uint8_t* key_src = ring + (m >> 4) * kPageBytes + key_row * 128;
    int g = w < n_tiles ? find_group(prefix, p.groups, t_begin + w) : 0;
    int g_begin = 0, g_end = -1, row0 = 0;
    int lim_r[4];
    uint4 qv = load_q_chunk(p, g, qn, qchunk);
    uint4 qv_next = load_q_chunk(p, g + 1, qn, qchunk);
    for (int i = w; i < n_tiles; i += kGroups) {
      const int t = t_begin + i;
      if (t >= g_end) {
        if (g_end >= 0) {
          const int g_prev = g;
          while (prefix[g + 1] <= t)
            ++g;
          qv = g == g_prev + 1 ? qv_next : load_q_chunk(p, g, qn, qchunk);
          qv_next = load_q_chunk(p, g + 1, qn, qchunk);
        }
        g_begin = prefix[g];
        g_end = prefix[g + 1];
        row0 = g * p.group_rows;
#pragma unroll
        for (int r = 0; r < 4; ++r)
          lim_r[r] = r < p.group_rows && row0 + r < p.rows ? lim[row0 + r] : 0;
        store_b_chunk(s_b, qv, qn, qchunk);
        device::ptx::fence_proxy_async_shared();
      }
      const int s = i % kStages;
      device::ptx::mbar_wait_parity(&sh.full[s], (i / kStages) & 1);
      const uint8_t* src = key_src + s * kTileBytes;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        uint4 raw[4];
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) {
          raw[c4] = *reinterpret_cast<const uint4*>(src + (((4 * half + c4) ^ (key_row & 7)) << 4));
        }
        uint32_t a[32];
#pragma unroll
        for (int c4 = 0; c4 < 4; ++c4) {
          const uint32_t x[4] = {raw[c4].x, raw[c4].y, raw[c4].z, raw[c4].w};
          const int col = 16 * (c4 >> 1) + 4 * (c4 & 1);
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            a[col + e] = f16x2_of_e4m3x2(x[e]);
            a[col + 8 + e] = f16x2_of_e4m3x2(x[e] >> 16);
          }
        }
        if (half == 1) {
          __syncwarp();
          if (lane == 0) device::ptx::mbar_arrive(&sh.empty[s]);
        }
        tcgen05::st_32x32b<32>(a_tmem + 32 * half, a);
      }
      tcgen05::wait_st();
      tcgen05::fence_before_thread_sync();
      device::ptx::bar_sync(1 + w, 128);
      if (issuer) {
        tcgen05::fence_after_thread_sync();
        if (lane == 0) {
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            tcgen05::mma_f16_ts(
                tmem_a + 64 + (u >> 1) * kN,
                tmem_a + u * 8,
                bdesc0 + (((u >> 2) * kKBlockBytes + (u & 3) * 32) >> 4),
                idesc,
                u & 1);
          }
          tcgen05::commit(&sh.d_full[w]);
        }
        __syncwarp();
      }
      device::ptx::mbar_wait_parity(&sh.d_full[w], (i / kGroups) & 1);
      tcgen05::fence_after_thread_sync();
      float acc[16];
      {
        uint32_t d[32];
        tcgen05::ld_32x32b<32>(d_tmem, d);
        tcgen05::wait_ld();
#pragma unroll
        for (int n = 0; n < 16; ++n)
          acc[n] = __fadd_rn(__fadd_rn(0.f, __uint_as_float(d[n])), __uint_as_float(d[16 + n]));
        tcgen05::ld_32x32b<32>(d_tmem + 32, d);
        tcgen05::wait_ld();
#pragma unroll
        for (int n = 0; n < 16; ++n)
          acc[n] = __fadd_rn(__fadd_rn(acc[n], __uint_as_float(d[n])), __uint_as_float(d[16 + n]));
      }
      const int pos = (t - g_begin) * kTileKeys + m;
      float* out = p.logits + static_cast<int64_t>(row0) * p.stride_logits + pos;
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const float v = (fmaxf(acc[4 * r], 0.f) + fmaxf(acc[4 * r + 1], 0.f)) +
                        (fmaxf(acc[4 * r + 2], 0.f) + fmaxf(acc[4 * r + 3], 0.f));
        if (pos < lim_r[r]) out[r * p.stride_logits] = v * inv_scale;
      }
    }
  }

  tcgen05::fence_before_thread_sync();
  __syncthreads();
  tcgen05::fence_after_thread_sync();
  if (warp == kTmaWarp) tcgen05::dealloc(sh.tmem, kTmemCols);
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_index_score_sm100

namespace qsa_decode_topk {

constexpr int kTopK = 512;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kBins = 256;
constexpr int kSampleKeys = 1024;
constexpr int kMaxSamples = 4096;
constexpr int kCap = 2048;
constexpr int kSampleLoads = 16;
constexpr int kVecLoads = 6;
constexpr unsigned kFull = 0xffffffffu;

struct Params {
  const float* logits;  // [rows, width]
  const int32_t* lens;  // [rows]
  int32_t* out;         // [rows, 512]
  int64_t stride_logits;
  int32_t width;
};

struct Shared {
  uint32_t key[kMaxSamples];  // the sample, then the candidates' order keys
  int32_t idx[kCap];
  int32_t hist[kWarps][kBins];
  int32_t warp_sum[kWarps];
  uint32_t all;
  uint32_t any;
  int32_t valid;
  int32_t count;
  int32_t digit;
  int32_t above;
  int32_t equal;
};

constexpr int kSmemBytes = static_cast<int>(sizeof(Shared));

SGL_DEVICE uint32_t order_key(const float x) {
  const uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// CTA-wide k-th largest valid key of [0, m): k becomes the copies of it to take, s.equal its copies. Digits start at
// the highest differing bit; after `rounds` 8-bit digits it returns the lower edge of the k-th largest's bin.
template <class KeyAt>
SGL_DEVICE uint32_t kth_largest(Shared& s, const KeyAt& key_at, const int m, int& k, int rounds = 4) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __syncthreads();
  if (tid == 0) {
    s.all = 0xffffffffu;
    s.any = 0;
    s.valid = 0;
  }
  __syncthreads();
  uint32_t all = 0xffffffffu, any = 0;
  int valid = 0;
  for (int i = tid; i < m; i += kThreads) {
    uint32_t key;
    if (key_at(i, key)) {
      all &= key;
      any |= key;
      ++valid;
    }
  }
  all = __reduce_and_sync(kFull, all);
  any = __reduce_or_sync(kFull, any);
  valid = __reduce_add_sync(kFull, valid);
  if (lane == 0) {
    atomicAnd(&s.all, all);
    atomicOr(&s.any, any);
    atomicAdd(&s.valid, valid);
  }
  __syncthreads();
  const uint32_t diff = s.all ^ s.any;
  if (diff == 0) {
    s.equal = s.valid;
    return s.any;
  }
  int hi = 31 - __clz(diff);
  uint32_t mask = hi == 31 ? 0u : ~((2u << hi) - 1);
  uint32_t prefix = s.any & mask;
#pragma unroll 1
  for (; hi >= 0 && rounds > 0; --rounds) {
    const int lo = max(hi - 7, 0);
    const uint32_t digit_mask = (2u << (hi - lo)) - 1;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
      s.hist[w][tid] = 0;
    __syncthreads();
    for (int i = tid; i < m; i += kThreads) {
      uint32_t key;
      if (key_at(i, key) && (key & mask) == prefix) atomicAdd(&s.hist[warp][(key >> lo) & digit_mask], 1);
    }
    __syncthreads();
    int v = 0;
#pragma unroll
    for (int w = 0; w < kWarps; ++w)
      v += s.hist[w][tid];
    int suffix = v;
#pragma unroll
    for (int o = 1; o < 32; o *= 2) {
      const int up = __shfl_down_sync(kFull, suffix, o);
      if (lane + o < 32) suffix += up;
    }
    if (lane == 0) s.warp_sum[warp] = suffix;
    __syncthreads();
    for (int w = warp + 1; w < kWarps; ++w)
      suffix += s.warp_sum[w];
    if (suffix >= k && suffix - v < k) {
      s.digit = tid;
      s.above = suffix - v;
      s.equal = v;
    }
    __syncthreads();
    prefix |= static_cast<uint32_t>(s.digit) << lo;
    mask |= digit_mask << lo;
    k -= s.above;
    hi = lo - 1;
  }
  return prefix;
}

// appends i to the output when take(i): positions by warp-aggregated reservation, order unspecified
template <class Take>
SGL_DEVICE void emit(Shared& s, int32_t* out, const Take& take, const int m) {
  const int lane = threadIdx.x & 31;
  if (threadIdx.x == 0) s.count = 0;
  __syncthreads();
  for (int base = 0; base < m; base += kThreads) {
    const int i = base + threadIdx.x;
    int32_t id = 0;
    const bool on = i < m && take(i, id);
    const unsigned ballot = __ballot_sync(kFull, on);
    int pos = 0;
    if (lane == 0 && ballot != 0) pos = atomicAdd(&s.count, __popc(ballot));
    pos = __shfl_sync(kFull, pos, 0) + __popc(ballot & ((1u << lane) - 1));
    if (on) out[pos] = id;
  }
}

// the 512 largest of a row given as order keys key_at(i) with ids id_at(i) for i < m, ties by the lower id
template <class KeyAt, class IdAt>
SGL_DEVICE void pick(Shared& s, int32_t* out, const KeyAt& key_at, const IdAt& id_at, const int m) {
  int need = kTopK;
  const uint32_t cut = kth_largest(s, key_at, m, need);
  const bool all_ties = need == s.equal;
  uint32_t tie_cut = 0;
  if (!all_ties) {
    int one = need;
    const auto tie_at = [&](const int i, uint32_t& key) {
      uint32_t k;
      key_at(i, k);
      key = ~static_cast<uint32_t>(id_at(i));
      return k == cut;
    };
    tie_cut = kth_largest(s, tie_at, m, one);
  }
  const auto take = [&](const int i, int32_t& id) {
    uint32_t k;
    key_at(i, k);
    id = id_at(i);
    return k > cut || (k == cut && (all_ties || ~static_cast<uint32_t>(id) >= tie_cut));
  };
  emit(s, out, take, m);
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kThreads, 4) select_kernel(const __grid_constant__ Params p) {
  extern __shared__ __align__(16) uint8_t smem[];
  Shared& s = *reinterpret_cast<Shared*>(smem);
  device::PDLWaitPrimary<kUsePDL>();
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int row = blockIdx.x;
  int32_t* const out = p.out + static_cast<int64_t>(row) * kTopK;
  const int n = min(p.lens[row], p.width);
  if (n <= kTopK) {
    for (int i = tid; i < kTopK; i += kThreads)
      out[i] = i < n ? i : -1;
    device::PDLTriggerSecondary<kUsePDL>();
    return;
  }
  const float* const x = p.logits + static_cast<int64_t>(row) * p.stride_logits;
  const int every = n > 8 * kMaxSamples ? 16 : 8;
  const int samples = (n + every - 1) / every;
  uint32_t floor_key = 0;
  if (n > kCap && samples <= kMaxSamples) {
    for (int base = 0; base < samples; base += kThreads * kSampleLoads) {
      float v[kSampleLoads];
#pragma unroll
      for (int u = 0; u < kSampleLoads; ++u) {
        const int j = base + u * kThreads + tid;
        v[u] = j < samples ? x[j * every] : 0.f;
      }
#pragma unroll
      for (int u = 0; u < kSampleLoads; ++u) {
        const int j = base + u * kThreads + tid;
        if (j < samples) s.key[j] = order_key(v[u]);
      }
    }
    __syncthreads();
    int rank = kSampleKeys / every;
    floor_key = kth_largest(
        s,
        [&](const int i, uint32_t& key) {
          key = s.key[i];
          return true;
        },
        samples,
        rank,
        2);
  }
  const uint32_t floor_bits = (floor_key & 0x80000000u) ? (floor_key & 0x7fffffffu) : ~floor_key;
  const float floor_value = floor_key == 0 ? -__int_as_float(0x7f800000) : __uint_as_float(floor_bits);
  if (tid == 0) s.count = 0;
  __syncthreads();
  if (n <= kCap || samples <= kMaxSamples) {
    const int vecs = (n + 3) / 4;
    for (int base = 0; base < vecs; base += kThreads * kVecLoads) {
      float4 v[kVecLoads];
#pragma unroll
      for (int u = 0; u < kVecLoads; ++u) {
        const int vi = base + u * kThreads + tid;
        v[u] = vi < vecs ? *reinterpret_cast<const float4*>(x + 4 * vi) : make_float4(0.f, 0.f, 0.f, 0.f);
      }
      uint64_t hit = 0;
#pragma unroll
      for (int u = 0; u < kVecLoads; ++u) {
        const int i0 = 4 * (base + u * kThreads + tid);
        const float e[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          if (i0 + c < n && e[c] >= floor_value) hit |= 1ull << (4 * u + c);
        }
      }
      const int found = __popcll(hit);
      int upto = found;
#pragma unroll
      for (int o = 1; o < 32; o *= 2) {
        const int up = __shfl_up_sync(kFull, upto, o);
        if (lane >= o) upto += up;
      }
      int pos = 0;
      if (lane == 31 && upto != 0) pos = atomicAdd(&s.count, upto);
      pos = __shfl_sync(kFull, pos, 31) + upto - found;
      if (hit != 0) {
#pragma unroll
        for (int u = 0; u < kVecLoads; ++u) {
          const int i0 = 4 * (base + u * kThreads + tid);
          const float e[4] = {v[u].x, v[u].y, v[u].z, v[u].w};
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            if ((hit >> (4 * u + c)) & 1) {
              if (pos < kCap) {
                s.key[pos] = order_key(e[c]);
                s.idx[pos] = i0 + c;
              }
              ++pos;
            }
          }
        }
      }
    }
  } else if (tid == 0) {
    s.count = kCap + 1;
  }
  __syncthreads();
  const int found = s.count;
  if (found >= kTopK && found <= kCap) {
    pick(
        s,
        out,
        [&](const int i, uint32_t& key) {
          key = s.key[i];
          return true;
        },
        [&](const int i) { return s.idx[i]; },
        found);
  } else {
    pick(
        s,
        out,
        [&](const int i, uint32_t& key) {
          key = order_key(x[i]);
          return true;
        },
        [](const int i) { return i; },
        n);
  }
  device::PDLTriggerSecondary<kUsePDL>();
}

}  // namespace qsa_decode_topk

template <bool kUsePDL>
struct QsaIndexScoreDecode {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView lens,
      const tvm::ffi::TensorView logits,
      double scale,
      int64_t group_rows,
      int64_t min_scored_len,
      int64_t splits) {
    using namespace host;
    using namespace qsa_index_score;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto P_ = SymbolicSize{"cache_pages"};
    auto MP_ = SymbolicSize{"max_pages"};
    auto W_ = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, kD, 1}).verify(q);
    TensorMatcher({P_, kPageBytes}).with_dtype<uint8_t>().with_device(device).with_strides({kPageBytes, 1}).verify(k);
    TensorMatcher({R_, MP_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(page_table);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(lens);
    TensorMatcher({R_, W_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(logits);
    const int64_t rows = R_.unwrap();
    RuntimeCheck(HQ_.unwrap() >= kHeads, "q needs at least 4 heads");
    RuntimeCheck(group_rows >= 1 && group_rows <= kMaxGroupRows, "group_rows must be in [1, 4]");
    RuntimeCheck(splits >= 1, "splits must be positive");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 4 == 0 && q.stride(0) % 4 == 0, "q rows must be 4-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0, "key pages must be 16-byte aligned");
    if (rows == 0 || P_.unwrap() == 0) return;
    const auto params = Params{
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .lens = static_cast<const int32_t*>(lens.data_ptr()),
        .logits = static_cast<float*>(logits.data_ptr()),
        .stride_q = q.stride(0),
        .stride_pt = page_table.stride(0),
        .stride_logits = logits.stride(0),
        .scale = static_cast<float>(scale),
        .rows = static_cast<int32_t>(rows),
        .group_rows = static_cast<int32_t>(group_rows),
        .cache_pages = static_cast<int32_t>(P_.unwrap()),
        .max_pages = static_cast<int32_t>(MP_.unwrap()),
        .width = static_cast<int32_t>(W_.unwrap()),
        .min_scored_len = static_cast<int32_t>(min_scored_len),
    };
    const dim3 grid(static_cast<uint32_t>((rows + group_rows - 1) / group_rows), static_cast<uint32_t>(splits));
    LaunchKernel(grid, kThreads, device.unwrap(), static_cast<std::size_t>(kWarps * kTileBytes))
        .enable_pdl(kUsePDL)(decode_kernel<kUsePDL>, params);
  }
};

template <bool kUsePDL>
struct QsaIndexScoreDecodeSm100 {
  static void
  run(const tvm::ffi::TensorView q,
      const tvm::ffi::TensorView k,
      const tvm::ffi::TensorView page_table,
      const tvm::ffi::TensorView lens,
      const tvm::ffi::TensorView logits,
      double scale,
      int64_t group_rows,
      int64_t min_scored_len) {
    using namespace host;
    using namespace qsa_index_score_sm100;
    auto R_ = SymbolicSize{"rows"};
    auto HQ_ = SymbolicSize{"q_heads"};
    auto P_ = SymbolicSize{"cache_pages"};
    auto MP_ = SymbolicSize{"max_pages"};
    auto W_ = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, HQ_, kD}).with_dtype<uint8_t>().with_device(device).with_strides({-1, kD, 1}).verify(q);
    TensorMatcher({P_, kPageBytes}).with_dtype<uint8_t>().with_device(device).with_strides({kPageBytes, 1}).verify(k);
    TensorMatcher({R_, MP_}).with_dtype<int32_t>().with_device(device).with_strides({-1, 1}).verify(page_table);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(lens);
    TensorMatcher({R_, W_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(logits);
    const int64_t rows = R_.unwrap();
    RuntimeCheck(HQ_.unwrap() >= 4, "q needs at least 4 heads");
    RuntimeCheck(group_rows >= 1 && group_rows <= 4, "group_rows must be in [1, 4]");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0 && q.stride(0) % 16 == 0, "q rows must be 16-byte aligned");
    RuntimeCheck(reinterpret_cast<uintptr_t>(k.data_ptr()) % 16 == 0, "key pages must be 16-byte aligned");
    if (rows == 0 || P_.unwrap() == 0) return;
    const int64_t groups = (rows + group_rows - 1) / group_rows;
    const std::size_t smem = 1024 + kOpBytes + sizeof(Shared) + 4 * (2 * groups + 1 + rows);
    const DLDevice dev = device.unwrap();
    RuntimeCheck(smem <= runtime::get_max_smem_per_block(dev.device_id), "too many rows for the shared memory");
    const auto params = Params{
        .k_map = tma::make_map(
            k.data_ptr(),
            CU_TENSOR_MAP_DATA_TYPE_UINT8,
            {static_cast<cuuint64_t>(kD), static_cast<cuuint64_t>(P_.unwrap() * kPageKeys)},
            {static_cast<cuuint64_t>(kD)},
            {static_cast<cuuint32_t>(kD), static_cast<cuuint32_t>(kPageKeys)},
            {.swizzle = CU_TENSOR_MAP_SWIZZLE_128B}),
        .k = static_cast<const uint8_t*>(k.data_ptr()),
        .q = static_cast<const uint8_t*>(q.data_ptr()),
        .page_table = static_cast<const int32_t*>(page_table.data_ptr()),
        .lens = static_cast<const int32_t*>(lens.data_ptr()),
        .logits = static_cast<float*>(logits.data_ptr()),
        .stride_q = q.stride(0),
        .stride_pt = page_table.stride(0),
        .stride_logits = logits.stride(0),
        .scale = static_cast<float>(scale),
        .rows = static_cast<int32_t>(rows),
        .group_rows = static_cast<int32_t>(group_rows),
        .groups = static_cast<int32_t>(groups),
        .cache_pages = static_cast<int32_t>(P_.unwrap()),
        .max_pages = static_cast<int32_t>(MP_.unwrap()),
        .width = static_cast<int32_t>(W_.unwrap()),
        .min_scored_len = static_cast<int32_t>(min_scored_len),
    };
    constexpr auto kernel = decode_kernel<kUsePDL>;
    runtime::set_max_dynamic_smem(kernel, smem);
    LaunchKernel(runtime::get_sm_count(dev.device_id), kThreads, dev, smem).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <bool kUsePDL>
struct QsaDecodeTopk {
  static void run(const tvm::ffi::TensorView logits, const tvm::ffi::TensorView lens, const tvm::ffi::TensorView out) {
    using namespace host;
    using namespace qsa_decode_topk;
    auto R_ = SymbolicSize{"rows"};
    auto W_ = SymbolicSize{"width"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({R_, W_}).with_dtype<fp32_t>().with_device(device).with_strides({-1, 1}).verify(logits);
    TensorMatcher({R_}).with_dtype<int32_t>().with_device(device).with_strides({1}).verify(lens);
    TensorMatcher({R_, kTopK}).with_dtype<int32_t>().with_device(device).with_strides({kTopK, 1}).verify(out);
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(logits.data_ptr()) % 16 == 0 && logits.stride(0) % 4 == 0,
        "logits rows must be 16-byte aligned");
    RuntimeCheck(W_.unwrap() % 4 == 0, "logits width must be a multiple of 4");
    const int64_t rows = R_.unwrap();
    if (rows == 0) return;
    const auto params = Params{
        .logits = static_cast<const float*>(logits.data_ptr()),
        .lens = static_cast<const int32_t*>(lens.data_ptr()),
        .out = static_cast<int32_t*>(out.data_ptr()),
        .stride_logits = logits.stride(0),
        .width = static_cast<int32_t>(W_.unwrap()),
    };
    constexpr auto kernel = select_kernel<kUsePDL>;
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes));
    LaunchKernel(static_cast<uint32_t>(rows), kThreads, device.unwrap(), kSmemBytes)
        .enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace sglang
