// QSA (Qwen4-Exp) decode / target-verify index scoring over the paged FP8 E4M3 compressed-key cache:
// logits[row, n] = (sum over the 4 index heads of relu(q[row, head] . key[n])) / scale for n < lens[row].
// The decode top-512 then picks, per row, the 512 largest logits of [0, lens[row]), ties at the cut broken by the
// lower key index; rows with at most 512 keys get 0..len-1, then -1.
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

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

// the CTA-wide k-th largest of the keys key_at(i) of [0, m) that are valid: returns it, leaves k as the number of
// its copies to take and s.equal as the number of its copies. Digits start at the highest bit where keys differ;
// after `rounds` 8-bit digits it stops and returns the lower edge of the k-th largest's bin instead.
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
