// QSA (Qwen4-Exp) decode / target-verify index scoring over the paged FP8 E4M3 compressed-key cache:
// logits[row, n] = (sum over the 4 index heads of relu(q[row, head] . key[n])) / scale for n < lens[row].
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

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

}  // namespace sglang
