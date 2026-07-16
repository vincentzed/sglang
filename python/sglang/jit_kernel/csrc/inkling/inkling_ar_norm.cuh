// Fused decode/verify {all-reduce -> residual-add + RMSNorm} -- the v5 push
// one-shot all-reduce (inkling_all_reduce.cuh) with the EPILOGUE SEAM filled in
// by the fused-add RMSNorm. This is inkling_ar_fused_decode.cuh minus the
// short-conv phase: without cross-token conv taps every token row is
// independent, so ONE kernel (block per token, per-block barrier) covers both
// decode and EAGLE target-verify shapes.
//
// Replaces TWO kernels (AR + fused_add_rmsnorm) and their intermediate HBM
// round trip with ONE launch. Layout: ONE BLOCK PER TOKEN (decode rows are few
// and the RMSNorm needs a per-row cross-hidden reduction), VPT 16B vecs
// (8 channels each) per thread. Phases:
//
//   0. prefetch: the gamma (norm weight) row loads FIRST -- it does not depend
//      on the producer kernel's output, so its HBM latency hides under the
//      barrier below. (This -- not PDL -- is the latency mechanism; the PDL
//      launch attribute only pipelines the launch tail.)
//   1. push:    griddepcontrol.wait, then multicast-store this rank's partial
//               row into staging slot (rank*T + t)*D; issue the residual load.
//   2. barrier: per-block peer handshake (block t <-> peers' block t).
//   3. reduce:  fp32 sum of the kNumGPU staged shards; round to bf16 `xb`
//               (bit-identical to what the unfused v5 AR would have stored).
//   4. norm:    fused_add_rmsnorm semantics on (xb, residual). The
//               ROUND_SUM_TO_BF16 template arg selects where the residual sum
//               is rounded: true mirrors the vectorized vLLM-style kernel
//               (sum rounded to bf16 before squaring and scaling -- what
//               sgl_kernel.fused_add_rmsnorm's 16B-vector path does), false
//               keeps the sum in fp32 through the variance (the scalar
//               fallback path). Calibrated against the installed sgl_kernel in
//               the bit-identity unit test.
//
// Staging reuse is caller-managed (A/B rotation shared with v5 -- this kernel
// IS a v5 AR occupying one rotation slot; same reuse-distance-2 invariant).
// bf16-only.

#include <sgl_kernel/tensor.h>  // For TensorMatcher, SymbolicSize, SymbolicDevice
#include <sgl_kernel/utils.h>   // For RuntimeCheck, div_ceil

#include <sgl_kernel/runtime.cuh>  // For get_blocks_per_sm
#include <sgl_kernel/utils.cuh>    // For LaunchKernel, kWarpThreads

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include "inkling_ar_barrier.cuh"
#include <bit>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

namespace {

constexpr uint32_t kNormVecElems = 8;  // bf16x8 = 16 B

struct ArAddRmsNormParams {
  // AR
  const void* __restrict__ in;     // [T, D] partial sums (LOCAL tensor)
  void* __restrict__ mc_stage;     // multicast staging base (>= kNumGPU*T*D elems)
  const void* __restrict__ stage;  // this GPU's local view of the staging base
  void* const* __restrict__ flag_ptrs;
  uint32_t* __restrict__ state;
  // norm
  const void* __restrict__ residual_in;  // [T, D]
  void* __restrict__ residual_out;       // [T, D]
  void* __restrict__ hs_out;             // [T, D]
  const void* __restrict__ norm_weight;  // [D]
  float eps;
  // strides (elements)
  int64_t in_stride_t;
  int64_t res_in_stride_t;
  int64_t res_out_stride_t;
  int64_t hs_stride_t;
  uint32_t rank;
  uint32_t T;
  uint32_t D;
};

template <typename DType, uint32_t kNumGPU, bool ROUND_SUM_TO_BF16, int VPT>
__global__ __launch_bounds__(1024, 1) void inkling_ar_add_rmsnorm_kernel(
    const __grid_constant__ ArAddRmsNormParams p) {
  static_assert(std::is_same_v<DType, __nv_bfloat16>, "multimem push path is bf16-only");
  const uint32_t t = blockIdx.x;
  const uint32_t vecs = p.D / kNormVecElems;

  uint32_t c0[VPT];
  bool act[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    const uint32_t v = threadIdx.x + i * blockDim.x;
    act[i] = v < vecs;
    c0[i] = (act[i] ? v : 0) * kNormVecElems;  // clamp: inactive lanes never store
  }

  // ---- 0. prefetch gamma (independent of the producer's output) ----
  const auto* gw = static_cast<const __nv_bfloat16*>(p.norm_weight);
  uint4 g_raw[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (act[i]) g_raw[i] = *reinterpret_cast<const uint4*>(gw + c0[i]);
  }

  // ---- 1. push: wait for the producer's output (PDL; no-op without a PDL
  // launch), multicast-store this rank's partial row, and issue the residual
  // load (it lands under the barrier). ----
  asm volatile("griddepcontrol.wait;" ::: "memory");
  const auto* in_row = static_cast<const __nv_bfloat16*>(p.in) + t * p.in_stride_t;
  auto* slot = static_cast<__nv_bfloat16*>(p.mc_stage) + (static_cast<uint64_t>(p.rank) * p.T + t) * p.D;
  uint4 res_raw[VPT];
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    const uint4 d = *reinterpret_cast<const uint4*>(in_row + c0[i]);
    asm volatile("multimem.st.relaxed.sys.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(slot + c0[i]),
                 "r"(d.x),
                 "r"(d.y),
                 "r"(d.z),
                 "r"(d.w)
                 : "memory");
    res_raw[i] = *reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(p.residual_in) + t * p.res_in_stride_t + c0[i]);
  }

  // ---- 2. per-block barrier: all ranks' row-t pushes have landed locally ----
  inkling_ar::block_system_barrier<kNumGPU>(p.state, p.flag_ptrs, p.rank);
  // Inactive lanes must NOT exit: they participate in the norm's __syncthreads
  // and full-mask warp shuffles below (sumsq contribution 0).

  float r[VPT][kNormVecElems];
  float sumsq = 0.0f;
  const auto* stage = static_cast<const __nv_bfloat16*>(p.stage);
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    // ---- 3. reduce: fp32 sum of the kNumGPU staged shards; round to bf16 ----
    float xf[kNormVecElems];
#pragma unroll
    for (int j = 0; j < static_cast<int>(kNormVecElems); ++j)
      xf[j] = 0.0f;
#pragma unroll
    for (uint32_t rr = 0; rr < kNumGPU; ++rr) {
      const uint4 d = *reinterpret_cast<const uint4*>(stage + (static_cast<uint64_t>(rr) * p.T + t) * p.D + c0[i]);
      const auto* h2 = reinterpret_cast<const __nv_bfloat162*>(&d);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h2[j]);
        xf[2 * j] += f.x;
        xf[2 * j + 1] += f.y;
      }
    }

    // ---- 4a. residual add (fused_add_rmsnorm semantics). Round the reduced
    // row to bf16 exactly as the unfused AR's store would, so the sum sees
    // the same bits the unfused chain sees. ----
#pragma unroll
    for (int j = 0; j < static_cast<int>(kNormVecElems); ++j) {
      const float xb = __bfloat162float(__float2bfloat16_rn(xf[j]));
      float rj = xb + __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&res_raw[i])[j]);
      if constexpr (ROUND_SUM_TO_BF16) {
        rj = __bfloat162float(__float2bfloat16_rn(rj));
      }
      r[i][j] = rj;
      sumsq += rj * rj;
    }
  }

  // ---- 4b. block reduction of sumsq (warp shuffle + one smem slot/warp) ----
  __shared__ float s_warp[32];
  __shared__ float s_inv;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t warp = threadIdx.x >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    sumsq += __shfl_down_sync(~0u, sumsq, off);
  if (lane == 0) s_warp[warp] = sumsq;
  __syncthreads();
  if (warp == 0) {
    const uint32_t nwarps = (blockDim.x + 31u) >> 5;
    float total = (lane < nwarps && lane < 32u) ? s_warp[lane] : 0.0f;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      total += __shfl_down_sync(~0u, total, off);
    if (lane == 0) s_inv = rsqrtf(total / static_cast<float>(p.D) + p.eps);
  }
  __syncthreads();
  const float inv = s_inv;

  auto* res_out = static_cast<__nv_bfloat16*>(p.residual_out) + t * p.res_out_stride_t;
  auto* hs_out = static_cast<__nv_bfloat16*>(p.hs_out) + t * p.hs_stride_t;
#pragma unroll
  for (int i = 0; i < VPT; ++i) {
    if (!act[i]) continue;
    __nv_bfloat162 ro[4], ho[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float g0 = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&g_raw[i])[2 * j]);
      const float g1 = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&g_raw[i])[2 * j + 1]);
      ro[j] = __floats2bfloat162_rn(r[i][2 * j], r[i][2 * j + 1]);
      if constexpr (ROUND_SUM_TO_BF16) {
        // Vectorized vLLM-style path: (bf16 sum) * inv rounded to bf16, then
        // * gamma rounded to bf16 (each op converts through fp32).
        const float n0 = __bfloat162float(__float2bfloat16_rn(r[i][2 * j] * inv));
        const float n1 = __bfloat162float(__float2bfloat16_rn(r[i][2 * j + 1] * inv));
        ho[j] = __floats2bfloat162_rn(n0 * g0, n1 * g1);
      } else {
        ho[j] = __floats2bfloat162_rn(r[i][2 * j] * inv * g0, r[i][2 * j + 1] * inv * g1);
      }
    }
    *reinterpret_cast<uint4*>(res_out + c0[i]) = *reinterpret_cast<const uint4*>(ro);
    *reinterpret_cast<uint4*>(hs_out + c0[i]) = *reinterpret_cast<const uint4*>(ho);
  }
}

template <typename DType, uint32_t kNumGPU, bool ROUND_SUM_TO_BF16>
struct ArAddRmsNormKernel {
  template <int VPT>
  static void launch(const ArAddRmsNormParams& params, uint32_t t_num, uint32_t vecs, DLDevice dev, bool pdl) {
    using namespace host;
    const uint32_t block = min(1024u, div_ceil(div_ceil(vecs, VPT), 32u) * 32u);
    constexpr auto kernel = inkling_ar_add_rmsnorm_kernel<DType, kNumGPU, ROUND_SUM_TO_BF16, VPT>;
    LaunchKernel(dim3{t_num}, dim3{block}, dev).enable_pdl(pdl)(kernel, params);
  }

  static void
  run(tvm::ffi::TensorView in,
      tvm::ffi::TensorView residual_in,
      tvm::ffi::TensorView residual_out,
      tvm::ffi::TensorView hs_out,
      tvm::ffi::TensorView norm_weight,
      double eps,
      int64_t mc_stage_ptr,
      int64_t local_stage_ptr,
      int64_t flag_ptrs_dev,
      int64_t state_ptr,
      int64_t rank,
      int64_t enable_pdl,
      int64_t vecs_per_thread) {
    using namespace host;
    auto T = SymbolicSize{"T"};
    auto D = SymbolicSize{"D"};
    auto dev = SymbolicDevice{};
    dev.set_options<kDLCUDA>();

    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(in);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(residual_in);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(residual_out);
    TensorMatcher({T, D}).with_strides({-1, 1}).with_dtype<DType>().with_device(dev).verify(hs_out);
    TensorMatcher({D}).with_dtype<DType>().with_device(dev).verify(norm_weight);
    const uint32_t t_num = static_cast<uint32_t>(T.unwrap());
    const uint32_t d_num = static_cast<uint32_t>(D.unwrap());
    const uint32_t vecs = d_num / kNormVecElems;
    RuntimeCheck(
        t_num >= 1 && t_num <= inkling_ar::kMaxBarrierBlocks,
        "T must be in [1, kMaxBarrierBlocks] (one barrier slot per token)");
    RuntimeCheck(d_num % kNormVecElems == 0, "D must be a multiple of 8");
    RuntimeCheck(mc_stage_ptr % 16 == 0, "mc_stage_ptr not 16B aligned");
    RuntimeCheck(local_stage_ptr != 0 && local_stage_ptr % 16 == 0, "bad local_stage_ptr");
    RuntimeCheck(flag_ptrs_dev != 0, "flag_ptrs_dev is null");
    RuntimeCheck(state_ptr != 0, "state_ptr is null");
    RuntimeCheck(rank >= 0 && rank < kNumGPU, "rank out of range");
    RuntimeCheck(in.stride(0) % kNormVecElems == 0, "in row stride must keep 16B alignment");
    RuntimeCheck(std::bit_cast<intptr_t>(in.data_ptr()) % 16 == 0, "in not 16B aligned");

    const auto params = ArAddRmsNormParams{
        .in = in.data_ptr(),
        .mc_stage = reinterpret_cast<void*>(mc_stage_ptr),
        .stage = reinterpret_cast<const void*>(local_stage_ptr),
        .flag_ptrs = reinterpret_cast<void* const*>(flag_ptrs_dev),
        .state = reinterpret_cast<uint32_t*>(state_ptr),
        .residual_in = residual_in.data_ptr(),
        .residual_out = residual_out.data_ptr(),
        .hs_out = hs_out.data_ptr(),
        .norm_weight = norm_weight.data_ptr(),
        .eps = static_cast<float>(eps),
        .in_stride_t = in.stride(0),
        .res_in_stride_t = residual_in.stride(0),
        .res_out_stride_t = residual_out.stride(0),
        .hs_stride_t = hs_out.stride(0),
        .rank = static_cast<uint32_t>(rank),
        .T = t_num,
        .D = d_num,
    };

    // vecs_per_thread (VPT) is the tuned knob; 0 -> 1. Each VPT must still fit
    // one block (div_ceil(vecs, VPT) <= 1024).
    const int vpt = vecs_per_thread > 0 ? static_cast<int>(vecs_per_thread) : 1;
    const bool pdl = enable_pdl != 0;
    switch (vpt) {
      case 1:
        RuntimeCheck(vecs <= 1024, "D/8 must fit one block at VPT=1");
        launch<1>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 2:
        launch<2>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 3:
        launch<3>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 4:
        launch<4>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      case 6:
        launch<6>(params, t_num, vecs, dev.unwrap(), pdl);
        break;
      default:
        RuntimeCheck(false, "unsupported vecs_per_thread (use 1/2/3/4/6)");
    }
  }
};

}  // namespace
