// GDN ReplaySSM commit of the compact bf16 ring on SM100: rows flagged in is_flush fold their committed records
// into the bf16 checkpoint, checkpoint[v, k] = exp(G) checkpoint[v, k] + sum over the write_pos records from
// cache_base of exp(G - G_i) d_i[v] k_i[k] (G_i the inclusive gate prefix, G its total, d and k as hi + lo parts).
// One CTA per (key head, row, layer); per value head one tcgen05 product per 64 state columns with A row v =
// [u_hi | u_lo | u_hi | u_lo] and B row k = [k_hi | k_hi | k_lo | k_lo] over the 16 ring slots, so K = 64 sums the
// four hi/lo products; u = exp(G - G_i) (d_hi + d_lo).
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <sgl_kernel/ptx/async.cuh>
#include <sgl_kernel/ptx/barrier.cuh>
#include <sgl_kernel/ptx/tcgen05.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cstdint>

namespace sglang {

namespace gdn_replayssm_fold {

using tcgen05 = device::ptx::TCGen05<1>;

constexpr int kD = 128;
constexpr int kL = 16;
constexpr int kWorkers = 128;
constexpr int kTmemCols = 64;
constexpr int kLayersPerCta = 4;
constexpr int kSmem = 3 * kD * 128 + 16;
constexpr unsigned kFull = 0xffffffffu;

struct Params {
  uint16_t* h0;        // bf16 [slots, HV, 128, 128] of this layer
  const uint16_t* d_hi;  // bf16 [rings, HV, 16, 128]
  const uint16_t* d_lo;
  const uint16_t* k_hi;  // bf16 [rings, H, 16, 128]
  const uint16_t* k_lo;
  const float* g;        // fp32 [rings, HV, 16]
  int64_t s_h0, s_dhi, s_dlo, s_khi, s_klo, s_g;
};

struct Args {
  Params p;
  int64_t l_h0, l_dhi, l_dlo, l_khi, l_klo, l_g;
  const void* state_idx;
  const void* ring_idx;
  const int32_t* write_pos;
  const int32_t* cache_base;
  const int8_t* is_flush;
  int64_t s_si, s_ri, s_wp, s_cb, s_fl;
  int32_t hv_per_h, null_block_id, layers;
  bool si64, ri64;
};

SGL_DEVICE int64_t load_index(const void* p, const int64_t i, const bool is64) {
  return is64 ? static_cast<const int64_t*>(p)[i] : static_cast<int64_t>(static_cast<const int32_t*>(p)[i]);
}

SGL_DEVICE float bf_lo(const uint32_t w) {
  return __uint_as_float(w << 16);
}

SGL_DEVICE float bf_hi(const uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

SGL_DEVICE uint32_t to_bf(const float x) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}

SGL_DEVICE void fold_heads(
    const Params& p,
    uint8_t* smem,
    uint64_t* bar,
    const uint32_t t_base,
    const int64_t slot,
    const int64_t ring,
    const int h,
    const int hv0,
    const int heads,
    const int r,
    const int base,
    uint32_t& parity) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  uint8_t* const s_a = smem;
  uint8_t* const s_b = smem + kD * 128;
  uint8_t* const s_o = smem + 2 * kD * 128;
  {
    const int64_t off = static_cast<int64_t>(h) * kL * kD + tid;
    const uint16_t* hi = p.k_hi + ring * p.s_khi + off;
    const uint16_t* lo = p.k_lo + ring * p.s_klo + off;
    uint32_t kh[kL / 2], kl[kL / 2];
#pragma unroll
    for (int i = 0; i < kL; i += 2) {
      const uint32_t h0 = i < r ? hi[((base + i) & (kL - 1)) * kD] : 0u;
      const uint32_t h1 = i + 1 < r ? hi[((base + i + 1) & (kL - 1)) * kD] : 0u;
      const uint32_t l0 = i < r ? lo[((base + i) & (kL - 1)) * kD] : 0u;
      const uint32_t l1 = i + 1 < r ? lo[((base + i + 1) & (kL - 1)) * kD] : 0u;
      kh[i / 2] = h0 | (h1 << 16);
      kl[i / 2] = l0 | (l1 << 16);
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint32_t* src = (c & 4) ? kl : kh;
      const int q = (c & 1) * 4;
      *reinterpret_cast<uint4*>(s_b + tcgen05::sw128_off(tid, c)) =
          make_uint4(src[q], src[q + 1], src[q + 2], src[q + 3]);
    }
  }
  const auto bdesc = tcgen05::make_desc_kmajor_sw128(s_b);
  const auto adesc = tcgen05::make_desc_kmajor_sw128(s_a);
  const uint32_t idesc = tcgen05::make_instr_desc_f16(kD, 64, false, false) | (1u << 7) | (1u << 10);
  const uint32_t t_row = t_base + (static_cast<uint32_t>(warp * 32) << 16);
#pragma unroll 1
  for (int j = 0; j < heads; ++j) {
    const int hv = hv0 + j;
    uint16_t* const state = p.h0 + slot * p.s_h0 + (static_cast<int64_t>(hv) * kD + tid) * kD;
    const int64_t hv_off = static_cast<int64_t>(hv) * kL * kD + tid;
    const uint16_t* const d_hi = p.d_hi + ring * p.s_dhi + hv_off;
    const uint16_t* const d_lo = p.d_lo + ring * p.s_dlo + hv_off;
    float before = lane < r ? p.g[ring * p.s_g + hv * kL + ((base + lane) & (kL - 1))] : 0.f;
    float d[kL];
#pragma unroll
    for (int i = 0; i < kL; ++i) {
      const int phys = (base + i) & (kL - 1);
      d[i] = i < r ? bf_lo(d_hi[phys * kD]) + bf_lo(d_lo[phys * kD]) : 0.f;
    }
#pragma unroll
    for (int step = 1; step < kL; step *= 2) {
      const float up = __shfl_up_sync(kFull, before, step);
      if (lane >= step) before += up;
    }
    const float total = __shfl_sync(kFull, before, kL - 1);
    const float w = expf(total - before);
    const float decay = expf(total);
    uint32_t uh[kL / 2], ul[kL / 2];
#pragma unroll
    for (int i = 0; i < kL; i += 2) {
      const float u0 = d[i] * __shfl_sync(kFull, w, i);
      const float u1 = d[i + 1] * __shfl_sync(kFull, w, i + 1);
      const uint32_t h0 = to_bf(u0);
      const uint32_t h1 = to_bf(u1);
      uh[i / 2] = h0 | (h1 << 16);
      ul[i / 2] = to_bf(u0 - bf_lo(h0)) | (to_bf(u1 - bf_lo(h1)) << 16);
    }
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      const uint32_t* src = (c & 2) ? ul : uh;
      const int q = (c & 1) * 4;
      *reinterpret_cast<uint4*>(s_a + tcgen05::sw128_off(tid, c)) =
          make_uint4(src[q], src[q + 1], src[q + 2], src[q + 3]);
    }
#pragma unroll 1
    for (int half = 0; half < 2; ++half) {
      const uint16_t* const tile_in = state - static_cast<int64_t>(tid) * kD;
      uint4 old[8];
#pragma unroll
      for (int k = 0; k < 8; ++k)
        old[k] = *reinterpret_cast<const uint4*>(
            tile_in + (warp * 32 + (lane >> 3) + 4 * k) * kD + half * 64 + (lane & 7) * 8);
      device::ptx::fence_proxy_async_shared();
      tcgen05::fence_before_thread_sync();
      device::ptx::bar_sync(1, kWorkers);
      tcgen05::fence_after_thread_sync();
      if (tid == 0) {
        const auto bd = bdesc.advance_bytes(half * 64 * 128);
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
          tcgen05::mma_f16_ss(t_base, adesc.advance_bytes(kk * 32).bits, bd.advance_bytes(kk * 32).bits, idesc, kk > 0);
        }
        tcgen05::commit(bar);
      }
#pragma unroll
      for (int k = 0; k < 8; ++k)
        *reinterpret_cast<uint4*>(s_o + tcgen05::sw128_off(warp * 32 + (lane >> 3) + 4 * k, lane & 7)) = old[k];
      __syncwarp();
      device::ptx::mbar_wait_parity(bar, parity);
      parity ^= 1;
      tcgen05::fence_after_thread_sync();
#pragma unroll
      for (int part = 0; part < 2; ++part) {
        float acc[32];
        tcgen05::ld_32x32b<32>(t_row + part * 32, acc);
        tcgen05::wait_ld();
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          const uint4 o = *reinterpret_cast<const uint4*>(s_o + tcgen05::sw128_off(tid, part * 4 + c));
          const uint32_t ow[4] = {o.x, o.y, o.z, o.w};
          uint32_t nw[4];
#pragma unroll
          for (int e = 0; e < 4; ++e) {
            const float x0 = __fmul_rn(bf_lo(ow[e]), decay) + acc[c * 8 + 2 * e];
            const float x1 = __fmul_rn(bf_hi(ow[e]), decay) + acc[c * 8 + 2 * e + 1];
            nw[e] = to_bf(x0) | (to_bf(x1) << 16);
          }
          *reinterpret_cast<uint4*>(s_o + tcgen05::sw128_off(tid, part * 4 + c)) =
              make_uint4(nw[0], nw[1], nw[2], nw[3]);
        }
      }
      __syncwarp();
      uint16_t* const tile = state - static_cast<int64_t>(tid) * kD;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int row = warp * 32 + (lane >> 3) + 4 * k;
        *reinterpret_cast<uint4*>(tile + row * kD + half * 64 + (lane & 7) * 8) =
            *reinterpret_cast<const uint4*>(s_o + tcgen05::sw128_off(row, lane & 7));
      }
      __syncwarp();
    }
  }
  tcgen05::fence_before_thread_sync();
}

template <bool kUsePDL>
__global__ void __launch_bounds__(kWorkers) fold_kernel(const __grid_constant__ Args a) {
  extern __shared__ __align__(1024) uint8_t smem[];
  uint64_t* const bar = reinterpret_cast<uint64_t*>(smem + 3 * kD * 128);
  uint32_t* const s_tmem = reinterpret_cast<uint32_t*>(smem + 3 * kD * 128 + 8);
  device::PDLWaitPrimary<kUsePDL>();
  const int h = blockIdx.x;
  const int n = blockIdx.y;
  const int64_t layer0 = static_cast<int64_t>(blockIdx.z) * kLayersPerCta;
  const int64_t slot = load_index(a.state_idx, n * a.s_si, a.si64);
  if (slot <= a.null_block_id) return;
  const int64_t ring = load_index(a.ring_idx, n * a.s_ri, a.ri64);
  if (a.is_flush[ring * a.s_fl] == 0) return;
  const int r = min(max(a.write_pos[ring * a.s_wp], 0), kL);
  if (r == 0) return;
  const int base = a.cache_base[ring * a.s_cb];
  const int warp = threadIdx.x >> 5;
  if (warp == 0) {
    tcgen05::alloc(s_tmem, kTmemCols);
    tcgen05::relinquish_alloc_permit();
  }
  if (threadIdx.x == 0) {
    device::ptx::mbar_init(bar, 1);
    device::ptx::fence_barrier_init();
  }
  tcgen05::fence_before_thread_sync();
  __syncthreads();
  tcgen05::fence_after_thread_sync();
  uint32_t parity = 0;
  const int64_t last = min(layer0 + kLayersPerCta, static_cast<int64_t>(a.layers));
#pragma unroll 1
  for (int64_t layer = layer0; layer < last; ++layer) {
    Params p = a.p;
    p.h0 += layer * a.l_h0;
    p.d_hi += layer * a.l_dhi;
    p.d_lo += layer * a.l_dlo;
    p.k_hi += layer * a.l_khi;
    p.k_lo += layer * a.l_klo;
    p.g += layer * a.l_g;
    fold_heads(p, smem, bar, *s_tmem, slot, ring, h, h * a.hv_per_h, a.hv_per_h, r, base, parity);
    tcgen05::fence_before_thread_sync();
    __syncthreads();
    tcgen05::fence_after_thread_sync();
  }
  tcgen05::fence_after_thread_sync();
  if (warp == 0) tcgen05::dealloc(*s_tmem, kTmemCols);
}

}  // namespace gdn_replayssm_fold

template <bool kUsePDL>
struct GdnReplaySSMFold {
  static void
  run(const tvm::ffi::TensorView h0,
      const tvm::ffi::TensorView d_hi,
      const tvm::ffi::TensorView d_lo,
      const tvm::ffi::TensorView k_hi,
      const tvm::ffi::TensorView k_lo,
      const tvm::ffi::TensorView g,
      const tvm::ffi::TensorView state_idx,
      const tvm::ffi::TensorView ring_idx,
      const tvm::ffi::TensorView write_pos,
      const tvm::ffi::TensorView cache_base,
      const tvm::ffi::TensorView is_flush,
      int64_t null_block_id) {
    using namespace host;
    using namespace gdn_replayssm_fold;
    auto L_ = SymbolicSize{"layers"};
    auto H_ = SymbolicSize{"key_heads"};
    auto HV_ = SymbolicSize{"value_heads"};
    auto Slots_ = SymbolicSize{"state_slots"};
    auto Rings_ = SymbolicSize{"ring_rows"};
    auto B_ = SymbolicSize{"rows"};
    auto device = SymbolicDevice{};
    device.set_options<kDLCUDA>();
    TensorMatcher({L_, Slots_, HV_, kD, kD})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, -1, kD * kD, kD, 1})
        .verify(h0);
    TensorMatcher({L_, Rings_, HV_, kL, kD})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, -1, kL * kD, kD, 1})
        .verify(d_hi)
        .verify(d_lo);
    TensorMatcher({L_, Rings_, H_, kL, kD})
        .with_dtype<bf16_t>()
        .with_device(device)
        .with_strides({-1, -1, kL * kD, kD, 1})
        .verify(k_hi)
        .verify(k_lo);
    TensorMatcher({L_, Rings_, HV_, kL}).with_dtype<fp32_t>().with_device(device).with_strides({-1, -1, kL, 1}).verify(g);
    TensorMatcher({B_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({-1}).verify(state_idx);
    TensorMatcher({B_}).with_dtype<int32_t, int64_t>().with_device(device).with_strides({-1}).verify(ring_idx);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).with_strides({-1}).verify(write_pos);
    TensorMatcher({-1}).with_dtype<int32_t>().with_device(device).with_strides({-1}).verify(cache_base);
    TensorMatcher({-1}).with_dtype<int8_t>().with_device(device).with_strides({-1}).verify(is_flush);
    const int64_t rows = B_.unwrap();
    const int64_t num_h = H_.unwrap();
    RuntimeCheck(num_h > 0 && HV_.unwrap() % num_h == 0, "value heads must be a multiple of key heads");
    RuntimeCheck(
        reinterpret_cast<uintptr_t>(h0.data_ptr()) % 16 == 0 && h0.stride(1) % 8 == 0 && h0.stride(0) % 8 == 0,
        "the checkpoint must be 16-byte aligned per slot");
    if (rows == 0 || L_.unwrap() == 0) return;
    const auto args = Args{
        .p =
            Params{
                .h0 = static_cast<uint16_t*>(h0.data_ptr()),
                .d_hi = static_cast<const uint16_t*>(d_hi.data_ptr()),
                .d_lo = static_cast<const uint16_t*>(d_lo.data_ptr()),
                .k_hi = static_cast<const uint16_t*>(k_hi.data_ptr()),
                .k_lo = static_cast<const uint16_t*>(k_lo.data_ptr()),
                .g = static_cast<const float*>(g.data_ptr()),
                .s_h0 = h0.stride(1),
                .s_dhi = d_hi.stride(1),
                .s_dlo = d_lo.stride(1),
                .s_khi = k_hi.stride(1),
                .s_klo = k_lo.stride(1),
                .s_g = g.stride(1),
            },
        .l_h0 = h0.stride(0),
        .l_dhi = d_hi.stride(0),
        .l_dlo = d_lo.stride(0),
        .l_khi = k_hi.stride(0),
        .l_klo = k_lo.stride(0),
        .l_g = g.stride(0),
        .state_idx = state_idx.data_ptr(),
        .ring_idx = ring_idx.data_ptr(),
        .write_pos = static_cast<const int32_t*>(write_pos.data_ptr()),
        .cache_base = static_cast<const int32_t*>(cache_base.data_ptr()),
        .is_flush = static_cast<const int8_t*>(is_flush.data_ptr()),
        .s_si = state_idx.stride(0),
        .s_ri = ring_idx.stride(0),
        .s_wp = write_pos.stride(0),
        .s_cb = cache_base.stride(0),
        .s_fl = is_flush.stride(0),
        .hv_per_h = static_cast<int32_t>(HV_.unwrap() / num_h),
        .null_block_id = static_cast<int32_t>(null_block_id),
        .layers = static_cast<int32_t>(L_.unwrap()),
        .si64 = state_idx.dtype().bits == 64,
        .ri64 = ring_idx.dtype().bits == 64,
    };
    const DLDevice dev = device.unwrap();
    runtime::set_max_dynamic_smem_per_device<fold_kernel<kUsePDL>>(dev, kSmem);
    const dim3 grid(
        static_cast<uint32_t>(num_h),
        static_cast<uint32_t>(rows),
        static_cast<uint32_t>((L_.unwrap() + kLayersPerCta - 1) / kLayersPerCta));
    LaunchKernel(grid, kWorkers, dev, kSmem).enable_pdl(kUsePDL)(fold_kernel<kUsePDL>, args);
  }
};

}  // namespace sglang
