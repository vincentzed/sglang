"""HC sum mix for more than 24 rows with the Up product, gate and stream mean in one kernel.

Down and its RMS/SiLU post-processing stay FlashInfer's (cuBLAS bmm into per-stream FP32 partials, then
the BF16 low-rank input and alpha). One warp-specialized tcgen05 kernel replaces the Up GEMM and the mix
kernel: a TMA warp keeps the 128-row low-rank input resident and streams the residual and Up tiles, an
MMA warp accumulates four streams x 32 channels into double-buffered TMEM, and eight epilogue warps apply
the gate, the RMS-normalized residual and the stream mean while the next tile's MMA runs. The [T, 4*H]
BF16 logits are never written.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    fence_async_shared,
    mbarrier,
    tcgen05_commit,
    tcgen05_mma,
    tensor_memory_descriptor,
    tensor_memory_descriptor_type,
    tma,
)
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.runtime.jit import constexpr_function

_BM = 128
_BK = 64
_BC = 32
_CH = 16
_STAGES = 4
_NW = 8
_MIN_ROWS = 128


@constexpr_function
def _tmem_reg_layout(dtype, shape, layout, num_warps):
    return tensor_memory_descriptor_type(dtype, shape, layout, shape).get_reg_layout(
        num_warps=num_warps
    )


@gluon.jit
def _term_pair(lg, x, inv, g):
    return gl.inline_asm_elementwise(
        asm="""{
        .reg .b32 p, a, b, x0, x1, gb, nb;
        cvt.rn.bf16x2.f32 p, $2, $1;
        shl.b32 a, p, 16;
        and.b32 b, p, 0xffff0000;
        mul.f32 a, a, 0fBFB8AA3B;
        mul.f32 b, b, 0fBFB8AA3B;
        ex2.approx.ftz.f32 a, a;
        ex2.approx.ftz.f32 b, b;
        add.f32 a, a, 0f3F800000;
        add.f32 b, b, 0f3F800000;
        rcp.approx.ftz.f32 a, a;
        rcp.approx.ftz.f32 b, b;
        cvt.rn.bf16x2.f32 gb, b, a;
        shl.b32 x0, $3, 16;
        and.b32 x1, $3, 0xffff0000;
        mul.f32 x0, x0, $4;
        mul.f32 x1, x1, $5;
        mul.f32 x0, x0, $6;
        mul.f32 x1, x1, $7;
        cvt.rn.bf16x2.f32 nb, x1, x0;
        mul.rn.bf16x2 $0, gb, nb;
        }""",
        constraints="=r,r,r,r,r,r,r,r",
        args=[lg, x, inv, g],
        dtype=gl.bfloat16,
        is_pure=True,
        pack=2,
    )


@gluon.aggregate
class _UpMix:
    t_desc: tma.tensor_descriptor
    up_desc: tma.tensor_descriptor
    x_desc: tma.tensor_descriptor
    out_desc: tma.tensor_descriptor
    t_bufs: gl.shared_memory_descriptor
    w_bufs: gl.shared_memory_descriptor
    x_bufs: gl.shared_memory_descriptor
    ld_empty: gl.shared_memory_descriptor
    ld_ready: gl.shared_memory_descriptor
    x_empty: gl.shared_memory_descriptor
    x_ready: gl.shared_memory_descriptor
    acc: tensor_memory_descriptor
    acc_empty: gl.shared_memory_descriptor
    acc_ready: gl.shared_memory_descriptor
    t_ready: gl.shared_memory_descriptor
    t_free: gl.shared_memory_descriptor


@gluon.jit
def _up_mix_load(
    p,
    start,
    end,
    NCB: gl.constexpr,
    H: gl.constexpr,
    BC: gl.constexpr,
    KS: gl.constexpr,
    STAGES: gl.constexpr,
):
    it = 0
    n = 0
    r = 0
    cur = -1
    for tile in range(start, end):
        rt = tile // NCB
        off_m = rt * 128
        c0 = (tile % NCB) * BC
        xs = n % 2
        mbarrier.wait(p.x_empty.index(xs), ((n // 2) & 1) ^ 1)
        xbar = p.x_ready.index(xs)
        mbarrier.expect(xbar, 4 * p.x_desc.block_type.nbytes)
        for b in gl.static_range(4):
            tma.async_load(
                p.x_desc, [off_m, b * H + c0], xbar, p.x_bufs.index(xs * 4 + b)
            )
        if rt != cur:
            mbarrier.wait(p.t_free, (r & 1) ^ 1)
            mbarrier.expect(p.t_ready, KS * p.t_desc.block_type.nbytes)
            for k in gl.static_range(KS):
                tma.async_load(p.t_desc, [off_m, k * 64], p.t_ready, p.t_bufs.index(k))
            cur = rt
            r += 1
        for k in gl.static_range(KS):
            s = it % STAGES
            mbarrier.wait(p.ld_empty.index(s), ((it // STAGES) & 1) ^ 1)
            bar = p.ld_ready.index(s)
            mbarrier.expect(bar, 4 * p.up_desc.block_type.nbytes)
            for b in gl.static_range(4):
                tma.async_load(
                    p.up_desc,
                    [b * H + c0, k * 64],
                    bar,
                    p.w_bufs.index(s).slice(b * BC, BC),
                )
            it += 1
        n += 1


@gluon.jit
def _up_mix_mma(
    p, start, end, NCB: gl.constexpr, KS: gl.constexpr, STAGES: gl.constexpr
):
    it = 0
    n = 0
    r = 0
    cur = -1
    for tile in range(start, end):
        rt = tile // NCB
        if rt != cur:
            mbarrier.wait(p.t_ready, r & 1)
            cur = rt
            r += 1
        a = n % 2
        mbarrier.wait(p.acc_empty.index(a), ((n // 2) & 1) ^ 1)
        acc = p.acc.index(a)
        for k in gl.static_range(KS):
            s = it % STAGES
            mbarrier.wait(p.ld_ready.index(s), (it // STAGES) & 1)
            tcgen05_mma(
                p.t_bufs.index(k),
                p.w_bufs.index(s).permute((1, 0)),
                acc,
                use_acc=k > 0,
                mbarriers=[p.ld_empty.index(s)],
            )
            it += 1
        tcgen05_commit(p.acc_ready.index(a))
        if (tile + 1 == end) or ((tile + 1) // NCB != rt):
            tcgen05_commit(p.t_free)
        n += 1


@gluon.jit
def _up_mix_epilogue(
    p,
    start,
    end,
    sums_ptr,
    gamma_ptr,
    next_sums_ptr,
    M,
    eps,
    NCB: gl.constexpr,
    H: gl.constexpr,
    BC: gl.constexpr,
    CH: gl.constexpr,
    NW: gl.constexpr,
):
    creg: gl.constexpr = _tmem_reg_layout(
        gl.float32, [128, CH], TensorMemoryLayout(block=(128, 4 * BC), col_stride=1), NW
    )
    rl: gl.constexpr = gl.SliceLayout(1, creg)
    cl: gl.constexpr = gl.SliceLayout(0, creg)
    zl: gl.constexpr = gl.BlockedLayout([4], [32], [NW], [0])
    dt: gl.constexpr = p.out_desc.dtype
    o_bufs = gl.allocate_shared_memory(dt, [2, 128, BC], p.out_desc.layout)
    n = 0
    for tile in range(start, end):
        cb = tile % NCB
        off_m = (tile // NCB) * 128
        c0 = cb * BC
        a = n % 2
        ph = (n // 2) & 1
        rows = off_m + gl.arange(0, 128, rl)
        live = rows < M
        if cb == 0:
            zi = off_m * 4 + gl.arange(0, 512, zl)
            gl.store(
                next_sums_ptr + zi, gl.zeros([512], gl.float32, zl), mask=zi < M * 4
            )
        inv0 = gl.rsqrt(
            gl.load(sums_ptr + rows * 4, mask=live, other=1.0) * (1.0 / H) + eps
        )
        inv1 = gl.rsqrt(
            gl.load(sums_ptr + rows * 4 + 1, mask=live, other=1.0) * (1.0 / H) + eps
        )
        inv2 = gl.rsqrt(
            gl.load(sums_ptr + rows * 4 + 2, mask=live, other=1.0) * (1.0 / H) + eps
        )
        inv3 = gl.rsqrt(
            gl.load(sums_ptr + rows * 4 + 3, mask=live, other=1.0) * (1.0 / H) + eps
        )
        mbarrier.wait(p.x_ready.index(a), ph)
        mbarrier.wait(p.acc_ready.index(a), ph)
        acc = p.acc.index(a)
        o = o_bufs.index(a)
        tma.store_wait(1)
        for c in gl.static_range(BC // CH):
            cols = c0 + c * CH + gl.arange(0, CH, cl)
            total = gl.zeros([128, CH], gl.float32, creg)
            for b in gl.static_range(4):
                if b == 0:
                    inv = inv0
                elif b == 1:
                    inv = inv1
                elif b == 2:
                    inv = inv2
                else:
                    inv = inv3
                lg = acc.slice(b * BC + c * CH, CH).load(creg)
                xb = p.x_bufs.index(a * 4 + b).slice(c * CH, CH, dim=1).load(creg)
                g = gl.load(gamma_ptr + b * H + cols)
                total += _term_pair(
                    lg,
                    xb,
                    gl.broadcast(inv[:, None], lg)[0],
                    gl.broadcast(g[None, :], lg)[0],
                ).to(gl.float32)
            o.slice(c * CH, CH, dim=1).store((total * 0.25).to(dt))
        gl.barrier()
        mbarrier.arrive(p.acc_empty.index(a), count=1)
        mbarrier.arrive(p.x_empty.index(a), count=1)
        fence_async_shared()
        tma.async_store(p.out_desc, [off_m, c0], o)
        n += 1
    tma.store_wait(0)


@gluon.jit
def _hc_up_mix_ws_kernel(
    t_desc,
    up_desc,
    x_desc,
    out_desc,
    sums_ptr,
    gamma_ptr,
    next_sums_ptr,
    M,
    eps,
    n_tiles,
    H: gl.constexpr,
    L: gl.constexpr,
    BC: gl.constexpr,
    CH: gl.constexpr,
    STAGES: gl.constexpr,
    NW: gl.constexpr,
):
    KS: gl.constexpr = L // 64
    NCB: gl.constexpr = H // BC
    pid = gl.program_id(0)
    grid = gl.num_programs(0)
    start = pid * n_tiles // grid
    end = (pid + 1) * n_tiles // grid
    t_bufs = gl.allocate_shared_memory(gl.bfloat16, [KS, 128, 64], t_desc.layout)
    w_bufs = gl.allocate_shared_memory(
        gl.bfloat16, [STAGES, 4 * BC, 64], up_desc.layout
    )
    x_bufs = gl.allocate_shared_memory(gl.bfloat16, [8, 128, BC], x_desc.layout)
    ld_empty = mbarrier.allocate_mbarrier(batch=STAGES)
    ld_ready = mbarrier.allocate_mbarrier(batch=STAGES)
    x_empty = mbarrier.allocate_mbarrier(batch=2)
    x_ready = mbarrier.allocate_mbarrier(batch=2)
    acc_empty = mbarrier.allocate_mbarrier(batch=2)
    acc_ready = mbarrier.allocate_mbarrier(batch=2)
    t_ready = mbarrier.allocate_mbarrier()
    t_free = mbarrier.allocate_mbarrier()
    for i in gl.static_range(STAGES):
        mbarrier.init(ld_empty.index(i), count=1)
        mbarrier.init(ld_ready.index(i), count=1)
    for i in gl.static_range(2):
        mbarrier.init(x_empty.index(i), count=1)
        mbarrier.init(x_ready.index(i), count=1)
        mbarrier.init(acc_empty.index(i), count=1)
        mbarrier.init(acc_ready.index(i), count=1)
    mbarrier.init(t_ready, count=1)
    mbarrier.init(t_free, count=1)
    acc = allocate_tensor_memory(
        gl.float32,
        [2, 128, 4 * BC],
        TensorMemoryLayout(block=(128, 4 * BC), col_stride=1),
    )
    p = _UpMix(
        t_desc,
        up_desc,
        x_desc,
        out_desc,
        t_bufs,
        w_bufs,
        x_bufs,
        ld_empty,
        ld_ready,
        x_empty,
        x_ready,
        acc,
        acc_empty,
        acc_ready,
        t_ready,
        t_free,
    )
    gl.warp_specialize(
        [
            (
                _up_mix_epilogue,
                (
                    p,
                    start,
                    end,
                    sums_ptr,
                    gamma_ptr,
                    next_sums_ptr,
                    M,
                    eps,
                    NCB,
                    H,
                    BC,
                    CH,
                    NW,
                ),
            ),
            (_up_mix_load, (p, start, end, NCB, H, BC, KS, STAGES)),
            (_up_mix_mma, (p, start, end, NCB, KS, STAGES)),
        ],
        [1, 1],
        [48, 48],
    )


def _nvmma(block):
    return gl.NVMMASharedLayout.get_default_for(block, gl.bfloat16)


_SM_COUNT = {}


def _sm_count(device):
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SM_COUNT[index]


def hc_gamma(norm: torch.Tensor) -> torch.Tensor:
    """FP32 Gemma gains [4, H] from FlashInfer's interleaved [H, 4] RMS weights."""
    h = norm.numel() // 4
    return (1.0 + norm.reshape(h, 4).T.float()).contiguous()


def hc_sum_ws_supported(residual: torch.Tensor) -> bool:
    return (
        residual.shape[0] >= _MIN_ROWS
        and residual.dtype == torch.bfloat16
        and residual.is_contiguous()
    )


def hc_up_mix(t, up, residual, sums, gamma, mixed, next_sums, eps):
    rows = residual.shape[0]
    h = mixed.shape[1]
    lowrank = t.shape[1]
    t_desc = TensorDescriptor.from_tensor(t, [_BM, _BK], _nvmma([_BM, _BK]))
    up_desc = TensorDescriptor.from_tensor(up, [_BC, _BK], _nvmma([_BC, _BK]))
    x_desc = TensorDescriptor.from_tensor(residual, [_BM, _BC], _nvmma([_BM, _BC]))
    out_desc = TensorDescriptor.from_tensor(mixed, [_BM, _BC], _nvmma([_BM, _BC]))
    n_tiles = triton.cdiv(rows, _BM) * (h // _BC)
    _hc_up_mix_ws_kernel[(min(n_tiles, _sm_count(residual.device)),)](
        t_desc,
        up_desc,
        x_desc,
        out_desc,
        sums,
        gamma,
        next_sums,
        rows,
        eps,
        n_tiles,
        H=h,
        L=lowrank,
        BC=_BC,
        CH=_CH,
        STAGES=_STAGES,
        NW=_NW,
        num_warps=_NW,
        enable_fp_fusion=False,
    )


def hc_sum_ws_mix(residual, sums, packed_down, up, gamma, eps):
    """Mixed input, final alpha and zeroed next sums of one HC sum mix; ``sums`` are the current ones."""
    from flashinfer.experimental.hc_sum_sm100.large import (
        _hc_down_batched_sum_post_kernel,
    )

    rows = residual.shape[0]
    h = packed_down.shape[2]
    np_ = packed_down.shape[1]
    lowrank = up.shape[1]
    device = residual.device
    partial = torch.empty((4, rows, np_), dtype=torch.float32, device=device)
    t = torch.empty((rows, lowrank), dtype=residual.dtype, device=device)
    alpha = torch.empty((rows, 4), dtype=torch.float32, device=device)
    next_sums = torch.empty((rows, 4), dtype=torch.float32, device=device)
    mixed = torch.empty((rows, h), dtype=residual.dtype, device=device)
    torch.bmm(
        residual.view(rows, 4, h).transpose(0, 1),
        packed_down.transpose(1, 2),
        out_dtype=torch.float32,
        out=partial,
    )
    n = rows * (lowrank + 4)
    _hc_down_batched_sum_post_kernel[(triton.cdiv(n, 256),)](
        partial,
        sums,
        t,
        alpha,
        n,
        H=h,
        LOWRANK=lowrank,
        C=4,
        NP=np_,
        BATCH_STRIDE=partial.stride(0),
        WIDE_INDEX=triton.cdiv(n, 256) * 256 > 2**31,
        EPS=eps,
        BLOCK=256,
        num_warps=4,
    )
    hc_up_mix(t, up, residual, sums, gamma, mixed, next_sums, eps)
    return mixed, alpha, next_sums
