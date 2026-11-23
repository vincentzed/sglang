"""CUDA tcgen05 GDN ReplaySSM verify: the Triton verify's contract on the
compact bf16 ring, checked against that kernel and an fp64 gated delta rule."""

import sys

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.version.hip is not None
        or torch.cuda.get_device_capability()[0] != 10,
        reason="the CUDA verify needs tcgen05 (SM100)",
    ),
]

DEVICE = "cuda"
SLOTS, H, HV, D, RING = 12, 2, 6, 128, 16
STATE_ROWS = [3, 7, -1, 1, 9]
RING_ROWS = [0, 5, 9, 2, 11]
WRITE_POS = [0, 3, 5, 8, 14]
SENTINEL = 7.0
# The serving model keeps A_log in fp32 and dt_bias in bf16.
GATE_DTYPES = (torch.float32, torch.bfloat16)


def _case(
    seed,
    *,
    spec_len=4,
    index_dtype=torch.int64,
    strided=False,
    flush_row=None,
    gate_dtypes=GATE_DTYPES,
):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device=DEVICE, generator=gen) * scale).to(
            torch.bfloat16
        )

    def cursor(values, dtype):
        values = torch.tensor(values, dtype=dtype, device=DEVICE)
        if not strided:
            return values
        out = torch.empty_strided(values.shape, (2,), dtype=dtype, device=DEVICE)
        out.copy_(values)
        return out

    B = len(STATE_ROWS)
    T = B * spec_len
    write_pos = [0] * SLOTS
    for ring, pos in zip(RING_ROWS, WRITE_POS):
        write_pos[ring] = pos
    is_flush = [0] * SLOTS
    if flush_row is not None:
        is_flush[RING_ROWS[flush_row]] = 1
    mixed = rand(T, 2 * H * D + HV * D)
    q, k, v = torch.split(mixed, [H * D, H * D, HV * D], dim=-1)
    checkpoint = rand(SLOTS, HV + (1 if strided else 0), D, D, scale=0.05)[:, :HV]
    return dict(
        q=q.reshape(T, H, D),
        k=k.reshape(T, H, D),
        v=v.reshape(T, HV, D),
        a=rand(T, HV),
        b=rand(T, HV),
        A_log=(torch.randn(HV, device=DEVICE, generator=gen) * 0.5 - 2.0).to(
            gate_dtypes[0]
        ),
        dt_bias=(torch.randn(HV, device=DEVICE, generator=gen) * 0.5).to(
            gate_dtypes[1]
        ),
        checkpoint_state=checkpoint,
        d_cache=rand(SLOTS, HV, RING, D, scale=0.05),
        k_cache=rand(SLOTS, H, RING, D, scale=0.09),
        g_cache=-torch.rand(SLOTS, HV, RING, device=DEVICE, generator=gen) * 0.5,
        d_residual_cache=rand(SLOTS, HV, RING, D, scale=1e-4),
        k_residual_cache=rand(SLOTS, H, RING, D, scale=2e-4),
        query_start_loc=torch.arange(
            0, T + 1, spec_len, dtype=torch.int32, device=DEVICE
        ),
        ssm_state_indices=torch.tensor(STATE_ROWS, dtype=index_dtype, device=DEVICE),
        replay_indices=torch.tensor(RING_ROWS, dtype=index_dtype, device=DEVICE),
        write_pos=cursor(write_pos, torch.int32),
        cache_base=cursor(
            torch.randint(0, RING, (SLOTS,), generator=gen, device=DEVICE).tolist(),
            torch.int32,
        ),
        is_flush=cursor(is_flush, torch.int8),
    )


_MUTABLE = ("d_cache", "k_cache", "g_cache", "d_residual_cache", "k_residual_cache")


def _clone(case):
    return {k: (v.clone() if k in _MUTABLE else v) for k, v in case.items()}


def _out(case):
    return torch.full_like(case["v"], SENTINEL)


def _cuda(case):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_verify_cuda import (
        gdn_replayssm_verify_cuda,
    )

    out = gdn_replayssm_verify_cuda(
        **case, out=_out(case), scale=D**-0.5, null_block_id=-1
    )
    torch.cuda.synchronize()
    return out


def _triton(case, spec_len):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_spec_decode import (
        gdn_replayssm_spec_decode,
    )

    args = dict(case)
    args["rawv_cache"] = args.pop("d_residual_cache")
    args["rawk_cache"] = args.pop("k_residual_cache")
    out = gdn_replayssm_spec_decode(
        **args,
        beta_cache=None,
        out=_out(case),
        max_cache_len=RING,
        max_spec_len=spec_len,
        scale=D**-0.5,
        null_block_id=-1,
        launch_mode="verify",
    )
    torch.cuda.synchronize()
    return out


def _supported(case, spec_len, **override):
    from sglang.kernels.ops.attention.linear.gdn_replayssm_verify_cuda import (
        gdn_replayssm_verify_cuda_supported,
    )

    args = {k: v for k, v in case.items() if k != "is_flush"}
    args.update(beta_cache=None, max_spec_len=spec_len)
    args.update(override)
    return gdn_replayssm_verify_cuda_supported(**args)


def _written_records(case, spec_len):
    """[ring row, record] mask of the ring records a verify of this case writes."""
    written = torch.zeros(SLOTS, RING, dtype=torch.bool, device=DEVICE)
    for slot, ring, pos in zip(STATE_ROWS, RING_ROWS, WRITE_POS):
        if slot < 0 or int(case["is_flush"][ring]):
            continue
        base = int(case["cache_base"][ring])
        for s in range(spec_len):
            if pos + s < RING:
                written[ring, (base + pos + s) % RING] = True
    return written


def _close(a, b, rel):
    a, b = a.float(), b.float()
    bound = rel * b.abs().max().item()
    assert (a - b).abs().max().item() <= bound, ((a - b).abs().max().item(), bound)


@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("spec_len", [4, 3])
@pytest.mark.parametrize("index_dtype", [torch.int64, torch.int32])
@pytest.mark.parametrize("strided", [False, True])
def test_matches_triton_verify(seed, spec_len, index_dtype, strided):
    case = _case(seed, spec_len=spec_len, index_dtype=index_dtype, strided=strided)
    assert _supported(case, spec_len)
    cuda, triton = _clone(case), _clone(case)
    out_cuda, out_triton = _cuda(cuda), _triton(triton, spec_len)
    _close(out_cuda, out_triton, 1e-2)
    null = STATE_ROWS.index(-1)
    assert not out_cuda[null * spec_len : (null + 1) * spec_len].any()
    _close(
        cuda["d_cache"].float() + cuda["d_residual_cache"].float(),
        triton["d_cache"].float() + triton["d_residual_cache"].float(),
        2e-3,
    )
    _close(
        cuda["k_cache"].float() + cuda["k_residual_cache"].float(),
        triton["k_cache"].float() + triton["k_residual_cache"].float(),
        1e-5,
    )
    _close(cuda["g_cache"], triton["g_cache"], 1e-6)
    written = _written_records(case, spec_len)
    for name in _MUTABLE:
        for arm in (cuda, triton):
            kept = ~written[:, None, :].expand(arm[name].shape[:3])
            assert torch.equal(arm[name][kept], case[name][kept]), name
        assert not torch.equal(cuda[name], case[name]), name


@pytest.mark.parametrize(
    "gate_dtypes",
    [
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_gate_parameter_dtypes(gate_dtypes):
    case = _case(4, gate_dtypes=gate_dtypes)
    assert _supported(case, 4)
    cuda, triton = _clone(case), _clone(case)
    _close(_cuda(cuda), _triton(triton, 4), 1e-2)
    _close(cuda["g_cache"], triton["g_cache"], 1e-6)


def test_many_rows_match_triton():
    rows, slots, spec_len = 96, 128, 4
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    cpu = torch.Generator().manual_seed(5)

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device=DEVICE, generator=gen) * scale).to(
            torch.bfloat16
        )

    tokens = rows * spec_len
    case = dict(
        q=rand(tokens, H, D),
        k=rand(tokens, H, D),
        v=rand(tokens, HV, D),
        a=rand(tokens, HV),
        b=rand(tokens, HV),
        A_log=torch.randn(HV, device=DEVICE, generator=gen) * 0.5 - 2.0,
        dt_bias=rand(HV, scale=0.5),
        checkpoint_state=rand(slots, HV, D, D, scale=0.05),
        d_cache=rand(rows + 1, HV, RING, D, scale=0.05),
        k_cache=rand(rows + 1, H, RING, D, scale=0.09),
        g_cache=-torch.rand(rows + 1, HV, RING, device=DEVICE, generator=gen) * 0.5,
        d_residual_cache=rand(rows + 1, HV, RING, D, scale=1e-4),
        k_residual_cache=rand(rows + 1, H, RING, D, scale=2e-4),
        query_start_loc=torch.arange(
            0, tokens + 1, spec_len, dtype=torch.int32, device=DEVICE
        ),
        ssm_state_indices=torch.randperm(slots, generator=cpu)[:rows].int().to(DEVICE),
        replay_indices=torch.randperm(rows, generator=cpu).int().to(DEVICE),
        write_pos=torch.randint(0, 13, (rows + 1,), generator=cpu).int().to(DEVICE),
        cache_base=torch.randint(0, RING, (rows + 1,), generator=cpu).int().to(DEVICE),
        is_flush=torch.zeros(rows + 1, dtype=torch.int8, device=DEVICE),
    )
    assert _supported(case, spec_len)
    cuda, triton = _clone(case), _clone(case)
    _close(_cuda(cuda), _triton(triton, spec_len), 1e-2)
    _close(
        cuda["d_cache"].float() + cuda["d_residual_cache"].float(),
        triton["d_cache"].float() + triton["d_residual_cache"].float(),
        2e-3,
    )
    _close(cuda["g_cache"], triton["g_cache"], 1e-6)


def test_flush_rows_are_left_alone():
    flush_row = 1
    case = _case(2, flush_row=flush_row)
    cuda = _clone(case)
    out = _cuda(cuda)
    assert (out[flush_row * 4 : (flush_row + 1) * 4] == SENTINEL).all()
    written = _written_records(case, 4)
    assert not written[RING_ROWS[flush_row]].any()
    for name in _MUTABLE:
        kept = ~written[:, None, :].expand(cuda[name].shape[:3])
        assert torch.equal(cuda[name][kept], case[name][kept]), name
    triton = _clone(case)
    _close(out, _triton(triton, 4), 1e-2)


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_fp64_delta_rule(seed):
    case = _case(seed)
    out = _cuda(_clone(case)).double().view(len(STATE_ROWS), 4, HV, D)
    q, k = case["q"].double(), case["k"].double()
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * D**-0.5
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    qn = qn.repeat_interleave(HV // H, dim=1).view(-1, 4, HV, D)
    kn = kn.repeat_interleave(HV // H, dim=1).view(-1, 4, HV, D)
    v = case["v"].double().view(-1, 4, HV, D)
    x = case["a"].double().view(-1, 4, HV) + case["dt_bias"].double()
    gate = -torch.exp(case["A_log"].double()) * torch.log1p(torch.exp(x))
    beta = torch.sigmoid(case["b"].double().view(-1, 4, HV))
    d_ring = case["d_cache"].double() + case["d_residual_cache"].double()
    k_ring = (
        case["k_cache"].double() + case["k_residual_cache"].double()
    ).repeat_interleave(HV // H, dim=1)
    for row, (slot, ring, pos) in enumerate(zip(STATE_ROWS, RING_ROWS, WRITE_POS)):
        if slot < 0:
            continue
        state = case["checkpoint_state"][slot].double()
        base = int(case["cache_base"][ring])
        for i in range(pos):
            phys = (base + i) % RING
            state = (
                state
                * torch.exp(case["g_cache"][ring, :, phys].double())[:, None, None]
            )
            state = (
                state + d_ring[ring, :, phys, :, None] * k_ring[ring, :, phys, None, :]
            )
        for s in range(4):
            state = state * torch.exp(gate[row, s])[:, None, None]
            pred = torch.einsum("hvk,hk->hv", state, kn[row, s])
            delta = beta[row, s][:, None] * (v[row, s] - pred)
            state = state + delta[:, :, None] * kn[row, s][:, None, :]
            ref = torch.einsum("hvk,hk->hv", state, qn[row, s])
            _close(out[row, s], ref, 5e-3)


def test_cuda_graph_replay():
    case = _case(3)
    eager = _cuda(_clone(case))
    captured = _clone(case)
    graph = torch.cuda.CUDAGraph()
    out = _out(case)
    from sglang.kernels.ops.attention.linear.gdn_replayssm_verify_cuda import (
        gdn_replayssm_verify_cuda,
    )

    with torch.cuda.graph(graph):
        gdn_replayssm_verify_cuda(**captured, out=out, scale=D**-0.5, null_block_id=-1)
    for name in _MUTABLE:
        captured[name].copy_(case[name])
    out.fill_(SENTINEL)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), eager.view(torch.int16))


def test_other_layouts_keep_the_triton_verify():
    case = _case(0)
    assert _supported(case, 4)
    assert not _supported(case, 5)
    assert not _supported(case, 4, d_residual_cache=None)
    assert not _supported(case, 4, beta_cache=case["g_cache"])
    assert not _supported(case, 4, checkpoint_state=case["checkpoint_state"].float())
    assert not _supported(case, 4, q=case["q"][:, :, 1:])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
