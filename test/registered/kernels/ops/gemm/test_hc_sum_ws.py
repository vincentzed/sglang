import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

HIDDEN = 2560
LOWRANK = 320
EPS = 1e-6

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="HC sum mix requires SM100/SM103",
)


def _plan():
    from flashinfer.hyperconnection import prepare_hc_sum_weights

    g = torch.Generator().manual_seed(0)
    down = (torch.randn(LOWRANK, 4 * HIDDEN, generator=g) * 0.02).bfloat16()
    up = (torch.randn(4 * HIDDEN, LOWRANK, generator=g) * 0.05).bfloat16()
    inject = (torch.randn(4, 4 * HIDDEN, generator=g) * 0.02).bfloat16()
    norm = (torch.randn(4, HIDDEN, generator=g) * 0.1).bfloat16()
    return prepare_hc_sum_weights(down, up, inject, norm, rms_eps=EPS, device="cuda")


def _residual(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scale = torch.tensor([1.0, 0.6, 2.5, 1.4], device="cuda").repeat_interleave(HIDDEN)
    return (
        torch.randn(rows, 4 * HIDDEN, device="cuda", generator=g) * scale
    ).bfloat16()


def _reference(plan, residual, sums):
    rows = residual.shape[0]
    r = residual.double().view(rows, 4, HIDDEN)
    inv = torch.rsqrt(sums.double() / HIDDEN + EPS)
    raw = torch.einsum("mbh,bnh,mb->mn", r, plan.packed_down.double(), inv)
    scaled = raw[:, : LOWRANK + 4] / 4
    sig = torch.sigmoid(scaled)
    t = scaled[:, :LOWRANK] * sig[:, :LOWRANK]
    gamma = 1.0 + plan.norm.double().view(HIDDEN, 4).T
    up = plan.up.double().view(4, HIDDEN, LOWRANK)
    mixed = sum(
        torch.sigmoid(t @ up[b].T) * r[:, b] * inv[:, b : b + 1] * gamma[b]
        for b in range(4)
    )
    return mixed / 4, 2.0 * sig[:, LOWRANK:]


def _run(plan, residual, sums):
    from sglang.kernels.ops.gemm.hc_sum_ws import hc_gamma, hc_sum_ws_mix

    return hc_sum_ws_mix(
        residual, sums, plan.packed_down, plan.up, hc_gamma(plan.norm), EPS
    )


@pytest.mark.parametrize("rows", [25, 64, 200, 416, 1000, 4100])
def test_matches_flashinfer_chain_and_reference(rows):
    plan = _plan()
    residual = _residual(rows, rows)
    sums = plan.bootstrap(residual)
    mixed, alpha, next_sums = _run(plan, residual, sums)
    ws = plan.allocate_workspace(rows)
    base_mixed, base_alpha = plan.mix(residual, sums, ws)
    ref_mixed, ref_alpha = _reference(plan, residual, sums)
    assert bool((next_sums == 0).all())
    assert torch.equal(alpha, base_alpha)
    assert torch.equal(mixed, base_mixed)
    new_err = (mixed.double() - ref_mixed).abs().max()
    base_err = (base_mixed.double() - ref_mixed).abs().max()
    assert new_err <= base_err + 1e-6


def test_cuda_graph_replay_with_new_inputs():
    plan = _plan()
    rows = 416
    residual = _residual(rows, 1)
    sums = plan.bootstrap(residual)
    _run(plan, residual, sums)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mixed, alpha, next_sums = _run(plan, residual, sums)
    ws = plan.allocate_workspace(rows)
    for seed in (2, 3, 4):
        residual.copy_(_residual(rows, seed))
        plan.bootstrap(residual, out_sum=sums)
        next_sums.fill_(3.0)
        graph.replay()
        base_mixed, base_alpha = plan.mix(residual, sums, ws)
        torch.cuda.synchronize()
        assert bool((next_sums == 0).all())
        assert torch.equal(mixed, base_mixed)
        assert torch.equal(alpha, base_alpha)


def test_adapter_routes_rows_through_ws_mix(monkeypatch):
    import sglang.kernels.ops.gemm.hc_sum_ws as ws_mod

    calls = []
    real = ws_mod.hc_sum_ws_mix

    def spy(*args, **kwargs):
        calls.append(args[0].shape[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(ws_mod, "hc_sum_ws_mix", spy)
    monkeypatch.setenv("SGLANG_ENABLE_FLASHINFER_HC_SUM", "1")
    monkeypatch.setenv("SGLANG_ENABLE_HC_SUM_WS_MIX", "1")
    from sglang.srt.layers.hyperconnection import HyperConnectionConfig
    from sglang.srt.layers.hyperconnection_flashinfer import FlashInferGatedResidual

    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=HIDDEN,
        params_dtype=torch.bfloat16,
        hc_lowrank=LOWRANK,
        rms_norm_eps=EPS,
        hc_per_branch_norm=True,
    )
    module = FlashInferGatedResidual(config).cuda()
    plan = _plan()
    with torch.no_grad():
        module.packed_down_weight.copy_(plan.packed_down)
        module.packed_norm_weight.copy_(plan.norm.view(HIDDEN, 4))
        module.input_mix_weight_up.weight.copy_(plan.up)
    module._up_loaded = True
    module._native_weights = {"down", "norm"}
    module._finish_initial_load()
    module.bind()
    rows = 1100
    residual = _residual(rows, 5)
    mixed, context = module.mix(residual)
    state = context.apply(torch.randn(rows, HIDDEN, device="cuda").bfloat16())
    sums = plan.bootstrap(residual)
    ws = plan.allocate_workspace(rows)
    base_mixed, _ = plan.mix(residual, sums, ws)
    assert calls == [rows]
    assert torch.equal(mixed, base_mixed)
    assert state.residual.shape == residual.shape
