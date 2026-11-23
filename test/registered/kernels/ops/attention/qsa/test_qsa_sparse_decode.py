"""QSA sparse decode kernels (split-K and tcgen05) against the torch reference, and
the backend wiring, on Qwen4-Exp QSA shapes."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.qsa_sparse_decode import (
    qsa_sparse_decode,
    qsa_sparse_decode_sm100,
)
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention_reference
from sglang.srt.utils import is_sm100_supported, is_sm120
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not (is_sm100_supported() or is_sm120()),
    reason="qsa_sparse_decode is tested on SM100 and SM120",
)

Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, RATIO = 24, 2, 256, 64, 4
TOPK = 2048 + RATIO - 1
SM100_IMPLS = ["sm100", "sm100_fp8_pv"]
IMPLS = ["split_k"] + [
    pytest.param(
        impl,
        marks=pytest.mark.skipif(not is_sm100_supported(), reason="needs tcgen05"),
    )
    for impl in SM100_IMPLS
]


def _make_inputs(requests, width, context, seed, shared=False):
    """``shared``: the rows of a request select the same distinct blocks, each
    row in its own order (a verify chain); otherwise random blocks per row."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    lens = torch.randint(
        context // 2, context, (requests,), generator=gen, device="cuda"
    )
    pages_per_req = (lens + width + PAGE - 1) // PAGE
    pages = int(pages_per_req.sum()) + 2
    order = (torch.randperm(pages - 1, generator=gen, device="cuda") + 1).int()
    req_to_token = torch.zeros(
        requests + 1, context + 2 * PAGE, dtype=torch.int32, device="cuda"
    )
    start = 0
    for r in range(requests):
        n = int(pages_per_req[r])
        page_ids = order[start : start + n]
        start += n
        req_to_token[r + 1, : n * PAGE] = (
            page_ids[:, None] * PAGE + torch.arange(PAGE, device="cuda")
        ).flatten()
    rows = requests * width
    row_req = torch.arange(1, requests + 1, device="cuda").repeat_interleave(width)
    seq_lens = (
        lens.repeat_interleave(width)
        + torch.arange(width, device="cuda").repeat(requests)
        + 1
    ).int()
    slots = pages * PAGE
    k_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, generator=gen, device="cuda")
    v_cache = torch.randn(slots, KV_HEADS, HEAD_DIM, generator=gen, device="cuda")
    q = torch.randn(rows, Q_HEADS, HEAD_DIM, generator=gen, device="cuda")
    blocks = (
        torch.rand(rows, 512, generator=gen, device="cuda")
        * (seq_lens // RATIO)[:, None]
    ).long()
    if shared:
        order = torch.rand(rows, 512, generator=gen, device="cuda").argsort(1)
        first = torch.rand(
            requests, context // RATIO // 2, generator=gen, device="cuda"
        )
        blocks = first.argsort(1)[:, :512].repeat_interleave(width, 0).gather(1, order)
    indices = (blocks[:, :, None] * RATIO + torch.arange(RATIO, device="cuda")).flatten(
        1
    )
    tail = (seq_lens.long() // RATIO * RATIO)[:, None] + torch.arange(
        RATIO - 1, device="cuda"
    )
    tail = torch.where(tail < seq_lens[:, None], tail, -1)
    indices = torch.cat([indices, tail], dim=1)
    indices[:, 7::97] = -1
    return SimpleNamespace(
        q=q.bfloat16(),
        k_cache=k_cache.to(torch.float8_e4m3fn),
        v_cache=v_cache.to(torch.float8_e4m3fn),
        req_to_token=req_to_token,
        row_req=row_req,
        indices=indices.int().contiguous(),
        seq_lens=seq_lens,
        width=width,
    )


def _run(x, rows=slice(None), impl="split_k"):
    pool_read = dict(
        q=x.q[rows],
        k_cache=x.k_cache,
        v_cache=x.v_cache,
        req_to_token=x.req_to_token,
        row_req_pool_indices=x.row_req[rows],
        indices=x.indices[rows],
        seq_lens=x.seq_lens[rows],
        sm_scale=HEAD_DIM**-0.5,
    )
    if impl in SM100_IMPLS:
        return qsa_sparse_decode_sm100(
            **pool_read, rows_per_request=x.width, fp8_pv=impl == "sm100_fp8_pv"
        )
    return qsa_sparse_decode(**pool_read)


def _reference(x):
    valid = (x.indices >= 0) & (x.indices < x.seq_lens[:, None])
    slots = x.req_to_token[x.row_req[:, None], x.indices.clamp(min=0)]
    slots = torch.where(valid, slots, -1)
    return qsa_sparse_attention_reference(
        x.q, x.k_cache, x.v_cache, slots, HEAD_DIM**-0.5
    )


@pytest.mark.parametrize("requests,width", [(1, 1), (3, 4), (24, 1), (8, 4), (40, 1)])
@pytest.mark.parametrize("context", [300, 6000])
@pytest.mark.parametrize("impl", IMPLS)
def test_matches_sparse_attention_reference(requests, width, context, impl):
    x = _make_inputs(requests, width, context, seed=requests * 7 + width + context)
    if width > 1:
        x.row_req = x.row_req.int()
    torch.testing.assert_close(_run(x, impl=impl), _reference(x), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize(
    "requests,width,shared",
    [(40, 4, True), (40, 4, False), (24, 4, True), (32, 3, True), (48, 2, True)],
)
@pytest.mark.parametrize("impl", SM100_IMPLS)
def test_sm100_rows_of_a_request_share_a_cta(requests, width, shared, impl):
    """Above one CTA per SM, 2, 3 or 4 rows of a request are merged: slots
    several rows selected are read once (``shared``), and rows that list a
    position twice keep their columns apart (random blocks repeat)."""
    if not is_sm100_supported():
        pytest.skip("needs tcgen05")
    x = _make_inputs(requests, width, 6000, seed=width, shared=shared)
    torch.testing.assert_close(_run(x, impl=impl), _reference(x), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("impl", SM100_IMPLS)
def test_sm100_group_of_rows_from_different_requests(impl):
    """``rows_per_request`` only sizes the groups: rows of two requests in one
    group are each read through their own request."""
    if not is_sm100_supported():
        pytest.skip("needs tcgen05")
    x = _make_inputs(80, 2, 6000, seed=17, shared=True)
    x.width = 4
    torch.testing.assert_close(_run(x, impl=impl), _reference(x), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("requests", [4, 24, 40])
@pytest.mark.parametrize("impl", SM100_IMPLS)
def test_sm100_score_far_above_the_first_tile(requests, impl):
    """Scores that keep rising by more than the rescale distance move the
    per-head softmax reference and rescale the accumulated output."""
    if not is_sm100_supported():
        pytest.skip("needs tcgen05")
    x = _make_inputs(requests, 4, 6000, seed=7, shared=True)
    x.q = x.q * 30
    torch.testing.assert_close(_run(x, impl=impl), _reference(x), atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("impl", IMPLS)
def test_rows_without_valid_columns_and_empty_blocks(impl):
    """Whole 32-column blocks without a valid column are skipped before and
    after the first valid block of a split, and a row with no valid column
    outputs zeros."""
    x = _make_inputs(4, 1, 300, seed=11)
    x.indices[:, 40:300] = -1
    x.indices[:, 340:] = -1
    x.indices[0] = -1
    out = _run(x, impl=impl)
    assert torch.count_nonzero(out[0]) == 0
    torch.testing.assert_close(out, _reference(x), atol=5e-3, rtol=1e-2)


@pytest.mark.parametrize("impl", IMPLS)
def test_deterministic_and_batch_invariant(impl):
    """A row's output does not depend on the run or on the other rows. The
    sm100 kernel merges rows of a request above one CTA per SM, so its slice
    keeps whole requests and the same group size."""
    x = _make_inputs(32, 4, 6000, seed=3)
    out = _run(x, impl=impl)
    assert torch.equal(out, _run(x, impl=impl))
    part = slice(0, 80) if impl in SM100_IMPLS else slice(20, 28)
    if impl == "sm100_fp8_pv":
        torch.testing.assert_close(out[part], _run(x, part, impl), atol=5e-3, rtol=1e-2)
    else:
        assert torch.equal(out[part], _run(x, part, impl))


@pytest.mark.parametrize("cache", ["k_cache", "v_cache"])
@pytest.mark.parametrize("impl", IMPLS)
def test_nan_in_selected_kv_propagates(cache, impl):
    """The fast fp8 -> bf16 conversion of the split-K kernel maps the e4m3 NaN
    codes to +-480; fragments holding one are converted again so NaN reaches
    the output as in the reference."""
    x = _make_inputs(3, 4, 6000, seed=5)
    raw = getattr(x, cache).view(torch.uint8)
    raw[5::97, :, 17] = 0x7F
    raw[11::89, :, 200] = 0xFF
    out, ref = _run(x, impl=impl), _reference(x)
    assert torch.isnan(ref).any()
    assert torch.equal(torch.isnan(out), torch.isnan(ref))


@pytest.mark.parametrize("cache", ["k_cache", "v_cache"])
@pytest.mark.parametrize("impl", SM100_IMPLS)
def test_sm100_nan_stays_in_the_rows_that_selected_it(cache, impl):
    """A NaN in a K or V row that one row of a merged request selected must
    not reach the request's other rows (0 x NaN is NaN on the tensor core)."""
    if not is_sm100_supported():
        pytest.skip("needs tcgen05")
    x = _make_inputs(40, 4, 6000, seed=13, shared=True)
    row = 8
    pos = int(x.indices[row, 100])
    for other in range(row + 1, row + 4):
        x.indices[other][x.indices[other] == pos] = -1
    slot = int(x.req_to_token[x.row_req[row], pos])
    getattr(x, cache).view(torch.uint8)[slot, :, 37] = 0x7F
    out, ref = _run(x, impl=impl), _reference(x)
    assert torch.isnan(ref[row]).any() and not torch.isnan(ref[row + 1 : row + 4]).any()
    assert torch.equal(torch.isnan(out), torch.isnan(ref))


@pytest.mark.parametrize("impl", IMPLS)
def test_cuda_graph_replay_matches_eager(impl):
    x = _make_inputs(16, 4, 6000, seed=9)
    eager = _run(x, impl=impl)
    _run(x, impl=impl)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(x, impl=impl)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


def test_sm100_fp8_pv_split_lists_graph_replay():
    """Below one CTA per SM the fp8 PV kernel splits the longest lists over the
    idle SMs and merges the halves; graph replays reuse the handshake counters."""
    if not is_sm100_supported():
        pytest.skip("needs tcgen05")
    x = _make_inputs(40, 4, 6000, seed=21, shared=True)
    eager = _run(x, impl="sm100_fp8_pv")
    torch.testing.assert_close(eager, _reference(x), atol=5e-3, rtol=1e-2)
    assert torch.equal(eager, _run(x, impl="sm100_fp8_pv"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(x, impl="sm100_fp8_pv")
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)


def _fp8_backend(enabled):
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        UnquantizedKVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.quant_method = UnquantizedKVCacheMethod()
    pool.dtype = torch.float8_e4m3fn
    with envs.SGLANG_ENABLE_QSA_SPARSE_DECODE_CUDA.override(enabled):
        backend = module.QwenSparseAttnBackend(SimpleNamespace(token_to_kv_pool=pool))
    return module, backend


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_routes_fp8_decode_to_kernel(monkeypatch, enabled):
    """With the env var set on SM120, decode / verify skip the fused
    pack-for-trtllm path and run qsa_sparse_decode on the expanded top-k;
    without it, or on SM100, nothing changes."""
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.test.mem_cache_utils import routed

    module, backend = _fp8_backend(enabled)
    enabled = enabled and is_sm120()
    assert backend._use_sparse_decode_cuda is enabled
    x = _make_inputs(2, 1, 6000, seed=1)
    backend.qsa_profile = None
    backend.req_to_token_pool = SimpleNamespace(req_to_token=x.req_to_token)
    backend.token_to_kv_pool = routed(
        SimpleNamespace(
            get_key_buffer=lambda _: x.k_cache, get_value_buffer=lambda _: x.v_cache
        )
    )
    metadata = SimpleNamespace(
        row_req_pool_indices=None, sequence_lengths=x.seq_lens, is_cuda_graph=False
    )
    backend._resolve_metadata = lambda _: metadata
    marker = object()
    monkeypatch.setattr(
        module, "_resolve_trtllm_sparse_decode", lambda: lambda *a, **k: marker
    )
    backend._forward_trtllm_sparse = lambda *args, **kwargs: marker
    batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=x.row_req,
        out_cache_loc=torch.ones(2, dtype=torch.int64, device="cuda"),
    )
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=Q_HEADS, head_dim=HEAD_DIM, scaling=HEAD_DIM**-0.5
    )
    if enabled:
        assert (
            backend._try_fused_kv_attention(x.q, x.q, x.q, layer, batch, x.indices)
            is None
        )
    out = backend._forward_paged_attention(x.q, layer, batch, x.indices)
    if enabled:
        assert torch.equal(out, _run(x).reshape(2, -1))
    else:
        assert out is marker


@pytest.mark.parametrize("mode,width", [("DECODE", 1), ("TARGET_VERIFY", 4)])
@pytest.mark.parametrize("impl", SM100_IMPLS)
def test_backend_routes_fp8_decode_to_sm100_kernel(mode, width, impl):
    """With SGLANG_ENABLE_QSA_SPARSE_DECODE_SM100 on SM100, decode / verify run
    qsa_sparse_decode_sm100, a verify chain as the rows of one request; with
    SGLANG_QSA_SPARSE_DECODE_SM100_FP8_PV also on, its fp8 PV product."""
    if not is_sm100_supported():
        pytest.skip("needs tcgen05")
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention import qwen_sparse_attn_backend as module
    from sglang.srt.layers.quantization.fp4_kv_cache_quant_method import (
        UnquantizedKVCacheMethod,
    )
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.test.mem_cache_utils import routed

    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.quant_method = UnquantizedKVCacheMethod()
    pool.dtype = torch.float8_e4m3fn
    with (
        envs.SGLANG_ENABLE_QSA_SPARSE_DECODE_SM100.override(True),
        envs.SGLANG_QSA_SPARSE_DECODE_SM100_FP8_PV.override(impl == "sm100_fp8_pv"),
    ):
        backend = module.QwenSparseAttnBackend(SimpleNamespace(token_to_kv_pool=pool))
    assert backend._use_sparse_decode_sm100 and not backend._use_sparse_decode_cuda
    x = _make_inputs(24, width, 6000, seed=1, shared=True)
    backend.qsa_profile = None
    backend.req_to_token_pool = SimpleNamespace(req_to_token=x.req_to_token)
    backend.token_to_kv_pool = routed(
        SimpleNamespace(
            get_key_buffer=lambda _: x.k_cache, get_value_buffer=lambda _: x.v_cache
        )
    )
    metadata = SimpleNamespace(
        row_req_pool_indices=x.row_req, sequence_lengths=x.seq_lens, is_cuda_graph=False
    )
    backend._resolve_metadata = lambda _: metadata
    batch = SimpleNamespace(
        forward_mode=ForwardMode[mode],
        req_pool_indices=x.row_req[::width],
        out_cache_loc=torch.ones(x.q.shape[0], dtype=torch.int64, device="cuda"),
        spec_info=SimpleNamespace(draft_token_num=width),
    )
    layer = SimpleNamespace(
        layer_id=0, tp_q_head_num=Q_HEADS, head_dim=HEAD_DIM, scaling=HEAD_DIM**-0.5
    )
    assert (
        backend._try_fused_kv_attention(x.q, x.q, x.q, layer, batch, x.indices) is None
    )
    out = backend._forward_paged_attention(x.q, layer, batch, x.indices)
    assert torch.equal(out, _run(x, impl=impl).reshape(x.q.shape[0], -1))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
