"""qsa_sparse_prefill_sm100 against the Triton chunk-prefill path it replaces, and the
backend wiring, on Qwen4-Exp QSA shapes."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.qsa_sparse_prefill import qsa_sparse_prefill_sm100
from sglang.srt.layers.attention.qsa.sparse_attn import (
    pack_qsa_prefill_kv,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.utils import is_sm100_supported
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(not is_sm100_supported(), reason="needs tcgen05")

Q_HEADS, KV_HEADS, HEAD_DIM, PAGE, RATIO, BLOCKS = 24, 2, 256, 64, 4, 512
SCALE = HEAD_DIM**-0.5


def _make_inputs(seqs, seed, repeat_a_position=False):
    """``seqs``: (cached prefix, new rows) per sequence. Each row selects up to
    512 distinct 4-token blocks before its position, about half of them shared
    with the rows around it, then the tokens of its own partial block."""
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen, device="cuda")

    rows = []
    for prefix, extend in seqs:
        visible = prefix + torch.arange(extend, device="cuda") + 1
        full = (visible // RATIO)[:, None]
        shared = rand(extend // 16 + 1, BLOCKS).repeat_interleave(16, 0)[:extend]
        pick = torch.where(rand(extend, BLOCKS) < 0.5, shared, rand(extend, BLOCKS))
        block = (pick * full).long().sort(1).values
        keep = full.expand(-1, BLOCKS) > 0
        keep[:, 1:] &= block[:, 1:] != block[:, :-1]
        tokens = block[:, :, None] * RATIO + torch.arange(RATIO, device="cuda")
        tail = full * RATIO + torch.arange(RATIO - 1, device="cuda")
        tokens = torch.cat([tokens.flatten(1), tail], 1)
        keep = torch.cat(
            [
                keep[:, :, None].expand(-1, -1, RATIO).flatten(1),
                tail < visible[:, None],
            ],
            1,
        )
        order = (~keep).to(torch.int8).sort(dim=1, stable=True).indices
        rows.append(torch.where(keep, tokens, -1).gather(1, order))
    indices = torch.cat(rows).int().contiguous()
    if repeat_a_position:
        indices[-1, 1] = indices[-1, 0]
    extend = [e for _, e in seqs]
    lens = [p + e for p, e in seqs]
    pages_per_seq = [(n + PAGE - 1) // PAGE for n in lens]
    pages = sum(pages_per_seq) + 2
    order = torch.randperm(pages - 1, generator=gen, device="cuda") + 1
    req_to_token = torch.zeros(
        len(seqs) + 1, max(pages_per_seq) * PAGE, dtype=torch.int32, device="cuda"
    )
    start = 0
    for i, n in enumerate(pages_per_seq):
        page_ids = order[start : start + n]
        start += n
        req_to_token[i + 1, : n * PAGE] = (
            page_ids[:, None] * PAGE + torch.arange(PAGE, device="cuda")
        ).flatten()
    slots = pages * PAGE
    return SimpleNamespace(
        q=torch.randn(
            sum(extend), Q_HEADS, HEAD_DIM, generator=gen, device="cuda"
        ).bfloat16(),
        k_cache=(
            torch.randn(slots, KV_HEADS, HEAD_DIM, generator=gen, device="cuda") * 0.3
        ).to(torch.float8_e4m3fn),
        v_cache=torch.randn(slots, KV_HEADS, HEAD_DIM, generator=gen, device="cuda").to(
            torch.float8_e4m3fn
        ),
        req_to_token=req_to_token,
        req_indices=torch.arange(1, len(seqs) + 1, device="cuda"),
        indices=indices,
        extend=extend,
        lens=lens,
        cu_q=F.pad(torch.tensor(extend, device="cuda").cumsum(0), (1, 0)).int(),
        lens_t=torch.tensor(lens, dtype=torch.int32, device="cuda"),
    )


def _pack(x, **dtypes):
    cu_k = F.pad(x.lens_t.cumsum(0), (1, 0)).int()
    packed = pack_qsa_prefill_kv(
        k=x.k_cache,
        v=x.v_cache,
        req_to_token=x.req_to_token,
        req_indices=x.req_indices,
        cu_k=cu_k,
        total_k=sum(x.lens),
        max_k=max(x.lens),
        **dtypes,
    )
    return cu_k, *packed


def _run(x, rows_per_group=None):
    cu_k, packed_k, packed_v = _pack(x, value_dtype=torch.bfloat16)
    return qsa_sparse_prefill_sm100(
        x.q,
        packed_k,
        packed_v,
        x.indices,
        x.cu_q,
        cu_k,
        x.lens_t,
        SCALE,
        extend_lens=x.extend,
        max_seq_len=max(x.lens),
        rows_per_group=rows_per_group,
    )


def _triton(x):
    cu_k, packed_k, packed_v = _pack(x, output_dtype=x.q.dtype)
    return sparse_gqa_fwd_interface_triton_ck(
        x.q, packed_k, packed_v, x.indices, x.cu_q, cu_k, x.lens_t, SCALE
    )


@pytest.mark.parametrize(
    "seqs",
    [
        [(0, 40)],
        [(6000, 133)],
        [(0, 21), (300, 64), (7000, 131), (1, 8), (2500, 1)],
        [(40000, 300), (1000, 77)],
    ],
)
@pytest.mark.parametrize("rows_per_group", [4, 8])
def test_matches_triton_chunk_prefill(seqs, rows_per_group):
    """The first rows of a sequence (fewer positions than top-k), a cached
    prefix, several sequences in one batch, groups cut short at the end of a
    sequence, and -1 padding."""
    x = _make_inputs(seqs, seed=len(seqs) + seqs[0][1])
    torch.testing.assert_close(
        _run(x, rows_per_group), _triton(x), atol=1e-2, rtol=2e-2
    )


@pytest.mark.parametrize("rows_per_group", [4, 8])
@pytest.mark.parametrize("key_byte", [0x68, 0x78])
def test_scores_far_above_the_first_tiles_move_the_reference(rows_per_group, key_byte):
    """The positions from 3000 on score 2^92 (e4m3 0x68 = 64) or 2^369 (0x78 =
    256) times the ones before them: the softmax reference moves in the middle
    of a group's list, in the tile where the scores first cross it."""
    x = _make_inputs([(6000, 70)], seed=7)
    x.q[:, :, 0] = 16.0
    positions = torch.arange(x.lens[0], device="cuda")
    slots = x.req_to_token[1, : x.lens[0]].long()
    x.k_cache.view(torch.uint8)[slots, :, 0] = torch.where(
        positions >= 3000, key_byte, 0
    )[:, None].to(torch.uint8)
    torch.testing.assert_close(
        _run(x, rows_per_group), _triton(x), atol=1e-2, rtol=2e-2
    )


def test_a_repeated_position_keeps_the_rows_of_its_group_apart():
    x = _make_inputs([(3000, 24)], seed=3, repeat_a_position=True)
    torch.testing.assert_close(_run(x), _triton(x), atol=1e-2, rtol=2e-2)


def test_rows_do_not_depend_on_the_rest_of_the_batch():
    x = _make_inputs([(2000, 48), (900, 32)], seed=5)
    alone = _make_inputs([(2000, 48), (900, 32)], seed=5)
    alone.q, alone.indices = x.q[:48], x.indices[:48]
    alone.extend, alone.lens = x.extend[:1], x.lens[:1]
    alone.cu_q, alone.lens_t = x.cu_q[:2], x.lens_t[:1]
    alone.req_indices = x.req_indices[:1]
    assert torch.equal(_run(x)[:48], _run(alone))


def test_cuda_graph_replay_matches_eager():
    x = _make_inputs([(5000, 70), (100, 40)], seed=9)
    eager = _run(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = _run(x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_routes_extend_to_sm100_kernel(monkeypatch, enabled):
    """With SGLANG_ENABLE_QSA_SPARSE_PREFILL_SM100, an extend of at least the
    row threshold reads the fp8 pool in qsa_sparse_prefill_sm100, with or
    without a cached prefix; otherwise the Triton kernels run."""
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
    with envs.SGLANG_ENABLE_QSA_SPARSE_PREFILL_SM100.override(enabled):
        backend = module.QwenSparseAttnBackend(SimpleNamespace(token_to_kv_pool=pool))
    assert backend._use_sparse_prefill_sm100 is enabled
    monkeypatch.setattr(module, "_SPARSE_PREFILL_SM100_MIN_ROWS", 64)
    calls = []
    kernel = module.qsa_sparse_prefill_sm100
    monkeypatch.setattr(
        module,
        "qsa_sparse_prefill_sm100",
        lambda *a, **k: calls.append(a[0].shape[0]) or kernel(*a, **k),
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=None)
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=Q_HEADS,
        tp_k_head_num=KV_HEADS,
        head_dim=HEAD_DIM,
        scaling=SCALE,
    )
    for seqs, routed_rows in (
        ([(4000, 90), (700, 38)], [128]),
        ([(0, 70)], [128, 70]),
        ([(900, 40)], [128, 70]),
    ):
        x = _make_inputs(seqs, seed=11)
        backend.req_to_token_pool.req_to_token = x.req_to_token
        backend.token_to_kv_pool = routed(
            SimpleNamespace(
                get_key_buffer=lambda _: x.k_cache,
                get_value_buffer=lambda _: x.v_cache,
                set_kv_buffer=lambda *a, **k: None,
            )
        )
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            req_pool_indices=x.req_indices,
            out_cache_loc=None,
            extend_seq_lens=torch.tensor(x.extend, device="cuda"),
            extend_seq_lens_cpu=x.extend,
            seq_lens=x.lens_t,
            seq_lens_cpu=x.lens,
        )
        slots = torch.cat(
            [
                x.req_to_token[i + 1, n - e : n]
                for i, (n, e) in enumerate(zip(x.lens, x.extend))
            ]
        ).long()
        out = backend.forward_extend(
            x.q.flatten(1),
            x.k_cache[slots].bfloat16(),
            x.v_cache[slots].bfloat16(),
            layer,
            batch,
            topk_indices=x.indices,
        )
        torch.testing.assert_close(
            out.reshape(x.q.shape), _triton(x), atol=1e-2, rtol=2e-2
        )
        assert calls == (routed_rows if enabled else [])
