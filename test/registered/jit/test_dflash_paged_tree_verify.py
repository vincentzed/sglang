import sys

import pytest
import torch

from sglang.jit_kernel.dflash_paged_tree_verify import dflash_paged_tree_verify_attn
from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.dense_attention import (
    DenseAttentionCase,
    build_dense_attention_fixture,
    run_dense_forward,
)

register_cuda_ci(est_time=30, suite="base-b-kernel-unit-1-gpu-b200")


def _sm100_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


pytestmark = pytest.mark.skipif(
    not _sm100_available(),
    reason="DFlash paged-tree FA4 verifier requires CUDA SM100+",
)


def _ancestor_path(parents: list[int], node: int) -> list[int]:
    path: list[int] = []
    cur = node
    while cur >= 0:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path


def _make_case(dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.Tensor | float]:
    device = "cuda"
    batch_size = 3
    tree_budget = 9
    page_size = 16
    num_q_heads = 8
    num_kv_heads = 2
    head_dim = 128
    num_pages = 16
    softcap = 5.0
    softmax_scale = head_dim**-0.5

    prefix_lens_cpu = [17, 31, 9]
    req_slot_bases = [0, 64, 128]
    full_lens_cpu = [x + tree_budget for x in prefix_lens_cpu]
    num_real_nodes_cpu = [9, 7, 4]
    parents_cpu = [
        [-1, 0, 0, 1, 1, 2, 2, 3, 5],
        [-1, 0, 1, 1, 0, 4, 4, 0, 0],
        [-1, 0, 0, 2, 0, 0, 0, 0, 0],
    ]

    max_pages = max((x + page_size - 1) // page_size for x in full_lens_cpu)
    page_table = torch.zeros(
        (batch_size, max_pages), dtype=torch.int32, device=device
    )
    prefix_slots: list[torch.Tensor] = []
    tree_slots: list[torch.Tensor] = []
    for req in range(batch_size):
        start_page = req_slot_bases[req] // page_size
        num_req_pages = (full_lens_cpu[req] + page_size - 1) // page_size
        page_table[req, :num_req_pages] = torch.arange(
            start_page,
            start_page + num_req_pages,
            dtype=torch.int32,
            device=device,
        )
        prefix_start = req_slot_bases[req]
        prefix_end = prefix_start + prefix_lens_cpu[req]
        prefix_slots.append(
            torch.arange(prefix_start, prefix_end, dtype=torch.long, device=device)
        )
        tree_slots.append(
            torch.arange(
                prefix_end,
                prefix_end + tree_budget,
                dtype=torch.long,
                device=device,
            )
        )
    tree_slots_tensor = torch.stack(tree_slots)

    compact_parts: list[torch.Tensor] = []
    compact_lens_cpu: list[int] = []
    ancestor_mask = torch.zeros(
        (batch_size, tree_budget, tree_budget), dtype=torch.int32, device=device
    )
    for req in range(batch_size):
        for node in range(tree_budget):
            if node < num_real_nodes_cpu[req]:
                path = _ancestor_path(parents_cpu[req], node)
            else:
                path = [node]
            ancestor_mask[req, node, path] = 1

            path_tensor = torch.tensor(path, dtype=torch.long, device=device)
            suffix = tree_slots_tensor[req].index_select(0, path_tensor)
            compact_row = torch.cat([prefix_slots[req], suffix])
            compact_parts.append(compact_row)
            compact_lens_cpu.append(int(compact_row.numel()))

    compact_kv_indices = torch.cat(compact_parts)
    compact_lens = torch.tensor(compact_lens_cpu, dtype=torch.int32, device=device)
    cu_q = torch.arange(
        0, batch_size * tree_budget + 1, dtype=torch.int32, device=device
    )
    cu_k = torch.empty(batch_size * tree_budget + 1, dtype=torch.int32, device=device)
    cu_k[0] = 0
    cu_k[1:] = torch.cumsum(compact_lens, dim=0)

    torch.manual_seed(1234)
    q = (
        torch.randn(
            batch_size * tree_budget,
            num_q_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        * 3.0
    ).to(dtype)
    k_cache = (
        torch.randn(
            num_pages,
            page_size,
            num_kv_heads,
            head_dim,
            device=device,
            dtype=torch.float32,
        )
        * 3.0
    ).to(dtype)
    v_cache = torch.randn(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.float32,
    ).to(dtype)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "page_table": page_table,
        "cache_seqlens": torch.tensor(full_lens_cpu, dtype=torch.int32, device=device),
        "prefix_lens": torch.tensor(prefix_lens_cpu, dtype=torch.int32, device=device),
        "ancestor_mask": ancestor_mask,
        "compact_kv_indices": compact_kv_indices,
        "cu_q": cu_q,
        "cu_k": cu_k,
        "max_seqlen_k": int(compact_lens.max().item()),
        "softmax_scale": softmax_scale,
        "softcap": softcap,
    }


def _compact_fa4_reference(case: dict[str, torch.Tensor | float], softcap):
    k_cache = case["k_cache"]
    v_cache = case["v_cache"]
    compact_kv_indices = case["compact_kv_indices"]
    flat_k = k_cache.reshape(-1, k_cache.shape[2], k_cache.shape[3])
    flat_v = v_cache.reshape(-1, v_cache.shape[2], v_cache.shape[3])
    compact_k = flat_k.index_select(0, compact_kv_indices)
    compact_v = flat_v.index_select(0, compact_kv_indices)

    return flash_attn_varlen_func(
        q=case["q"],
        k=compact_k,
        v=compact_v,
        cu_seqlens_q=case["cu_q"],
        cu_seqlens_k=case["cu_k"],
        max_seqlen_q=1,
        max_seqlen_k=case["max_seqlen_k"],
        softmax_scale=case["softmax_scale"],
        causal=False,
        softcap=softcap,
        num_splits=1,
    )


def _expand_gqa(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    if x.shape[0] == num_heads:
        return x
    assert num_heads % x.shape[0] == 0
    return x.repeat_interleave(num_heads // x.shape[0], dim=0)


def _softcapped_tree_reference(
    fixture,
    ancestor_mask: torch.Tensor,
    *,
    softcap: float,
) -> torch.Tensor:
    case = fixture.case
    module = fixture.reference_module
    dtype = fixture.input_hidden.dtype
    q, k, v = module.project_qkv(fixture.input_hidden)
    q = q.view(-1, case.num_heads, module.head_dim)
    k = k.view(-1, case.num_kv_heads, module.head_dim)
    v = v.view(-1, case.num_kv_heads, module.head_dim)
    outputs = []
    offset = 0
    for req_idx, prefix in enumerate(fixture.prefix_hidden):
        tree_budget = case.input_lens[req_idx]
        req_q = q[offset : offset + tree_budget]
        req_k = k[offset : offset + tree_budget]
        req_v = v[offset : offset + tree_budget]
        offset += tree_budget

        _, prefix_k, prefix_v = module.project_qkv(prefix)
        prefix_k = prefix_k.view(-1, case.num_kv_heads, module.head_dim)
        prefix_v = prefix_v.view(-1, case.num_kv_heads, module.head_dim)
        all_k = torch.cat([prefix_k, req_k], dim=0)
        all_v = torch.cat([prefix_v, req_v], dim=0)
        prefix_allowed = torch.ones(
            prefix_k.shape[0], dtype=torch.bool, device=ancestor_mask.device
        )

        for node_idx, query in enumerate(req_q):
            tree_allowed = ancestor_mask[req_idx, node_idx].to(torch.bool)
            allowed = torch.cat([prefix_allowed, tree_allowed], dim=0)
            keys = _expand_gqa(all_k[allowed].movedim(0, 1), case.num_heads)
            values = _expand_gqa(all_v[allowed].movedim(0, 1), case.num_heads)
            scores = (
                torch.einsum("hd,hkd->hk", query.float(), keys.float())
                * module.scaling
            )
            scores = softcap * torch.tanh(scores / softcap)
            probs = torch.softmax(scores, dim=-1)
            out = torch.einsum("hk,hkd->hd", probs, values.float())
            outputs.append(out.reshape(-1))

    attn_output = torch.stack(outputs, dim=0).to(dtype)
    return module.reconstruct_output(attn_output)


def test_dflash_paged_tree_verify_matches_compact_fa4_softcap_bf16():
    case = _make_case(dtype=torch.bfloat16)

    compact_out = _compact_fa4_reference(case, softcap=case["softcap"])
    paged_out = dflash_paged_tree_verify_attn(
        q=case["q"],
        k_cache=case["k_cache"],
        v_cache=case["v_cache"],
        page_table=case["page_table"],
        cache_seqlens=case["cache_seqlens"],
        prefix_lens=case["prefix_lens"],
        ancestor_mask=case["ancestor_mask"],
        softmax_scale=case["softmax_scale"],
        softcap=case["softcap"],
        num_splits=1,
    )

    assert paged_out.shape == compact_out.shape
    assert paged_out.dtype == compact_out.dtype == torch.bfloat16
    assert torch.isfinite(paged_out).all()

    no_softcap_out = _compact_fa4_reference(case, softcap=None)
    no_softcap_diff = (no_softcap_out.float() - compact_out.float()).abs().max().item()
    assert no_softcap_diff > 1.0e-2

    diff = (paged_out.float() - compact_out.float()).abs()
    assert diff.max().item() == 0.0
    assert diff.mean().item() == 0.0
    torch.testing.assert_close(
        paged_out.float(), compact_out.float(), atol=0.0, rtol=0.0
    )


class _PytestCase:
    def skipTest(self, reason: str) -> None:
        pytest.skip(reason)


def test_dflash_paged_tree_verify_backend_path_matches_reference_softcap_bf16():
    tree_budget = 5
    case = DenseAttentionCase(
        name="dflash_paged_tree_verify_backend_path",
        backend="fa4",
        forward_mode=ForwardMode.TARGET_VERIFY,
        num_heads=4,
        num_kv_heads=2,
        page_size=16,
        prefix_lens=(17, 31),
        extend_lens=(tree_budget, tree_budget),
    )
    fixture = build_dense_attention_fixture(
        _PytestCase(),
        case,
        head_dim=64,
        hidden_size=256,
        dtype=torch.bfloat16,
        device="cuda",
    )
    softcap = 5.0
    fixture.actual_module.attn.logit_cap = softcap

    parents_by_req = (
        (-1, 0, 0, 1, 2),
        (-1, 0, 1, 1, 0),
    )
    ancestor_mask = torch.zeros(
        (case.batch_size, tree_budget, tree_budget),
        dtype=torch.int32,
        device="cuda",
    )
    for req_idx, parents in enumerate(parents_by_req):
        for node_idx in range(tree_budget):
            ancestor_mask[req_idx, node_idx, _ancestor_path(list(parents), node_idx)] = 1

    batch = fixture.forward_batch
    prefix_lens = torch.tensor(case.prefix_lens, dtype=torch.int32, device="cuda")
    req_pool_indices = torch.arange(
        case.batch_size, dtype=torch.int32, device="cuda"
    ).repeat_interleave(tree_budget)
    batch.batch_size = case.batch_size * tree_budget
    batch.req_pool_indices = req_pool_indices
    batch.seq_lens = prefix_lens.repeat_interleave(tree_budget)
    batch.seq_lens_cpu = batch.seq_lens.to(device="cpu", dtype=torch.int32)
    batch.seq_lens_sum = int(batch.seq_lens_cpu.sum().item())
    batch.spec_info = DFlashVerifyInput(
        draft_token=batch.input_ids,
        positions=batch.positions,
        draft_token_num=1,
        topk=1,
        custom_mask=None,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        paged_tree_prefix_lens=prefix_lens,
        paged_tree_ancestor_mask=ancestor_mask,
    )

    expected = _softcapped_tree_reference(
        fixture,
        ancestor_mask,
        softcap=softcap,
    )
    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_forward_metadata(batch)
        actual = run_dense_forward(fixture, batch, {"input_hidden": fixture.input_hidden})

    assert fixture.backend.forward_metadata.dflash_paged_tree
    torch.testing.assert_close(actual, expected, atol=4.0e-2, rtol=4.0e-2)

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=fixture.backend)):
        fixture.backend.init_cuda_graph_state(
            max_bs=batch.batch_size,
            max_num_tokens=batch.input_ids.numel(),
        )
        fixture.backend.init_forward_metadata_out_graph(batch, in_capture=True)
        fixture.backend.init_forward_metadata_in_graph(batch)
        graph_metadata_actual = run_dense_forward(
            fixture,
            batch,
            {"input_hidden": fixture.input_hidden},
        )

    assert fixture.backend.forward_metadata.dflash_paged_tree
    torch.testing.assert_close(
        graph_metadata_actual, expected, atol=4.0e-2, rtol=4.0e-2
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
