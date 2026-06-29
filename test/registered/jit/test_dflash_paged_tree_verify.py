import sys

import pytest
import torch

from sglang.jit_kernel.dflash_paged_tree_verify import dflash_paged_tree_verify_attn
from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func
from sglang.test.ci.ci_register import register_cuda_ci

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
