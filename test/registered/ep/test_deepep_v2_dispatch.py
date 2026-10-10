"""DeepEP dispatch/combine parity across prefill, decode, and CUDA graphs."""

import itertools
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=120, stage="base-c", runner_config="4-gpu-h100")


def _dequantize(output):
    x = output.hidden_states.float()
    scales = output.hidden_states_scale
    if scales is None:
        return x.to(torch.bfloat16)
    if scales.dtype == torch.int32:
        shifts = torch.arange(4, device=x.device) * 8
        exponents = (scales.unsqueeze(-1).to(torch.int64) >> shifts) & 255
        scales = torch.exp2(exponents.float() - 127).flatten(1)
    blocks = x.shape[1] // output.activation_scale_block_size
    return (
        (
            x.reshape(x.shape[0], blocks, output.activation_scale_block_size)
            * scales[:, :blocks].unsqueeze(-1)
        )
        .reshape_as(x)
        .to(torch.bfloat16)
    )


def _run_experts(impl, x, topk):
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
        DeepEPv2CombineInput,
    )

    output = impl.dispatch(x, topk)
    x = _dequantize(output)
    weights = output.topk_weights
    if output.is_expanded:
        psum = output.psum_num_recv_tokens_per_expert
        starts = torch.cat((psum.new_zeros(1), psum[:-1]))
        align = output.expert_alignment
        starts = (starts + align - 1) // align * align
        rows = torch.arange(x.shape[0], device=x.device)
        valid_experts = (rows[:, None] >= starts) & (rows[:, None] < psum)
        factors = (
            torch.arange(impl.num_local_experts, device=x.device)
            + dist.get_rank(impl.group) * impl.num_local_experts
            + 1
        ) / impl.num_experts
        row_factors = (valid_experts * factors).sum(dim=1)
        x = torch.where(
            valid_experts.any(dim=1)[:, None],
            x.float() * (weights * row_factors)[:, None],
            0,
        ).to(torch.bfloat16)
        weights = None
    else:
        factors = (
            output.topk_ids + dist.get_rank(impl.group) * impl.num_local_experts + 1
        ) / impl.num_experts
        local_weights = torch.where(output.topk_ids >= 0, weights * factors, 0).sum(
            dim=1
        )
        x = (x.float() * local_weights[:, None]).to(torch.bfloat16)
        weights = None
    return impl.combine(DeepEPv2CombineInput(x, weights))


def _expected_output(impl, x, topk):
    factors = (topk.topk_ids.float() + 1) / impl.num_experts
    return (x.float() * (topk.topk_weights * factors).sum(dim=1)[:, None]).to(
        torch.bfloat16
    )


def _worker(rank, world_size, port):
    import deep_ep

    try:
        from deep_ep import destroy_all_managed_nccl_comm
    except ImportError:
        from deep_ep.utils.comm import destroy_all_managed_nccl_comm

    from sglang.srt.environ import envs
    from sglang.srt.layers.moe.token_dispatcher.deepep_v2 import (
        DeepEPv2Buffer,
        _DeepEPv2Impl,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.moe.utils import DeepEPv2Fp8ScaleFormat
    from sglang.srt.runtime_context import get_context, get_forward

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        store=dist.TCPStore("127.0.0.1", port),
        rank=rank,
        world_size=world_size,
        device_id=torch.device("cuda", rank),
    )
    group = dist.group.WORLD
    published = get_context().override_server_args(model_path="dummy")
    published.install()
    major, _ = torch.cuda.get_device_capability()
    scale_formats = [False, True] if major >= 10 else [False]
    try:
        for fp8, extend, ue8m0 in itertools.product(
            (False, True), (False, True), scale_formats
        ):
            if not fp8 and ue8m0:
                continue
            get_forward().set("is_extend_in_batch", extend)
            with envs.SGLANG_DEEPEP_V2_ENABLE_PREFILL_EXPAND.override(
                False if major >= 10 else None
            ):
                impl = _DeepEPv2Impl(
                    group=group,
                    router_topk=2,
                    num_experts=world_size * 4,
                    num_local_experts=4,
                    hidden_size=512,
                    scale_format=DeepEPv2Fp8ScaleFormat(tma_aligned=ue8m0, ue8m0=ue8m0),
                    num_max_dispatch_tokens_per_rank=32,
                    use_fp8_dispatch=fp8,
                )
            # An idle rank must still participate in prefill and decode collectives.
            num_tokens = 0 if rank == world_size - 1 else 17 - rank
            x = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")
            rows = torch.arange(num_tokens, device="cuda")[:, None]
            ids = (
                (rows + rank + torch.arange(2, device="cuda") * 3) % (world_size * 4)
            ).to(deep_ep.topk_idx_t)
            weights = (
                torch.tensor([0.25, 0.5], device="cuda")
                .expand(num_tokens, -1)
                .contiguous()
            )
            topk = StandardTopKOutput(weights, ids, None)
            expected = _expected_output(impl, x, topk)

            out = _run_experts(impl, x, topk)
            torch.testing.assert_close(
                out, expected, rtol=0.08 if fp8 else 0.01, atol=0.02
            )
            if not extend:
                stream = torch.cuda.Stream()
                with torch.cuda.stream(stream):
                    _run_experts(impl, x, topk)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    captured = _run_experts(impl, x, topk)
                for _ in range(2):
                    x.mul_(0.5)
                    ids.add_(1).remainder_(impl.num_experts)
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        captured,
                        _expected_output(impl, x, topk),
                        rtol=0.08 if fp8 else 0.01,
                        atol=0.02,
                    )
            dist.barrier()
            del impl, out, topk
            DeepEPv2Buffer.destroy()
        print(f"rank {rank}: dispatch/combine and CUDA graph parity passed", flush=True)
        torch.cuda.synchronize()
        DeepEPv2Buffer.destroy()
        destroy_all_managed_nccl_comm()
    finally:
        dist.destroy_process_group()
        published.restore()


class TestDeepEPv2Dispatch(CustomTestCase):
    def test_dispatch_combine_and_graph(self):
        store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False)
        mp.spawn(
            _worker,
            args=(torch.cuda.device_count(), store.port),
            nprocs=torch.cuda.device_count(),
        )


if __name__ == "__main__":
    unittest.main()
