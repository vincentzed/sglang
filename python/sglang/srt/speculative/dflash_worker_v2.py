import logging
import math
import os
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from typing import List, Optional

import torch

from sglang.srt.distributed import get_tp_group
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    compute_position,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_info_v2 import DFlashDraftInputV2
from sglang.srt.speculative.dflash_utils import (
    apply_dflash_verify_logits_adjustments,
    can_dflash_use_fused_qkv_proj,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dflash_sampling_verify_available,
    parse_dflash_draft_config,
)
from sglang.srt.speculative.dflash_tree_utils import (
    DraftTreeCPU,
    build_retrieve_links_from_parents,
    build_tree_custom_mask,
    build_tree_from_topk_cpu,
    compute_tree_budget,
    sample_topk_from_logits,
    tree_accept_greedy_batched,
)
from sglang.srt.speculative.eagle_utils import verify_tree_greedy_func
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    assign_req_to_token_pool_func,
    commit_mamba_states_after_verify,
    move_kv_cache_overlap_safe,
    move_accept_tokens_to_target_kvcache,
)
from sglang.srt.speculative.triton_ops.cache_locs import assign_extend_cache_locs_func
from sglang.srt.speculative.triton_ops.dflash import (
    _compute_dflash_accept_bonus_triton_unchecked,
    _prepare_dflash_draft_block_unchecked,
)
from sglang.srt.utils import get_available_gpu_memory, is_cuda, is_hip, is_npu

_is_npu = is_npu()


logger = logging.getLogger(__name__)

_FusedKVMaterializeHelper = None


def _get_fused_kv_materialize_helper():
    global _FusedKVMaterializeHelper
    if _FusedKVMaterializeHelper is None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            FusedKVMaterializeHelper,
        )

        _FusedKVMaterializeHelper = FusedKVMaterializeHelper
    return _FusedKVMaterializeHelper


class DFlashWorkerV2(BaseSpecWorker):
    """DFLASH speculative decoding worker (spec-v2).

    Drives both overlap and non-overlap scheduling, same as EAGLE: the
    scheduler runs it synchronously when overlap is disabled.
    """

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port
        self._target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.page_size = server_args.page_size
        # Normalized in arg_groups.speculative_hook.handle_speculative_decoding.
        self.draft_window_size: Optional[int] = (
            server_args.speculative_draft_window_size
        )
        self.use_compact_draft_cache = self.draft_window_size is not None
        self.device = target_worker.device

        self._warned_sampling_fallback = False
        self._logged_first_verify = False

        # Draft runner (separate KV cache + attention backend).
        draft_server_args = deepcopy(server_args)
        draft_server_args.skip_tokenizer_init = True
        draft_backend = draft_server_args.speculative_draft_attention_backend
        supported_draft_backends = ("flashinfer", "fa3", "fa4", "triton", "ascend")
        if draft_backend is None:
            draft_backend, _ = draft_server_args.get_attention_backends()
        if draft_backend is None:
            # Use triton on ROCm (no FlashInfer), flashinfer on CUDA
            import torch as _torch

            draft_backend = "triton" if _torch.version.hip else "flashinfer"
        elif draft_backend == "trtllm_mha":
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"
            logger.warning(
                "DFLASH draft worker does not support 'trtllm_mha' because the "
                "draft path requires per-layer DFlash attention. Falling back to "
                "'%s'.",
                _fb,
            )
            draft_backend = _fb
        elif draft_backend not in supported_draft_backends:
            import torch as _torch

            _fb = "triton" if _torch.version.hip else "flashinfer"
            logger.warning(
                "DFLASH draft worker only supports attention_backend in %s for now, "
                "but got %r. Falling back to '%s'.",
                supported_draft_backends,
                draft_backend,
                _fb,
            )
            draft_backend = _fb
        # Make the draft worker backend explicit and self-contained (no further overrides).
        draft_server_args.speculative_draft_attention_backend = None
        draft_server_args.prefill_attention_backend = None
        draft_server_args.decode_attention_backend = None
        draft_server_args.attention_backend = draft_backend
        # Keep draft context length aligned with the target.
        draft_server_args.context_length = (
            target_worker.model_runner.model_config.context_len
        )
        saved_server_args = get_global_server_args()
        self._draft_worker = TpModelWorker(
            server_args=draft_server_args,
            gpu_id=gpu_id,
            tp_rank=tp_rank,
            moe_ep_rank=moe_ep_rank,
            pp_rank=0,
            attn_cp_rank=attn_cp_rank,
            moe_dp_rank=moe_dp_rank,
            dp_rank=dp_rank,
            nccl_port=nccl_port,
            is_draft_worker=True,
        )
        set_global_server_args_for_scheduler(saved_server_args)
        self.draft_model_runner = self._draft_worker.model_runner
        # Keep the same alias that other spec-v2 workers expose.
        self._draft_worker.draft_runner = self.draft_model_runner
        self.draft_model = self.draft_model_runner.model
        draft_config = parse_dflash_draft_config(
            draft_hf_config=self.draft_model_runner.model_config.hf_config
        )
        if server_args.speculative_num_draft_tokens is None:
            # Should not happen (ServerArgs should have inferred it), but keep a fallback.
            self.block_size = int(draft_config.resolve_block_size(default=16))
        else:
            self.block_size = int(server_args.speculative_num_draft_tokens)
            model_block_size = draft_config.block_size
            if model_block_size is None:
                model_block_size = getattr(self.draft_model, "block_size", None)
            if model_block_size is not None and int(model_block_size) != int(
                self.block_size
            ):
                logger.warning(
                    "DFLASH block size mismatch: using speculative_num_draft_tokens=%s but draft config block_size=%s.",
                    self.block_size,
                    model_block_size,
                )
        self.speculative_num_draft_tokens = int(self.block_size)
        self.tree_width = int(server_args.speculative_dflash_tree_width)
        self.tree_budget = (
            int(server_args.speculative_dflash_tree_budget)
            if server_args.speculative_dflash_tree_budget is not None
            else compute_tree_budget(self.block_size, self.tree_width)
        )
        self.tree_draft = str(server_args.speculative_dflash_tree_draft)
        self.dflash_head_type = str(server_args.speculative_dflash_head_type)
        self.use_tree_draft = self.tree_width > 1
        self._tree_verify_attn_backend = None
        self._tree_verify_full_attn_backend = None
        self._checked_tree_verify_attn_backend = False
        self.dflash_causal_head = self._read_dflash_causal_head(
            self.draft_model_runner.model_config.hf_config
        )
        if self.use_tree_draft:
            if self.dflash_head_type == "bidirectional" or (
                self.dflash_head_type == "auto" and self.dflash_causal_head is False
            ):
                raise NotImplementedError(
                    "DFLASH tree mode requires a causal DFlash draft head. "
                    f"head_type={self.dflash_head_type!r}, "
                    f"config.causal_head={self.dflash_causal_head!r}."
                )

        self._mask_token = draft_config.mask_token
        self._mask_token_id_override = draft_config.mask_token_id
        self._mask_token_id = self._resolve_mask_token_id(
            mask_token=self._mask_token,
            mask_token_id=self._mask_token_id_override,
        )
        if self.tp_rank == 0:
            logger.info(
                "Initialized DFLASH draft runner. attention_backend=%s, model=%s, block_size=%s, draft_window_size=%s, compact_cache=%s",
                getattr(draft_server_args, "attention_backend", None),
                self.draft_model.__class__.__name__,
                self.block_size,
                self.draft_window_size,
                self.use_compact_draft_cache,
            )
            logger.info(
                "DFLASH tree config. width=%s, budget=%s, draft=%s, head_type=%s, causal_head=%s",
                self.tree_width,
                self.tree_budget,
                self.tree_draft,
                self.dflash_head_type,
                self.dflash_causal_head,
            )
            logger.info(
                "DFLASH draft runner ready. mask_token=%s, mask_token_id=%s, mask_token_id_override=%s",
                self._mask_token,
                self._mask_token_id,
                self._mask_token_id_override,
            )

        self._block_pos_offsets = torch.arange(
            self.block_size, device=self.device, dtype=torch.int64
        )
        self._draft_block_ids_buf: Optional[torch.Tensor] = None  # [cap_bs, block_size]
        self._draft_block_positions_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_tokens_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_verify_out_cache_loc_buf: Optional[torch.Tensor] = (
            None  # [cap_bs, block_size]
        )
        self._draft_block_end_buf: Optional[torch.Tensor] = None  # [cap_bs]
        self._draft_seq_lens_cpu_buf: Optional[torch.Tensor] = None  # [cap_bs] on CPU
        self._draft_block_spec_info = DFlashVerifyInput(
            draft_token=torch.empty((0,), dtype=torch.long, device=self.device),
            positions=torch.empty((0,), dtype=torch.int64, device=self.device),
            draft_token_num=int(self.block_size),
            custom_mask=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        self._draft_greedy_gathered_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gathered_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_gather_cap: int = 0
        self._draft_greedy_local_max_buf: Optional[torch.Tensor] = None
        self._draft_greedy_local_arg_buf: Optional[torch.Tensor] = None
        self._draft_greedy_local_cap: int = 0
        self._draft_greedy_best_rank_buf: Optional[torch.Tensor] = None
        self._draft_greedy_rank_index_buf: Optional[torch.Tensor] = None
        self._draft_greedy_selected_ids_buf: Optional[torch.Tensor] = None
        self._draft_greedy_index_cap: int = 0
        self._use_fused_kv_materialize = is_cuda() or is_hip()
        self._fused_kv_helper: Optional[object] = None
        if self._use_fused_kv_materialize:
            self._init_fused_kv_helper()

        supports_gpu_triton = is_cuda() or is_hip()
        self._use_triton_prepare_block = supports_gpu_triton
        self._use_triton_accept_bonus = supports_gpu_triton
        self._accept_bonus_buffer_cap: int = 0
        self._accept_bonus_buffer_slot: int = 0
        self._accept_len_buf: Optional[torch.Tensor] = None
        self._commit_lens_bufs: List[torch.Tensor] = []
        self._bonus_id_bufs: List[torch.Tensor] = []
        self._out_tokens_bufs: List[torch.Tensor] = []
        self._new_seq_lens_bufs: List[torch.Tensor] = []

    def _maybe_use_triton_target_backend_for_tree(self):
        if not (is_cuda() or is_hip()):
            return

        target_model_runner = self.target_worker.model_runner
        if getattr(target_model_runner, "attn_backend", None) is None:
            return
        target_attn_backend = target_model_runner.attn_backend
        while getattr(target_attn_backend, "full_attn_backend", None) is not None:
            target_attn_backend = target_attn_backend.full_attn_backend
        if type(target_attn_backend).__name__ != "FlashInferAttnBackend":
            return

        tree_backend = os.environ.get("SGLANG_DFLASH_TREE_VERIFY_BACKEND")
        if tree_backend is None or tree_backend == "flashinfer":
            self._checked_tree_verify_attn_backend = True
            return
        target_model_runner.attn_backend = target_model_runner._get_attention_backend_from_str(
            tree_backend
        )
        self._checked_tree_verify_attn_backend = True
        logger.warning(
            "DFLASH tree mode uses %s as the target attention backend when "
            "the configured backend is FlashInfer.",
            tree_backend,
        )

    def _get_tree_verify_full_attn_backend(self):
        if self._tree_verify_full_attn_backend is None:
            graph_runner = self.target_worker.model_runner.decode_cuda_graph_runner
            graph_backend = getattr(
                graph_runner, "dflash_tree_graph_full_attn_backend", None
            )
            if graph_backend is not None:
                self._tree_verify_full_attn_backend = graph_backend
                return self._tree_verify_full_attn_backend

            from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS

            backend_name = os.environ.get(
                "SGLANG_DFLASH_TREE_VERIFY_FULL_ATTN_BACKEND", "flashinfer"
            )
            self._tree_verify_full_attn_backend = ATTENTION_BACKENDS[backend_name](
                self.target_worker.model_runner
            )
            logger.warning(
                "DFLASH tree target verify uses %s for full-attention layers "
                "because the configured full-attention backend cannot consume the "
                "tree verify mask/causal path.",
                backend_name,
            )
        return self._tree_verify_full_attn_backend

    @staticmethod
    def _set_hybrid_full_attn_backend(hybrid_backend, full_attn_backend):
        hybrid_backend.full_attn_backend = full_attn_backend
        hybrid_backend.attn_backend_list = [
            full_attn_backend,
            hybrid_backend.linear_attn_backend,
        ]
        hybrid_backend.token_to_kv_pool = full_attn_backend.token_to_kv_pool
        hybrid_backend.req_to_token_pool = full_attn_backend.req_to_token_pool
        hybrid_backend.max_context_len = getattr(
            full_attn_backend, "max_context_len", None
        )
        hybrid_backend.needs_cpu_seq_lens = (
            full_attn_backend.needs_cpu_seq_lens
            or hybrid_backend.linear_attn_backend.needs_cpu_seq_lens
        )

    @contextmanager
    def _maybe_use_tree_verify_full_attn_backend(self, verify_input):
        target_attn_backend = self.target_worker.model_runner.attn_backend
        full_attn_backend = getattr(target_attn_backend, "full_attn_backend", None)
        needs_tree_verify_backend = (
            getattr(verify_input, "custom_mask", None) is not None
            or getattr(verify_input, "force_causal", False)
        )
        if (
            full_attn_backend is None
            or not needs_tree_verify_backend
            or type(full_attn_backend).__name__ != "TRTLLMHAAttnBackend"
        ):
            yield False
            return

        original_full_attn_backend = full_attn_backend
        replacement_full_attn_backend = self._get_tree_verify_full_attn_backend()
        self._set_hybrid_full_attn_backend(
            target_attn_backend, replacement_full_attn_backend
        )
        try:
            yield True
        finally:
            self._set_hybrid_full_attn_backend(
                target_attn_backend, original_full_attn_backend
            )

    def _snapshot_persistent_mamba_state(
        self,
        model_worker_batch: ScheduleBatch,
        bs: int,
    ):
        target_model_runner = self.target_worker.model_runner
        linear_backend = getattr(
            target_model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return None
        try:
            mamba_caches = (
                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            state_indices = linear_backend.req_to_token_pool.get_mamba_indices(
                model_worker_batch.req_pool_indices
            )[:bs].to(torch.long)
            conv_snapshots = [
                conv_cache[:, state_indices].clone()
                for conv_cache in mamba_caches.conv
            ]
            mamba_pool = getattr(linear_backend.req_to_token_pool, "mamba_pool", None)
            replayssm_write_pos = (
                getattr(mamba_pool, "replayssm_write_pos", None)
                if mamba_pool is not None
                else None
            )
            replayssm_write_pos_snapshot = (
                replayssm_write_pos[state_indices].clone()
                if replayssm_write_pos is not None
                else None
            )
            return (
                state_indices,
                mamba_caches.temporal[:, state_indices].clone(),
                conv_snapshots,
                replayssm_write_pos_snapshot,
            )
        except Exception:
            logger.exception("DFLASH tree failed to snapshot persistent mamba state")
            return None

    def _restore_persistent_mamba_state(self, snapshot) -> None:
        if snapshot is None:
            return
        target_model_runner = self.target_worker.model_runner
        linear_backend = getattr(
            target_model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return
        try:
            mamba_caches = (
                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            if len(snapshot) == 3:
                state_indices, temporal_snapshot, conv_snapshots = snapshot
                replayssm_write_pos_snapshot = None
            else:
                (
                    state_indices,
                    temporal_snapshot,
                    conv_snapshots,
                    replayssm_write_pos_snapshot,
                ) = snapshot
            mamba_caches.temporal[:, state_indices] = temporal_snapshot
            for conv_cache, conv_snapshot in zip(mamba_caches.conv, conv_snapshots):
                conv_cache[:, state_indices] = conv_snapshot
            if replayssm_write_pos_snapshot is not None:
                mamba_pool = getattr(
                    linear_backend.req_to_token_pool, "mamba_pool", None
                )
                replayssm_write_pos = (
                    getattr(mamba_pool, "replayssm_write_pos", None)
                    if mamba_pool is not None
                    else None
                )
                if replayssm_write_pos is not None:
                    replayssm_write_pos[state_indices] = replayssm_write_pos_snapshot
        except Exception:
            logger.exception("DFLASH tree failed to restore persistent mamba state")

    def _commit_selected_persistent_mamba_snapshots(
        self,
        snapshots,
        commit_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
        model_worker_batch: ScheduleBatch,
    ) -> None:
        if not snapshots:
            return
        target_model_runner = self.target_worker.model_runner
        linear_backend = getattr(
            target_model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return
        try:
            mamba_caches = (
                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )

            def copy_snapshot_row(snapshot, row: int, dst_slot: torch.Tensor) -> None:
                _, temporal_snapshot, conv_snapshots, *_ = snapshot
                mamba_caches.temporal[:, dst_slot] = temporal_snapshot[:, row]
                for conv_cache, conv_snapshot in zip(
                    mamba_caches.conv, conv_snapshots
                ):
                    conv_cache[:, dst_slot] = conv_snapshot[:, row]

            state_indices = snapshots[0][0]
            bs = int(commit_lens.shape[0])
            for row in range(bs):
                step = max(0, int(commit_lens[row].item()) - 1)
                step = min(step, len(snapshots) - 1)
                copy_snapshot_row(snapshots[step], row, state_indices[row])

            if model_worker_batch.mamba_track_indices is None:
                return

            mamba_track_interval = get_global_server_args().mamba_track_interval
            seq_lens_post_verify = prefix_lens + commit_lens.to(prefix_lens.dtype)
            to_track_mask = (
                prefix_lens // mamba_track_interval
                != seq_lens_post_verify // mamba_track_interval
            )
            tracking_point = (
                seq_lens_post_verify
                // mamba_track_interval
                * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - prefix_lens - 1, min=0).to(
                torch.long
            )
            for row in range(bs):
                if not bool(to_track_mask[row].item()):
                    continue
                step = max(0, int(to_track_ith[row].item()))
                step = min(step, len(snapshots) - 1)
                dst_slot = model_worker_batch.mamba_track_indices[row].to(torch.long)
                copy_snapshot_row(snapshots[step], row, dst_slot)
        except Exception:
            logger.exception(
                "DFLASH tree failed to commit selected persistent mamba snapshot"
            )

    def _commit_latest_verify_mamba_state(
        self,
        model_worker_batch: ScheduleBatch,
        bs: int,
    ) -> None:
        target_model_runner = self.target_worker.model_runner
        linear_backend = getattr(
            target_model_runner.attn_backend, "linear_attn_backend", None
        )
        if linear_backend is None:
            return
        linear_metadata = getattr(linear_backend, "forward_metadata", None)
        if linear_metadata is None:
            return
        try:
            state_indices = linear_metadata.mamba_cache_indices[:bs].to(torch.long)
            mamba_caches = (
                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            mamba_caches.temporal[:, state_indices] = (
                mamba_caches.intermediate_ssm[:, state_indices, 0]
            )
            for conv_idx, conv_cache in enumerate(mamba_caches.conv):
                conv_cache[:, state_indices] = (
                    mamba_caches.intermediate_conv_window[conv_idx][
                        :, state_indices, 0
                    ]
                )
        except Exception:
            logger.exception("DFLASH tree failed to commit latest verify mamba state")

    def _reverify_accepted_tree_path_for_commit(
        self,
        *,
        model_worker_batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        tree_tokens: torch.Tensor,
        tree_cache_loc_2d: torch.Tensor,
        linear_fallback_tokens: torch.Tensor,
        accept_index_local: torch.Tensor,
        commit_lens: torch.Tensor,
        prefix_lens: torch.Tensor,
    ) -> tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[list],
    ]:
        """Replay the accepted tree path as a causal verify for exact state commit.

        Hybrid MoE models can produce shape-sensitive full-attention KV during the
        wide tree verify even when the accepted token predictions match a linear
        replay. Commit the linear replay's KV/hidden/mamba intermediates so the
        persistent state remains identical to width=1 decode.
        """

        bs = int(tree_tokens.shape[0])
        block_size = int(self.block_size)
        device = tree_tokens.device
        path_pos = torch.arange(block_size, dtype=torch.long, device=device)
        path_mask = path_pos.unsqueeze(0) < commit_lens.to(torch.long).unsqueeze(1)

        branch_cache_alloc: Optional[torch.Tensor] = None
        branch_cache_loc_2d = tree_cache_loc_2d[:, :block_size].to(torch.int64)

        accepted_path_tokens = torch.gather(tree_tokens, 1, accept_index_local)
        use_linear_commit_fallback = (
            self.target_worker.model_runner.mambaish_config is not None
            and os.environ.get("SGLANG_DFLASH_TREE_MOE_LINEAR_COMMIT_FALLBACK", "1")
            != "0"
        )
        if use_linear_commit_fallback:
            branch_tokens = linear_fallback_tokens[:, :block_size].to(
                accepted_path_tokens.dtype
            )
        else:
            branch_tokens = torch.where(
                path_mask,
                accepted_path_tokens,
                linear_fallback_tokens[:, :block_size].to(accepted_path_tokens.dtype),
            )
        branch_positions_2d = prefix_lens.to(torch.int64).unsqueeze(
            1
        ) + path_pos.to(torch.int64).unsqueeze(0)

        persistent_mamba_snapshot = self._snapshot_persistent_mamba_state(
            model_worker_batch, bs
        )
        batch_input_ids_backup = model_worker_batch.input_ids
        batch_spec_info_backup = model_worker_batch.spec_info
        batch_forward_mode_backup = model_worker_batch.forward_mode
        batch_capture_hidden_mode_backup = model_worker_batch.capture_hidden_mode
        batch_out_cache_loc_backup = model_worker_batch.out_cache_loc
        batch_seq_lens_backup = model_worker_batch.seq_lens
        seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
        seq_lens_sum_backup = model_worker_batch.seq_lens_sum
        req_to_token = self.model_runner.req_to_token_pool.req_to_token
        req_rows = model_worker_batch.req_pool_indices.to(torch.long).unsqueeze(1)
        branch_req_to_token_backup = req_to_token[
            req_rows,
            branch_positions_2d,
        ].clone()
        req_to_token[req_rows, branch_positions_2d] = branch_cache_loc_2d.to(
            req_to_token.dtype
        )
        branch_hidden = None
        branch_target_predict = None
        mamba_state_snapshots = None
        try:
            branch_verify_input = DFlashVerifyInput(
                draft_token=branch_tokens.reshape(-1),
                positions=branch_positions_2d.reshape(-1),
                draft_token_num=block_size,
                custom_mask=None,
                force_causal=False,
                capture_hidden_mode=CaptureHiddenMode.FULL,
                allow_cuda_graph=False,
            )
            model_worker_batch.out_cache_loc = branch_cache_loc_2d.reshape(-1)
            model_worker_batch.seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                model_worker_batch.seq_lens_cpu = draft_input.planning_seq_lens_cpu
                model_worker_batch.seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                model_worker_batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
                model_worker_batch.seq_lens_sum = int(
                    draft_input.reserved_seq_lens_sum
                )

            branch_forward_batch, _ = branch_verify_input.prepare_for_verify(
                model_worker_batch, self.target_worker
            )
            branch_target_out = self.target_worker.forward_batch_generation(
                batch=None,
                forward_batch=branch_forward_batch,
                is_verify=True,
                skip_attn_backend_init=True,
            )
            step_hidden = branch_target_out.logits_output.hidden_states
            if step_hidden is None:
                raise RuntimeError(
                    "DFLASH accepted-path reverify requires hidden states, "
                    "but got None."
                )
            branch_hidden = step_hidden.view(bs, block_size, -1)
            branch_target_predict = torch.argmax(
                branch_target_out.logits_output.next_token_logits,
                dim=-1,
            ).view(bs, block_size)
        finally:
            req_to_token[req_rows, branch_positions_2d] = branch_req_to_token_backup
            model_worker_batch.input_ids = batch_input_ids_backup
            model_worker_batch.spec_info = batch_spec_info_backup
            model_worker_batch.forward_mode = batch_forward_mode_backup
            model_worker_batch.capture_hidden_mode = batch_capture_hidden_mode_backup
            model_worker_batch.out_cache_loc = batch_out_cache_loc_backup
            model_worker_batch.seq_lens = batch_seq_lens_backup
            model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
            model_worker_batch.seq_lens_sum = seq_lens_sum_backup
            self._restore_persistent_mamba_state(persistent_mamba_snapshot)

        if branch_hidden is None or branch_target_predict is None:
            raise RuntimeError("DFLASH accepted-path reverify produced no outputs.")
        return (
            branch_cache_alloc,
            branch_cache_loc_2d.to(torch.int64),
            branch_hidden,
            branch_target_predict,
            branch_tokens,
            mamba_state_snapshots,
        )

    @property
    def target_worker(self) -> TpModelWorker:
        return self._target_worker

    @property
    def draft_worker(self):
        # DFLASH drives the draft model through a plain TpModelWorker: the
        # draft KV is materialized from target hidden states, so there is no
        # EagleDraftWorkerBase draft/draft_extend split to wrap it in.
        return self._draft_worker

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        return (
            self._target_worker.model_runner.attn_backend,
            self.draft_model_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        # Without draft windowing, the draft worker aliases the target
        # request->token mapping and allocation state. With draft windowing
        # enabled, the draft worker keeps a private compact req->token table
        # over the same global KV index space, so radix-cache/prefix-hit KV
        # remains reusable while draft attention sees only the recent window.
        self._draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=(
                None if self.use_compact_draft_cache else req_to_token_pool
            ),
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )

    def init_attention_backends(self):
        if self.use_tree_draft:
            self._maybe_use_triton_target_backend_for_tree()
        self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        capture_decode_cuda_graph = not self.server_args.disable_cuda_graph
        if is_cuda() and capture_decode_cuda_graph:
            available_mem = get_available_gpu_memory(self.device, self.gpu_id)
            if available_mem < 1.0:
                capture_decode_cuda_graph = False
                logger.warning(
                    "Disable DFLASH draft cuda graph because only %.2f GB GPU "
                    "memory is available after target backend initialization.",
                    available_mem,
                )
        self._draft_worker.init_cuda_graphs(
            capture_decode_cuda_graph=capture_decode_cuda_graph
        )

    def _init_fused_kv_helper(self) -> None:
        """Initialize the fused KV materialization helper with pre-stacked weights."""
        try:
            layers = self.draft_model.layers
            fused_disable_reason: Optional[str] = None

            if len(layers) == 0:
                fused_disable_reason = "no layers found"

            for layer_idx, layer in enumerate(layers):
                attn = layer.self_attn
                eligible, reason = can_dflash_use_fused_qkv_proj(attn.qkv_proj)
                if not eligible:
                    fused_disable_reason = f"{reason}: layer={layer_idx}"
                    break

                # Keep semantics aligned with set_kv_buffer scaling behavior.
                k_scale = getattr(attn.attn, "k_scale", None)
                v_scale = getattr(attn.attn, "v_scale", None)
                if k_scale is not None and not math.isclose(float(k_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit k_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, k_scale={k_scale}"
                    )
                    break
                if v_scale is not None and not math.isclose(float(v_scale), 1.0):
                    fused_disable_reason = (
                        "non-unit v_scale is not supported for fused KV path: "
                        f"layer={layer_idx}, v_scale={v_scale}"
                    )
                    break

                rope_is_neox_style = bool(
                    getattr(attn.rotary_emb, "is_neox_style", True)
                )
                if not rope_is_neox_style:
                    fused_disable_reason = (
                        "non-neox RoPE is not supported for fused KV path: "
                        f"layer={layer_idx}, rope_is_neox_style={rope_is_neox_style}"
                    )
                    break

            if fused_disable_reason is not None:
                if self.tp_rank == 0:
                    logger.info(
                        "DFLASH fused KV materialization disabled: %s",
                        fused_disable_reason,
                    )
                self._use_fused_kv_materialize = False
                self._fused_kv_helper = None
                return

            FusedKVMaterializeHelper = _get_fused_kv_materialize_helper()
            first_attn = layers[0].self_attn
            rotary_emb = first_attn.rotary_emb

            self._fused_kv_helper = FusedKVMaterializeHelper(
                layers=layers,
                rotary_emb=rotary_emb,
                num_kv_heads=first_attn.num_kv_heads,
                head_dim=first_attn.head_dim,
                device=self.device,
                max_position_hint=self.target_worker.model_runner.model_config.context_len
                + int(self.block_size),
            )
            if self.tp_rank == 0:
                logger.info(
                    "DFLASH fused KV materialization enabled. "
                    "n_layers=%d, num_kv_heads=%d, head_dim=%d",
                    len(layers),
                    first_attn.num_kv_heads,
                    first_attn.head_dim,
                )
        except Exception as e:
            logger.warning(
                "DFLASH fused KV initialization failed, falling back to sequential path: %s",
                e,
            )
            self._use_fused_kv_materialize = False
            self._fused_kv_helper = None

    def _ensure_draft_block_buffers(self, bs: int) -> None:
        cap = (
            0
            if self._draft_block_ids_buf is None
            else int(self._draft_block_ids_buf.shape[0])
        )
        if cap >= int(bs):
            return

        new_cap = max(int(bs), cap * 2 if cap > 0 else int(bs))
        device = self.device
        block_size = int(self.block_size)
        self._draft_block_ids_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_block_positions_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_tokens_buf = torch.empty(
            (new_cap, block_size), dtype=torch.long, device=device
        )
        self._draft_verify_out_cache_loc_buf = torch.empty(
            (new_cap, block_size), dtype=torch.int64, device=device
        )
        self._draft_block_end_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device=device
        )
        self._draft_seq_lens_cpu_buf = torch.empty(
            (new_cap,), dtype=torch.int32, device="cpu"
        )

    def __getattr__(self, name):
        # Delegate anything not implemented yet to the target worker. Guard
        # the backing field so a lookup before __init__ sets it raises
        # AttributeError instead of recursing through the property.
        if name == "_target_worker":
            raise AttributeError(name)
        return getattr(self.target_worker, name)

    def clear_cache_pool(self):
        # The target worker owns the shared KV allocator/cache. For the compact
        # sliding-window path, the draft req->token view is rebuilt from committed
        # target state before each draft forward, so there is nothing persistent
        # to flush here.
        pass

    def _gather_req_to_token_masked(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        pos2d: torch.Tensor,
        mask: torch.Tensor,
        context: str,
    ) -> torch.Tensor:
        if pos2d.ndim != 2:
            raise RuntimeError(
                f"{context} expected 2D positions, got shape={tuple(pos2d.shape)}."
            )
        if mask.shape != pos2d.shape:
            raise RuntimeError(
                f"{context} mask/position shape mismatch: {tuple(mask.shape)} vs {tuple(pos2d.shape)}."
            )

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)

        table_width = int(req_to_token.shape[1])
        if table_width <= 0:
            if bool(mask.any().item()):
                raise RuntimeError(
                    f"{context} req_to_token table is empty but gather mask is non-empty."
                )
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        # Only the masked-off rectangular padding can be out of range in the normal
        # ragged-batch case. Replace those don't-care columns with a valid in-range
        # position before the gather so the kernel only sees real positions.
        safe_pos2d = pos2d.masked_fill(~mask, 0)
        return req_to_token[req_pool_indices[:, None], safe_pos2d][mask].to(torch.int64)

    def _gather_req_to_token_segments(
        self,
        *,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        start: torch.Tensor | None,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        lengths = lengths.to(torch.int64)
        if lengths.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)

        if req_pool_indices.dtype != torch.int64:
            req_pool_indices = req_pool_indices.to(torch.int64)
        offsets = torch.arange(
            max_len, device=self.device, dtype=torch.int64
        ).unsqueeze(0)
        if start is None:
            pos2d = offsets.expand(req_pool_indices.shape[0], -1)
        else:
            pos2d = start.to(torch.int64).unsqueeze(1) + offsets
        mask = offsets < lengths.unsqueeze(1)
        return self._gather_req_to_token_masked(
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            pos2d=pos2d,
            mask=mask,
            context="DFLASH req_to_token segment gather",
        )

    def _compute_compact_draft_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        assert self.draft_window_size is not None
        visible_lens = torch.clamp(
            seq_lens.to(dtype=torch.int32, device=self.device),
            max=int(self.draft_window_size),
        )
        if self.page_size <= 1:
            return visible_lens

        # Paged FA backends derive the page table from local token positions, so the
        # compact suffix must start on a page boundary. Keep up to page_size - 1 extra
        # tokens on the left to preserve valid local page structure.
        seq_lens_i64 = seq_lens.to(torch.int64)
        visible_lens_i64 = visible_lens.to(torch.int64)
        visible_start = seq_lens_i64 - visible_lens_i64
        aligned_start = visible_start - torch.remainder(visible_start, self.page_size)
        return (seq_lens_i64 - aligned_start).to(torch.int32)

    @staticmethod
    def _read_dflash_causal_head(hf_config) -> Optional[bool]:
        config_dict = None
        if isinstance(hf_config, dict):
            config_dict = hf_config
        elif hasattr(hf_config, "to_dict"):
            try:
                config_dict = hf_config.to_dict()
            except Exception:
                config_dict = None

        if isinstance(config_dict, dict):
            dflash_cfg = config_dict.get("dflash_config", {})
            if isinstance(dflash_cfg, dict) and "causal_head" in dflash_cfg:
                return bool(dflash_cfg["causal_head"])
            if "causal_head" in config_dict:
                return bool(config_dict["causal_head"])

        dflash_cfg = getattr(hf_config, "dflash_config", None)
        if isinstance(dflash_cfg, dict) and "causal_head" in dflash_cfg:
            return bool(dflash_cfg["causal_head"])
        if hasattr(dflash_cfg, "causal_head"):
            return bool(getattr(dflash_cfg, "causal_head"))
        if hasattr(hf_config, "causal_head"):
            return bool(getattr(hf_config, "causal_head"))
        return None

    def _resolve_mask_token_id(
        self, *, mask_token: str, mask_token_id: Optional[int] = None
    ) -> int:
        if not isinstance(mask_token, str) or not mask_token:
            raise ValueError(
                f"DFLASH mask_token must be a non-empty string, got {mask_token!r}."
            )

        vocab_size = int(self.target_worker.model_runner.model_config.vocab_size)
        if mask_token_id is not None:
            resolved_id = int(mask_token_id)
            if resolved_id >= vocab_size:
                raise ValueError(
                    "DFLASH mask_token_id is outside the target vocab size. "
                    f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                    f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                    "SGLang does not support resizing target embeddings for DFLASH yet."
                )

            tokenizer = getattr(self.target_worker, "tokenizer", None)
            if tokenizer is not None:
                token_id_from_vocab = tokenizer.get_vocab().get(mask_token, None)
                if (
                    token_id_from_vocab is not None
                    and int(token_id_from_vocab) != resolved_id
                ):
                    raise ValueError(
                        "DFLASH config mismatch: dflash_config.mask_token_id conflicts with tokenizer vocab id "
                        f"for dflash_config.mask_token. mask_token={mask_token!r}, "
                        f"mask_token_id={resolved_id}, tokenizer_vocab_id={int(token_id_from_vocab)}."
                    )
            return resolved_id

        tokenizer = getattr(self.target_worker, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "DFLASH requires tokenizer initialization when dflash_config.mask_token_id is not set "
                "(skip_tokenizer_init is not supported in this mode)."
            )

        resolved_id = None
        if getattr(tokenizer, "mask_token", None) == mask_token:
            resolved_id = getattr(tokenizer, "mask_token_id", None)

        if resolved_id is None:
            # Prefer checking the explicit vocab mapping first.
            vocab = tokenizer.get_vocab()
            resolved_id = vocab.get(mask_token, None)

        if resolved_id is None:
            # Mirror the reference DFlash HF demo by adding the mask token to the tokenizer.
            # This is safe only when the resulting id stays within the target model vocab size.
            added = tokenizer.add_special_tokens({"mask_token": mask_token})
            resolved_id = getattr(tokenizer, "mask_token_id", None)
            if resolved_id is None:
                resolved_id = tokenizer.convert_tokens_to_ids(mask_token)

            if added and self.tp_rank == 0:
                logger.info(
                    "Added DFLASH mask token to tokenizer. token=%s, mask_token_id=%s, tokenizer_len=%s, model_vocab_size=%s",
                    mask_token,
                    resolved_id,
                    len(tokenizer),
                    vocab_size,
                )

        if resolved_id is None or int(resolved_id) < 0:
            raise ValueError(
                "DFLASH requires resolving a mask token id, but it could not be resolved. "
                f"mask_token={mask_token!r}."
            )

        if resolved_id >= vocab_size:
            raise ValueError(
                "DFLASH mask_token_id is outside the target vocab size. "
                f"mask_token_id={resolved_id}, vocab_size={vocab_size}. "
                f"This likely means mask_token={mask_token!r} requires vocab expansion beyond the model's embedding size. "
                "SGLang does not support resizing target embeddings for DFLASH yet."
            )

        return int(resolved_id)

    def _greedy_sample_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        """Greedy argmax over the target LM head in a TP-safe way.

        We cannot materialize full logits for large vocabularies efficiently, and with
        TP>1 each rank only owns a shard of the LM head weight. This computes the
        per-rank max, gathers candidates across TP ranks, and selects the global max.
        """

        if hidden_states.numel() == 0:
            return torch.empty((0,), dtype=torch.long, device=hidden_states.device)

        weight = lm_head.weight  # [local_vocab_padded, hidden]
        weight_dtype = weight.dtype
        num_tokens = int(hidden_states.shape[0])
        out_tokens = torch.empty(
            (num_tokens,), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        if not hasattr(lm_head, "shard_indices"):
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                logits = torch.matmul(hs, weight.T)
                out_tokens[start:end] = torch.argmax(logits, dim=-1).to(torch.long)
            return out_tokens

        shard = lm_head.shard_indices
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        # Valid ranges in the local shard (excluding padding):
        #   base vocab:  [0, num_org)
        #   added vocab: [num_org_padded, num_org_padded + num_added)
        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)

        def _ensure_local_reduce_buffers(
            chunk_len: int,
            value_dtype: torch.dtype,
            device: torch.device,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if (
                self._draft_greedy_local_cap < chunk_len
                or self._draft_greedy_local_max_buf is None
                or self._draft_greedy_local_arg_buf is None
                or self._draft_greedy_local_max_buf.dtype != value_dtype
                or self._draft_greedy_local_max_buf.device != device
                or self._draft_greedy_local_arg_buf.device != device
            ):
                cap = max(int(chunk_size), chunk_len)
                self._draft_greedy_local_max_buf = torch.empty(
                    (cap,), dtype=value_dtype, device=device
                )
                self._draft_greedy_local_arg_buf = torch.empty(
                    (cap,), dtype=torch.int64, device=device
                )
                self._draft_greedy_local_cap = cap
            return (
                self._draft_greedy_local_max_buf[:chunk_len],
                self._draft_greedy_local_arg_buf[:chunk_len],
            )

        # Fast path (common): single-rank greedy sampling over the base vocab shard.
        # Avoids extra max/id bookkeeping that is only needed for TP sync or added vocab.
        #
        # DFLASH draft sampling only materializes a small fixed block of hidden states
        # each step. On tp=1, splitting those states into many 256-token chunks adds
        # extra matmul/argmax launches without reducing peak memory meaningfully.
        if tp_size == 1 and num_added == 0:
            fast_chunk_size = max(int(chunk_size), 1024)
            for start in range(0, num_tokens, fast_chunk_size):
                end = min(num_tokens, start + fast_chunk_size)
                hs = _cast_hs(hidden_states[start:end])
                if num_org > 0:
                    base_logits = torch.matmul(hs, weight[:num_org].T)
                    local_max, local_arg = _ensure_local_reduce_buffers(
                        end - start, base_logits.dtype, hs.device
                    )
                    torch.max(base_logits, dim=-1, out=(local_max, local_arg))
                    out_tokens[start:end].copy_(local_arg)
                    out_tokens[start:end].add_(org_vocab_start)
                else:
                    out_tokens[start:end] = 0
            return out_tokens

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            # Base vocab logits.
            if num_org > 0:
                base_logits = torch.matmul(hs, weight[:num_org].T)
                local_max, local_arg = _ensure_local_reduce_buffers(
                    chunk_len, base_logits.dtype, hs.device
                )
                torch.max(base_logits, dim=-1, out=(local_max, local_arg))
            else:
                local_max = torch.full(
                    (chunk_len,),
                    torch.finfo(weight_dtype).min,
                    dtype=weight_dtype,
                    device=hs.device,
                )
                local_arg = torch.zeros(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )

            # Added vocab logits (e.g., LoRA-added embeddings), if present.
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                added_logits = torch.matmul(
                    hs, weight[added_slice_start:added_slice_end].T
                )
                added_max, added_arg = torch.max(added_logits, dim=-1)
                use_added = added_max > local_max
                local_max = torch.where(use_added, added_max, local_max)
                # For base/added conversion below, keep local_arg expressed in the full local
                # weight index space (base + padding + added), matching `lm_head.weight`.
                local_arg = torch.where(
                    use_added, added_arg.to(local_arg.dtype) + num_org_padded, local_arg
                )

            # Convert local argmax indices to global token ids.
            if num_added == 0:
                local_arg.add_(org_vocab_start)
                global_ids = local_arg
            else:
                global_ids = torch.empty(
                    (chunk_len,), dtype=torch.int64, device=hs.device
                )
                is_base = local_arg < num_org
                global_ids[is_base] = org_vocab_start + local_arg[is_base]
                global_ids[~is_base] = added_vocab_start + (
                    local_arg[~is_base] - num_org_padded
                )

            if tp_size == 1:
                out_tokens[start:end] = global_ids.to(torch.long)
                continue

            # Gather per-rank maxima and associated global ids, then select the global max.
            needed = tp_size * chunk_len
            chunk_cap = int(chunk_size)
            if (
                self._draft_greedy_gather_cap < needed
                or self._draft_greedy_gathered_max_buf is None
                or self._draft_greedy_gathered_ids_buf is None
                or self._draft_greedy_gathered_max_buf.dtype != local_max.dtype
                or self._draft_greedy_gathered_max_buf.device != hs.device
            ):
                # Allocate enough space for the max chunk size to avoid reallocations.
                cap = tp_size * chunk_cap
                self._draft_greedy_gathered_max_buf = torch.empty(
                    (cap,), dtype=local_max.dtype, device=hs.device
                )
                self._draft_greedy_gathered_ids_buf = torch.empty(
                    (cap,), dtype=global_ids.dtype, device=hs.device
                )
                self._draft_greedy_gather_cap = cap

            if (
                self._draft_greedy_index_cap < chunk_len
                or self._draft_greedy_best_rank_buf is None
                or self._draft_greedy_rank_index_buf is None
                or self._draft_greedy_selected_ids_buf is None
                or self._draft_greedy_best_rank_buf.device != hs.device
                or self._draft_greedy_selected_ids_buf.device != hs.device
            ):
                self._draft_greedy_best_rank_buf = torch.empty(
                    (chunk_cap,), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_rank_index_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_selected_ids_buf = torch.empty(
                    (1, chunk_cap), dtype=torch.int64, device=hs.device
                )
                self._draft_greedy_index_cap = chunk_cap

            gathered_max = self._draft_greedy_gathered_max_buf[:needed]
            gathered_ids = self._draft_greedy_gathered_ids_buf[:needed]

            tp_group.all_gather_into_tensor(gathered_max, local_max.contiguous())
            tp_group.all_gather_into_tensor(gathered_ids, global_ids.contiguous())
            gathered_max = gathered_max.view(tp_size, chunk_len)
            gathered_ids = gathered_ids.view(tp_size, chunk_len)

            best_rank = self._draft_greedy_best_rank_buf[:chunk_len]
            torch.argmax(gathered_max, dim=0, out=best_rank)

            rank_index = self._draft_greedy_rank_index_buf[:, :chunk_len]
            rank_index[0].copy_(best_rank)
            selected_ids = self._draft_greedy_selected_ids_buf[:, :chunk_len]
            torch.gather(gathered_ids, 0, rank_index, out=selected_ids)
            out_tokens[start:end].copy_(selected_ids.view(-1))

        return out_tokens

    def _topk_from_vocab_parallel_head(
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head,
        k: int,
        chunk_size: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k over the target LM head with global token ids.

        Returns `(topk_logprobs, topk_token_ids)`, both shaped `[num_tokens, k]`.
        The implementation is intentionally simple because tree drafting v1
        prioritizes correctness over a fused GPU draft-head kernel.
        """
        if hidden_states.numel() == 0:
            return (
                torch.empty(
                    (0, int(k)), dtype=torch.float32, device=hidden_states.device
                ),
                torch.empty((0, int(k)), dtype=torch.long, device=hidden_states.device),
            )
        if int(k) <= 0:
            raise ValueError(f"DFLASH tree top-k must be positive, got {k}.")

        weight = lm_head.weight
        weight_dtype = weight.dtype
        num_tokens = int(hidden_states.shape[0])
        topk_logprobs_out = torch.empty(
            (num_tokens, int(k)), dtype=torch.float32, device=hidden_states.device
        )
        topk_ids_out = torch.empty(
            (num_tokens, int(k)), dtype=torch.long, device=hidden_states.device
        )

        def _cast_hs(x: torch.Tensor) -> torch.Tensor:
            return x if x.dtype == weight_dtype else x.to(weight_dtype)

        if not hasattr(lm_head, "shard_indices"):
            if int(k) > int(weight.shape[0]):
                raise ValueError(
                    f"DFLASH tree_width={k} exceeds vocab size={int(weight.shape[0])}."
                )
            for start in range(0, num_tokens, int(chunk_size)):
                end = min(num_tokens, start + int(chunk_size))
                hs = _cast_hs(hidden_states[start:end])
                logits = torch.matmul(hs, weight.T)
                topk_logprobs, topk_ids = sample_topk_from_logits(logits, int(k))
                topk_logprobs_out[start:end].copy_(topk_logprobs.to(torch.float32))
                topk_ids_out[start:end].copy_(topk_ids.to(torch.long))
            return topk_logprobs_out, topk_ids_out

        shard = lm_head.shard_indices
        tp_group = get_tp_group()
        tp_size = int(tp_group.world_size)

        num_org = int(shard.num_org_elements)
        num_org_padded = int(shard.num_org_elements_padded)
        num_added = int(shard.num_added_elements)
        org_vocab_start = int(shard.org_vocab_start_index)
        added_vocab_start = int(shard.added_vocab_start_index)
        local_vocab_size = num_org + num_added
        if int(k) > local_vocab_size * tp_size:
            raise ValueError(
                "DFLASH tree_width exceeds the tensor-parallel global vocab size. "
                f"tree_width={k}, global_vocab={local_vocab_size * tp_size}."
            )

        for start in range(0, num_tokens, int(chunk_size)):
            end = min(num_tokens, start + int(chunk_size))
            hs = _cast_hs(hidden_states[start:end])
            chunk_len = int(hs.shape[0])

            logits_pieces = []
            id_pieces = []
            if num_org > 0:
                logits_pieces.append(torch.matmul(hs, weight[:num_org].T))
                id_pieces.append(
                    torch.arange(
                        org_vocab_start,
                        org_vocab_start + num_org,
                        dtype=torch.long,
                        device=hs.device,
                    )
                )
            if num_added > 0:
                added_slice_start = num_org_padded
                added_slice_end = num_org_padded + num_added
                logits_pieces.append(
                    torch.matmul(hs, weight[added_slice_start:added_slice_end].T)
                )
                id_pieces.append(
                    torch.arange(
                        added_vocab_start,
                        added_vocab_start + num_added,
                        dtype=torch.long,
                        device=hs.device,
                    )
                )
            if not logits_pieces:
                raise RuntimeError(
                    "DFLASH target LM head shard has no valid vocab rows."
                )

            local_logits = (
                logits_pieces[0]
                if len(logits_pieces) == 1
                else torch.cat(logits_pieces, dim=-1)
            )
            local_ids = id_pieces[0] if len(id_pieces) == 1 else torch.cat(id_pieces)
            local_lse = torch.logsumexp(local_logits, dim=-1)
            local_top_logits, local_top_arg = torch.topk(local_logits, int(k), dim=-1)
            local_top_ids = local_ids[local_top_arg]

            if tp_size == 1:
                topk_logprobs_out[start:end].copy_(
                    (local_top_logits - local_lse[:, None]).to(torch.float32)
                )
                topk_ids_out[start:end].copy_(local_top_ids.to(torch.long))
                continue

            gathered_lse = torch.empty(
                (tp_size * chunk_len,), dtype=local_lse.dtype, device=hs.device
            )
            gathered_logits = torch.empty(
                (tp_size * chunk_len * int(k),),
                dtype=local_top_logits.dtype,
                device=hs.device,
            )
            gathered_ids = torch.empty(
                (tp_size * chunk_len * int(k),),
                dtype=local_top_ids.dtype,
                device=hs.device,
            )
            tp_group.all_gather_into_tensor(gathered_lse, local_lse.contiguous())
            tp_group.all_gather_into_tensor(
                gathered_logits, local_top_logits.contiguous().view(-1)
            )
            tp_group.all_gather_into_tensor(
                gathered_ids, local_top_ids.contiguous().view(-1)
            )
            global_lse = torch.logsumexp(gathered_lse.view(tp_size, chunk_len), dim=0)
            candidate_logits = (
                gathered_logits.view(tp_size, chunk_len, int(k))
                .permute(1, 0, 2)
                .reshape(chunk_len, tp_size * int(k))
            )
            candidate_ids = (
                gathered_ids.view(tp_size, chunk_len, int(k))
                .permute(1, 0, 2)
                .reshape(chunk_len, tp_size * int(k))
            )
            selected_logits, selected_arg = torch.topk(candidate_logits, int(k), dim=-1)
            topk_logprobs_out[start:end].copy_(
                (selected_logits - global_lse[:, None]).to(torch.float32)
            )
            topk_ids_out[start:end].copy_(
                torch.gather(candidate_ids, 1, selected_arg).to(torch.long)
            )

        return topk_logprobs_out, topk_ids_out

    def _append_target_hidden_to_draft_kv_by_loc(
        self,
        *,
        target_hidden: torch.Tensor,
        cache_loc: torch.Tensor,
        positions: torch.Tensor,
        cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Materialize target context features into the draft KV cache at explicit slots.

        For the spec-v2 overlap path, callers can pass dense `[bs, block_size]`
        `cache_loc_2d` plus `commit_lens`; the prefix-valid writer then commits
        only the live prefix rows without constructing masked/packed index tensors.
        """
        if target_hidden is None:
            raise RuntimeError("DFLASH missing target hidden context features.")
        if target_hidden.numel() == 0:
            return
        if target_hidden.ndim != 2:
            raise ValueError(
                "DFLASH target_hidden must be 2D, "
                f"got shape={tuple(target_hidden.shape)}."
            )

        if cache_loc.ndim != 1:
            raise ValueError(
                f"DFLASH cache_loc must be 1D, got shape={tuple(cache_loc.shape)}."
            )
        if positions.ndim != 1:
            raise ValueError(
                f"DFLASH positions must be 1D, got shape={tuple(positions.shape)}."
            )
        num_tokens = int(target_hidden.shape[0])
        if int(cache_loc.numel()) != num_tokens:
            raise ValueError(
                "DFLASH cache_loc length mismatch: "
                f"cache_loc={int(cache_loc.numel())}, target_hidden={num_tokens}."
            )
        if int(positions.numel()) != num_tokens:
            raise ValueError(
                "DFLASH positions length mismatch: "
                f"positions={int(positions.numel())}, target_hidden={num_tokens}."
            )
        if cache_loc_2d is not None:
            if cache_loc_2d.ndim != 2:
                raise ValueError(
                    "DFLASH cache_loc_2d must be 2D, "
                    f"got shape={tuple(cache_loc_2d.shape)}."
                )
            if int(cache_loc_2d.numel()) != num_tokens:
                raise ValueError(
                    "DFLASH cache_loc_2d size mismatch: "
                    f"cache_loc_2d={int(cache_loc_2d.numel())}, target_hidden={num_tokens}."
                )
            if commit_lens is None:
                raise ValueError(
                    "DFLASH cache_loc_2d requires commit_lens for prefix-valid writes."
                )

        device = self.model_runner.device
        if cache_loc.device != device:
            cache_loc = cache_loc.to(device, non_blocking=True)
        if positions.device != device:
            positions = positions.to(device, non_blocking=True)
        if target_hidden.device != device:
            target_hidden = target_hidden.to(device, non_blocking=True)

        if cache_loc.dtype != torch.int64:
            cache_loc = cache_loc.to(torch.int64)
        if positions.dtype != torch.int64:
            positions = positions.to(torch.int64)
        if cache_loc_2d is not None:
            if cache_loc_2d.device != device:
                cache_loc_2d = cache_loc_2d.to(device, non_blocking=True)
            if cache_loc_2d.dtype != torch.int64:
                cache_loc_2d = cache_loc_2d.to(torch.int64)
        if commit_lens is not None:
            if commit_lens.device != device:
                commit_lens = commit_lens.to(device, non_blocking=True)
            if commit_lens.dtype != torch.int32:
                commit_lens = commit_lens.to(torch.int32)

        with torch.inference_mode():
            ctx_hidden = self.draft_model.project_target_hidden(target_hidden)

            if cache_loc_2d is not None:
                bs = int(commit_lens.shape[0])
                if int(cache_loc_2d.shape[0]) != bs:
                    raise ValueError(
                        "DFLASH cache_loc_2d batch size mismatch: "
                        f"cache_loc_2d={tuple(cache_loc_2d.shape)}, commit_lens={tuple(commit_lens.shape)}."
                    )
                if bs == 0:
                    return
                if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                    try:
                        self._append_target_hidden_fused(
                            ctx_hidden=ctx_hidden,
                            ctx_positions=positions,
                            ctx_cache_loc=cache_loc,
                            ctx_cache_loc_2d=cache_loc_2d,
                            commit_lens=commit_lens,
                        )
                        return
                    except Exception as e:
                        logger.warning(
                            "DFLASH fused prefix-direct KV append failed; falling back to the per-layer prefix-direct path: %s",
                            e,
                        )
                        self._use_fused_kv_materialize = False
                        self._fused_kv_helper = None

                for layer in self.draft_model.layers:
                    attn = layer.self_attn
                    k, v = attn.kv_proj_only(ctx_hidden)
                    k = attn.apply_k_norm(k)
                    k = attn.apply_k_rope(positions, k)
                    k = k.view(-1, attn.num_kv_heads, attn.head_dim)
                    v = v.view(-1, attn.num_kv_heads, attn.head_dim)

                    self.draft_model_runner.token_to_kv_pool.set_kv_buffer_prefix_valid(
                        attn.attn,
                        cache_loc_2d,
                        commit_lens,
                        k,
                        v,
                        attn.attn.k_scale,
                        attn.attn.v_scale,
                    )
                return

            if self._use_fused_kv_materialize and self._fused_kv_helper is not None:
                try:
                    self._append_target_hidden_fused(
                        ctx_hidden=ctx_hidden,
                        ctx_positions=positions,
                        ctx_cache_loc=cache_loc,
                    )
                    return
                except Exception as e:
                    logger.warning(
                        "DFLASH fused KV append-by-loc failed; falling back to sequential path: %s",
                        e,
                    )
                    self._use_fused_kv_materialize = False
                    self._fused_kv_helper = None

            self._append_target_hidden_sequential(
                ctx_hidden=ctx_hidden,
                ctx_positions=positions,
                ctx_cache_loc=cache_loc,
            )

    def _append_target_hidden_sequential(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
    ) -> None:
        for layer in self.draft_model.layers:
            attn = layer.self_attn
            if _is_npu:
                _, k, v = attn.forward_prepare_npu(ctx_positions, ctx_hidden)
            else:
                k, v = attn.kv_proj_only(ctx_hidden)
                k = attn.apply_k_norm(k)
                k = attn.apply_k_rope(ctx_positions, k)
            k = k.view(-1, attn.num_kv_heads, attn.head_dim)
            v = v.view(-1, attn.num_kv_heads, attn.head_dim)
            self.draft_model_runner.token_to_kv_pool.set_kv_buffer(
                attn.attn,
                ctx_cache_loc,
                k,
                v,
                attn.attn.k_scale,
                attn.attn.v_scale,
            )

    def _append_target_hidden_fused(
        self,
        ctx_hidden: torch.Tensor,
        ctx_positions: torch.Tensor,
        ctx_cache_loc: torch.Tensor,
        ctx_cache_loc_2d: Optional[torch.Tensor] = None,
        commit_lens: Optional[torch.Tensor] = None,
    ) -> None:
        """Fused KV materialization using batched projection + Triton kernel."""
        token_to_kv_pool = self.draft_model_runner.token_to_kv_pool
        if self._fused_kv_helper is None:
            raise RuntimeError("DFLASH fused KV helper is not initialized.")

        def _write_layer_kv(
            layer_idx: int,
            cache_k: torch.Tensor,
            cache_v: torch.Tensor,
        ) -> None:
            attn = self.draft_model.layers[layer_idx].self_attn.attn
            if ctx_cache_loc_2d is not None and commit_lens is not None:
                token_to_kv_pool.set_kv_buffer_prefix_valid(
                    attn,
                    ctx_cache_loc_2d,
                    commit_lens,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )
            else:
                token_to_kv_pool.set_kv_buffer(
                    attn,
                    ctx_cache_loc,
                    cache_k,
                    cache_v,
                    attn.k_scale,
                    attn.v_scale,
                )

        self._fused_kv_helper.materialize(
            ctx_hidden=ctx_hidden,
            positions=ctx_positions,
            write_layer_kv=_write_layer_kv,
        )

    def _update_target_mamba_state_after_verify(
        self,
        *,
        batch: ScheduleBatch,
        seq_lens_pre_verify: torch.Tensor,
        commit_lens: torch.Tensor,
    ) -> None:
        """Commit Mamba intermediate states for accepted verify steps.

        During TARGET_VERIFY, Mamba kernels run with `disable_state_update=True` and
        cache per-step intermediate states. After acceptance, we need to commit the
        state corresponding to each request's last accepted step.
        """
        attn_backend = self.target_worker.model_runner.attn_backend
        if not hasattr(attn_backend, "update_mamba_state_after_mtp_verify"):
            return

        last_correct_step_indices = commit_lens.to(torch.int64) - 1
        mamba_steps_to_track = None

        if batch.mamba_track_indices is not None:
            mamba_track_interval = self.server_args.mamba_track_interval
            to_track_mask = (
                seq_lens_pre_verify // mamba_track_interval
                != batch.seq_lens // mamba_track_interval
            )
            tracking_point = (
                batch.seq_lens // mamba_track_interval * mamba_track_interval
            )
            to_track_ith = torch.clamp(tracking_point - seq_lens_pre_verify - 1, min=0)
            can_track_mask = to_track_mask & (
                to_track_ith < commit_lens.to(to_track_ith.dtype)
            )
            mamba_steps_to_track = torch.where(
                can_track_mask,
                to_track_ith.to(torch.int64),
                torch.full_like(to_track_ith, -1, dtype=torch.int64),
            )

        attn_backend.update_mamba_state_after_mtp_verify(
            last_correct_step_indices=last_correct_step_indices,
            mamba_track_indices=batch.mamba_track_indices,
            mamba_steps_to_track=mamba_steps_to_track,
            model=self.target_worker.model_runner.model,
        )

    def _ensure_accept_bonus_buffers(self, bs: int) -> None:
        if self._accept_bonus_buffer_cap >= int(bs):
            return

        new_cap = max(
            int(bs),
            (
                self._accept_bonus_buffer_cap * 2
                if self._accept_bonus_buffer_cap > 0
                else int(bs)
            ),
        )
        device = self.device
        block_size = int(self.block_size)
        self._accept_len_buf = torch.empty((new_cap,), dtype=torch.int32, device=device)
        self._commit_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._bonus_id_bufs = [
            torch.empty((new_cap,), dtype=torch.int32, device=device) for _ in range(2)
        ]
        self._out_tokens_bufs = [
            torch.empty((new_cap, block_size), dtype=torch.int64, device=device)
            for _ in range(2)
        ]
        self._new_seq_lens_bufs = [
            torch.empty((new_cap,), dtype=torch.int64, device=device) for _ in range(2)
        ]
        self._accept_bonus_buffer_cap = new_cap

    def _next_accept_bonus_buffers(self, bs: int) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        self._ensure_accept_bonus_buffers(bs)
        assert self._accept_len_buf is not None
        slot = self._accept_bonus_buffer_slot
        self._accept_bonus_buffer_slot = (slot + 1) % 2
        return (
            self._accept_len_buf[:bs],
            self._commit_lens_bufs[slot][:bs],
            self._bonus_id_bufs[slot][:bs],
            self._out_tokens_bufs[slot][:bs],
            self._new_seq_lens_bufs[slot][:bs],
        )

    def _validate_phase1_sampling_support(
        self, model_worker_batch: ScheduleBatch
    ) -> None:
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is None or sampling_info.is_all_greedy:
            return

        if (
            not is_dflash_sampling_verify_available()
            and not self._warned_sampling_fallback
            and self.tp_rank == 0
        ):
            logger.warning(
                "DFLASH non-greedy verification is unavailable on this build/device; "
                "falling back to greedy argmax verification."
            )
            self._warned_sampling_fallback = True

    def _make_next_draft_input_prefill(
        self,
        *,
        bonus_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(seq_lens.numel())
        device = bonus_tokens.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _make_next_draft_input_decode(
        self,
        *,
        bonus_tokens: torch.Tensor,
        new_seq_lens: torch.Tensor,
        verify_done: Optional[torch.cuda.Event] = None,
        cur_allocated_seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> DFlashDraftInputV2:
        bs = int(new_seq_lens.numel())
        device = bonus_tokens.device
        return DFlashDraftInputV2(
            topk_p=torch.empty((bs, 0), device=device, dtype=torch.float32),
            topk_index=torch.empty((bs, 0), device=device, dtype=torch.int64),
            bonus_tokens=bonus_tokens.to(dtype=torch.int64),
            new_seq_lens=new_seq_lens.to(dtype=torch.int64),
            hidden_states=torch.empty((bs, 0), device=device, dtype=torch.float16),
            verify_done=verify_done,
            cur_allocated_seq_lens_cpu=cur_allocated_seq_lens_cpu,
        )

    def _forward_batch_generation_tree(
        self,
        model_worker_batch: ScheduleBatch,
        draft_input: DFlashDraftInputV2,
        on_publish=None,
    ) -> GenerationBatchResult:
        sampling_info = model_worker_batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise NotImplementedError(
                "DFLASH tree mode currently supports greedy verification only. "
                "Lossless sampling tree acceptance is a follow-up."
            )

        bs = len(model_worker_batch.seq_lens)
        device = self.device
        block_size = int(self.block_size)
        tree_budget = int(self.tree_budget)
        tree_width = int(self.tree_width)

        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DFLASH requires the target model to expose `lm_head` with `weight`."
            )

        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        linear_draft_tokens = self._draft_block_tokens_buf[:bs]
        prefix_lens = model_worker_batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
        if not self._use_triton_prepare_block:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=prefix_lens + block_size,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])
        draft_positions = positions_2d.reshape(-1)
        draft_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                draft_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            draft_seq_lens = prefix_lens
            seq_lens_cpu.copy_(
                (prefix_lens + block_size).to(device="cpu", dtype=torch.int32)
            )
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=draft_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=draft_positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, block_size, -1)
        depth_count = max(block_size - 1, 0)
        topk_logprobs, topk_tokens = self._topk_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
            k=tree_width,
        )
        topk_logprobs = topk_logprobs.view(bs, depth_count, tree_width)
        topk_tokens = topk_tokens.view(bs, depth_count, tree_width)
        linear_draft_tokens[:, 0].copy_(block_ids[:, 0])
        if depth_count > 0:
            linear_draft_tokens[:, 1:].copy_(topk_tokens[:, :, 0])

        tree_tokens = torch.zeros((bs, tree_budget), dtype=torch.long, device=device)
        tree_parents = torch.zeros((bs, tree_budget), dtype=torch.long, device=device)
        tree_depths = torch.ones((bs, tree_budget), dtype=torch.long, device=device)
        num_real_nodes: list[int] = []
        tree_parent_rows: list[torch.Tensor] = []
        for i in range(bs):
            tree: DraftTreeCPU = build_tree_from_topk_cpu(
                int(block_ids[i, 0].item()),
                topk_tokens[i],
                topk_logprobs[i],
                tree_budget,
                depth_first=False,
                score_mode=self.tree_draft,
            )
            num_nodes = int(tree.num_nodes)
            num_real_nodes.append(num_nodes)
            token_row = torch.tensor(tree.token_ids, dtype=torch.long, device=device)
            parent_row = torch.tensor(
                tree.parent_indices, dtype=torch.long, device=device
            )
            depth_row = torch.tensor(tree.depths, dtype=torch.long, device=device)
            tree_tokens[i, :num_nodes].copy_(token_row)
            tree_parents[i, :num_nodes].copy_(parent_row)
            tree_depths[i, :num_nodes].copy_(depth_row)
            tree_parent_rows.append(tree_parents[i])

        tree_positions_2d = prefix_lens.to(torch.int64).unsqueeze(1) + tree_depths
        tree_cache_loc = assign_extend_cache_locs_func(
            req_pool_indices=model_worker_batch.req_pool_indices,
            req_to_token=self.model_runner.req_to_token_pool.req_to_token,
            start_offset=prefix_lens,
            end_offset=prefix_lens + tree_budget,
            batch_size=bs,
            draft_token_num=tree_budget,
            device=device,
        )
        tree_cache_loc_2d = tree_cache_loc.view(bs, tree_budget)

        custom_mask = build_tree_custom_mask(
            parent_indices_batch=tree_parent_rows,
            num_real_nodes=num_real_nodes,
            prefix_lens=prefix_lens,
            num_verify_tokens=tree_budget,
            device=device,
        )
        retrieve_index_rows = []
        retrieve_next_token_rows = []
        retrieve_next_sibling_rows = []
        for i, parents in enumerate(tree_parent_rows):
            retrieve_index, retrieve_next_token, retrieve_next_sibling = (
                build_retrieve_links_from_parents(
                    parents,
                    num_verify_tokens=tree_budget,
                    num_nodes=num_real_nodes[i],
                    row_offset=i * tree_budget,
                    token_ids=tree_tokens[i],
                )
            )
            retrieve_index_rows.append(retrieve_index)
            retrieve_next_token_rows.append(retrieve_next_token)
            retrieve_next_sibling_rows.append(retrieve_next_sibling)

        retrieve_index = torch.stack(retrieve_index_rows, dim=0)
        retrieve_next_token = torch.stack(retrieve_next_token_rows, dim=0)
        retrieve_next_sibling = torch.stack(retrieve_next_sibling_rows, dim=0)

        target_model_runner = self.target_worker.model_runner
        active_target_attn_backend = target_model_runner.attn_backend
        while getattr(active_target_attn_backend, "full_attn_backend", None) is not None:
            active_target_attn_backend = active_target_attn_backend.full_attn_backend
        use_flashinfer_compact_tree = (
            type(active_target_attn_backend).__name__ == "FlashInferAttnBackend"
            and os.environ.get("SGLANG_DFLASH_TREE_COMPACT_FLASHINFER", "0") == "1"
        )
        use_flashinfer_expanded_causal_tree = (
            type(active_target_attn_backend).__name__ == "FlashInferAttnBackend"
            and self._tree_verify_attn_backend is None
            and not use_flashinfer_compact_tree
            and os.environ.get("SGLANG_DFLASH_TREE_EXPANDED_CAUSAL", "1") != "0"
        )

        compact_kv_indices = None
        compact_kv_indptr = None
        compact_qo_indptr = None
        compact_seq_lens = None
        compact_seq_lens_cpu = None
        compact_req_pool_indices = None
        expanded_tokens = None
        expanded_positions = None
        expanded_cache_loc = None
        expanded_cache_loc_3d = None
        expanded_req_pool_indices = None
        expanded_seq_lens = None
        expanded_seq_lens_cpu = None
        expanded_mamba_indices = None
        if use_flashinfer_expanded_causal_tree:
            use_mamba_expanded_rows = target_model_runner.mambaish_config is not None
            req_to_token_pool = self.model_runner.req_to_token_pool
            req_to_token = req_to_token_pool.req_to_token
            path_offsets = torch.arange(block_size, dtype=torch.long, device=device)
            expanded_tokens = torch.full(
                (bs, tree_budget, block_size),
                int(self._mask_token_id),
                dtype=torch.long,
                device=device,
            )
            expanded_positions = (
                prefix_lens.to(torch.int64).view(bs, 1, 1)
                + path_offsets.view(1, 1, block_size)
            ).expand(bs, tree_budget, block_size).contiguous()
            expanded_cache_loc = alloc_token_slots(
                model_worker_batch.tree_cache, bs * tree_budget * block_size
            )
            expanded_cache_loc_3d = expanded_cache_loc.view(
                bs, tree_budget, block_size
            )
            if os.environ.get("SGLANG_DFLASH_TREE_ALLOC_DEBUG"):
                torch.get_device_module(device).synchronize()
                alloc_limit = (
                    model_worker_batch.token_to_kv_pool_allocator.size
                    + model_worker_batch.token_to_kv_pool_allocator.page_size
                )
                loc_cpu = expanded_cache_loc.detach().cpu()
                loc_min = int(loc_cpu.min().item()) if loc_cpu.numel() else 0
                loc_max = int(loc_cpu.max().item()) if loc_cpu.numel() else 0
                loc_unique = int(torch.unique(loc_cpu).numel())
                logger.info(
                    "DFLASH expanded scratch alloc: n=%s unique=%s min=%s max=%s limit=%s available=%s",
                    int(loc_cpu.numel()),
                    loc_unique,
                    loc_min,
                    loc_max,
                    alloc_limit,
                    model_worker_batch.token_to_kv_pool_allocator.available_size(),
                )
                if loc_min < 0 or loc_max >= alloc_limit:
                    raise RuntimeError(
                        "DFLASH expanded scratch allocation returned OOB KV slots: "
                        f"n={int(loc_cpu.numel())} unique={loc_unique} "
                        f"min={loc_min} max={loc_max} limit={alloc_limit}"
                    )

            num_expanded_rows = bs * tree_budget
            expanded_seq_lens = prefix_lens.repeat_interleave(tree_budget)
            expanded_seq_lens_cpu = expanded_seq_lens.to(
                device="cpu", dtype=torch.int32
            )
            if use_mamba_expanded_rows:
                compact_req_pool_indices = (
                    model_worker_batch.req_pool_indices.repeat_interleave(tree_budget)
                )
                compact_parts = []
                compact_lens = []
                for row in range(bs):
                    req_row = int(model_worker_batch.req_pool_indices[row].item())
                    prefix_len = int(prefix_lens[row].item())
                    prefix_locs = req_to_token[req_row, :prefix_len].to(torch.int32)
                    for _ in range(tree_budget):
                        compact_parts.append(prefix_locs)
                        compact_lens.append(prefix_len)
                compact_seq_lens = torch.tensor(
                    compact_lens, dtype=torch.int32, device=device
                )
                compact_seq_lens_cpu = compact_seq_lens.to(device="cpu")
                compact_kv_indptr = torch.empty(
                    (num_expanded_rows + 1,), dtype=torch.int32, device=device
                )
                compact_kv_indptr[0] = 0
                compact_kv_indptr[1:] = torch.cumsum(compact_seq_lens, dim=0)
                compact_kv_indices = torch.cat(compact_parts, dim=0)
                compact_qo_indptr = torch.arange(
                    0,
                    num_expanded_rows * block_size + 1,
                    step=block_size,
                    dtype=torch.int32,
                    device=device,
                )
                expanded_mamba_indices = req_to_token_pool.mamba_allocator.alloc(
                    num_expanded_rows
                )
                if expanded_mamba_indices is None:
                    raise RuntimeError(
                        "DFLASH expanded causal tree verify needs temporary mamba "
                        f"slots for {num_expanded_rows} tree paths, but only "
                        f"{req_to_token_pool.mamba_allocator.available_size()} are free."
                    )
                original_mamba_indices = req_to_token_pool.get_mamba_indices(
                    model_worker_batch.req_pool_indices
                )[:bs].to(torch.long)
                req_to_token_pool.mamba_pool.copy_from(
                    original_mamba_indices.repeat_interleave(tree_budget),
                    expanded_mamba_indices.to(torch.long),
                )
            else:
                if num_expanded_rows > req_to_token_pool.available_size():
                    raise RuntimeError(
                        "DFLASH tree verify needs temporary request rows for FlashInfer "
                        f"expanded causal verify, but only {req_to_token_pool.available_size()} "
                        f"are free for {num_expanded_rows} tree nodes."
                    )
                expanded_req_pool_indices_cpu = req_to_token_pool.free_slots[
                    :num_expanded_rows
                ]
                del req_to_token_pool.free_slots[:num_expanded_rows]
                expanded_req_pool_indices = torch.tensor(
                    expanded_req_pool_indices_cpu,
                    dtype=torch.int32,
                    device=device,
                )

            for row in range(bs):
                req_row = int(model_worker_batch.req_pool_indices[row].item())
                prefix_len = int(prefix_lens[row].item())
                prefix_locs = req_to_token[req_row, :prefix_len].to(torch.int32)
                parents = tree_parents[row, : num_real_nodes[row]].tolist()
                for node in range(tree_budget):
                    path: list[int]
                    if node < num_real_nodes[row]:
                        path = []
                        cur = node
                        while cur >= 0 and len(path) < block_size:
                            path.append(cur)
                            cur = int(parents[cur])
                        path.reverse()
                    else:
                        path = [0]
                    path_tensor = torch.tensor(path, dtype=torch.long, device=device)
                    path_len = int(path_tensor.numel())
                    expanded_tokens[row, node, :path_len].copy_(
                        torch.index_select(tree_tokens[row], 0, path_tensor)
                    )
                    if expanded_req_pool_indices is not None:
                        temp_req_row = int(
                            expanded_req_pool_indices_cpu[row * tree_budget + node]
                        )
                        req_to_token[temp_req_row, :prefix_len].copy_(prefix_locs)
                        req_to_token[
                            temp_req_row, prefix_len : prefix_len + block_size
                        ].copy_(
                            expanded_cache_loc_3d[row, node].to(req_to_token.dtype)
                        )
        elif use_flashinfer_compact_tree:
            req_to_token = self.model_runner.req_to_token_pool.req_to_token
            compact_parts: list[torch.Tensor] = []
            compact_lens: list[int] = []
            for row in range(bs):
                req_row = int(model_worker_batch.req_pool_indices[row].item())
                prefix_len = int(prefix_lens[row].item())
                prefix_locs = req_to_token[req_row, :prefix_len].to(torch.int32)
                parents = tree_parents[row, : num_real_nodes[row]].tolist()
                for node in range(tree_budget):
                    if node < num_real_nodes[row]:
                        path: list[int] = []
                        cur = node
                        while cur >= 0:
                            path.append(cur)
                            cur = int(parents[cur])
                        path.reverse()
                        path_locs = tree_cache_loc_2d[
                            row,
                            torch.tensor(path, dtype=torch.long, device=device),
                        ].to(torch.int32)
                    else:
                        path_locs = tree_cache_loc_2d[row, node : node + 1].to(
                            torch.int32
                        )
                    compact_parts.append(torch.cat((prefix_locs, path_locs), dim=0))
                    compact_lens.append(prefix_len + int(path_locs.numel()))

            compact_seq_lens = torch.tensor(
                compact_lens, dtype=torch.int32, device=device
            )
            compact_seq_lens_cpu = compact_seq_lens.to(device="cpu")
            compact_kv_indptr = torch.empty(
                (bs * tree_budget + 1,), dtype=torch.int32, device=device
            )
            compact_kv_indptr[0] = 0
            compact_kv_indptr[1:] = torch.cumsum(compact_seq_lens, dim=0)
            compact_kv_indices = torch.cat(compact_parts, dim=0)
            compact_qo_indptr = torch.arange(
                0, bs * tree_budget + 1, dtype=torch.int32, device=device
            )
            compact_req_pool_indices = model_worker_batch.req_pool_indices.repeat_interleave(
                tree_budget
            )

        verify_draft_token = (
            expanded_tokens.reshape(-1)
            if use_flashinfer_expanded_causal_tree
            else tree_tokens.reshape(-1)
        )
        verify_positions = (
            expanded_positions.reshape(-1)
            if use_flashinfer_expanded_causal_tree
            else tree_positions_2d.reshape(-1)
        )
        verify_draft_token_num = (
            block_size
            if use_flashinfer_expanded_causal_tree
            else (1 if use_flashinfer_compact_tree else tree_budget)
        )
        verify_input = DFlashVerifyInput(
            draft_token=verify_draft_token,
            positions=verify_positions,
            draft_token_num=verify_draft_token_num,
            topk=(
                1
                if use_flashinfer_expanded_causal_tree
                and target_model_runner.mambaish_config is not None
                else tree_width
            ),
            custom_mask=(
                None
                if (use_flashinfer_compact_tree or use_flashinfer_expanded_causal_tree)
                else custom_mask
            ),
            retrieve_index=(
                None
                if use_flashinfer_expanded_causal_tree
                and target_model_runner.mambaish_config is not None
                else retrieve_index
            ),
            retrieve_next_token=(
                None
                if use_flashinfer_expanded_causal_tree
                and target_model_runner.mambaish_config is not None
                else retrieve_next_token
            ),
            retrieve_next_sibling=(
                None
                if use_flashinfer_expanded_causal_tree
                and target_model_runner.mambaish_config is not None
                else retrieve_next_sibling
            ),
            max_tree_depth=block_size,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            allow_cuda_graph=not use_flashinfer_compact_tree,
            compact_kv_indices=compact_kv_indices,
            compact_kv_indptr=compact_kv_indptr,
            compact_qo_indptr=compact_qo_indptr,
            force_causal=use_flashinfer_expanded_causal_tree,
            mamba_cache_indices=(
                expanded_mamba_indices.to(torch.int32)
                if expanded_mamba_indices is not None
                else None
            ),
            num_tokens_per_batch=verify_draft_token_num,
        )
        use_reverify_commit = (
            target_model_runner.mambaish_config is not None
            and not use_flashinfer_expanded_causal_tree
            and os.environ.get("SGLANG_DFLASH_TREE_REVERIFY_ACCEPTED_COMMIT", "1")
            != "0"
        )
        expanded_can_run_cuda_graph = False

        target_attn_backend_backup = None
        if self._tree_verify_attn_backend is not None:
            verify_input.allow_cuda_graph = False
            target_attn_backend_backup = target_model_runner.attn_backend
            target_model_runner.attn_backend = self._tree_verify_attn_backend
        verify_context = (
            forward_context(ForwardContext(attn_backend=self._tree_verify_attn_backend))
            if self._tree_verify_attn_backend is not None
            else nullcontext()
        )
        seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
        seq_lens_sum_backup = model_worker_batch.seq_lens_sum
        pre_tree_mamba_snapshot = (
            self._snapshot_persistent_mamba_state(model_worker_batch, bs)
            if use_reverify_commit
            else None
        )
        compare_persistent_mamba_snapshot = None
        if os.environ.get("SGLANG_DFLASH_TREE_COMPARE_CAUSAL"):
            linear_backend = getattr(
                target_model_runner.attn_backend, "linear_attn_backend", None
            )
            if linear_backend is not None:
                try:
                    mamba_caches = (
                        linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                    )
                    state_indices = linear_backend.req_to_token_pool.get_mamba_indices(
                        model_worker_batch.req_pool_indices
                    )[:bs].to(torch.long)
                    compare_persistent_mamba_snapshot = (
                        state_indices,
                        mamba_caches.temporal[:, state_indices].clone(),
                        mamba_caches.conv[0][:, state_indices].clone(),
                    )
                except Exception:
                    logger.exception(
                        "DFLASH tree causal compare failed to snapshot persistent mamba state"
                    )
                    compare_persistent_mamba_snapshot = None

        def _restore_compare_persistent_mamba_state() -> None:
            if compare_persistent_mamba_snapshot is None:
                return
            linear_backend = getattr(
                target_model_runner.attn_backend, "linear_attn_backend", None
            )
            if linear_backend is None:
                return
            try:
                mamba_caches = (
                    linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                )
                state_indices, temporal_snapshot, conv_snapshot = (
                    compare_persistent_mamba_snapshot
                )
                mamba_caches.temporal[:, state_indices] = temporal_snapshot
                mamba_caches.conv[0][:, state_indices] = conv_snapshot
            except Exception:
                logger.exception(
                    "DFLASH tree causal compare failed to restore persistent mamba state"
                )

        target_forward_ok = False
        skip_attn_backend_init = True
        try:
            with verify_context:
                with self._maybe_use_tree_verify_full_attn_backend(
                    verify_input
                ) as swapped_full_attn_backend:
                    if swapped_full_attn_backend:
                        graph_runner = target_model_runner.decode_cuda_graph_runner
                        if not getattr(
                            graph_runner,
                            "dflash_tree_graph_uses_full_attn_replacement",
                            False,
                        ):
                            verify_input.allow_cuda_graph = False

                    model_worker_batch.out_cache_loc = tree_cache_loc
                    if use_flashinfer_expanded_causal_tree:
                        expanded_forward_req_pool_indices = (
                            expanded_req_pool_indices
                            if expanded_req_pool_indices is not None
                            else compact_req_pool_indices
                        )
                        assert expanded_forward_req_pool_indices is not None
                        assert expanded_seq_lens is not None
                        assert expanded_seq_lens_cpu is not None
                        assert expanded_cache_loc is not None
                        verify_forward_batch = ForwardBatch(
                            forward_mode=ForwardMode.TARGET_VERIFY,
                            batch_size=bs * tree_budget,
                            input_ids=verify_draft_token,
                            req_pool_indices=expanded_forward_req_pool_indices,
                            seq_lens=expanded_seq_lens,
                            out_cache_loc=expanded_cache_loc,
                            seq_lens_sum=int(expanded_seq_lens_cpu.sum().item()),
                            seq_lens_cpu=expanded_seq_lens_cpu,
                            positions=verify_positions,
                            spec_algorithm=SpeculativeAlgorithm.DFLASH,
                            spec_info=verify_input,
                            capture_hidden_mode=CaptureHiddenMode.FULL,
                        )
                        if target_model_runner.model_is_mrope:
                            verify_forward_batch.mrope_positions = (
                                verify_positions.unsqueeze(0).repeat(3, 1)
                            )
                        verify_forward_batch.allow_cuda_graph = (
                            verify_input.allow_cuda_graph
                        )
                        expanded_can_run_cuda_graph = bool(
                            verify_forward_batch.allow_cuda_graph
                            and target_model_runner.decode_cuda_graph_runner
                            and target_model_runner.decode_cuda_graph_runner.can_run_graph(
                                verify_forward_batch
                            )
                        )
                        if expanded_can_run_cuda_graph:
                            skip_attn_backend_init = None
                        else:
                            verify_forward_batch.allow_cuda_graph = False
                            target_model_runner.attn_backend.init_forward_metadata(
                                verify_forward_batch
                            )
                    elif use_flashinfer_compact_tree:
                        assert compact_req_pool_indices is not None
                        assert compact_seq_lens is not None
                        assert compact_seq_lens_cpu is not None
                        verify_forward_batch = ForwardBatch(
                            forward_mode=ForwardMode.TARGET_VERIFY,
                            batch_size=bs * tree_budget,
                            input_ids=verify_draft_token,
                            req_pool_indices=compact_req_pool_indices,
                            seq_lens=compact_seq_lens,
                            out_cache_loc=(
                                expanded_cache_loc
                                if use_flashinfer_expanded_causal_tree
                                else tree_cache_loc
                            ),
                            seq_lens_sum=int(compact_seq_lens_cpu.sum().item()),
                            seq_lens_cpu=compact_seq_lens_cpu,
                            positions=verify_positions,
                            spec_algorithm=SpeculativeAlgorithm.DFLASH,
                            spec_info=verify_input,
                            capture_hidden_mode=CaptureHiddenMode.FULL,
                        )
                        if target_model_runner.model_is_mrope:
                            verify_forward_batch.mrope_positions = (
                                verify_positions.unsqueeze(0).repeat(3, 1)
                            )
                        verify_forward_batch.allow_cuda_graph = False
                        target_model_runner.attn_backend.init_forward_metadata(
                            verify_forward_batch
                        )
                    else:
                        if draft_input.planning_seq_lens_cpu is not None:
                            model_worker_batch.seq_lens_cpu = (
                                draft_input.planning_seq_lens_cpu
                            )
                            model_worker_batch.seq_lens_sum = int(
                                draft_input.planning_seq_lens_sum
                            )
                        elif draft_input.reserved_seq_lens_cpu is not None:
                            model_worker_batch.seq_lens_cpu = (
                                draft_input.reserved_seq_lens_cpu
                            )
                            model_worker_batch.seq_lens_sum = int(
                                draft_input.reserved_seq_lens_sum
                            )

                        verify_forward_batch, _ = verify_input.prepare_for_verify(
                            model_worker_batch, self.target_worker
                        )
                        model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
                        model_worker_batch.seq_lens_sum = seq_lens_sum_backup

                    target_out = self.target_worker.forward_batch_generation(
                        batch=None,
                        forward_batch=verify_forward_batch,
                        is_verify=True,
                        skip_attn_backend_init=skip_attn_backend_init,
                    )
                    target_forward_ok = True
        finally:
            if target_attn_backend_backup is not None:
                target_model_runner.attn_backend = target_attn_backend_backup
            if expanded_req_pool_indices is not None:
                if (
                    expanded_mamba_indices is not None
                    and hasattr(
                        self.model_runner.req_to_token_pool,
                        "req_index_to_mamba_index_mapping",
                    )
                ):
                    self.model_runner.req_to_token_pool.req_index_to_mamba_index_mapping[
                        expanded_req_pool_indices.to(torch.long)
                    ] = 0
                self.model_runner.req_to_token_pool.free_slots.extend(
                    expanded_req_pool_indices.detach().cpu().tolist()
                )
                expanded_req_pool_indices = None
            if expanded_mamba_indices is not None and not target_forward_ok:
                self.model_runner.req_to_token_pool.mamba_allocator.free(
                    expanded_mamba_indices
                )
                expanded_mamba_indices = None
            model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
            model_worker_batch.seq_lens_sum = seq_lens_sum_backup
            if (
                use_reverify_commit
                and not target_forward_ok
                and pre_tree_mamba_snapshot is not None
            ):
                self._restore_persistent_mamba_state(pre_tree_mamba_snapshot)
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph
        if (
            use_flashinfer_expanded_causal_tree
            and can_run_cuda_graph
            and os.environ.get(
                "SGLANG_DFLASH_TREE_GRAPH_REVERIFY_ACCEPTED_COMMIT", "1"
            )
            != "0"
        ):
            use_reverify_commit = True
        if use_reverify_commit and pre_tree_mamba_snapshot is not None:
            self._restore_persistent_mamba_state(pre_tree_mamba_snapshot)
        if use_flashinfer_expanded_causal_tree:
            depth_index = tree_depths.to(torch.long).clamp(max=block_size - 1)
            expanded_next_token_logits = logits_output.next_token_logits.view(
                bs, tree_budget, block_size, -1
            )
            logits_output.next_token_logits = torch.gather(
                expanded_next_token_logits,
                2,
                depth_index[:, :, None, None].expand(
                    bs, tree_budget, 1, expanded_next_token_logits.shape[-1]
                ),
            ).squeeze(2).reshape(bs * tree_budget, -1)
        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=tree_budget,
            )

        tree_target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
            bs, tree_budget
        )
        num_real_nodes_tensor = torch.tensor(
            num_real_nodes, dtype=torch.long, device=device
        )
        predict, accept_index, num_correct_drafts = tree_accept_greedy_batched(
            tree_tokens=tree_tokens,
            parent_indices=tree_parents,
            depths=tree_depths,
            target_tokens=tree_target_predict,
            num_real_nodes=num_real_nodes_tensor,
            max_tree_depth=block_size - 1,
        )

        accept_cap = os.environ.get("SGLANG_DFLASH_TREE_ACCEPT_CAP")
        if accept_cap is not None:
            cap = max(0, min(int(accept_cap), block_size - 1))
            num_correct_drafts.clamp_(max=cap)
            accept_index[:, cap + 1 :].fill_(-1)

        commit_lens = num_correct_drafts + 1
        safe_accept_index = accept_index.to(torch.long).clamp(min=0)
        row_offsets = torch.arange(
            0, bs * tree_budget, tree_budget, dtype=torch.long, device=device
        ).unsqueeze(1)
        accept_index_local = (safe_accept_index - row_offsets).clamp(min=0)
        accept_tokens = predict[safe_accept_index].to(torch.int64)
        out_tokens = torch.zeros((bs, tree_budget), dtype=torch.int64, device=device)
        out_tokens[:, :block_size].copy_(accept_tokens)
        req_indices = torch.arange(bs, dtype=torch.long, device=device)
        bonus = accept_tokens[req_indices, (commit_lens - 1).to(torch.long)]

        if os.environ.get("SGLANG_DFLASH_TREE_DEBUG"):
            step = int(getattr(self, "_dflash_tree_debug_step", 0))
            self._dflash_tree_debug_step = step + 1
            if step < int(os.environ.get("SGLANG_DFLASH_TREE_DEBUG_STEPS", "32")):
                target_on_path = torch.gather(
                    tree_target_predict,
                    1,
                    accept_index_local,
                )
                draft_on_path = torch.gather(tree_tokens, 1, accept_index_local)
                parents_on_path = torch.gather(tree_parents, 1, accept_index_local)
                depths_on_path = torch.gather(tree_depths, 1, accept_index_local)
                prefix_overlap_counts = []
                tree_duplicate_counts = []
                req_to_token_debug = self.model_runner.req_to_token_pool.req_to_token
                for row in range(bs):
                    req_row = int(model_worker_batch.req_pool_indices[row].item())
                    prefix_len = int(prefix_lens[row].item())
                    prefix_locs = req_to_token_debug[req_row, :prefix_len].to(
                        torch.long
                    )
                    row_tree_locs = tree_cache_loc_2d[row].to(torch.long)
                    prefix_overlap_counts.append(
                        int(torch.isin(row_tree_locs, prefix_locs).sum().item())
                    )
                    tree_duplicate_counts.append(
                        int(row_tree_locs.numel() - torch.unique(row_tree_locs).numel())
                    )
                logger.info(
                    "DFLASH tree debug step=%s prefix=%s commit_lens=%s correct=%s accept_index=%s local=%s draft_on_path=%s target_on_path=%s parents_on_path=%s depths_on_path=%s emitted_tokens=%s tree_loc0=%s prefix_overlap=%s tree_dup=%s",
                    step,
                    prefix_lens.detach().cpu().tolist(),
                    commit_lens.detach().cpu().tolist(),
                    num_correct_drafts.detach().cpu().tolist(),
                    accept_index.detach().cpu().tolist(),
                    accept_index_local.detach().cpu().tolist(),
                    draft_on_path.detach().cpu().tolist(),
                    target_on_path.detach().cpu().tolist(),
                    parents_on_path.detach().cpu().tolist(),
                    depths_on_path.detach().cpu().tolist(),
                    accept_tokens.detach().cpu().tolist(),
                    tree_cache_loc_2d[:, 0].detach().cpu().tolist(),
                    prefix_overlap_counts,
                    tree_duplicate_counts,
                )
                if os.environ.get("SGLANG_DFLASH_TREE_DEBUG_FULL"):
                    logger.info(
                        "DFLASH tree debug full step=%s real_nodes=%s tree_tokens=%s parents=%s depths=%s target_predict=%s",
                        step,
                        num_real_nodes,
                        [
                            tree_tokens[row, : num_real_nodes[row]]
                            .detach()
                            .cpu()
                            .tolist()
                            for row in range(bs)
                        ],
                        [
                            tree_parents[row, : num_real_nodes[row]]
                            .detach()
                            .cpu()
                            .tolist()
                            for row in range(bs)
                        ],
                        [
                            tree_depths[row, : num_real_nodes[row]]
                            .detach()
                            .cpu()
                            .tolist()
                            for row in range(bs)
                        ],
                        [
                            tree_target_predict[row, : num_real_nodes[row]]
                            .detach()
                            .cpu()
                            .tolist()
                            for row in range(bs)
                        ],
                    )

        if os.environ.get("SGLANG_DFLASH_TREE_COMPARE_CAUSAL"):
            branch_cache_alloc = None
            scratch_start = tree_budget - block_size
            if scratch_start >= block_size:
                scratch_nodes = torch.arange(
                    scratch_start,
                    tree_budget,
                    dtype=torch.long,
                    device=device,
                )
                branch_cache_loc_2d = tree_cache_loc_2d[:, scratch_nodes].to(torch.long)
            else:
                branch_cache_alloc = alloc_token_slots(
                    model_worker_batch.tree_cache, bs * block_size
                )
                if branch_cache_alloc.numel() != bs * block_size:
                    logger.info(
                        "DFLASH tree causal compare skipped: failed to allocate "
                        "branch scratch slots, got=%s need=%s",
                        int(branch_cache_alloc.numel()),
                        int(bs * block_size),
                    )
                    branch_cache_alloc = None
                    scratch_nodes = torch.full(
                        (block_size,),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
                else:
                    scratch_nodes = torch.full(
                        (block_size,),
                        -1,
                        dtype=torch.long,
                        device=device,
                    )
                    branch_cache_loc_2d = branch_cache_alloc.view(bs, block_size)
            path_pos = torch.arange(block_size, dtype=torch.long, device=device)
            path_mask = path_pos.unsqueeze(0) < commit_lens.to(torch.long).unsqueeze(1)
            overlaps_scratch = (
                (accept_index_local.unsqueeze(-1) == scratch_nodes)
                & path_mask.unsqueeze(-1)
            ).any()
            if branch_cache_alloc is None and scratch_nodes[0].item() < 0:
                pass
            elif bool(overlaps_scratch.item()):
                logger.info(
                    "DFLASH tree causal compare skipped: accepted path overlaps scratch tail slots local=%s scratch=%s",
                    accept_index_local.detach().cpu().tolist(),
                    scratch_nodes.detach().cpu().tolist(),
                )
            else:
                _restore_compare_persistent_mamba_state()
                branch_candidates = torch.gather(
                    tree_tokens,
                    1,
                    accept_index_local,
                )
                branch_candidates = torch.where(
                    path_mask,
                    branch_candidates,
                    torch.full_like(branch_candidates, int(self._mask_token_id)),
                )
                branch_positions_2d = prefix_lens.to(torch.int64).unsqueeze(
                    1
                ) + path_pos.to(torch.int64).unsqueeze(0)
                branch_custom_mask = None
                if not use_flashinfer_expanded_causal_tree:
                    branch_mask_parts = []
                    for prefix_len in prefix_lens.to(
                        device="cpu", dtype=torch.int64
                    ).tolist():
                        prefix_mask = torch.ones(
                            (block_size, int(prefix_len)),
                            dtype=torch.bool,
                            device=device,
                        )
                        causal_mask = torch.tril(
                            torch.ones(
                                (block_size, block_size),
                                dtype=torch.bool,
                                device=device,
                            )
                        )
                        branch_mask_parts.append(
                            torch.cat((prefix_mask, causal_mask), dim=1).reshape(-1)
                        )
                    branch_custom_mask = torch.cat(branch_mask_parts, dim=0)
                branch_verify_input = DFlashVerifyInput(
                    draft_token=branch_candidates.reshape(-1),
                    positions=branch_positions_2d.reshape(-1),
                    draft_token_num=block_size,
                    custom_mask=branch_custom_mask,
                    force_causal=use_flashinfer_expanded_causal_tree,
                    capture_hidden_mode=CaptureHiddenMode.FULL,
                    allow_cuda_graph=False,
                )

                tree_mamba_snapshot = None
                tree_conv_snapshot = None
                tree_mamba_state_indices = None
                if os.environ.get("SGLANG_DFLASH_TREE_COMPARE_MAMBA"):
                    linear_backend = getattr(
                        target_model_runner.attn_backend, "linear_attn_backend", None
                    )
                    linear_metadata = getattr(linear_backend, "forward_metadata", None)
                    if (
                        linear_backend is not None
                        and linear_metadata is not None
                        and getattr(linear_metadata, "mamba_cache_indices", None)
                        is not None
                    ):
                        try:
                            mamba_caches = (
                                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                            )
                            tree_mamba_state_indices = (
                                linear_metadata.mamba_cache_indices[:bs].to(torch.long)
                            )
                            tree_mamba_rows = []
                            tree_conv_rows = []
                            for row in range(bs):
                                state_idx = tree_mamba_state_indices[row]
                                local_path = accept_index_local[row].to(torch.long)
                                tree_mamba_rows.append(
                                    mamba_caches.intermediate_ssm[
                                        :, state_idx, local_path
                                    ].clone()
                                )
                                tree_conv_rows.append(
                                    mamba_caches.intermediate_conv_window[0][
                                        :, state_idx, local_path
                                    ].clone()
                                )
                            tree_mamba_snapshot = torch.stack(
                                tree_mamba_rows, dim=1
                            )
                            tree_conv_snapshot = torch.stack(tree_conv_rows, dim=1)
                        except Exception:
                            logger.exception(
                                "DFLASH tree causal compare failed to snapshot mamba intermediates"
                            )
                            tree_mamba_snapshot = None
                            tree_conv_snapshot = None

                batch_input_ids_backup = model_worker_batch.input_ids
                batch_spec_info_backup = model_worker_batch.spec_info
                batch_forward_mode_backup = model_worker_batch.forward_mode
                batch_capture_hidden_mode_backup = model_worker_batch.capture_hidden_mode
                batch_out_cache_loc_backup = model_worker_batch.out_cache_loc
                seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
                seq_lens_sum_backup = model_worker_batch.seq_lens_sum
                req_to_token = self.model_runner.req_to_token_pool.req_to_token
                req_rows = model_worker_batch.req_pool_indices.to(torch.long).unsqueeze(
                    1
                )
                branch_req_to_token_backup = req_to_token[
                    req_rows,
                    branch_positions_2d,
                ].clone()
                req_to_token[req_rows, branch_positions_2d] = branch_cache_loc_2d.to(
                    req_to_token.dtype
                )
                branch_verify_context = (
                    forward_context(
                        ForwardContext(attn_backend=self._tree_verify_attn_backend)
                    )
                    if self._tree_verify_attn_backend is not None
                    else nullcontext()
                )
                try:
                    with branch_verify_context:
                        with self._maybe_use_tree_verify_full_attn_backend(
                            branch_verify_input
                        ) as swapped_full_attn_backend:
                            if swapped_full_attn_backend:
                                branch_verify_input.allow_cuda_graph = False
                            model_worker_batch.out_cache_loc = (
                                branch_cache_loc_2d.reshape(-1)
                            )
                            if draft_input.planning_seq_lens_cpu is not None:
                                model_worker_batch.seq_lens_cpu = (
                                    draft_input.planning_seq_lens_cpu
                                )
                                model_worker_batch.seq_lens_sum = int(
                                    draft_input.planning_seq_lens_sum
                                )
                            elif draft_input.reserved_seq_lens_cpu is not None:
                                model_worker_batch.seq_lens_cpu = (
                                    draft_input.reserved_seq_lens_cpu
                                )
                                model_worker_batch.seq_lens_sum = int(
                                    draft_input.reserved_seq_lens_sum
                                )

                            branch_forward_batch, _ = (
                                branch_verify_input.prepare_for_verify(
                                    model_worker_batch, self.target_worker
                                )
                            )
                            model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
                            model_worker_batch.seq_lens_sum = seq_lens_sum_backup
                            branch_target_out = (
                                self.target_worker.forward_batch_generation(
                                    batch=None,
                                    forward_batch=branch_forward_batch,
                                    is_verify=True,
                                    skip_attn_backend_init=True,
                                )
                            )
                finally:
                    req_to_token[req_rows, branch_positions_2d] = (
                        branch_req_to_token_backup
                    )
                    model_worker_batch.input_ids = batch_input_ids_backup
                    model_worker_batch.spec_info = batch_spec_info_backup
                    model_worker_batch.forward_mode = batch_forward_mode_backup
                    model_worker_batch.capture_hidden_mode = (
                        batch_capture_hidden_mode_backup
                    )
                    model_worker_batch.out_cache_loc = batch_out_cache_loc_backup
                    model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
                    model_worker_batch.seq_lens_sum = seq_lens_sum_backup
                    _restore_compare_persistent_mamba_state()

                branch_logits_output = branch_target_out.logits_output
                if sampling_info is not None:
                    apply_dflash_verify_logits_adjustments(
                        next_token_logits=branch_logits_output.next_token_logits,
                        sampling_info=sampling_info,
                        draft_token_num=block_size,
                    )

                branch_target_predict = torch.argmax(
                    branch_logits_output.next_token_logits, dim=-1
                ).view(bs, block_size)
                tree_predict_on_path = torch.gather(
                    tree_target_predict,
                    1,
                    accept_index_local,
                )

                same_shape_logits_output = None
                same_shape_cache_loc_3d = None
                same_shape_cache_alloc = None
                same_shape_req_pool_indices = None
                if (
                    use_flashinfer_expanded_causal_tree
                    and os.environ.get("SGLANG_DFLASH_TREE_COMPARE_SAME_SHAPE")
                ):
                    req_to_token_pool = self.model_runner.req_to_token_pool
                    num_temp_req_rows = bs * tree_budget
                    if num_temp_req_rows > req_to_token_pool.available_size():
                        logger.info(
                            "DFLASH tree same-shape compare skipped: only %s free request rows for %s tree rows",
                            req_to_token_pool.available_size(),
                            num_temp_req_rows,
                        )
                    else:
                        try:
                            same_shape_cache_alloc = alloc_token_slots(
                                model_worker_batch.tree_cache,
                                bs * tree_budget * block_size,
                            )
                            same_shape_cache_loc_3d = same_shape_cache_alloc.view(
                                bs, tree_budget, block_size
                            )
                            same_shape_req_pool_indices_cpu = (
                                req_to_token_pool.free_slots[:num_temp_req_rows]
                            )
                            del req_to_token_pool.free_slots[:num_temp_req_rows]
                            same_shape_req_pool_indices = torch.tensor(
                                same_shape_req_pool_indices_cpu,
                                dtype=torch.int32,
                                device=device,
                            )
                            same_shape_seq_lens = prefix_lens.repeat_interleave(
                                tree_budget
                            )
                            same_shape_seq_lens_cpu = same_shape_seq_lens.to(
                                device="cpu", dtype=torch.int32
                            )
                            same_shape_tokens = (
                                branch_candidates[:, None, :]
                                .expand(bs, tree_budget, block_size)
                                .contiguous()
                            )
                            same_shape_positions = (
                                branch_positions_2d[:, None, :]
                                .expand(bs, tree_budget, block_size)
                                .contiguous()
                            )
                            req_to_token = self.model_runner.req_to_token_pool.req_to_token
                            for row in range(bs):
                                req_row = int(
                                    model_worker_batch.req_pool_indices[row].item()
                                )
                                prefix_len = int(prefix_lens[row].item())
                                prefix_locs = req_to_token[
                                    req_row, :prefix_len
                                ].to(torch.int32)
                                for node in range(tree_budget):
                                    temp_req_row = int(
                                        same_shape_req_pool_indices_cpu[
                                            row * tree_budget + node
                                        ]
                                    )
                                    req_to_token[temp_req_row, :prefix_len].copy_(
                                        prefix_locs
                                    )
                                    req_to_token[
                                        temp_req_row,
                                        prefix_len : prefix_len + block_size,
                                    ].copy_(
                                        same_shape_cache_loc_3d[row, node].to(
                                            req_to_token.dtype
                                        )
                                    )
                            same_shape_verify_input = DFlashVerifyInput(
                                draft_token=same_shape_tokens.reshape(-1),
                                positions=same_shape_positions.reshape(-1),
                                draft_token_num=block_size,
                                custom_mask=None,
                                force_causal=True,
                                capture_hidden_mode=CaptureHiddenMode.FULL,
                                allow_cuda_graph=False,
                                num_tokens_per_batch=block_size,
                            )
                            same_shape_forward_batch = ForwardBatch(
                                forward_mode=ForwardMode.TARGET_VERIFY,
                                batch_size=bs * tree_budget,
                                input_ids=same_shape_verify_input.draft_token,
                                req_pool_indices=same_shape_req_pool_indices,
                                seq_lens=same_shape_seq_lens,
                                out_cache_loc=same_shape_cache_alloc,
                                seq_lens_sum=int(
                                    same_shape_seq_lens_cpu.sum().item()
                                ),
                                seq_lens_cpu=same_shape_seq_lens_cpu,
                                positions=same_shape_verify_input.positions,
                                spec_algorithm=SpeculativeAlgorithm.DFLASH,
                                spec_info=same_shape_verify_input,
                                capture_hidden_mode=CaptureHiddenMode.FULL,
                            )
                            same_shape_forward_batch.allow_cuda_graph = False
                            target_model_runner.attn_backend.init_forward_metadata(
                                same_shape_forward_batch
                            )
                            same_shape_target_out = (
                                self.target_worker.forward_batch_generation(
                                    batch=None,
                                    forward_batch=same_shape_forward_batch,
                                    is_verify=True,
                                    skip_attn_backend_init=True,
                                )
                            )
                            same_shape_logits_output = (
                                same_shape_target_out.logits_output
                            )
                        except Exception:
                            logger.exception(
                                "DFLASH tree same-shape causal compare failed"
                            )
                            if same_shape_req_pool_indices is not None:
                                req_to_token_pool.free_slots.extend(
                                    same_shape_req_pool_indices.detach()
                                    .cpu()
                                    .tolist()
                                )
                                same_shape_req_pool_indices = None
                            if same_shape_cache_alloc is not None:
                                model_worker_batch.tree_cache.token_to_kv_pool_allocator.free(
                                    same_shape_cache_alloc
                                )
                                same_shape_cache_alloc = None
                                same_shape_cache_loc_3d = None

                hidden_max_abs = None
                leaf_hidden_max_abs = None
                same_shape_hidden_max_abs = None
                kv_max_abs = None
                leaf_kv_max_abs = None
                same_shape_kv_max_abs = None
                mamba_max_abs = None
                conv_max_abs = None
                tree_hidden = logits_output.hidden_states
                branch_hidden = branch_logits_output.hidden_states
                if tree_hidden is not None and branch_hidden is not None:
                    if use_flashinfer_expanded_causal_tree:
                        tree_hidden = tree_hidden.view(bs, tree_budget, block_size, -1)
                        hidden_offsets_2d = (
                            torch.arange(block_size, dtype=torch.long, device=device)
                            .unsqueeze(0)
                            .expand_as(accept_index_local)
                        )
                        tree_hidden_on_path = tree_hidden[
                            req_indices.to(torch.long).unsqueeze(1),
                            accept_index_local.to(torch.long),
                            hidden_offsets_2d,
                        ]
                        leaf_index_2d = torch.gather(
                            accept_index_local,
                            1,
                            (commit_lens.to(torch.long) - 1).clamp(min=0).unsqueeze(1),
                        ).expand_as(accept_index_local)
                        leaf_hidden_on_path = tree_hidden[
                            req_indices.to(torch.long).unsqueeze(1),
                            leaf_index_2d.to(torch.long),
                            hidden_offsets_2d,
                        ]
                    else:
                        tree_hidden = tree_hidden.view(bs, tree_budget, -1)
                        tree_hidden_on_path = torch.gather(
                            tree_hidden,
                            1,
                            accept_index_local.unsqueeze(-1).expand(
                                bs, block_size, tree_hidden.shape[-1]
                            ),
                        )
                    branch_hidden = branch_hidden.view(bs, block_size, -1)
                    hidden_delta = (
                        tree_hidden_on_path[path_mask]
                        - branch_hidden.to(tree_hidden_on_path.device)[path_mask]
                    ).abs()
                    hidden_max_abs = (
                        float(hidden_delta.max().item())
                        if hidden_delta.numel()
                        else 0.0
                    )
                    if use_flashinfer_expanded_causal_tree:
                        leaf_hidden_delta = (
                            leaf_hidden_on_path[path_mask]
                            - branch_hidden.to(leaf_hidden_on_path.device)[path_mask]
                        ).abs()
                        leaf_hidden_max_abs = (
                            float(leaf_hidden_delta.max().item())
                            if leaf_hidden_delta.numel()
                            else 0.0
                        )
                        if (
                            same_shape_logits_output is not None
                            and same_shape_logits_output.hidden_states is not None
                        ):
                            same_shape_hidden = (
                                same_shape_logits_output.hidden_states.view(
                                    bs, tree_budget, block_size, -1
                                )
                            )
                            same_shape_hidden_on_path = same_shape_hidden[
                                req_indices.to(torch.long).unsqueeze(1),
                                accept_index_local.to(torch.long),
                                hidden_offsets_2d,
                            ]
                            same_shape_hidden_delta = (
                                tree_hidden_on_path[path_mask]
                                - same_shape_hidden_on_path.to(
                                    tree_hidden_on_path.device
                                )[path_mask]
                            ).abs()
                            same_shape_hidden_max_abs = (
                                float(same_shape_hidden_delta.max().item())
                                if same_shape_hidden_delta.numel()
                                else 0.0
                            )
                if (
                    tree_mamba_snapshot is not None
                    and tree_mamba_state_indices is not None
                ):
                    linear_backend = getattr(
                        target_model_runner.attn_backend, "linear_attn_backend", None
                    )
                    try:
                        mamba_caches = (
                            linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                        )
                        branch_mamba_rows = []
                        branch_conv_rows = []
                        for row in range(bs):
                            state_idx = tree_mamba_state_indices[row]
                            branch_steps = path_pos.to(torch.long)
                            branch_mamba_rows.append(
                                mamba_caches.intermediate_ssm[
                                    :, state_idx, branch_steps
                                ].clone()
                            )
                            branch_conv_rows.append(
                                mamba_caches.intermediate_conv_window[0][
                                    :, state_idx, branch_steps
                                ].clone()
                            )
                        branch_mamba = torch.stack(branch_mamba_rows, dim=1)
                        branch_conv = torch.stack(branch_conv_rows, dim=1)
                        valid_mamba = (
                            path_mask.to(tree_mamba_snapshot.device)
                            .unsqueeze(0)
                            .reshape(
                                1,
                                bs,
                                block_size,
                                *([1] * (tree_mamba_snapshot.dim() - 3)),
                            )
                        )
                        mamba_delta = (
                            tree_mamba_snapshot - branch_mamba.to(tree_mamba_snapshot.device)
                        ).abs()
                        conv_delta = (
                            tree_conv_snapshot - branch_conv.to(tree_conv_snapshot.device)
                        ).abs()
                        mamba_max_abs = float(
                            mamba_delta.masked_select(valid_mamba).max().item()
                        )
                        conv_max_abs = float(
                            conv_delta.masked_select(valid_mamba).max().item()
                        )
                    except Exception:
                        logger.exception(
                            "DFLASH tree causal compare failed to compare mamba intermediates"
                        )
                    try:
                        # The branch replay is a diagnostic forward that reuses the
                        # request's intermediate state rows. Restore the tree-verify
                        # rows before the real commit path reads them, otherwise this
                        # compare mode becomes an accidental branch-reverify fallback.
                        linear_backend = getattr(
                            target_model_runner.attn_backend, "linear_attn_backend", None
                        )
                        if (
                            linear_backend is not None
                            and tree_mamba_snapshot is not None
                            and tree_conv_snapshot is not None
                            and tree_mamba_state_indices is not None
                        ):
                            mamba_caches = (
                                linear_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                            )
                            for row in range(bs):
                                state_idx = tree_mamba_state_indices[row]
                                local_path = accept_index_local[row].to(torch.long)
                                mamba_caches.intermediate_ssm[
                                    :, state_idx, local_path
                                ] = tree_mamba_snapshot[:, row]
                                mamba_caches.intermediate_conv_window[0][
                                    :, state_idx, local_path
                                ] = tree_conv_snapshot[:, row]
                    except Exception:
                        logger.exception(
                            "DFLASH tree causal compare failed to restore tree mamba intermediates"
                        )
                if os.environ.get("SGLANG_DFLASH_TREE_COMPARE_KV"):
                    kv_pool = model_worker_batch.token_to_kv_pool_allocator.get_kvcache()
                    if use_flashinfer_expanded_causal_tree:
                        assert expanded_cache_loc_3d is not None
                        accept_offsets_2d = (
                            torch.arange(block_size, dtype=torch.long, device=device)
                            .unsqueeze(0)
                            .expand_as(accept_index_local)
                        )
                        src_locs_2d = expanded_cache_loc_3d[
                            req_indices.to(torch.long).unsqueeze(1),
                            accept_index_local.to(torch.long),
                            accept_offsets_2d,
                        ].to(torch.long)
                        leaf_index_2d = torch.gather(
                            accept_index_local,
                            1,
                            (commit_lens.to(torch.long) - 1).clamp(min=0).unsqueeze(1),
                        ).expand_as(accept_index_local)
                        leaf_src_locs_2d = expanded_cache_loc_3d[
                            req_indices.to(torch.long).unsqueeze(1),
                            leaf_index_2d.to(torch.long),
                            accept_offsets_2d,
                        ].to(torch.long)
                    else:
                        src_locs_2d = torch.gather(
                            tree_cache_loc_2d,
                            1,
                            accept_index_local,
                        ).to(torch.long)
                        leaf_src_locs_2d = None
                    dst_locs_2d = branch_cache_loc_2d.to(torch.long)
                    valid_path = path_mask
                    layer_diffs = []
                    leaf_layer_diffs = []
                    same_shape_layer_diffs = []
                    if same_shape_cache_loc_3d is not None:
                        same_shape_src_locs_2d = same_shape_cache_loc_3d[
                            req_indices.to(torch.long).unsqueeze(1),
                            accept_index_local.to(torch.long),
                            accept_offsets_2d,
                        ].to(torch.long)
                    else:
                        same_shape_src_locs_2d = None
                    if hasattr(kv_pool, "full_attention_layer_id_mapping"):
                        kv_layer_ids = sorted(kv_pool.full_attention_layer_id_mapping)
                    else:
                        kv_layer_ids = range(int(getattr(kv_pool, "layer_num", 0)))
                    for layer_id in kv_layer_ids:
                        try:
                            k_buffer = kv_pool.get_key_buffer(layer_id)
                            v_buffer = kv_pool.get_value_buffer(layer_id)
                        except Exception:
                            continue
                        k_delta = (
                            k_buffer[src_locs_2d][valid_path]
                            - k_buffer[dst_locs_2d][valid_path]
                        ).abs()
                        v_delta = (
                            v_buffer[src_locs_2d][valid_path]
                            - v_buffer[dst_locs_2d][valid_path]
                        ).abs()
                        layer_diffs.append(
                            (
                                layer_id,
                                float(k_delta.max().item()) if k_delta.numel() else 0.0,
                                float(v_delta.max().item()) if v_delta.numel() else 0.0,
                            )
                        )
                        if leaf_src_locs_2d is not None:
                            leaf_k_delta = (
                                k_buffer[leaf_src_locs_2d][valid_path]
                                - k_buffer[dst_locs_2d][valid_path]
                            ).abs()
                            leaf_v_delta = (
                                v_buffer[leaf_src_locs_2d][valid_path]
                                - v_buffer[dst_locs_2d][valid_path]
                            ).abs()
                            leaf_layer_diffs.append(
                                (
                                    layer_id,
                                    float(leaf_k_delta.max().item())
                                    if leaf_k_delta.numel()
                                    else 0.0,
                                    float(leaf_v_delta.max().item())
                                    if leaf_v_delta.numel()
                                    else 0.0,
                                )
                            )
                        if same_shape_src_locs_2d is not None:
                            same_shape_k_delta = (
                                k_buffer[src_locs_2d][valid_path]
                                - k_buffer[same_shape_src_locs_2d][valid_path]
                            ).abs()
                            same_shape_v_delta = (
                                v_buffer[src_locs_2d][valid_path]
                                - v_buffer[same_shape_src_locs_2d][valid_path]
                            ).abs()
                            same_shape_layer_diffs.append(
                                (
                                    layer_id,
                                    float(same_shape_k_delta.max().item())
                                    if same_shape_k_delta.numel()
                                    else 0.0,
                                    float(same_shape_v_delta.max().item())
                                    if same_shape_v_delta.numel()
                                    else 0.0,
                                )
                            )
                    kv_max_abs = layer_diffs
                    if leaf_layer_diffs:
                        leaf_kv_max_abs = leaf_layer_diffs
                    if same_shape_layer_diffs:
                        same_shape_kv_max_abs = same_shape_layer_diffs

                logger.info(
                    "DFLASH tree causal compare prefix=%s commit_lens=%s local=%s branch_candidates=%s tree_predict=%s branch_predict=%s hidden_max_abs=%s leaf_hidden_max_abs=%s same_shape_hidden_max_abs=%s kv_max_abs=%s leaf_kv_max_abs=%s same_shape_kv_max_abs=%s mamba_max_abs=%s conv_max_abs=%s",
                    prefix_lens.detach().cpu().tolist(),
                    commit_lens.detach().cpu().tolist(),
                    accept_index_local.detach().cpu().tolist(),
                    branch_candidates.detach().cpu().tolist(),
                    tree_predict_on_path.detach().cpu().tolist(),
                    branch_target_predict.detach().cpu().tolist(),
                    hidden_max_abs,
                    leaf_hidden_max_abs,
                    same_shape_hidden_max_abs,
                    kv_max_abs,
                    leaf_kv_max_abs,
                    same_shape_kv_max_abs,
                    mamba_max_abs,
                    conv_max_abs,
                )
                torch.get_device_module(device).synchronize()
                if same_shape_req_pool_indices is not None:
                    self.model_runner.req_to_token_pool.free_slots.extend(
                        same_shape_req_pool_indices.detach().cpu().tolist()
                    )
                if same_shape_cache_alloc is not None:
                    model_worker_batch.tree_cache.token_to_kv_pool_allocator.free(
                        same_shape_cache_alloc
                    )
                if branch_cache_alloc is not None:
                    model_worker_batch.tree_cache.token_to_kv_pool_allocator.free(
                        branch_cache_alloc
                    )

        reverify_commit_cache_alloc = None
        reverify_commit_cache_loc_2d = None
        reverify_commit_hidden = None
        reverify_target_predict = None
        reverify_candidates = None
        reverify_mamba_state_snapshots = None
        if use_reverify_commit:
            (
                reverify_commit_cache_alloc,
                reverify_commit_cache_loc_2d,
                reverify_commit_hidden,
                reverify_target_predict,
                reverify_candidates,
                reverify_mamba_state_snapshots,
            ) = self._reverify_accepted_tree_path_for_commit(
                model_worker_batch=model_worker_batch,
                draft_input=draft_input,
                tree_tokens=tree_tokens,
                tree_cache_loc_2d=tree_cache_loc_2d,
                linear_fallback_tokens=linear_draft_tokens,
                accept_index_local=accept_index_local,
                commit_lens=commit_lens,
                prefix_lens=prefix_lens,
            )
            reverify_accept_len_raw, _ = compute_dflash_correct_drafts_and_bonus(
                candidates=reverify_candidates,
                target_predict=reverify_target_predict,
            )
            use_linear_commit_fallback = (
                target_model_runner.mambaish_config is not None
                and os.environ.get(
                    "SGLANG_DFLASH_TREE_MOE_LINEAR_COMMIT_FALLBACK", "1"
                )
                != "0"
            )
            if use_linear_commit_fallback:
                reverify_accept_len = reverify_accept_len_raw
            else:
                reverify_accept_len = torch.minimum(
                    reverify_accept_len_raw,
                    num_correct_drafts.to(reverify_accept_len_raw.dtype),
                )
            reverify_bonus = reverify_target_predict[
                torch.arange(bs, device=device), reverify_accept_len.to(torch.long)
            ].to(torch.int64)
            if torch.any(reverify_accept_len != num_correct_drafts):
                logger.info(
                    "DFLASH tree accepted-path replay adjusted accept prefix=%s "
                    "tree_correct=%s replay_correct=%s capped_replay_correct=%s "
                    "tree_bonus=%s replay_bonus=%s",
                    prefix_lens.detach().cpu().tolist(),
                    num_correct_drafts.detach().cpu().tolist(),
                    reverify_accept_len_raw.detach().cpu().tolist(),
                    reverify_accept_len.detach().cpu().tolist(),
                    bonus.detach().cpu().tolist(),
                    reverify_bonus.detach().cpu().tolist(),
                )
            num_correct_drafts = reverify_accept_len.to(num_correct_drafts.dtype)
            commit_lens = num_correct_drafts.to(torch.int32) + 1
            bonus = reverify_bonus
            out_tokens.zero_()
            if block_size > 1:
                out_tokens[:, : block_size - 1].copy_(
                    reverify_candidates[:, 1:block_size]
                )
            out_tokens.scatter_(
                1,
                num_correct_drafts.to(torch.long).unsqueeze(1),
                bonus.unsqueeze(1),
            )
            if os.environ.get("SGLANG_DFLASH_TREE_REVERIFY_DEBUG"):
                logger.info(
                    "DFLASH tree reverify accept prefix=%s candidates=%s "
                    "target_predict=%s commit_correct=%s raw_reverify_correct=%s "
                    "commit_bonus=%s raw_reverify_bonus=%s",
                    prefix_lens.detach().cpu().tolist(),
                    reverify_candidates.detach().cpu().tolist(),
                    reverify_target_predict.detach().cpu().tolist(),
                    num_correct_drafts.detach().cpu().tolist(),
                    reverify_accept_len_raw.detach().cpu().tolist(),
                    bonus.detach().cpu().tolist(),
                    reverify_target_predict[
                        torch.arange(bs, device=device),
                        reverify_accept_len_raw.to(torch.long),
                    ]
                    .to(torch.int64)
                    .detach()
                    .cpu()
                    .tolist(),
                )
            self._commit_selected_persistent_mamba_snapshots(
                reverify_mamba_state_snapshots,
                commit_lens,
                prefix_lens,
                model_worker_batch,
            )

        accepted_offsets_2d = torch.arange(
            block_size, dtype=torch.int64, device=device
        ).unsqueeze(0)
        accepted_positions_2d = prefix_lens.to(torch.int64).unsqueeze(
            1
        ) + accepted_offsets_2d
        accepted_cache_loc_2d = self.model_runner.req_to_token_pool.req_to_token[
            model_worker_batch.req_pool_indices.to(torch.long).unsqueeze(1),
            accepted_positions_2d,
        ].to(torch.int64)

        if reverify_commit_cache_loc_2d is not None:
            accepted_cache_loc_2d = reverify_commit_cache_loc_2d
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.model_runner.req_to_token_pool.req_to_token,
                prefix_lens,
                prefix_lens.to(torch.int64) + block_size,
                accepted_cache_loc_2d.reshape(-1),
                bs,
            )
        elif use_flashinfer_expanded_causal_tree:
            assert expanded_cache_loc_3d is not None
            accept_offsets_2d = accepted_offsets_2d.expand_as(accept_index_local)
            source_cache_loc_2d = expanded_cache_loc_3d[
                req_indices.to(torch.long).unsqueeze(1),
                accept_index_local.to(torch.long),
                accept_offsets_2d,
            ].to(torch.int64)
            valid_offsets = (
                accepted_offsets_2d < commit_lens.to(torch.int64).unsqueeze(1)
            )
            move_kv_cache_overlap_safe(
                model_worker_batch.token_to_kv_pool_allocator.get_kvcache(),
                accepted_cache_loc_2d[valid_offsets],
                source_cache_loc_2d[valid_offsets],
            )
        elif os.environ.get("SGLANG_DFLASH_TREE_LOGICAL_COMMIT"):
            accepted_cache_loc_2d = torch.gather(
                tree_cache_loc_2d,
                1,
                accept_index_local,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.model_runner.req_to_token_pool.req_to_token,
                prefix_lens,
                prefix_lens.to(torch.int64) + block_size,
                accepted_cache_loc_2d.reshape(-1),
                bs,
            )
        else:
            accepted_cache_loc_2d = move_accept_tokens_to_target_kvcache(
                model_worker_batch,
                accept_index,
                num_correct_drafts,
                model_worker_batch.token_to_kv_pool_allocator,
                overlap_safe=True,
            ).to(torch.int64)
        mamba_commit_accept_index = None
        mamba_commit_draft_token_num = None
        if use_flashinfer_expanded_causal_tree and expanded_mamba_indices is not None:
            req_to_token_pool = self.model_runner.req_to_token_pool
            mamba_caches = (
                req_to_token_pool.get_speculative_mamba2_params_all_layers()
            )
            original_mamba_indices = req_to_token_pool.get_mamba_indices(
                model_worker_batch.req_pool_indices
            )[:bs].to(torch.long)
            expanded_mamba_2d = expanded_mamba_indices.view(bs, tree_budget).to(
                torch.long
            )
            for row in range(bs):
                dst_slot = original_mamba_indices[row]
                last_path_offset = int(commit_lens[row].item()) - 1
                last_node = accept_index_local[row, last_path_offset].to(torch.long)
                src_slot = expanded_mamba_2d[row, last_node]
                mamba_caches.temporal[:, dst_slot] = mamba_caches.intermediate_ssm[
                    :, src_slot, last_path_offset
                ]
                for conv_idx, conv_cache in enumerate(mamba_caches.conv):
                    conv_cache[:, dst_slot] = mamba_caches.intermediate_conv_window[
                        conv_idx
                    ][:, src_slot, last_path_offset]

            if model_worker_batch.mamba_track_indices is not None:
                mamba_track_interval = get_global_server_args().mamba_track_interval
                seq_lens_post_verify = prefix_lens + commit_lens.to(prefix_lens.dtype)
                to_track_mask = (
                    prefix_lens // mamba_track_interval
                    != seq_lens_post_verify // mamba_track_interval
                )
                tracking_point = (
                    seq_lens_post_verify
                    // mamba_track_interval
                    * mamba_track_interval
                )
                to_track_ith = torch.clamp(
                    tracking_point - prefix_lens - 1, min=0
                ).to(torch.long)
                for row in range(bs):
                    if not bool(to_track_mask[row].item()):
                        continue
                    dst_slot = model_worker_batch.mamba_track_indices[row].to(
                        torch.long
                    )
                    step = int(to_track_ith[row].item())
                    node = accept_index_local[row, step].to(torch.long)
                    src_slot = expanded_mamba_2d[row, node]
                    mamba_caches.temporal[:, dst_slot] = (
                        mamba_caches.intermediate_ssm[:, src_slot, step]
                    )
                    for conv_idx, conv_cache in enumerate(mamba_caches.conv):
                        conv_cache[:, dst_slot] = (
                            mamba_caches.intermediate_conv_window[conv_idx][
                                :, src_slot, step
                            ]
                        )
            req_to_token_pool.mamba_allocator.free(expanded_mamba_indices)
            expanded_mamba_indices = None
        else:
            if use_reverify_commit and reverify_mamba_state_snapshots is not None:
                # Persistent Mamba state was already committed from the direct
                # causal-reverify snapshots above.
                pass
            elif use_reverify_commit:
                self._update_target_mamba_state_after_verify(
                    batch=model_worker_batch,
                    seq_lens_pre_verify=prefix_lens,
                    commit_lens=commit_lens,
                )
            else:
                mamba_commit_accept_index = accept_index
                mamba_commit_draft_token_num = tree_budget

            if mamba_commit_accept_index is not None:
                commit_mamba_states_after_verify(
                    self.target_worker,
                    model_worker_batch,
                    commit_lens,
                    mamba_commit_accept_index,
                    mamba_commit_draft_token_num,
                )

        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        hidden = (
            reverify_commit_hidden
            if reverify_commit_hidden is not None
            else logits_output.hidden_states
        )
        if hidden is None:
            raise RuntimeError(
                "DFLASH tree verify requires target hidden states, but got None."
            )
        if reverify_commit_hidden is not None:
            hidden = hidden.view(bs, block_size, -1)
            accepted_hidden = hidden
        elif use_flashinfer_expanded_causal_tree:
            hidden = hidden.view(bs, tree_budget, block_size, -1)
            accepted_hidden = hidden[
                req_indices.to(torch.long).unsqueeze(1),
                accept_index_local.to(torch.long),
                accept_offsets_2d,
            ]
        else:
            hidden = hidden.view(bs, tree_budget, -1)
            accepted_hidden = torch.gather(
                hidden,
                1,
                accept_index_local.unsqueeze(-1).expand(
                    bs, block_size, hidden.shape[-1]
                ),
            )
        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=accepted_hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=accepted_cache_loc_2d.reshape(-1),
            cache_loc_2d=accepted_cache_loc_2d,
            positions=accepted_positions_2d.reshape(-1),
            commit_lens=commit_lens,
            )

        logits_output.hidden_states = None
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        if (
            (expanded_cache_loc is not None or reverify_commit_cache_alloc is not None)
            and os.environ.get("SGLANG_DFLASH_TREE_KEEP_SCRATCH", "0") != "1"
        ):
            verify_done.synchronize()
            if expanded_cache_loc is not None:
                model_worker_batch.tree_cache.token_to_kv_pool_allocator.free(
                    expanded_cache_loc
                )
                expanded_cache_loc = None
            if reverify_commit_cache_alloc is not None:
                model_worker_batch.tree_cache.token_to_kv_pool_allocator.free(
                    reverify_commit_cache_alloc
                )
                reverify_commit_cache_alloc = None

        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=tree_budget,
            new_seq_lens=new_seq_lens,
        )

    def forward_batch_generation(
        self,
        model_worker_batch: ScheduleBatch,
        on_publish=None,
    ) -> GenerationBatchResult:
        if getattr(model_worker_batch, "return_logprob", False):
            raise ValueError(
                "DFLASH speculative decoding does not support return_logprob yet."
            )
        self._validate_phase1_sampling_support(model_worker_batch)

        if (
            model_worker_batch.forward_mode.is_extend()
            or model_worker_batch.is_extend_in_batch
        ):
            # Target prefill: capture DFlash aux hidden states for prompt tokens.
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            batch_output = self.target_worker.forward_batch_generation(
                model_worker_batch
            )

            logits_output, next_token_ids = (
                batch_output.logits_output,
                batch_output.next_token_ids,
            )
            batch_output.new_seq_lens = model_worker_batch.seq_lens
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            if logits_output.hidden_states is None:
                raise RuntimeError(
                    "DFLASH requires target aux hidden capture for prefill, but got None. "
                    "Make sure the target model has DFlash layers-to-capture configured."
                )

            if (
                model_worker_batch.extend_lens is None
                or model_worker_batch.prefix_lens is None
            ):
                raise RuntimeError(
                    "DFLASH expected extend_lens / prefix_lens to be populated in extend mode, "
                    "but got None."
                )

            # Materialize prompt tokens into the draft KV cache immediately. This is required
            # for radix cache safety (the scheduler may update radix after prefill returns).
            device = next_token_ids.device
            ctx_lens = torch.tensor(
                model_worker_batch.extend_lens, dtype=torch.int32, device=device
            )
            draft_seq_lens = torch.tensor(
                model_worker_batch.prefix_lens, dtype=torch.int32, device=device
            )

            if model_worker_batch.out_cache_loc is None:
                raise RuntimeError(
                    "DFLASH prefill expected out_cache_loc, but got None."
                )
            positions, _ = compute_position(
                self.model_runner.server_args.attention_backend,
                draft_seq_lens,
                ctx_lens,
                int(sum(model_worker_batch.extend_lens)),
            )
            self._append_target_hidden_to_draft_kv_by_loc(
                target_hidden=logits_output.hidden_states,
                cache_loc=model_worker_batch.out_cache_loc,
                positions=positions,
            )

            # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
            logits_output.hidden_states = None

            batch_output.next_draft_input = self._make_next_draft_input_prefill(
                bonus_tokens=next_token_ids,
                seq_lens=model_worker_batch.seq_lens,
                cur_allocated_seq_lens_cpu=model_worker_batch.seq_lens_cpu,
            )
            verify_done = torch.get_device_module(device).Event()
            verify_done.record()
            batch_output.next_draft_input.verify_done = verify_done
            return batch_output

        # Decode / target-verify stage.
        if model_worker_batch.spec_info is None:
            model_worker_batch.spec_info = DFlashDraftInputV2.create_idle_input(
                device=self.device
            )

        draft_input = model_worker_batch.spec_info
        if not isinstance(draft_input, DFlashDraftInputV2):
            raise RuntimeError(
                "DFLASH spec-v2 expected DFlashDraftInputV2 state on the running batch."
            )

        if model_worker_batch.forward_mode.is_idle():
            empty_ids = torch.empty((0,), dtype=torch.int64, device=self.device)
            empty_lens = torch.empty((0,), dtype=torch.int32, device=self.device)
            next_draft_input = self._make_next_draft_input_decode(
                bonus_tokens=torch.empty((0,), device=self.device, dtype=torch.int64),
                new_seq_lens=torch.empty((0,), device=self.device, dtype=torch.int64),
            )
            if on_publish is not None:
                on_publish(next_draft_input.new_seq_lens)
            verify_done = torch.get_device_module(self.device).Event()
            verify_done.record()
            next_draft_input.verify_done = verify_done
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=empty_ids,
                accept_lens=empty_lens,
                next_draft_input=next_draft_input,
                can_run_cuda_graph=False,
                speculative_num_draft_tokens=int(self.block_size),
                new_seq_lens=next_draft_input.new_seq_lens,
            )

        # `seq_lens` is carried over from the previous overlap iteration and may have been
        # produced on another stream.
        model_worker_batch.seq_lens.record_stream(
            torch.get_device_module(self.device).current_stream()
        )

        bs = len(model_worker_batch.seq_lens)
        device = self.device
        if self.use_tree_draft:
            return self._forward_batch_generation_tree(
                model_worker_batch,
                draft_input,
                on_publish=on_publish,
            )

        # --- 1) Draft a fixed block with the draft model.
        target_model = self.target_worker.model_runner.model
        embed_module = target_model.get_input_embeddings()
        lm_head = getattr(target_model, "lm_head", None)
        if lm_head is None or not hasattr(lm_head, "weight"):
            raise RuntimeError(
                "DFLASH requires the target model to expose `lm_head` with `weight`."
            )

        block_size = int(self.block_size)
        self._ensure_draft_block_buffers(bs)
        assert self._draft_block_ids_buf is not None
        assert self._draft_block_positions_buf is not None
        assert self._draft_block_tokens_buf is not None
        assert self._draft_verify_out_cache_loc_buf is not None
        assert self._draft_block_end_buf is not None
        assert self._draft_seq_lens_cpu_buf is not None

        block_ids = self._draft_block_ids_buf[:bs]
        prefix_lens = model_worker_batch.seq_lens
        positions_2d = self._draft_block_positions_buf[:bs]
        verify_out_cache_loc_2d = self._draft_verify_out_cache_loc_buf[:bs]
        if self._use_triton_prepare_block:
            try:
                _prepare_dflash_draft_block_unchecked(
                    bonus_tokens=draft_input.bonus_tokens.view(-1),
                    prefix_lens=prefix_lens.view(-1),
                    req_pool_indices=model_worker_batch.req_pool_indices.view(-1),
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    block_ids_out=block_ids,
                    positions_out=positions_2d,
                    cache_loc_out=verify_out_cache_loc_2d,
                    mask_token_id=int(self._mask_token_id),
                )
            except Exception as e:
                self._use_triton_prepare_block = False
                logger.warning(
                    "DFLASH Triton prepare_block failed; falling back to eager path: %s",
                    e,
                )
                block_ids.fill_(int(self._mask_token_id))
                block_ids[:, 0].copy_(draft_input.bonus_tokens)
                torch.add(
                    prefix_lens.unsqueeze(1),
                    self._block_pos_offsets,
                    out=positions_2d,
                )
                end_offset = prefix_lens + block_size
                verify_out_cache_loc = assign_extend_cache_locs_func(
                    req_pool_indices=model_worker_batch.req_pool_indices,
                    req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                    start_offset=prefix_lens,
                    end_offset=end_offset,
                    batch_size=bs,
                    draft_token_num=block_size,
                    device=device,
                )
                verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))
        else:
            block_ids.fill_(int(self._mask_token_id))
            block_ids[:, 0].copy_(draft_input.bonus_tokens)
            torch.add(
                prefix_lens.unsqueeze(1),
                self._block_pos_offsets,
                out=positions_2d,
            )
            end_offset = prefix_lens + block_size
            verify_out_cache_loc = assign_extend_cache_locs_func(
                req_pool_indices=model_worker_batch.req_pool_indices,
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                start_offset=prefix_lens,
                end_offset=end_offset,
                batch_size=bs,
                draft_token_num=block_size,
                device=device,
            )
            verify_out_cache_loc_2d.copy_(verify_out_cache_loc.view(bs, block_size))

        noise_embedding = embed_module(block_ids)
        input_embeds = noise_embedding.view(-1, noise_embedding.shape[-1])

        positions = positions_2d.reshape(-1)
        verify_out_cache_loc = verify_out_cache_loc_2d.reshape(-1)

        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        if self.use_compact_draft_cache:
            # Rebuild the draft-local sliding-window view from committed target state.
            draft_prefix_lens = self._compute_compact_draft_seq_lens(prefix_lens)
            seq_lens_cpu.copy_(draft_prefix_lens.to(device="cpu", dtype=torch.int32))

            suffix_start = prefix_lens.to(torch.int64) - draft_prefix_lens.to(
                torch.int64
            )
            suffix_cache_loc = self._gather_req_to_token_segments(
                req_to_token=self.model_runner.req_to_token_pool.req_to_token,
                req_pool_indices=model_worker_batch.req_pool_indices,
                start=suffix_start,
                lengths=draft_prefix_lens,
            )
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                torch.zeros_like(draft_prefix_lens),
                draft_prefix_lens,
                suffix_cache_loc,
                bs,
            )

            block_end = self._draft_block_end_buf[:bs]
            torch.add(draft_prefix_lens, block_size, out=block_end)
            assign_req_to_token_pool_func(
                model_worker_batch.req_pool_indices,
                self.draft_model_runner.req_to_token_pool.req_to_token,
                draft_prefix_lens,
                block_end,
                verify_out_cache_loc,
                bs,
            )
            draft_seq_lens = draft_prefix_lens
            draft_seq_lens_sum = int(seq_lens_cpu.sum().item())
        else:
            # Non-windowed path uses the shared overallocated mapping directly.
            # Backend planning only needs a safe upper bound for the committed
            # prefix lengths, not the full allocator reservation length.
            draft_seq_lens = prefix_lens
            if draft_input.planning_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.planning_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.planning_seq_lens_sum)
            elif draft_input.reserved_seq_lens_cpu is not None:
                seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
                draft_seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
            elif model_worker_batch.seq_lens_cpu is not None:
                seq_lens_cpu.copy_(model_worker_batch.seq_lens_cpu)
                draft_seq_lens_sum = (
                    int(model_worker_batch.seq_lens_sum)
                    if model_worker_batch.seq_lens_sum is not None
                    else int(model_worker_batch.seq_lens_cpu.sum())
                )
            else:
                seq_lens_cpu.copy_(prefix_lens.to("cpu", dtype=torch.int32))
                draft_seq_lens_sum = int(prefix_lens.sum().item())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=model_worker_batch.req_pool_indices,
            seq_lens=draft_seq_lens,
            out_cache_loc=verify_out_cache_loc,
            seq_lens_sum=draft_seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions,
            input_embeds=input_embeds,
            spec_algorithm=SpeculativeAlgorithm.DFLASH,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )

        with torch.inference_mode():
            draft_logits_output = self.draft_model_runner.forward(
                forward_batch
            ).logits_output

        draft_hidden = draft_logits_output.hidden_states
        if draft_hidden is None:
            raise RuntimeError("DFLASH draft model returned no hidden states.")
        draft_hidden = draft_hidden.view(bs, int(self.block_size), -1)
        draft_next = self._greedy_sample_from_vocab_parallel_head(
            hidden_states=draft_hidden[:, 1:, :].reshape(-1, draft_hidden.shape[-1]),
            lm_head=lm_head,
        ).view(bs, int(self.block_size) - 1)

        draft_tokens = self._draft_block_tokens_buf[:bs]
        draft_tokens[:, 0].copy_(block_ids[:, 0])
        draft_tokens[:, 1:].copy_(draft_next)

        # --- 2) Target verify.
        # TARGET_VERIFY uses standard causal masking; custom masks are unnecessary here.
        custom_mask = None

        verify_input_ids = draft_tokens.reshape(-1)
        verify_input = DFlashVerifyInput(
            draft_token=verify_input_ids,
            positions=positions,
            draft_token_num=int(self.block_size),
            custom_mask=custom_mask,
            capture_hidden_mode=CaptureHiddenMode.FULL,
        )

        model_worker_batch.out_cache_loc = verify_out_cache_loc
        sampling_info = model_worker_batch.sampling_info

        need_mamba_verify_commit = hasattr(
            self.target_worker.model_runner.attn_backend,
            "update_mamba_state_after_mtp_verify",
        )
        seq_lens_pre_verify = (
            model_worker_batch.seq_lens.clone() if need_mamba_verify_commit else None
        )
        seq_lens_cpu_backup = model_worker_batch.seq_lens_cpu
        seq_lens_sum_backup = model_worker_batch.seq_lens_sum
        if draft_input.planning_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.planning_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.planning_seq_lens_sum)
        elif draft_input.reserved_seq_lens_cpu is not None:
            model_worker_batch.seq_lens_cpu = draft_input.reserved_seq_lens_cpu
            model_worker_batch.seq_lens_sum = int(draft_input.reserved_seq_lens_sum)

        verify_forward_batch, _ = verify_input.prepare_for_verify(
            model_worker_batch, self.target_worker
        )
        model_worker_batch.seq_lens_cpu = seq_lens_cpu_backup
        model_worker_batch.seq_lens_sum = seq_lens_sum_backup

        target_out = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = target_out.logits_output
        can_run_cuda_graph = target_out.can_run_cuda_graph

        if sampling_info is not None:
            apply_dflash_verify_logits_adjustments(
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                draft_token_num=int(self.block_size),
            )

        candidates = draft_tokens
        new_seq_lens = None
        if (
            sampling_info is not None
            and not sampling_info.is_all_greedy
            and is_dflash_sampling_verify_available()
        ):
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=logits_output.next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
            commit_lens = accept_len.to(torch.int32) + 1  # [bs]
            out_tokens = torch.empty(
                (bs, int(self.block_size)), dtype=torch.int64, device=device
            )
            if int(self.block_size) > 1:
                out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
            out_tokens[:, int(self.block_size) - 1].fill_(0)
            out_tokens.scatter_(1, accept_len.to(torch.int64)[:, None], bonus[:, None])
        else:
            target_predict = torch.argmax(logits_output.next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
            if self._use_triton_accept_bonus:
                try:
                    (
                        accept_len,
                        commit_lens,
                        bonus,
                        out_tokens,
                        new_seq_lens,
                    ) = self._next_accept_bonus_buffers(bs)
                    _compute_dflash_accept_bonus_triton_unchecked(
                        candidates=candidates,
                        target_top1=target_predict,
                        accept_lens_out=accept_len,
                        commit_lens_out=commit_lens,
                        bonus_ids_out=bonus,
                        out_tokens_out=out_tokens,
                        prefix_lens=prefix_lens,
                        new_seq_lens_out=new_seq_lens,
                    )
                except Exception as e:
                    self._use_triton_accept_bonus = False
                    logger.warning(
                        "DFLASH Triton accept/bonus failed; falling back to eager path: %s",
                        e,
                    )
                    accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                        candidates=candidates,
                        target_predict=target_predict,
                    )
                    commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                    out_tokens = torch.empty(
                        (bs, int(self.block_size)),
                        dtype=torch.int64,
                        device=device,
                    )
                    if int(self.block_size) > 1:
                        out_tokens[:, : int(self.block_size) - 1].copy_(
                            candidates[:, 1:]
                        )
                    out_tokens[:, int(self.block_size) - 1].fill_(0)
                    out_tokens.scatter_(
                        1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                    )
            else:
                accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
                    candidates=candidates,
                    target_predict=target_predict,
                )
                commit_lens = accept_len.to(torch.int32) + 1  # [bs]
                out_tokens = torch.empty(
                    (bs, int(self.block_size)), dtype=torch.int64, device=device
                )
                if int(self.block_size) > 1:
                    out_tokens[:, : int(self.block_size) - 1].copy_(candidates[:, 1:])
                out_tokens[:, int(self.block_size) - 1].fill_(0)
                out_tokens.scatter_(
                    1, accept_len.to(torch.int64)[:, None], bonus[:, None]
                )
        if need_mamba_verify_commit:
            assert seq_lens_pre_verify is not None
            self._update_target_mamba_state_after_verify(
                batch=model_worker_batch,
                seq_lens_pre_verify=seq_lens_pre_verify,
                commit_lens=commit_lens,
            )

        if new_seq_lens is None:
            new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        if on_publish is not None:
            on_publish(new_seq_lens)

        # --- 3) Materialize committed verify-input tokens into draft KV cache.
        hidden = logits_output.hidden_states
        if hidden is None:
            raise RuntimeError(
                "DFLASH verify requires target hidden states, but got None."
            )
        hidden = hidden.view(bs, int(self.block_size), -1)

        self._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden.reshape(-1, hidden.shape[-1]),
            cache_loc=verify_out_cache_loc,
            cache_loc_2d=verify_out_cache_loc_2d,
            positions=positions,
            commit_lens=commit_lens,
        )

        # Avoid copying large hidden-state buffers to CPU in overlap scheduling.
        logits_output.hidden_states = None

        next_draft_input = self._make_next_draft_input_decode(
            bonus_tokens=bonus,
            new_seq_lens=new_seq_lens,
            cur_allocated_seq_lens_cpu=draft_input.reserved_seq_lens_cpu,
        )
        verify_done = torch.get_device_module(device).Event()
        verify_done.record()
        next_draft_input.verify_done = verify_done

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=out_tokens.reshape(-1),
            accept_lens=commit_lens,
            can_run_cuda_graph=can_run_cuda_graph,
            next_draft_input=next_draft_input,
            speculative_num_draft_tokens=int(self.block_size),
            # The non-overlap (sync) scheduler path advances batch.seq_lens
            # from the result; overlap carries it via next_draft_input instead.
            new_seq_lens=new_seq_lens,
        )
