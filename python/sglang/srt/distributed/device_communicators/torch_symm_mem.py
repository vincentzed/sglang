# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/bf214ca22625e311a2c4c0dfbf7af19128f4919c/vllm/distributed/device_communicators/symm_mem.py
import logging
from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.all_reduce_utils import (
    TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES,
)
from sglang.srt.distributed.device_communicators.symm_mem_custom_ar import (
    AR_VEC,
    CUSTOM_AR_BUFFER_BYTES,
    build_custom_ar_resources,
    custom_all_reduce,
)
from sglang.srt.utils import is_cuda, is_hip

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _is_cuda = is_cuda()
    _is_hip = is_hip()

    torch_symm_mem_available = False
    if _is_cuda:
        torch_symm_mem_available = True
except ImportError:
    torch_symm_mem_available = False


logger = logging.getLogger(__name__)


class TorchSymmMemCommunicator:
    """
    Thin wrapper around torch-symmetric-memory collectives.

    This communicator:
      - Validates device capability and world size.
      - Allocates a shared symmetric buffer.
      - Chooses between 'multimem' and 'two-shot' all-reduce kernels.
      - Exposes a fast-path all_reduce() compatible with bfloat16 inputs.

    If any prerequisite is not met, the instance remains disabled and will
    decline to perform symmetric-memory all-reduce.
    """

    # Mapping: compute capability major -> supported world sizes for multimem
    # If the current (cc_major, world_size) is not listed, we fall back
    # to the two-shot path.
    _WORLD_SIZES_MULTIMEM = {
        9: [4, 6, 8],
        10: [6, 8],
    }

    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        *,
        plain_all_reduce_enabled: bool = True,
        custom_ar_enabled: bool = False,
    ):
        """
        Args:
            group: Torch process group used for rendezvous and naming.
            device: Target CUDA device (index, 'cuda:X', or torch.device).
            plain_all_reduce_enabled: serve the torch multimem/two-shot
                all-reduce path (--enable-torch-symm-mem semantics).
            custom_ar_enabled: build the JIT custom-AR resources
                (SGLANG_OPT_USE_SYMM_MEM_CUSTOM_AR); enlarges the buffer.
        """

        self.disabled = True
        self.buffer = None
        self.max_size = 0
        self.plain_all_reduce_enabled = plain_all_reduce_enabled
        self.custom_ar_resources = None

        if not torch_symm_mem_available:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        torch.cuda.set_device(device)
        self.dtype = torch.bfloat16
        self.device = device
        self.group = group
        self.world_size = dist.get_world_size(self.group)
        self.device_capability = torch.cuda.get_device_capability(device)[0]
        supported_max_sizes = TORCH_SYMM_MEM_ALL_REDUCE_MAX_SIZES.get(
            self.device_capability
        )
        if supported_max_sizes is None:
            logger.warning(
                "TorchSymmMemCommunicator: Device capability %s not supported, "
                "communicator is not available.",
                self.device_capability,
            )
            return
        if self.world_size not in supported_max_sizes:
            logger.warning(
                "TorchSymmMemCommunicator: World size %d not supported, "
                "communicator is not available.",
                self.world_size,
            )
            return
        self.max_size = supported_max_sizes[self.world_size]
        if custom_ar_enabled:
            # Keep the custom-AR buffer above the largest prefill payload
            # ([chunked_prefill_size, hidden] bf16), including room for the
            # v4/v5 rotating tail regions.
            self.max_size = max(self.max_size, CUSTOM_AR_BUFFER_BYTES)
        self.buffer = torch_symm_mem.empty(
            self.max_size // self.dtype.itemsize,
            device=self.device,
            dtype=self.dtype,
        )
        self.handle = torch_symm_mem.rendezvous(self.buffer, self.group.group_name)
        if self.handle.multicast_ptr == 0:
            logger.warning(
                "TorchSymmMemCommunicator: torch symmetric memory "
                "multicast operations are not supported."
            )
            self.buffer = None
            self.disabled = True
            return
        self.disabled = False
        if custom_ar_enabled:
            self.custom_ar_resources = build_custom_ar_resources(
                buffer=self.buffer,
                handle=self.handle,
                group_name=self.group.group_name,
                world_size=self.world_size,
            )
            if self.custom_ar_resources is None:
                logger.warning(
                    "TorchSymmMemCommunicator: custom AR requested but not "
                    "available for world size %d; using the default path.",
                    self.world_size,
                )

    def should_torch_symm_mem_allreduce(self, inp: torch.Tensor):
        """
        Fast-path eligibility check for a given tensor.

        Conditions:
          - Communicator must be enabled.
          - dtype must be bfloat16 (matches kernel + buffer dtype).
          - Total byte size must be 4-byte aligned (hardware requirement).
          - Payload must be smaller than the symmetric-memory max size.

        Returns:
            True if the symmetric-memory path can handle this tensor.
        """
        if self.disabled or not self.plain_all_reduce_enabled:
            return False
        if inp.device != self.device:
            return False
        if inp.dtype != self.dtype:
            return False
        inp_size = inp.numel() * inp.element_size()
        # enforce 4-byte alignment
        if inp_size % 4 != 0:
            return False
        return inp_size < self.max_size

    def should_custom_all_reduce(self, inp: torch.Tensor) -> bool:
        """Eligibility for the JIT custom-AR family (symm_mem_custom_ar.py).

        A pure function of static resources + input metadata: every rank and
        every call site must take the same branch, or the in-kernel barriers
        deadlock. Ineligible inputs fall through to the default dispatch.
        """
        if self.custom_ar_resources is None or self.disabled:
            return False
        if inp.device != self.device or inp.dtype != self.dtype:
            return False
        if not inp.is_contiguous():
            return False
        n = inp.numel()
        if n == 0 or n % AR_VEC != 0:
            return False
        return n * inp.element_size() < self.max_size

    def custom_all_reduce(self, inp: torch.Tensor) -> torch.Tensor:
        """Custom-AR dispatch; caller must have passed should_custom_all_reduce.

        Returns a view of the symm buffer -- the result must be consumed
        before the next all-reduce on this group.
        """
        return custom_all_reduce(
            inp,
            buffer=self.buffer,
            group_name=self.group.group_name,
            res=self.custom_ar_resources,
        )

    def all_reduce(
        self, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """
        Perform an in-place sum all-reduce via torch symmetric memory.

        Args:
            inp: Input tensor on the target CUDA device (bfloat16).
            out: Optional output tensor; if omitted, a new tensor is allocated.

        Returns:
            The reduced tensor (same shape as inp), or None if disabled.

        Implementation details:
            - Stages 'inp' into the symmetric buffer.
            - Selects 'multimem' or 'two_shot' kernel based on topology.
            - Writes the result into 'out' and returns it.
        """
        if not self.should_torch_symm_mem_allreduce(inp):
            return None
        if out is None:
            out = torch.empty_like(inp)
        self.buffer[: inp.numel()].copy_(inp.view(-1))
        if self.world_size in self._WORLD_SIZES_MULTIMEM.get(
            self.device_capability, ()
        ):
            torch.ops.symm_mem.multimem_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        else:
            torch.ops.symm_mem.two_shot_all_reduce_(
                self.buffer[: inp.numel()], "sum", self.group.group_name
            )
        out.copy_(self.buffer[: inp.numel()].view(out.shape))
        return out
