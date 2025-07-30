"""
Copyright 2023-2024 SGLang Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Dict, List, Optional

import torch

from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter


class MultimodalEmbeddingPool:
    """A memory pool for multimodal embeddings."""

    def __init__(
        self,
        size: int,
        embedding_dim: int,
        dtype: torch.dtype,
        device: str,
        enable_memory_saver: bool,
    ):
        self.size = size
        self.embedding_dim = embedding_dim
        self.dtype = dtype
        self.device = device
        self.enable_memory_saver = enable_memory_saver
        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
        )

        with self.memory_saver_adapter.region("mm_embedding_pool"):
            self.embedding_pool = torch.zeros(
                (size, embedding_dim), dtype=dtype, device=device
            )

        self.free_slots = list(range(size))
        self.mm_hash_to_slot: Dict[int, int] = {}

    def alloc(self, mm_hash: int) -> Optional[int]:
        if mm_hash in self.mm_hash_to_slot:
            return self.mm_hash_to_slot[mm_hash]

        if not self.free_slots:
            return None

        slot = self.free_slots.pop(0)
        self.mm_hash_to_slot[mm_hash] = slot
        return slot

    def put(self, mm_hash: int, embedding: torch.Tensor) -> bool:
        if not self.has(mm_hash):
            slot = self.alloc(mm_hash)
            if slot is None:
                return False
        else:
            slot = self.mm_hash_to_slot[mm_hash]

        self.embedding_pool[slot] = embedding
        return True

    def has(self, mm_hash: int) -> bool:
        return mm_hash in self.mm_hash_to_slot

    def get(self, mm_hash: int) -> Optional[torch.Tensor]:
        if not self.has(mm_hash):
            return None
        slot = self.mm_hash_to_slot[mm_hash]
        return self.embedding_pool[slot]

    def free(self, mm_hash: int):
        if mm_hash in self.mm_hash_to_slot:
            slot = self.mm_hash_to_slot.pop(mm_hash)
            self.free_slots.append(slot)

    def clear(self):
        self.free_slots = list(range(self.size))
        self.mm_hash_to_slot.clear()

    def available_size(self) -> int:
        return len(self.free_slots)
