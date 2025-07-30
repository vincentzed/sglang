"""
Metadata generation for EPD (Encode-Prefill-Decode) disaggregation.

This module provides the encoder-side logic to generate metadata from
multimodal inputs instead of processing raw images/videos.
"""

import asyncio
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image

from sglang.srt.disaggregation.multimodal_metadata import (
    ImageMetadata,
    VideoMetadata,
    MultimodalRequestMetadata,
    ImageProcessingParams,
    VisionConfig,
    ModalityType,
    generate_image_id,
    generate_request_id,
)
from sglang.srt.layers.rotary_embedding import MRotaryEmbedding


class MetadataGenerator:
    """Generates metadata for multimodal inputs in encoder service."""

    def __init__(self, hf_config, server_args):
        self.hf_config = hf_config
        self.server_args = server_args
        
        # Qwen2.5-VL specific configuration
        self.vision_config = VisionConfig(
            spatial_merge_size=getattr(hf_config.vision_config, 'spatial_merge_size', 2),
            patch_size=getattr(hf_config.vision_config, 'patch_size', 14),
            hidden_size=getattr(hf_config.vision_config, 'hidden_size', 3584),
            tokens_per_frame=getattr(hf_config.vision_config, 'tokens_per_frame', 770),
            vision_start_token_id=hf_config.vision_start_token_id,
            vision_end_token_id=hf_config.vision_end_token_id,
            image_token_id=hf_config.image_token_id,
        )
        
        self.image_processing_params = ImageProcessingParams(
            resize_method="smart_resize",
            min_pixels=4 * 28 * 28,  # Qwen2.5-VL default
            max_pixels=16384 * 28 * 28,  # Qwen2.5-VL default
            image_factor=28,
            max_ratio=200,
        )

    async def generate_image_metadata(
        self,
        image: Image.Image,
        image_id: str,
        request_id: str,
        token_start_idx: int,
    ) -> ImageMetadata:
        """Generate metadata for a single image."""
        
        # Get original dimensions
        original_width, original_height = image.size
        
        # Calculate processed dimensions (same logic as Qwen2_5VLImageProcessor)
        processed_height, processed_width = self._smart_resize(
            original_height,
            original_width,
            factor=28,
            min_pixels=4 * 28 * 28,
            max_pixels=16384 * 28 * 28,
        )
        
        # Calculate grid size and token count
        grid_height = processed_height // 28
        grid_width = processed_width // 28
        grid_size = (grid_height, grid_width)
        
        # Calculate tokens per image (Qwen2.5-VL specific)
        token_count = grid_height * grid_width * 4  # 4 tokens per patch
        
        # Calculate aspect ratio
        aspect_ratio = original_width / original_height
        
        # Generate image hash for cache validation
        image_bytes = image.tobytes()
        image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
        
        # Calculate M-RoPE positions (placeholder, will be calculated during reconstruction)
        mrope_positions = list(range(token_start_idx, token_start_idx + token_count))
        mrope_position_delta = 0
        
        # Create metadata
        metadata = ImageMetadata(
            image_id=image_id,
            request_id=request_id,
            original_width=original_width,
            original_height=original_height,
            processed_width=processed_width,
            processed_height=processed_height,
            aspect_ratio=aspect_ratio,
            token_count=token_count,
            grid_size=grid_size,
            token_start_idx=token_start_idx,
            token_end_idx=token_start_idx + token_count - 1,
            mrope_positions=mrope_positions,
            mrope_position_delta=mrope_position_delta,
            image_hash=image_hash,
            processing_params=self.image_processing_params,
            vision_config=self.vision_config,
            original_size_bytes=len(image_bytes),
            created_at=time.time(),
        )
        
        return metadata

    async def generate_video_metadata(
        self,
        video_frames: List[Image.Image],
        video_id: str,
        request_id: str,
        token_start_idx: int,
        original_fps: float,
        original_duration: float,
    ) -> VideoMetadata:
        """Generate metadata for a video."""
        
        if not video_frames:
            raise ValueError("Empty video frames")
        
        # Get dimensions from first frame
        original_width, original_height = video_frames[0].size
        original_frame_count = len(video_frames)
        
        # Calculate processed dimensions
        processed_height, processed_width = self._smart_resize(
            original_height,
            original_width,
            factor=28,
            min_pixels=128 * 28 * 28,
            max_pixels=768 * 28 * 28,
        )
        
        # Calculate frame selection (simplified - use all frames for now)
        frame_indices = list(range(len(video_frames)))
        processed_fps = original_fps
        processed_frame_count = len(video_frames)
        
        # Calculate tokens per frame and total
        grid_height = processed_height // 28
        grid_width = processed_width // 28
        tokens_per_frame = grid_height * grid_width * 4
        total_token_count = tokens_per_frame * processed_frame_count
        
        # Calculate frame token counts
        frame_token_counts = [tokens_per_frame] * processed_frame_count
        
        # Generate video hash
        video_hash = hashlib.sha256(
            f"{video_id}_{original_frame_count}_{original_width}_{original_height}".encode()
        ).hexdigest()[:16]
        
        # Calculate M-RoPE positions
        mrope_positions = list(range(token_start_idx, token_start_idx + total_token_count))
        
        # Create metadata
        metadata = VideoMetadata(
            video_id=video_id,
            request_id=request_id,
            original_width=original_width,
            original_height=original_height,
            original_fps=original_fps,
            original_frame_count=original_frame_count,
            original_duration=original_duration,
            original_size_bytes=None,  # Not calculated for video
            processed_width=processed_width,
            processed_height=processed_height,
            processed_fps=processed_fps,
            processed_frame_count=processed_frame_count,
            total_token_count=total_token_count,
            frame_indices=frame_indices,
            frame_token_counts=frame_token_counts,
            token_start_idx=token_start_idx,
            token_end_idx=token_start_idx + total_token_count - 1,
            mrope_positions=mrope_positions,
            video_hash=video_hash,
            processing_params=self.image_processing_params,
            vision_config=self.vision_config,
            created_at=time.time(),
        )
        
        return metadata

    async def generate_request_metadata(
        self,
        images: List[Image.Image],
        videos: List[List[Image.Image]],
        original_text: str,
        request_id: Optional[str] = None,
    ) -> MultimodalRequestMetadata:
        """Generate complete metadata for a multimodal request."""
        
        if request_id is None:
            request_id = generate_request_id()
        
        # Process images
        image_metadata = []
        current_token_idx = 0
        
        for i, image in enumerate(images):
            image_id = generate_image_id(image.tobytes())
            metadata = await self.generate_image_metadata(
                image=image,
                image_id=image_id,
                request_id=request_id,
                token_start_idx=current_token_idx,
            )
            image_metadata.append(metadata)
            current_token_idx = metadata.token_end_idx + 1
        
        # Process videos
        video_metadata = []
        for i, video_frames in enumerate(videos):
            video_id = f"video_{request_id}_{i}"
            metadata = await self.generate_video_metadata(
                video_frames=video_frames,
                video_id=video_id,
                request_id=request_id,
                token_start_idx=current_token_idx,
                original_fps=30.0,  # Default FPS
                original_duration=len(video_frames) / 30.0,
            )
            video_metadata.append(metadata)
            current_token_idx = metadata.token_end_idx + 1
        
        # Calculate totals
        total_images = len(images)
        total_videos = len(videos)
        total_audio = 0  # Not implemented yet
        
        total_multimodal_tokens = sum(
            img.token_count for img in image_metadata
        ) + sum(
            vid.total_token_count for vid in video_metadata
        )
        
        # Create token sequence mapping
        token_sequence = []
        for img_meta in image_metadata:
            token_sequence.append({
                "type": "image",
                "id": img_meta.image_id,
                "start": img_meta.token_start_idx,
                "end": img_meta.token_end_idx,
                "token_count": img_meta.token_count,
            })
        
        for vid_meta in video_metadata:
            token_sequence.append({
                "type": "video",
                "id": vid_meta.video_id,
                "start": vid_meta.token_start_idx,
                "end": vid_meta.token_end_idx,
                "token_count": vid_meta.total_token_count,
            })
        
        # Create request metadata
        request_metadata = MultimodalRequestMetadata(
            request_id=request_id,
            total_images=total_images,
            total_videos=total_videos,
            total_audio=total_audio,
            image_metadata=image_metadata,
            video_metadata=video_metadata,
            total_token_count=total_multimodal_tokens,
            text_token_count=0,  # Will be calculated during reconstruction
            multimodal_token_count=total_multimodal_tokens,
            token_sequence=token_sequence,
            model_name=self.hf_config.model_type,
            model_version=getattr(self.hf_config, 'transformers_version', 'unknown'),
        )
        
        return request_metadata

    def _smart_resize(
        self,
        height: int,
        width: int,
        factor: int = 28,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
    ) -> Tuple[int, int]:
        """Smart resize implementation matching Qwen2.5-VL."""
        import math
        
        if max(height, width) / min(height, width) > 200:
            raise ValueError(
                f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
            )
        
        h_bar = max(factor, round(height / factor) * factor)
        w_bar = max(factor, round(width / factor) * factor)
        
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = math.floor((height / beta) / factor) * factor
            w_bar = math.floor((width / beta) / factor) * factor
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil((height * beta) / factor) * factor
            w_bar = math.ceil((width * beta) / factor) * factor
        
        return h_bar, w_bar


class MetadataReconstructor:
    """Reconstructs prompts from metadata in prefill/decode services."""
    
    def __init__(self, hf_config):
        self.hf_config = hf_config
        self.vision_config = VisionConfig(
            spatial_merge_size=getattr(hf_config.vision_config, 'spatial_merge_size', 2),
            patch_size=getattr(hf_config.vision_config, 'patch_size', 14),
            hidden_size=getattr(hf_config.vision_config, 'hidden_size', 3584),
            tokens_per_frame=getattr(hf_config.vision_config, 'tokens_per_frame', 770),
            vision_start_token_id=hf_config.vision_start_token_id,
            vision_end_token_id=hf_config.vision_end_token_id,
            image_token_id=hf_config.image_token_id,
        )

    async def reconstruct_from_metadata(
        self,
        metadata: MultimodalRequestMetadata,
        text_prompt: str,
    ) -> Dict[str, Any]:
        """Reconstruct prompt data from metadata."""
        
        # Build placeholder tokens for images and videos
        image_placeholders = []
        for img_meta in metadata.image_metadata:
            placeholder = self._create_image_placeholder(img_meta)
            image_placeholders.append(placeholder)
        
        video_placeholders = []
        for vid_meta in metadata.video_metadata:
            placeholder = self._create_video_placeholder(vid_meta)
            video_placeholders.append(placeholder)
        
        # Calculate M-RoPE positions for all tokens
        mrope_positions, mrope_position_delta = self._calculate_mrope_positions(metadata)
        
        # Build final input
        reconstructed_data = {
            "input_ids": self._build_input_ids(text_prompt, image_placeholders, video_placeholders),
            "mrope_positions": mrope_positions,
            "mrope_position_delta": mrope_position_delta,
            "image_metadata": metadata.image_metadata,
            "video_metadata": metadata.video_metadata,
            "total_tokens": metadata.total_token_count,
        }
        
        return reconstructed_data

    def _create_image_placeholder(self, img_meta: ImageMetadata) -> str:
        """Create placeholder text for an image."""
        # Qwen2.5-VL uses <|vision_start|><|image_pad|>...<|vision_end|>
        num_tokens = img_meta.token_count
        placeholder = f"<|vision_start|>{'<|image_pad|>' * num_tokens}<|vision_end|>"
        return placeholder

    def _create_video_placeholder(self, vid_meta: VideoMetadata) -> str:
        """Create placeholder text for a video."""
        # Qwen2.5-VL uses <|vision_start|><|video_pad|>...<|vision_end|>
        num_tokens = vid_meta.total_token_count
        placeholder = f"<|vision_start|>{'<|video_pad|>' * num_tokens}<|vision_end|>"
        return placeholder

    def _calculate_mrope_positions(
        self,
        metadata: MultimodalRequestMetadata,
    ) -> Tuple[List[int], int]:
        """Calculate M-RoPE positions for all tokens."""
        
        # This is a simplified version - actual implementation would use
        # the same logic as MRotaryEmbedding.get_rope_index
        total_tokens = metadata.total_token_count
        mrope_positions = list(range(total_tokens))
        mrope_position_delta = 0
        
        return mrope_positions, mrope_position_delta

    def _build_input_ids(
        self,
        text_prompt: str,
        image_placeholders: List[str],
        video_placeholders: List[str],
    ) -> List[int]:
        """Build input IDs from text and placeholders."""
        # This would need tokenizer integration
        # For now, return placeholder
        return []