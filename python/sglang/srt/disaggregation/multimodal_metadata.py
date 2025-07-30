"""
Core metadata structures for EPD (Encode-Prefill-Decode) disaggregation
with metadata-based multimodal processing.

This module defines the complete metadata packet structure that enables
the encoder to send processed image metadata to prefill/decode services
instead of raw image data.
"""

import dataclasses
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from enum import Enum


class ModalityType(Enum):
    """Supported modality types for metadata"""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"


@dataclasses.dataclass
class ImageProcessingParams:
    """Parameters used for image preprocessing"""
    resize_method: str = "smart_resize"
    target_size: Optional[Tuple[int, int]] = None  # (height, width)
    min_pixels: int = 4 * 28 * 28  # Qwen2.5-VL default
    max_pixels: int = 16384 * 28 * 28  # Qwen2.5-VL default
    image_factor: int = 28  # Qwen2.5-VL grid size
    max_ratio: int = 200
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "resize_method": self.resize_method,
            "target_size": self.target_size,
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
            "image_factor": self.image_factor,
            "max_ratio": self.max_ratio,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageProcessingParams":
        """Create from dictionary"""
        return cls(**data)


@dataclasses.dataclass
class VisionConfig:
    """Model-specific vision configuration"""
    spatial_merge_size: int = 2  # Qwen2.5-VL default
    patch_size: int = 14
    hidden_size: int = 3584  # Qwen2.5-VL-7B
    tokens_per_frame: int = 770  # Qwen2.5-VL default
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    image_token_id: int = 151654
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return dataclasses.asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisionConfig":
        """Create from dictionary"""
        return cls(**data)


@dataclasses.dataclass
class ImageMetadata:
    """Complete metadata for an image without raw pixel data"""
    
    # Required fields (no defaults)
    image_id: str  # Unique identifier for this image instance
    request_id: str  # Parent request ID
    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    aspect_ratio: float  # width / height
    token_count: int  # Total number of image tokens
    grid_size: Tuple[int, int]  # (height_tiles, width_tiles)
    token_start_idx: int  # Start position in token sequence
    token_end_idx: int  # End position in token sequence
    mrope_positions: List[int]  # Positional indices for M-RoPE
    mrope_position_delta: int  # Delta for M-RoPE calculation
    image_hash: str  # SHA256 hash of processed image for cache matching
    processing_params: ImageProcessingParams
    vision_config: VisionConfig
    
    # Optional fields (with defaults)
    modality: ModalityType = ModalityType.IMAGE
    original_size_bytes: Optional[int] = None  # Size of original image
    compressed_features: Optional[bytes] = None
    compression_method: Optional[str] = None
    created_at: Optional[float] = None  # Unix timestamp
    processing_time_ms: Optional[float] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived fields and validate"""
        if self.checksum is None:
            self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for data integrity validation"""
        data = {
            "image_id": self.image_id,
            "request_id": self.request_id,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "token_count": self.token_count,
            "grid_size": self.grid_size,
            "mrope_positions": self.mrope_positions,
            "image_hash": self.image_hash,
        }
        checksum_str = json.dumps(data, sort_keys=True)
        self.checksum = hashlib.sha256(checksum_str.encode()).hexdigest()[:16]
        return self.checksum
    
    def validate_checksum(self) -> bool:
        """Validate the checksum"""
        if not self.checksum:
            return False
        current_checksum = self.checksum
        self.checksum = None
        calculated = self.calculate_checksum()
        self.checksum = current_checksum
        return calculated == current_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for transfer"""
        return {
            "image_id": self.image_id,
            "request_id": self.request_id,
            "modality": self.modality.value,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "original_size_bytes": self.original_size_bytes,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "aspect_ratio": self.aspect_ratio,
            "token_count": self.token_count,
            "grid_size": self.grid_size,
            "token_start_idx": self.token_start_idx,
            "token_end_idx": self.token_end_idx,
            "mrope_positions": self.mrope_positions,
            "mrope_position_delta": self.mrope_position_delta,
            "image_hash": self.image_hash,
            "processing_params": self.processing_params.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "compressed_features": self.compressed_features.hex() if self.compressed_features else None,
            "compression_method": self.compression_method,
            "created_at": self.created_at,
            "processing_time_ms": self.processing_time_ms,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ImageMetadata":
        """Create from dictionary"""
        # Handle enum conversion
        data = data.copy()
        if isinstance(data.get("modality"), str):
            data["modality"] = ModalityType(data["modality"])
        
        # Handle nested objects
        if "processing_params" in data:
            data["processing_params"] = ImageProcessingParams.from_dict(
                data["processing_params"]
            )
        if "vision_config" in data:
            data["vision_config"] = VisionConfig.from_dict(data["vision_config"])
        
        # Handle bytes conversion
        if data.get("compressed_features"):
            data["compressed_features"] = bytes.fromhex(data["compressed_features"])
        
        return cls(**data)
    
    def get_size_estimate(self) -> int:
        """Estimate serialized size in bytes"""
        return len(json.dumps(self.to_dict()).encode())


@dataclasses.dataclass
class VideoMetadata:
    """Complete metadata for a video without raw pixel data"""
    
    # Required fields (no defaults)
    video_id: str  # Unique identifier for this video instance
    request_id: str  # Parent request ID
    original_width: int
    original_height: int
    original_fps: float
    original_frame_count: int
    original_duration: float  # seconds
    original_size_bytes: Optional[int]
    processed_width: int
    processed_height: int
    processed_fps: float
    processed_frame_count: int
    total_token_count: int
    frame_indices: List[int]  # Selected frame indices
    frame_token_counts: List[int]  # Tokens per frame
    token_start_idx: int
    token_end_idx: int
    mrope_positions: List[int]
    video_hash: str
    processing_params: ImageProcessingParams
    vision_config: VisionConfig
    
    # Optional fields (with defaults)
    modality: ModalityType = ModalityType.VIDEO
    compressed_features: Optional[bytes] = None
    compression_method: Optional[str] = None
    created_at: Optional[float] = None  # Unix timestamp
    processing_time_ms: Optional[float] = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate derived fields and validate"""
        if self.checksum is None:
            self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for data integrity validation"""
        data = {
            "video_id": self.video_id,
            "request_id": self.request_id,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "original_fps": self.original_fps,
            "original_frame_count": self.original_frame_count,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "processed_fps": self.processed_fps,
            "processed_frame_count": self.processed_frame_count,
            "total_token_count": self.total_token_count,
            "frame_indices": self.frame_indices,
            "frame_token_counts": self.frame_token_counts,
            "mrope_positions": self.mrope_positions,
            "video_hash": self.video_hash,
        }
        checksum_str = json.dumps(data, sort_keys=True)
        self.checksum = hashlib.sha256(checksum_str.encode()).hexdigest()[:16]
        return self.checksum
    
    def validate_checksum(self) -> bool:
        """Validate the checksum"""
        if not self.checksum:
            return False
        current_checksum = self.checksum
        self.checksum = None
        calculated = self.calculate_checksum()
        self.checksum = current_checksum
        return calculated == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary for transfer"""
        return {
            "video_id": self.video_id,
            "request_id": self.request_id,
            "modality": self.modality.value,
            "original_width": self.original_width,
            "original_height": self.original_height,
            "original_fps": self.original_fps,
            "original_frame_count": self.original_frame_count,
            "original_duration": self.original_duration,
            "original_size_bytes": self.original_size_bytes,
            "processed_width": self.processed_width,
            "processed_height": self.processed_height,
            "processed_fps": self.processed_fps,
            "processed_frame_count": self.processed_frame_count,
            "total_token_count": self.total_token_count,
            "frame_indices": self.frame_indices,
            "frame_token_counts": self.frame_token_counts,
            "token_start_idx": self.token_start_idx,
            "token_end_idx": self.token_end_idx,
            "mrope_positions": self.mrope_positions,
            "video_hash": self.video_hash,
            "processing_params": self.processing_params.to_dict(),
            "vision_config": self.vision_config.to_dict(),
            "compressed_features": self.compressed_features.hex() if self.compressed_features else None,
            "compression_method": self.compression_method,
            "created_at": self.created_at,
            "processing_time_ms": self.processing_time_ms,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        """Create from dictionary"""
        data = data.copy()
        if isinstance(data.get("modality"), str):
            data["modality"] = ModalityType(data["modality"])
        
        # Handle nested objects
        if "processing_params" in data:
            data["processing_params"] = ImageProcessingParams.from_dict(
                data["processing_params"]
            )
        if "vision_config" in data:
            data["vision_config"] = VisionConfig.from_dict(data["vision_config"])
        
        # Handle bytes conversion
        if data.get("compressed_features"):
            data["compressed_features"] = bytes.fromhex(data["compressed_features"])
        
        return cls(**data)
    
    def get_size_estimate(self) -> int:
        """Estimate serialized size in bytes"""
        return len(json.dumps(self.to_dict()).encode())


@dataclasses.dataclass
class MultimodalRequestMetadata:
    """Complete metadata for a multimodal request"""
    
    request_id: str
    total_images: int
    total_videos: int
    total_audio: int
    
    # Individual metadata
    image_metadata: List[ImageMetadata]
    video_metadata: List[VideoMetadata]
    # audio_metadata: List[AudioMetadata]  # Future extension
    
    # Combined token information
    total_token_count: int
    text_token_count: int
    multimodal_token_count: int
    
    # Token sequence mapping
    token_sequence: List[Dict[str, Any]]  # Maps tokens to their sources
    
    # Request-level configuration
    model_name: str
    model_version: str
    
    # Validation
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """Calculate request-level checksum"""
        if self.checksum is None:
            self.calculate_checksum()
    
    def calculate_checksum(self) -> str:
        """Calculate checksum for entire request"""
        data = {
            "request_id": self.request_id,
            "total_images": self.total_images,
            "total_videos": self.total_videos,
            "total_audio": self.total_audio,
            "total_token_count": self.total_token_count,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }
        
        # Include individual checksums
        for img_meta in self.image_metadata:
            data[f"img_{img_meta.image_id}"] = img_meta.checksum
        
        for vid_meta in self.video_metadata:
            data[f"vid_{vid_meta.video_id}"] = vid_meta.video_hash
        
        checksum_str = json.dumps(data, sort_keys=True)
        self.checksum = hashlib.sha256(checksum_str.encode()).hexdigest()[:16]
        return self.checksum
    
    def validate_checksum(self) -> bool:
        """Validate all checksums in the request"""
        if not self.checksum:
            return False
        
        # Validate individual metadata
        for img_meta in self.image_metadata:
            if not img_meta.validate_checksum():
                return False
        
        for vid_meta in self.video_metadata:
            if not vid_meta.validate_checksum():
                return False
        
        # Validate request-level checksum
        current_checksum = self.checksum
        self.checksum = None
        calculated = self.calculate_checksum()
        self.checksum = current_checksum
        
        return calculated == current_checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        return {
            "request_id": self.request_id,
            "total_images": self.total_images,
            "total_videos": self.total_videos,
            "total_audio": self.total_audio,
            "image_metadata": [img.to_dict() for img in self.image_metadata],
            "video_metadata": [vid.to_dict() for vid in self.video_metadata],
            "total_token_count": self.total_token_count,
            "text_token_count": self.text_token_count,
            "multimodal_token_count": self.multimodal_token_count,
            "token_sequence": self.token_sequence,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "checksum": self.checksum,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultimodalRequestMetadata":
        """Create from dictionary"""
        data = data.copy()
        
        # Handle nested objects
        if "image_metadata" in data:
            data["image_metadata"] = [
                ImageMetadata.from_dict(img) for img in data["image_metadata"]
            ]
        
        if "video_metadata" in data:
            data["video_metadata"] = [
                VideoMetadata.from_dict(vid) for vid in data["video_metadata"]
            ]
        
        return cls(**data)
    
    def get_total_size_estimate(self) -> int:
        """Estimate total serialized size in bytes"""
        return len(json.dumps(self.to_dict()).encode())
    
    def get_image_by_id(self, image_id: str) -> Optional[ImageMetadata]:
        """Get image metadata by ID"""
        for img_meta in self.image_metadata:
            if img_meta.image_id == image_id:
                return img_meta
        return None
    
    def get_video_by_id(self, video_id: str) -> Optional[VideoMetadata]:
        """Get video metadata by ID"""
        for vid_meta in self.video_metadata:
            if vid_meta.video_id == video_id:
                return vid_meta
        return None


# Utility functions for metadata handling
def generate_image_id(image_data: bytes) -> str:
    """Generate unique image ID from image data"""
    return hashlib.sha256(image_data).hexdigest()[:16]


def generate_request_id() -> str:
    """Generate unique request ID"""
    import uuid
    return str(uuid.uuid4())[:16]


def validate_metadata_compatibility(
    metadata: MultimodalRequestMetadata,
    model_config: Dict[str, Any]
) -> bool:
    """Validate metadata compatibility with current model"""
    # Check model name/version
    if metadata.model_name != model_config.get("model_name"):
        return False
    
    # Check vision config compatibility
    for img_meta in metadata.image_metadata:
        if img_meta.vision_config.spatial_merge_size != model_config.get(
            "spatial_merge_size", 2
        ):
            return False
    
    return True