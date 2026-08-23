"""Native CLIP and BGE-VL dual encoders."""

from .checkpoint import (
    CLIPCheckpointAdapter,
    clip_checkpoint_directory,
    clip_vision_from_state_dict,
    clip_vision_state_dict,
    clip_weight_names,
)
from .config import (
    BGE_VL_BASE_MODEL_ID,
    BGE_VL_BASE_REVISION,
    CLIP_VIT_B32_MODEL_ID,
    CLIP_VIT_B32_REVISION,
    CLIPConfig,
    CLIPTextConfig,
    CLIPVisionConfig,
)
from .loading import load_clip
from .model import CLIPBatch, CLIPEncoder, CLIPVisionTower
from .processing import make_clip_processor

__all__ = [
    "BGE_VL_BASE_MODEL_ID",
    "BGE_VL_BASE_REVISION",
    "CLIPBatch",
    "CLIPCheckpointAdapter",
    "CLIPConfig",
    "CLIPEncoder",
    "CLIPTextConfig",
    "CLIPVisionConfig",
    "CLIPVisionTower",
    "CLIP_VIT_B32_MODEL_ID",
    "CLIP_VIT_B32_REVISION",
    "clip_checkpoint_directory",
    "clip_vision_from_state_dict",
    "clip_vision_state_dict",
    "clip_weight_names",
    "load_clip",
    "make_clip_processor",
]
