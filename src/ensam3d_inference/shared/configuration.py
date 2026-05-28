# Copyright (c) 2026 Ilya Snegov (aka Sierra Arn)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# src/ensam3d_inference/shared/configuration.py
from dataclasses import dataclass
from functools import cached_property
import warnings
import numpy as np
import torch


@dataclass(frozen=True)
class PipelineConfig:
    """
    Centralized configuration for all architecture and pipeline constants.

    Stores all structural parameters required to instantiate the backbone,
    decoder, regression heads, positional encodings, preprocessing
    pipeline, and checkpoint-compatible tensor shapes.
    """

    # ========== PREPROCESSING ==========

    scale_padding: float = 1.25
    intermediate_aspect_ratio: float = 0.75
    detector_cat_id: int = 0
    detector_conf_thr: float = 0.25
    detector_nms_thr: float = 0.45
    image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    image_std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ========== BACKBONE ==========

    backbone_input_size: tuple[int, int] = (512, 384)  # (W, H)
    backbone_patch_size: int = 16
    backbone_embed_dim: int = 1280
    backbone_depth: int = 32
    backbone_heads: int = 16
    backbone_mlp_ratio: float = 4.0
    backbone_crop_width_px: int = 64

    # ========== DECODER ==========

    decoder_dim: int = 1024
    decoder_depth: int = 6
    decoder_heads: int = 8
    decoder_head_dims: int = 64
    decoder_mlp_dims: int = 1024
    decoder_context_dim: int = 1280

    # ========== HEADS & MHR ==========

    mhr_shape_comps: int = 45
    mhr_scale_comps: int = 28
    mhr_hand_comps: int = 54
    mhr_face_comps: int = 72
    mhr_body_cont_dim: int = 260
    camera_ncam: int = 3
    cond_dim: int = 3

    # ========== KEYPOINTS & GEOMETRY ==========

    n_keypoints: int = 70
    pelvis_indices: tuple[int, int] = (9, 10)

    # ========== TYPES ==========

    numpy_dtype: np.dtype = np.float32
    """
    Precision for all intermediate NumPy arrays in the preprocessor.
    Fixed to float32 because OpenCV geometric primitives natively expect 
    and output 32-bit floats. Since NumPy is used exclusively at this 
    stage for CPU-bound warping, this dtype ensures maximum compatibility 
    and performance without unnecessary conversions.
    """

    core_input_dtype: torch.dtype = torch.float32
    """
    Precision of tensors exiting the preprocessor and entering the model.
    Fixed to float32 because the pretrained checkpoint was trained on
    float32 inputs.
    """

    @cached_property
    def core_compute_dtype(self) -> torch.dtype:
        """
        Precision used for forward-pass computations inside torch.autocast.
        Dynamically resolved at runtime based on GPU compute capabilities.
        Ampere and newer architectures use bfloat16 to leverage tensor cores
        while preserving the float32 dynamic range. Pre-Ampere CUDA devices
        fall back to float16 with a numerical stability warning. CPU targets
        or unsupported hardware default to float32 for guaranteed compatibility.
        Applied exclusively to backbone and decoder activations; sensitive
        geometric heads and the MHR TorchScript solver remain in float32 via
        explicit dtype boundaries.

        Returns
        -------
        torch.dtype
            Selected compute dtype for the current execution environment.
        """
        
        if torch.cuda.is_available():
            try:
                cc_major = torch.cuda.get_device_properties(0).major
                if cc_major >= 8:
                    return torch.bfloat16
                elif cc_major >= 6:
                    warnings.warn(
                        "GPU compute capability < 8.0: falling back to float16.",
                        UserWarning,
                        stacklevel=2,
                    )
                    return torch.float16
            except Exception:
                pass

        warnings.warn(
            "No compatible GPU or unsupported architecture detected: using float32. "
            "Mixed-precision acceleration is disabled; expect higher VRAM consumption "
            "and reduced throughput compared to CUDA-optimized runtimes.",
            UserWarning,
            stacklevel=2,
        )
        return torch.float32

    # ========== MODEL ASSETS & LOADING ==========

    checkpoint_filename: str = "model.ckpt"
    """
    Relative path to the primary PyTorch checkpoint file within the model
    directory. Combined with the resolved local model path to form the
    absolute filesystem location of backbone, decoder, and regression head
    weights.
    """

    mhr_asset_path: str = "assets/mhr_model.pt"
    """
    Relative path to the TorchScript MHR kinematic solver asset within the
    model directory. Combined with the resolved local model path to locate
    the frozen .pt file containing PCA components and mesh topology.
    """

    mhr_skip_prefix: str = "mhr_head."
    """
    Module attribute prefix used by the precision-refresh routine to exclude
    MHR-related parameters from bfloat16 casting. The TorchScript solver and
    its immediate wrapper rely on float32 sparse matrix multiplications and
    rigid-body kinematics that are numerically unsupported in
    bfloat16.
    """

    # ========== CHECKPOINT KEY MAPPING ==========

    # Rules for translating legacy parameter names from the original SAM-3D-Body
    # checkpoint into the refactored ensam3d_inference architecture.
    # Applied during weight loading to ensure full compatibility without modifying
    # the original .ckpt file. Mappings are evaluated sequentially: prefixes first,
    # then substrings, guaranteeing deterministic renaming even for nested modules.

    checkpoint_legacy_prefixes = (
        ("head_pose.",           "mhr_head."),
        ("head_camera.",         "camera_head."),
        ("ray_cond_emb.",        "camera_encoder."),
        ("init_to_token_mhr.",   "init_to_token."),
        ("prev_to_token_mhr.",   "prev_to_token."),
        ("init_to_token_mhr",    "init_to_token"),
        ("prev_to_token_mhr",    "prev_to_token"),
    )

    checkpoint_legacy_substrings = (
        (".ffn.layers.0.0.", ".mlp.fc1."),
        (".ffn.layers.1.",   ".mlp.fc2."),
    )

    # ========== DERIVED PROPERTIES ==========

    @cached_property
    def cropped_backbone_size(self) -> tuple[int, int]:
        w, h = self.backbone_input_size
        return (w - 2 * self.backbone_crop_width_px, h)

    @cached_property
    def patch_grid(self) -> tuple[int, int]:
        h, w = self.cropped_backbone_size
        return (
            h // self.backbone_patch_size,
            w // self.backbone_patch_size,
        )

    @cached_property
    def npose(self) -> int:
        return (
            6  # global rotation in 6D representation
            + self.mhr_body_cont_dim
            + self.mhr_shape_comps
            + self.mhr_scale_comps
            + self.mhr_hand_comps * 2  # left + right hand
            + self.mhr_face_comps
        )

    # ========== VALIDATION ==========

    def __post_init__(self) -> None:
        """
        Validate internal architectural consistency.

        Ensures that all derived tensor dimensions remain compatible with
        the pretrained checkpoint and dependent model components.

        Raises
        ------
        ValueError
            Raised when one or more configuration values are incompatible
            with the expected checkpoint architecture.
        """
        if self.decoder_context_dim != self.backbone_embed_dim:
            raise ValueError(
                "decoder_context_dim must equal backbone_embed_dim"
            )

        if self.patch_grid != (24, 24):
            raise ValueError(
                f"Current checkpoint expects (24, 24), "
                f"got {self.patch_grid}"
            )

        if self.npose != 519:
            raise ValueError(
                f"Checkpoint npose is 519, calculated {self.npose}"
            )


# Single shared configuration instance used across the entire pipeline
config = PipelineConfig()
