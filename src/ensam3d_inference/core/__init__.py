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

# src/ensam3d_inference/core/__init__.py
from pathlib import Path
import torch
from .types import ModelInput
from .utils import (
    load_weights_into_module, 
    resolve_model_path, 
    cast_floating_params_buffers_bf16
)
from .feature_extraction import FeatureExtractor
from .pose_estimation import PoseEstimator
from .pose_estimation.types import (
    PoseEstimatorInput,
    PoseEstimatorOutput,
)
from ..shared import DeviceType, config


class Engine:
    """
    Full inference stack combining FeatureExtractor and PoseEstimator.

    Loads model weights from a local directory or downloads them from
    HuggingFace Hub if the path does not exist locally. Weights are cast
    to bfloat16 after loading except for LayerNormFp32 layers and the
    TorchScript MHR submodule which remain in float32.

    Parameters
    ----------
    model_path : str or Path
        Local filesystem path to the model directory or a HuggingFace
        repository ID. The directory must contain model.ckpt and
        assets/mhr_model.pt.
    device : DeviceType, optional
        Device used for inference. Default is DeviceType.CUDA.

    Attributes
    ----------
    feature_extractor : FeatureExtractor
        ViT backbone with Fourier ray conditioning.
    pose_estimator : PoseEstimator
        Iterative transformer decoder with MHR and camera heads.
    device : DeviceType
        Stored inference device.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: DeviceType = DeviceType.CUDA,
    ) -> None:
        self.device = device
        local_path = resolve_model_path(model_path)

        checkpoint_path = local_path / config.checkpoint_filename
        mhr_model_path = local_path / config.mhr_asset_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not mhr_model_path.exists():
            raise FileNotFoundError(f"MHR model asset not found: {mhr_model_path}")

        self.feature_extractor = FeatureExtractor()
        self.pose_estimator = PoseEstimator(
            mhr_model_path=str(mhr_model_path),
            device=device,
        )

        load_weights_into_module(
            self.feature_extractor,
            checkpoint_path,
            map_location=DeviceType.CPU,
        )
        load_weights_into_module(
            self.pose_estimator,
            checkpoint_path,
            map_location=DeviceType.CPU,
        )

        cast_floating_params_buffers_bf16(self.feature_extractor)
        cast_floating_params_buffers_bf16(self.pose_estimator)

        self.feature_extractor.to(device).eval()
        self.pose_estimator.to(device).eval()

    @torch.inference_mode()
    def __call__(self, batch: ModelInput) -> PoseEstimatorOutput:
        """
        Run full inference on a preprocessed batch.

        Parameters
        ----------
        batch : ModelInput
            Preprocessed crops and geometric metadata.

        Returns
        -------
        PoseEstimatorOutput
            Full pose, mesh, and camera outputs for the batch.
        """
        with torch.autocast(device_type=self.device, dtype=config.core_compute_dtype):

            image_embeddings = self.feature_extractor(
                batch.img,
                affine_trans=batch.affine_trans,
                cam_int=batch.cam_int,
            )

            pose_output = self.pose_estimator(
                PoseEstimatorInput(
                    image_embeddings=image_embeddings,
                    bbox_center=batch.bbox_center,
                    bbox_scale=batch.bbox_scale,
                    affine_trans=batch.affine_trans,
                    img_size=batch.crop_img_size,
                    ori_img_size=batch.ori_img_size,
                    cam_int=batch.cam_int,
                )
            )
    
        return pose_output
