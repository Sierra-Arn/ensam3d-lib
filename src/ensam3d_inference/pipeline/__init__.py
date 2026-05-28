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

# src/ensam3d_inference/pipeline/__init__.py
from .types import FramePoseResult, PipelineOutput
from ..preprocessor import Preprocessor
from ..preprocessor.types import PreprocessorInput
from ..core import Engine
from ..core.types import ModelInput
from ..core.pose_estimation.types import PoseEstimatorOutput
from ..shared import DeviceType


class Pipeline:
    """
    End-to-end human pose estimation pipeline.

    Combines the preprocessing pipeline with the inference engine (feature extraction 
    and pose estimation) into a single callable interface. 
    Accepts raw RGB frames, detects valid person instances, generates crops, and returns
    full pose estimation outputs for successfully processed frames.

    Frames where no person is detected are excluded from model inference
    while preserving alignment information through the preprocessing
    metadata and valid_indices mapping.

    Parameters
    ----------
    model_path : str
        Local filesystem path to the model directory or a HuggingFace
        repository ID containing the pretrained pipeline weights.
    model_device : DeviceType, optional
        Device used for feature extraction, pose estimation, and all
        preprocessing output tensors consumed by the model.
        Default is DeviceType.CUDA.
    detector_device : DeviceType, optional
        Device used for YOLO-based human detection.
        Default is DeviceType.CUDA.
    detector_model_path : str, optional
        YOLO model name or local file path passed to the detector. If the
        weights are not found locally, they are downloaded automatically.
        Default is "yolo26n.pt".

    Attributes
    ----------
    preprocessor : Preprocessor
        Human detection and canonical cropping pipeline.
    model : Engine
        Inference engine containing the feature extractor and pose
        estimator.
    """

    def __init__(
        self,
        model_path: str,
        model_device: DeviceType = DeviceType.CUDA,
        detector_device: DeviceType = DeviceType.CUDA,
        detector_model_path: str = "yolo26n.pt",
    ) -> None:
        self.preprocessor = Preprocessor(
            detector_device=detector_device,
            model_device=model_device,
            detector_model_path=detector_model_path,
        )
        self.model = Engine(
            model_path=model_path,
            device=model_device,
        )

    def __call__(self, request: PreprocessorInput) -> PipelineOutput:
        """
        Run end-to-end human pose estimation on a batch of RGB frames.

        Performs human detection, cropping, feature extraction,
        and pose estimation for each input frame. Frames where no person is
        detected are excluded from model inference and represented as None
        entries in the output, preserving alignment with the original input
        frame sequence.

        Parameters
        ----------
        request : PreprocessorInput
            Batch of RGB frames and optional shared camera intrinsics.

        Returns
        -------
        PipelineOutput
            Container with one entry per input frame aligned with request.imgs.
            Each entry is either a FramePoseResult (detection + pose) or None
            if no person was detected in the corresponding frame.
        """
        preprocessed = self.preprocessor(request)

        if preprocessed is None:
            return PipelineOutput([None] * len(request.imgs))

        pose_output = self.model(
            ModelInput(
                img=preprocessed.img,
                bbox_center=preprocessed.bbox_center,
                bbox_scale=preprocessed.bbox_scale,
                affine_trans=preprocessed.affine_trans,
                crop_img_size=preprocessed.crop_img_size,
                ori_img_size=preprocessed.ori_img_size,
                cam_int=preprocessed.cam_int,
            )
        )

        results: PipelineOutput = [None] * len(request.imgs)
        for batch_idx, frame_idx in enumerate(preprocessed.valid_indices):
            detection = preprocessed.detections[frame_idx]
            results[frame_idx] = FramePoseResult(
                detection=detection,
                pose=PoseEstimatorOutput(
                    pred_keypoints_3d=pose_output.pred_keypoints_3d[[batch_idx]],
                    pred_keypoints_2d=pose_output.pred_keypoints_2d[[batch_idx]],
                    pred_keypoints_2d_depth=pose_output.pred_keypoints_2d_depth[[batch_idx]],
                    pred_vertices=pose_output.pred_vertices[[batch_idx]],
                    pred_keypoints_2d_verts=pose_output.pred_keypoints_2d_verts[[batch_idx]],
                    pred_cam=pose_output.pred_cam[[batch_idx]],
                    pred_cam_t=pose_output.pred_cam_t[[batch_idx]],
                    focal_length=pose_output.focal_length[[batch_idx]],
                    pred_pose_raw=pose_output.pred_pose_raw[[batch_idx]],
                    global_rot=pose_output.global_rot[[batch_idx]],
                    body_pose=pose_output.body_pose[[batch_idx]],
                    shape=pose_output.shape[[batch_idx]],
                    scale=pose_output.scale[[batch_idx]],
                    hand=pose_output.hand[[batch_idx]],
                    faces=pose_output.faces,
                ),
            )

        return results