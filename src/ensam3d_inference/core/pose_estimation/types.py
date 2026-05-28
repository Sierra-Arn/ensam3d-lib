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

# src/ensam3d_inference/core/pose_estimation/types.py
from typing import NamedTuple
import torch
from jaxtyping import Int
from .heads.mhr_head.types import (
    BodyPoseTensor,
    GlobalRotTensor,
    HandTensor,
    Keypoints3DTensor,
    MHRHeadOutput,
    PoseRawTensor,
    ScaleTensor,
    ShapeTensor,
    SkinnedVertsTensor,
)
from .heads.perspective_head.types import PerspectiveHeadOutput
from .utils.types import (
    CamTranslationTensor,
    CroppedSizeTensor,
    FocalLengthTensor,
    OriImgSizeTensor,
    Points2DTensor,
    PointsDepthTensor,
    ProjectionOutput,
)
from ..feature_extraction.backbone.types import BackboneOutput
from ...preprocessor.types import (
    AffineTransTensor, 
    BBoxCenterTensor, 
    BBoxScaleTensor
)
from ...preprocessor.types import CamIntrinsicsTensor


FacesTensor = Int[torch.Tensor, "F 3"]
"""
Mesh face indices represented as a PyTorch long tensor with shape (36874, 3) and integer dtype,
where 36874 is the fixed number of triangular faces in the MHR body mesh and 3 is the number
of vertex indices per triangle; this tensor lives on CPU and is shared across all frames in the
batch as it encodes static mesh topology, not per-frame geometry.
"""


class IteratedPoseOutput(NamedTuple):
    """
    Intermediate pose output produced by one decoder layer iteration.

    Collected by the decoder at each layer and passed to the keypoint token
    update functions; only the final instance is used to assemble the full
    PoseEstimatorOutput.

    Attributes
    ----------
    mhr_out : MHRHeadOutput
        Output of the MHR head for this iteration; pred_keypoints_3d is
        always populated, all other fields only on the final iteration.
    pred_cam : PerspectiveHeadOutput
        Weak-perspective camera parameters (s, tx, ty), shape (B, 3).
    proj_out : ProjectionOutput
        Output of perspective projection; pred_keypoints_2d and
        pred_keypoints_2d_depth are always populated, pred_cam_t and
        focal_length only on the final iteration.
    """

    mhr_out: MHRHeadOutput
    pred_cam: PerspectiveHeadOutput
    proj_out: ProjectionOutput


class PoseEstimatorInput(NamedTuple):
    """
    Input batch for the pose estimator.

    All tensors share the same device. The batch axis B counts frames,
    not people within a single frame.

    Attributes
    ----------
    image_embeddings : BackboneOutput
        Ray-conditioned feature map from the feature extractor, shape (B, C, Hp, Wp).
    bbox_center : BBoxCenterTensor
        Bounding box centres in full-image pixel coordinates, shape (B, 2).
    bbox_scale : BBoxScaleTensor
        Bounding box scales (width, height) in pixels, shape (B, 2).
    affine_trans : AffineTransTensor
        Affine transforms mapping full-image coordinates to crop space, shape (B, 2, 3).
    img_size : CropSizeTensor
        Crop resolution (width, height) in pixels, shape (B, 2).
    ori_img_size : OriImgSizeTensor
        Original full-image resolution (width, height) in pixels, shape (B, 2).
    cam_int : CamIntrinsicsTensor
        Camera intrinsic matrix, shape (B, 3, 3).
    """

    image_embeddings: BackboneOutput
    bbox_center: BBoxCenterTensor
    bbox_scale: BBoxScaleTensor
    affine_trans: AffineTransTensor
    img_size: CroppedSizeTensor
    ori_img_size: OriImgSizeTensor
    cam_int: CamIntrinsicsTensor


class PoseEstimatorOutput(NamedTuple):
    """
    Output of the pose estimator after a full forward pass.

    Attributes
    ----------
    pred_keypoints_3d : Keypoints3DTensor
        3D body keypoints in camera space, shape (B, 70, 3).
    pred_keypoints_2d : Points2DTensor
        Projected 2D keypoints in full image space, shape (B, 70, 2).
    pred_keypoints_2d_depth : PointsDepthTensor
        Depth of projected keypoints in camera space, shape (B, 70).
    pred_vertices : SkinnedVertsTensor
        Mesh vertices in camera space, shape (B, V, 3).
    pred_keypoints_2d_verts : Points2DTensor
        Projected 2D mesh vertices in full image space, shape (B, V, 2).
    pred_cam : PerspectiveHeadOutput
        Weak-perspective camera parameters (s, tx, ty), shape (B, 3).
    pred_cam_t : CamTranslationTensor
        Perspective camera translation (tx, ty, tz) in camera space, shape (B, 3).
    focal_length : FocalLengthTensor
        Per-frame focal length from the intrinsic matrix, shape (B,).
    pred_pose_raw : PoseRawTensor
        Concatenated 6D global rotation and continuous body pose, shape (B, 266).
    global_rot : GlobalRotTensor
        Global orientation as ZYX Euler angles, shape (B, 3).
    body_pose : BodyPoseTensor
        MHR body pose parameters, shape (B, 133).
    shape : ShapeTensor
        Shape PCA coefficients, shape (B, 45).
    scale : ScaleTensor
        Scale PCA coefficients, shape (B, 28).
    hand : HandTensor
        Hand PCA coefficients for left and right hands, shape (B, 108).
    faces : FacesTensor
        Mesh face indices on CPU, shape (36874, 3).
    """

    pred_keypoints_3d: Keypoints3DTensor
    pred_keypoints_2d: Points2DTensor
    pred_keypoints_2d_depth: PointsDepthTensor
    pred_vertices: SkinnedVertsTensor
    pred_keypoints_2d_verts: Points2DTensor
    pred_cam: PerspectiveHeadOutput
    pred_cam_t: CamTranslationTensor
    focal_length: FocalLengthTensor
    pred_pose_raw: PoseRawTensor
    global_rot: GlobalRotTensor
    body_pose: BodyPoseTensor
    shape: ShapeTensor
    scale: ScaleTensor
    hand: HandTensor
    faces: FacesTensor
