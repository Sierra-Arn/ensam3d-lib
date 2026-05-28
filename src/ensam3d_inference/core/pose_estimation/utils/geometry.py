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

# src/ensam3d_inference/core/pose_estimation/utils/geometry.py
import torch
from .types import (
    CroppedSizeTensor, 
    CLIFFConditionTensor,
    Points2DTensor,
    NormalisedCropCoords
)
from ....preprocessor.types import CamIntrinsicsTensor
from ....preprocessor.types import (
    BBoxCenterTensor, 
    BBoxScaleTensor, 
    AffineTransTensor
)


def get_decoder_condition(
    bbox_center: BBoxCenterTensor,
    bbox_scale: BBoxScaleTensor,
    cam_int: CamIntrinsicsTensor,
) -> CLIFFConditionTensor:
    """
    Compute CLIFF-style decoder conditioning vector (cx/f, cy/f, b/f).

    Parameters
    ----------
    bbox_center : BBoxCenterTensor
        Bounding box centres in full-image coordinates, shape (B, 2).
    bbox_scale : BBoxScaleTensor
        Bounding box scales (width, height) in pixels, shape (B, 2).
    cam_int : CamIntrinsicsTensor
        Camera intrinsic matrix, shape (B, 3, 3).

    Returns
    -------
    CLIFFConditionTensor
        Conditioning vector, shape (B, 3).
    """
    cx = bbox_center[:, [0]]
    cy = bbox_center[:, [1]]
    b = bbox_scale[:, [0]]
    focal_length = cam_int[:, 0, 0]

    condition = torch.cat(
        [cx - cam_int[:, [0], 2], cy - cam_int[:, [1], 2], b], dim=-1
    )
    condition[:, :2] = condition[:, :2] / focal_length.unsqueeze(-1)
    condition[:, 2] = condition[:, 2] / focal_length
    return condition


def full_to_crop(
    pred_keypoints_2d: Points2DTensor,
    affine_trans: AffineTransTensor,
    img_size: CroppedSizeTensor,
) -> NormalisedCropCoords:
    """
    Map full-image keypoints to normalised crop coordinates [-0.5, 0.5].

    Parameters
    ----------
    pred_keypoints_2d : Points2DTensor
        Keypoints in full-image space, shape (B, N, 2).
    affine_trans : AffineTransTensor
        Affine transforms from full image to crop, shape (B, 2, 3).
    img_size : CropSizeTensor
        Crop resolution (width, height) in pixels, shape (B, 2).

    Returns
    -------
    NormalisedCropCoords
        Normalised keypoint coordinates in crop space, shape (B, N, 2).
    """
    pts = torch.cat(
        [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])],
        dim=-1,
    )
    pts = pts @ affine_trans.to(pts).mT
    return pts[..., :2] / img_size.unsqueeze(1) - 0.5