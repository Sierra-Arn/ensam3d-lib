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

# src/ensam3d_inference/core/pose_estimation/utils/projection.py
import torch
from .types import (
    Points3DTensor,
    BBoxSizeTensor,
    ProjectionOutput,
)
from ..heads.perspective_head.types import PerspectiveHeadOutput
from ....preprocessor.types import CamIntrinsicsTensor
from ....preprocessor.types import BBoxCenterTensor


def perspective_projection(
    points_3d: Points3DTensor,
    pred_cam: PerspectiveHeadOutput,
    bbox_center: BBoxCenterTensor,
    bbox_size: BBoxSizeTensor,
    cam_int: CamIntrinsicsTensor,
    full_output: bool = True,
) -> ProjectionOutput:
    """
    Project 3D points into the full image plane via perspective projection.

    Converts weak-perspective camera parameters (s, tx, ty) and camera
    intrinsics into a full perspective camera translation, then projects
    the input 3D points onto the image plane.

    Parameters
    ----------
    points_3d : Points3DTensor
        3D points in camera space, shape (B, N, 3).
    pred_cam : PerspectiveHeadOutput
        Weak-perspective camera parameters (s, tx, ty), shape (B, 3).
    bbox_center : BBoxCenterTensor
        Bounding box centre in full-image pixel coordinates, shape (B, 2).
    bbox_size : BBoxSizeTensor
        Bounding box side length in pixels, shape (B,).
    cam_int : CamIntrinsicsTensor
        Camera intrinsic matrix, shape (B, 3, 3).
    full_output : bool, optional
        If True, computes and returns pred_cam_t and focal_length in
        addition to the projected points and depth. If False, these
        fields are None. Default is True.

    Returns
    -------
    ProjectionOutput
        Projected 2D points and depth always populated; pred_cam_t and
        focal_length populated only when full_output is True.
    """
    batch_size = points_3d.shape[0]
    points_3d = points_3d.float()
    pred_cam = pred_cam.float().clone()
    bbox_center = bbox_center.float()
    bbox_size = bbox_size.float()
    cam_int = cam_int.float()

    pred_cam[..., [0, 2]] *= -1

    s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
    focal_length = cam_int[:, 0, 0]
    bs = bbox_size * s + 1e-8
    tz = 2 * focal_length / bs

    cx = 2 * (bbox_center[:, 0] - cam_int[:, 0, 2]) / bs
    cy = 2 * (bbox_center[:, 1] - cam_int[:, 1, 2]) / bs

    pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    j3d_cam = points_3d + pred_cam_t.unsqueeze(1)

    y = j3d_cam / j3d_cam[:, :, -1].unsqueeze(-1)
    y = torch.einsum("bij,bkj->bki", cam_int, y.float())
    j2d = y[:, :, :2].to(dtype=j3d_cam.dtype)

    if not full_output:
        return ProjectionOutput(
            pred_keypoints_2d=j2d.reshape(batch_size, -1, 2),
            pred_keypoints_2d_depth=j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
            pred_cam_t=None,
            focal_length=None,
        )

    return ProjectionOutput(
        pred_keypoints_2d=j2d.reshape(batch_size, -1, 2),
        pred_keypoints_2d_depth=j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
        pred_cam_t=pred_cam_t,
        focal_length=focal_length,
    )