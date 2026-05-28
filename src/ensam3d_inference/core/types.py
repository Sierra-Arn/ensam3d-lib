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

# src/ensam3d_inference/core/types.py
from typing import NamedTuple
from ..preprocessor.types import (
    CamIntrinsicsTensor, 
    OriImgSizeTensor, 
    CroppedSizeTensor, 
    CroppedTensor, 
    BBoxCenterTensor, 
    BBoxScaleTensor, 
    AffineTransTensor
)


class ModelInput(NamedTuple):
    """
    Container for the batched input tensors passed to the 3D pose model.

    Attributes
    ----------
    img : CroppedTensor
        Batched canonical RGB crops with batch dimension B, already warped
        to the fixed backbone input resolution expected by the model.
    bbox_center : BBoxCenterTensor
        Bounding box centers in original image pixel coordinates, shape
        (B, 2), where each row corresponds to (x, y).
    bbox_scale : BBoxScaleTensor
        Bounding box scale factors derived from the detector output, shape
        (B,), used to normalize the crop geometry during preprocessing and
        reconstruction.
    affine_trans : AffineTransTensor
        Forward affine transformation matrices, shape (B, 2, 3), mapping
        coordinates from the original image space into the canonical crop
        coordinate system.
    ori_img_size : OriImgSizeTensor
        Original full-image resolution (width, height) in pixels, shape
        (B, 2), shared across all frames in the batch.
    crop_img_size : CroppedSizeTensor
        Canonical crop resolution (width, height) in pixels, shape (B, 2),
        constant across all frames as all crops share the same backbone
        input resolution.
    cam_int : CamIntrinsicsTensor
        Camera intrinsic matrices, shape (B, 3, 3), one matrix per frame,
        used for camera-space projection and reconstruction.
    """

    img: CroppedTensor
    bbox_center: BBoxCenterTensor
    bbox_scale: BBoxScaleTensor
    affine_trans: AffineTransTensor
    ori_img_size: OriImgSizeTensor
    crop_img_size: CroppedSizeTensor
    cam_int: CamIntrinsicsTensor