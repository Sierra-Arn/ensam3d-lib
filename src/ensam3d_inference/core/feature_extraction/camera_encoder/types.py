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

# src/ensam3d_inference/core/feature_extraction/camera_encoder/types.py
import torch
from jaxtyping import Float


RayDirections = Float[torch.Tensor, "B N 3"]
"""
Ray direction vectors represented as a PyTorch tensor with shape (B, N, 3) and floating-point
dtype (matching the model output), where B is the number of frames in the batch, N is the number
of spatial positions (Hp * Wp after patch grid flattening), and 3 corresponds to (x, y, 1) in
normalised camera coordinates with a homogeneous third component appended before encoding.
"""

FourierEncoding = Float[torch.Tensor, "B N C"]
"""
Fourier positional encoding represented as a PyTorch tensor with shape (B, N, 99) and
floating-point dtype (matching the model output), where B is the number of frames in the batch,
N is the number of spatial positions, and C is the encoding width computed as
num_bands * num_dims * 2 (sin-cos pairs) + num_dims (concatenated input coordinates);
for the default configuration (n=3, num_bands=16) this equals 99.
"""

RayCondition = Float[torch.Tensor, "B 2 H W"]
"""
Camera ray condition grid represented as a PyTorch tensor with shape (B, 2, H, W) and
floating-point dtype (matching the model output), where B is the number of frames in the batch,
2 corresponds to the (x, y) normalised ray directions in camera coordinate space computed from
the intrinsic matrix and affine transform, and H, W are the spatial dimensions of the input
image before patch downsampling.
"""

ImgHW = tuple[int, int]
"""
Spatial dimensions of the input image represented as a tuple (H, W), where H is the image
height in pixels and W is the image width in pixels; passed explicitly to CameraEncoder to
construct the pixel coordinate grid for ray direction computation without requiring access
to the original image tensor.
"""