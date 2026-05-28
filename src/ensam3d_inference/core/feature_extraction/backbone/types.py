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

# src/ensam3d_inference/core/feature_extraction/backbone/types.py
import torch
from jaxtyping import Float


TokenSequence = Float[torch.Tensor, "B N C"]
"""
Token sequence represented as a PyTorch tensor with shape (B, N, C) and floating-point dtype
(matching the model output), where B is the number of frames in the batch, N is the sequence
length (number of patch tokens produced by PatchEmbed), and C is the embedding width (embed_dim)
shared across all transformer blocks.
"""


PatchGridSize = tuple[int, int]
"""
Spatial token grid dimensions represented as a tuple (Hp, Wp), where Hp and Wp are the height
and width of the patch grid produced by the convolutional projection in PatchEmbed; each value
equals floor(img_size / patch_size) * ratio for the corresponding spatial axis.
"""


BackboneInput = Float[torch.Tensor, "B 3 H W"]
"""
Normalised RGB image batch represented as a PyTorch tensor with shape (B, 3, H, W) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
(one crop per frame), 3 is RGB in channel-first order, H and W are the spatial dimensions of the
input image after affine warp and optional width crop; channel values follow ImageNet normalisation
(mean subtraction and std division per channel in RGB order) applied inside _data_preprocess before
the tensor reaches the backbone.
"""


BackboneOutput = Float[torch.Tensor, "B C Hp Wp"]
"""
Channel-major feature grid represented as a PyTorch tensor with shape (B, C, Hp, Wp) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
(one feature map per frame), C is the embedding dimension (embed_dim), and Hp, Wp are the spatial
patch grid dimensions produced by PatchEmbed after convolutional projection and reshaping from the
token sequence layout B N C.
"""