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

# src/ensam3d_inference/core/pose_estimation/heads/types.py
import torch
from jaxtyping import Float


PoseToken = Float[torch.Tensor, "B C"]
"""
Pose token represented as a PyTorch tensor with shape (B, C) and floating-point dtype
(matching the model output), where B is the number of frames in the batch and C is the
decoder token width (dec_dim); extracted as the first token from the decoder output
sequence before being passed to the MHR and camera heads.
"""