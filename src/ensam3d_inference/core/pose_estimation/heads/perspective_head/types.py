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

# src/ensam3d_inference/core/pose_estimation/heads/perspective_head/types.py
import torch
from jaxtyping import Float


PerspectiveHeadOutput = Float[torch.Tensor, "B 3"]
"""
Weak-perspective camera parameters represented as a PyTorch tensor with shape (B, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
(one camera vector per frame) and 3 corresponds to (s, tx, ty): s is a scale factor relating
bounding box size to focal depth, and tx, ty are image-plane translations in normalised units.
"""