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

# src/ensam3d_inference/core/pose_estimation/heads/perspective_head/__init__.py
import torch.nn as nn
from .types import PerspectiveHeadOutput
from ..layers import MLP
from ..types import PoseToken
from .....shared.configuration import config


class PerspectiveHead(nn.Module):
    """
    Predict weak-perspective camera parameters from a decoder pose token.

    Attributes
    ----------
    proj : MLP
        Two-layer MLP projecting the pose token to camera parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        self.proj = MLP(
            in_dims=config.decoder_dim,
            hidden_dims=config.decoder_dim,
            out_dims=config.camera_ncam,
        )

    def forward(
        self,
        x: PoseToken,
        init_estimate: PerspectiveHeadOutput,
    ) -> PerspectiveHeadOutput:
        """
        Predict weak-perspective camera parameters from a pose token.

        Parameters
        ----------
        x : PoseToken
            Pose token from the decoder, shape (B, C).
        init_estimate : PerspectiveHeadOutput
            Additive residual initialisation for the camera parameters,
            shape (B, 3).

        Returns
        -------
        PerspectiveHeadOutput
            Predicted camera parameters (s, tx, ty), shape (B, 3).
        """
        return self.proj(x) + init_estimate