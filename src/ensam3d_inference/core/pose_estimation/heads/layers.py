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

# src/ensam3d_inference/core/pose_estimation/heads/layers.py
import torch.nn as nn
from .types import PoseToken


class MLP(nn.Module):
    """
    Multi-layer perceptron without residual connection.

    Parameters
    ----------
    in_dims : int
        Input feature width.
    hidden_dims : int
        Hidden layer width.
    out_dims : int
        Output feature width.
    num_layers : int, optional
        Number of fully-connected layers. Default is 2.
    act_layer : type of nn.Module, optional
        Activation class applied between hidden layers. Default is nn.ReLU.
    drop : float, optional
        Dropout probability after each hidden layer and after the final
        linear layer. Default is 0.0.

    Attributes
    ----------
    layers : nn.Sequential
        Stacked linear, activation, and dropout layers.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        num_layers: int = 2,
        act_layer: type[nn.Module] = nn.ReLU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()

        layers = []
        in_channels = in_dims
        for _ in range(num_layers - 1):
            layers.append(
                nn.Sequential(
                    nn.Linear(in_channels, hidden_dims),
                    act_layer(),
                    nn.Dropout(drop),
                )
            )
            in_channels = hidden_dims
        layers.append(nn.Linear(in_channels, out_dims))
        layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: PoseToken) -> PoseToken:
        """
        Apply the MLP to a pose token.

        Parameters
        ----------
        x : PoseToken
            Input tensor, shape (B, C).

        Returns
        -------
        PoseToken
            Output tensor, shape (B, out_dims).
        """
        return self.layers(x)