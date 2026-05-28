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

# src/ensam3d_inference/core/pose_estimation/utils/position_encoding.py
import numpy as np
import torch
import torch.nn as nn
from .types import FourierEncoding, NormalisedCoords, PatchPositionEncoding, SpatialSize


class PositionEmbeddingRandom(nn.Module):
    """
    Non-learnable positional encoding using random spatial frequencies.

    Encodes a spatial grid by projecting normalised (x, y) coordinates onto
    640 random Gaussian frequencies, then applying sin and cos. The frequency
    matrix is frozen at initialisation and never updated.

    Attributes
    ----------
    positional_encoding_gaussian_matrix : torch.Tensor
        Frozen random frequency matrix, shape (2, 640).
    """

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((2, 640)),
        )

    def _pe_encoding(self, coords: NormalisedCoords) -> FourierEncoding:
        """
        Encode normalised coordinates in [0, 1] with random Fourier features.

        Parameters
        ----------
        coords : NormalisedCoords
            Normalised (x, y) coordinates, shape (..., 2).

        Returns
        -------
        FourierEncoding
            Positional encoding, shape (..., 1280).
        """
        target_dtype = coords.dtype
        coords = coords.float()
        wmat = self.positional_encoding_gaussian_matrix.float()
        coords = 2 * coords - 1
        coords = coords @ wmat
        coords = 2 * np.pi * coords
        out = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return out.to(target_dtype)

    def forward(self, size: SpatialSize) -> PatchPositionEncoding:
        """
        Generate a dense positional encoding grid for a given spatial size.

        Parameters
        ----------
        size : SpatialSize
            Spatial grid (height, width) in patch cells.

        Returns
        -------
        PatchPositionEncoding
            Positional encoding grid, shape (1, 1280, Hp, Wp).
        """
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1).unsqueeze(0)