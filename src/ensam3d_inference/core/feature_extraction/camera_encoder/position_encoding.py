# src/ensam3d_inference/core/feature_extraction/camera_encoder/position_encoding.py
import numpy as np
import torch
import torch.nn as nn
from .types import RayDirections, FourierEncoding


class FourierPositionEncoding(nn.Module):
    """
    Non-learnable Fourier positional encoding for ray direction vectors.

    Encodes each input position by projecting it onto a set of linearly sampled
    frequency bands, applying sin and cos, then concatenating the original
    coordinates. No parameters are learned.

    Parameters
    ----------
    n : int
        Number of spatial dimensions in the input position vectors.
    num_bands : int
        Number of frequency bands per spatial dimension.
    max_resolution : int
        Maximum resolution used to compute the upper frequency bound per
        dimension.

    Attributes
    ----------
    num_bands : int
        Number of frequency bands per spatial dimension.
    max_resolution : list of int
        Per-dimension maximum resolution, replicated n times from the scalar
        max_resolution argument.
    """

    def __init__(self, n: int, num_bands: int, max_resolution: int) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n

    @property
    def channels(self) -> int:
        """
        Output encoding width C for a single position vector.

        Returns
        -------
        int
            Total encoding dimension: num_bands * num_dims * 2 + num_dims.
        """
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims * 2 + num_dims
        return encoding_size

    def forward(self, pos: RayDirections) -> FourierEncoding:
        """
        Encode ray direction vectors with Fourier features.

        Parameters
        ----------
        pos : RayDirections
            Ray direction vectors, shape (B, N, 3).

        Returns
        -------
        FourierEncoding
            Fourier positional encoding, shape (B, N, C), where C equals
            the value of the channels property.
        """
        b, n = pos.shape[:2]
        device = pos.device

        freq_bands = torch.stack(
            [
                torch.linspace(start=1.0, end=res / 2, steps=self.num_bands, device=device)
                for res in self.max_resolution
            ],
            dim=0,
        )

        per_pos_features = torch.stack(
            [pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0
        )
        per_pos_features = per_pos_features.reshape(b, n, -1)

        per_pos_features = torch.cat(
            [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)],
            dim=-1,
        )

        return torch.cat([pos, per_pos_features], dim=-1)