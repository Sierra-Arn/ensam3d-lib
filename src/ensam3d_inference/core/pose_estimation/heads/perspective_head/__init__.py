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