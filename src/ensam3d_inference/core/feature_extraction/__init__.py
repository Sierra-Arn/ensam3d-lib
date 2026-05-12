# src/ensam3d_inference/core/feature_extraction/__init__.py
import torch.nn as nn
from .backbone import Backbone
from .backbone.types import BackboneOutput
from .camera_encoder import CameraEncoder
from .camera_encoder.types import ImgHW
from ...preprocessor.utils.types import CroppedTensor
from ...preprocessor.types import CamIntrinsicsTensor, AffineTransTensor


class FeatureExtractor(nn.Module):
    """
    ViT backbone followed by Fourier ray conditioning.

    Normalises the input crop with ImageNet mean and standard deviation,
    runs the backbone once to produce patch embeddings, then computes
    per-pixel ray directions from the affine transform and camera intrinsics
    and fuses them via CameraEncoder to produce ray-conditioned feature maps
    ready for the pose estimator.

    Attributes
    ----------
    backbone : Backbone
        ViT-HMR 512x384 patch embedding and transformer stack.
    camera_encoder : CameraEncoder
        Fourier ray encoder and 1x1 fusion convolution.
    """

    def __init__(self) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.camera_encoder = CameraEncoder()

    def forward(
        self,
        x: CroppedTensor,
        affine_trans: AffineTransTensor,
        cam_int: CamIntrinsicsTensor,
    ) -> BackboneOutput:
        """
        Normalise, extract, and ray-condition image features.

        Parameters
        ----------
        x : CroppedRGBTensor
            Warped RGB crops in [0, 1], shape (B, 3, H, W).
        affine_trans : AffineTransTensor
            Affine transforms from full image to crop, shape (B, 2, 3).
        cam_int : CamIntrinsicsTensor
            Camera intrinsic matrix, shape (B, 3, 3).

        Returns
        -------
        BackboneOutput
            Ray-conditioned patch feature map, shape (B, C, Hp, Wp).
        """
        x = x[:, :, :, 64:-64]  

        img_hw: ImgHW = (x.shape[2], x.shape[3])
        return self.camera_encoder(
            self.backbone(x),
            affine_trans=affine_trans,
            cam_int=cam_int,
            img_hw=img_hw,
        )