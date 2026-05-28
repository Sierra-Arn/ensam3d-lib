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

# src/ensam3d_inference/core/feature_extraction/camera_encoder/__init__.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNorm2d
from .position_encoding import FourierPositionEncoding
from .types import ImgHW, RayCondition, FourierEncoding
from ..backbone.types import BackboneOutput
from ....preprocessor.types import CamIntrinsicsTensor, AffineTransTensor
from ....shared import config


class CameraEncoder(nn.Module):
    """
    Fuses image feature maps with Fourier-encoded camera ray directions.

    Computes per-pixel ray directions from the affine transform and camera
    intrinsics, encodes them with FourierPositionEncoding, interpolates to
    the patch grid resolution, then fuses the result with the image embeddings
    via a 1x1 convolution and layer normalisation.

    Attributes
    ----------
    camera : FourierPositionEncoding
        Non-learnable Fourier encoder for ray directions.
    conv : nn.Conv2d
        1x1 convolution fusing concatenated image and ray embeddings back to
        EMBED_DIM.
    norm : LayerNorm2d
        Spatial layer normalisation applied after fusion.
    """

    def __init__(self) -> None:
        super().__init__()
        self.camera = FourierPositionEncoding(
            n=3, 
            num_bands=16, 
            max_resolution=64
        )
        self.conv = nn.Conv2d(
            config.backbone_embed_dim + 99, 
            config.backbone_embed_dim, 
            kernel_size=1, 
            bias=False
        )
        self.norm = LayerNorm2d(config.backbone_embed_dim)

    def _compute_ray_condition(
        self,
        affine_trans: AffineTransTensor,
        cam_int: CamIntrinsicsTensor,
        img_hw: ImgHW,
    ) -> RayCondition:
        """
        Compute normalised ray directions for each pixel in the image grid.

        Parameters
        ----------
        affine_trans : AffineTransTensor
            Affine transforms from full image to crop, shape (B, 2, 3).
        cam_int : CamIntrinsicsTensor
            Camera intrinsic matrix, shape (B, 3, 3).
        img_hw : ImgHW
            Spatial dimensions (H, W) of the input image in pixels.

        Returns
        -------
        RayCondition
            Normalised ray direction grid, shape (B, 2, H, W).
        """
        h, w = img_hw
        n = affine_trans.shape[0]
        dt = affine_trans.dtype
        device = affine_trans.device

        grid = (
            torch.stack(
                torch.meshgrid(
                    torch.arange(h, device=device),
                    torch.arange(w, device=device),
                    indexing="xy",
                ),
                dim=2,
            )[None]
            .repeat(n, 1, 1, 1)
            .to(dtype=dt)
        )

        grid = grid / affine_trans[:, None, None, [0, 1], [0, 1]]
        grid = grid - (
            affine_trans[:, None, None, [0, 1], [2, 2]]
            / affine_trans[:, None, None, [0, 1], [0, 1]]
        )
        grid = grid - cam_int[:, None, None, [0, 1], [2, 2]]
        grid = grid / cam_int[:, None, None, [0, 1], [0, 1]]

        return grid.permute(0, 3, 1, 2)

    def forward(
        self,
        img_embeddings: BackboneOutput,
        affine_trans: AffineTransTensor,
        cam_int: CamIntrinsicsTensor,
        img_hw: ImgHW,
    ) -> BackboneOutput:
        """
        Fuse image embeddings with Fourier-encoded ray directions.

        Parameters
        ----------
        img_embeddings : BackboneOutput
            Image feature map from the backbone, shape (B, C, Hp, Wp).
        affine_trans : AffineTransTensor
            Affine transforms from full image to crop, shape (B, 2, 3).
        cam_int : CamIntrinsicsTensor
            Camera intrinsic matrix, shape (B, 3, 3).
        img_hw : ImgHW
            Spatial dimensions (H, W) of the input image in pixels.

        Returns
        -------
        BackboneOutput
            Fused feature map, shape (B, C, Hp, Wp); same spatial layout as
            img_embeddings with ray conditioning absorbed into the channels.
        """
        B, D, _h, _w = img_embeddings.shape

        with torch.no_grad():
            rays: RayCondition = self._compute_ray_condition(affine_trans, cam_int, img_hw)

            scale = 1 / config.backbone_patch_size
            rays = F.interpolate(
                rays,
                scale_factor=(scale, scale),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            rays = rays.permute(0, 2, 3, 1).contiguous()
            rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
            rays_embeddings: FourierEncoding = self.camera(pos=rays.reshape(B, -1, 3))
            rays_embeddings = (
                rays_embeddings.view(B, _h, _w, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            rays_embeddings = rays_embeddings.to(dtype=img_embeddings.dtype)

        z = torch.cat([img_embeddings, rays_embeddings], dim=1)
        return self.norm(self.conv(z))
