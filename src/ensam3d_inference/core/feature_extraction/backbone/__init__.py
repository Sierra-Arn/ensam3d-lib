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

# src/ensam3d_inference/core/feature_extraction/backbone/__init__.py
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from .layers import Block, PatchEmbed
from .types import BackboneInput, BackboneOutput
from ....shared.configuration import config


class Backbone(nn.Module):
    """
    Vision Transformer stack for inference.

    Omits training-only variants (hybrid embed, activation checkpointing,
    classifier). Attention and MLP dropout are 0; stochastic depth follows
    a linear schedule from 0 to 0.55 across the block depth.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Stem convolution and token flattening.
    pos_embed : nn.Parameter
        Positional embedding, shape (1, num_patches + 1, EMBED_DIM); index 0
        is an unused cls-style slot; indices 1: are summed into every token row.
    blocks : nn.ModuleList
        Transformer blocks with per-index stochastic depth.
    last_norm : nn.Module
        Layer normalisation applied before reshaping tokens to a feature grid.
    """

    def __init__(self) -> None:
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-5)

        self.patch_embed = PatchEmbed(
            img_size=config.backbone_input_size,
            patch_size=config.backbone_patch_size,
            embed_dim=config.backbone_embed_dim,
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, config.backbone_embed_dim)
        )

        dpr = [t.item() for t in torch.linspace(0, 0.55, config.backbone_depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=config.backbone_embed_dim,
                    num_heads=config.backbone_heads,
                    mlp_ratio=config.backbone_mlp_ratio,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(config.backbone_depth)
            ]
        )

        self.last_norm = norm_layer(config.backbone_embed_dim)
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: BackboneInput) -> BackboneOutput:
        """
        Map a normalised image batch to a channel-major patch feature grid.

        Parameters
        ----------
        x : BackboneInput
            Normalised RGB crops, shape (B, 3, H, W).

        Returns
        -------
        BackboneOutput
            Patch feature grid, shape (B, C, Hp, Wp), where C is EMBED_DIM and
            Hp, Wp are the token grid dimensions produced by PatchEmbed.
        """
        b = x.shape[0]
        tokens, (hp, wp) = self.patch_embed(x)

        orig_h, orig_w = self.patch_embed.patch_shape
        if hp != orig_h or wp != orig_w:
            pe = self.pos_embed[:, 1:].reshape(1, orig_h, orig_w, -1)
            pe = F.interpolate(
                pe.permute(0, 3, 1, 2),
                size=(hp, wp),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1)
            tokens = (
                tokens
                + pe.reshape(1, -1, config.backbone_embed_dim)
                + self.pos_embed[:, :1]
            )
        else:
            tokens = tokens + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.last_norm(tokens)
        return tokens.permute(0, 2, 1).reshape(b, -1, hp, wp).contiguous()