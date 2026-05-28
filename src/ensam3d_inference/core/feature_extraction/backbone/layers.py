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

# src/ensam3d_inference/core/feature_extraction/backbone/layers.py
import torch.nn.functional as F
import torch.nn as nn
from timm.layers import DropPath, Mlp
from .types import TokenSequence, PatchGridSize, BackboneInput


class Attention(nn.Module):
    """
    Multi-head self-attention with a single linear projection for Q, K, and V.

    Attention weights are computed with a scaled dot-product, softmax, then
    dropout on the output projection.

    Parameters
    ----------
    dim : int
        Embedding width C in the token layout B N C.
    num_heads : int
        Number of attention heads.

    Attributes
    ----------
    num_heads : int
        Number of parallel attention heads.
    dim : int
        Embedding width C stored for the B N C tensor contract.
    scale : float
        Query scaling factor; head_dim ** -0.5.
    qkv : nn.Linear
        Single linear layer projecting C to 3 * num_heads * head_dim for
        packed Q, K, V computation with bias.
    proj : nn.Linear
        Output projection mapping concatenated heads back to C.
    proj_drop : nn.Dropout
        Dropout applied after the output projection.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads
        all_head_dim = head_dim * num_heads

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=True)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: TokenSequence) -> TokenSequence:
        """
        Run self-attention on a token sequence.

        Parameters
        ----------
        x : TokenSequence
            Input token sequence, shape (B, N, C), where C equals the embedding
            width passed to the constructor.

        Returns
        -------
        TokenSequence
            Output token sequence, shape (B, N, C); embedding width C is
            unchanged after projection.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    """
    One pre-normalisation transformer block: attention and MLP, each with a residual.

    Submodules apply layer normalisation before the submodule, then add the submodule
    output scaled by stochastic depth (DropPath) when the drop rate is positive.

    Parameters
    ----------
    dim : int
        Token embedding width C in layout B N C.
    num_heads : int
        Attention heads for the internal Attention module.
    mlp_ratio : float
        MLP hidden width multiplier: hidden = floor(dim * mlp_ratio).
    drop_path : float, optional
        Stochastic depth rate for both residual branches; 0.0 uses Identity.
        Default is 0.0.
    norm_layer : type of nn.Module, optional
        Constructor accepting (dim,) returning a norm module.
        Default is nn.LayerNorm.

    Attributes
    ----------
    norm1 : nn.Module
        Normalisation applied before the attention submodule.
    norm2 : nn.Module
        Normalisation applied before the MLP submodule.
    attn : Attention
        Multi-head self-attention submodule.
    drop_path : DropPath or nn.Identity
        Stochastic depth scaling applied to both residual branches;
        nn.Identity when drop_path rate is 0.0.
    mlp : timm.layers.Mlp
        Two-layer feed-forward network with hidden width floor(dim * mlp_ratio).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path: float,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=0.0,
            norm_layer=None,
        )

    def forward(self, x: TokenSequence) -> TokenSequence:
        """
        Apply attention and MLP sub-blocks with residuals.

        Parameters
        ----------
        x : TokenSequence
            Input token sequence, shape (B, N, C), where C equals the embedding
            width passed to the constructor.

        Returns
        -------
        TokenSequence
            Output token sequence, shape (B, N, C); embedding width C is
            unchanged after both residual branches.
        """
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """
    Convolutional patch embedding from an image to a sequence of tokens.

    Maps images of shape (B, 3, H, W) to tokens of shape (B, N, C), where
    N = Hp * Wp token sites and C = embed_dim. Stride equals patch_size and
    padding is fixed at 4 following the SAM 3D Body ViT-HMR configuration.

    Parameters
    ----------
    img_size : tuple of int
        Input image (height, width) in pixels.
    patch_size : int
        Spatial patch size used as both kernel and stride.
    embed_dim : int
        Output token width C in B N C.

    Attributes
    ----------
    patch_shape : tuple of int
        Token grid (Hp, Wp) after convolutional projection.
    img_size : tuple of int
        Nominal input (height, width) in pixels.
    patch_size : int
        Convolution kernel and stride.
    num_patches : int
        Total number of patch tokens from geometry.
    proj : nn.Conv2d
        Patch projection convolution mapping 3 input channels to embed_dim.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_shape = (
            img_size[0] // patch_size,
            img_size[1] // patch_size,
        )
        self.num_patches = self.patch_shape[0] * self.patch_shape[1]
        self.proj = nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=4,
        )

    def forward(self, x: BackboneInput) -> tuple[TokenSequence, PatchGridSize]:
        """
        Project image patches to a flattened token sequence.

        Parameters
        ----------
        x : BackboneInput
            Normalised image batch, shape (B, 3, H, W).

        Returns
        -------
        tokens : TokenSequence
            Patch token sequence, shape (B, N, C), where N = Hp * Wp and
            C = embed_dim.
        spatial : PatchGridSize
            Token grid dimensions (Hp, Wp) after convolutional projection.
        """
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)