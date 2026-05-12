# src/ensam3d_inference/core/pose_estimation/decoder/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import LayerNormFp32, Mlp
from ...feature_extraction.backbone.types import TokenSequence
from .types import AttentionHeads, ContextSequence


class Attention(nn.Module):
    """
    Multi-head attention supporting both self-attention and cross-attention.

    Query, key, and value projections are separate linear layers, allowing
    different input dimensions for each. Attention is computed with
    scaled_dot_product_attention.

    Parameters
    ----------
    embed_dims : int
        Total attention width; must equal num_heads * head_dims.
    num_heads : int
        Number of parallel attention heads.
    query_dims : int or None, optional
        Query input width; if None, defaults to embed_dims. Default is None.
    key_dims : int or None, optional
        Key input width; if None, defaults to embed_dims. Default is None.
    value_dims : int or None, optional
        Value input width; if None, defaults to embed_dims. Default is None.

    Attributes
    ----------
    head_dims : int
        Per-head dimension computed as embed_dims // num_heads.
    q_proj : nn.Linear
        Query projection from query_dims to embed_dims.
    k_proj : nn.Linear
        Key projection from key_dims to embed_dims.
    v_proj : nn.Linear
        Value projection from value_dims to embed_dims.
    proj : nn.Linear
        Output projection from embed_dims back to query_dims.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        query_dims: int | None = None,
        key_dims: int | None = None,
        value_dims: int | None = None,
    ) -> None:
        super().__init__()
        self.query_dims = query_dims or embed_dims
        self.key_dims = key_dims or embed_dims
        self.value_dims = value_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        self.q_proj = nn.Linear(self.query_dims, embed_dims, bias=True)
        self.k_proj = nn.Linear(self.key_dims, embed_dims, bias=True)
        self.v_proj = nn.Linear(self.value_dims, embed_dims, bias=True)
        self.proj = nn.Linear(embed_dims, self.query_dims, bias=True)

    def _separate_heads(self, x: torch.Tensor) -> AttentionHeads:
        b, n, _ = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dims)
        return x.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute multi-head attention.

        Parameters
        ----------
        q : torch.Tensor
            Query tensor, shape (B, N, query_dims).
        k : torch.Tensor
            Key tensor, shape (B, Nk, key_dims).
        v : torch.Tensor
            Value tensor, shape (B, Nk, value_dims).

        Returns
        -------
        torch.Tensor
            Output tensor, shape (B, N, query_dims).
        """
        B, N, _ = q.shape
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)
        return self.proj(x)


class TransformerDecoderLayer(nn.Module):
    """
    One cross-attention decoder layer adapted from the Segment Anything Model.

    Applies token self-attention, token-to-image cross-attention, and an FFN
    in sequence with layer-adaptive positional encoding (LaPE).

    Parameters
    ----------
    token_dims : int
        Token channel width C in layout B N C.
    context_dims : int
        Image context channel width Dc in layout B N Dc.
    num_heads : int, optional
        Number of attention heads. Default is 8.
    head_dims : int, optional
        Per-head dimension; total attention width is num_heads * head_dims.
        Default is 64.
    mlp_dims : int, optional
        FFN hidden width. Default is 1024.
    skip_first_pe : bool, optional
        Whether to skip positional encoding on the first self-attention block.
        Default is False.

    Attributes
    ----------
    ln_pe_1 : LayerNormFp32
        Norm applied to token positional encoding following LaPE.
    ln_pe_2 : LayerNormFp32
        Norm applied to context positional encoding following LaPE.
    ln1 : LayerNormFp32
        Norm applied before token self-attention.
    self_attn : Attention
        Token self-attention submodule.
    ln2_1 : LayerNormFp32
        Norm applied to tokens before cross-attention.
    ln2_2 : LayerNormFp32
        Norm applied to context before cross-attention.
    cross_attn : Attention
        Token-to-image cross-attention submodule.
    ln3 : LayerNormFp32
        Norm applied before the FFN.
    mlp : timm.layers.Mlp
        Two-layer feed-forward network with hidden width mlp_dims.
    """

    def __init__(
        self,
        token_dims: int,
        context_dims: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        skip_first_pe: bool = False,
    ) -> None:
        super().__init__()
        self.skip_first_pe = skip_first_pe

        self.ln_pe_1 = LayerNormFp32(token_dims, eps=1e-6)
        self.ln_pe_2 = LayerNormFp32(context_dims, eps=1e-6)
        self.ln1 = LayerNormFp32(token_dims, eps=1e-6)

        self.self_attn = Attention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=token_dims,
            value_dims=token_dims,
        )

        self.ln2_1 = LayerNormFp32(token_dims, eps=1e-6)
        self.ln2_2 = LayerNormFp32(context_dims, eps=1e-6)

        self.cross_attn = Attention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=context_dims,
            value_dims=context_dims,
        )

        self.ln3 = LayerNormFp32(token_dims, eps=1e-6)
        self.mlp = Mlp(
            in_features=token_dims,
            hidden_features=mlp_dims,
            act_layer=nn.GELU,
            drop=0.0,
        )

    def forward(
        self,
        x: TokenSequence,
        context: ContextSequence,
        x_pe: TokenSequence | None = None,
        context_pe: ContextSequence | None = None,
    ) -> tuple[TokenSequence, ContextSequence]:
        """
        Apply one decoder layer to token and context sequences.

        Parameters
        ----------
        x : TokenSequence
            Pose token sequence, shape (B, N, C).
        context : ContextSequence
            Image context sequence, shape (B, Nc, Dc).
        x_pe : TokenSequence or None, optional
            Positional encoding for tokens, shape (B, N, C).
        context_pe : ContextSequence or None, optional
            Positional encoding for context, shape (B, Nc, Dc).

        Returns
        -------
        x : TokenSequence
            Updated token sequence, shape (B, N, C).
        context : ContextSequence
            Context sequence passed through unchanged, shape (B, Nc, Dc).
        """
        if context_pe is not None:
            x_pe = self.ln_pe_1(x_pe)
            context_pe = self.ln_pe_2(context_pe)

        if not self.skip_first_pe and x_pe is not None:
            q = k = self.ln1(x) + x_pe
            v = self.ln1(x)
        else:
            q = k = v = self.ln1(x)

        x = x + self.self_attn(q=q, k=k, v=v)

        if context_pe is not None:
            q = self.ln2_1(x) + x_pe
            k = self.ln2_2(context) + context_pe
            v = self.ln2_2(context)
        else:
            q = self.ln2_1(x)
            k = v = self.ln2_2(context)
        x = x + self.cross_attn(q=q, k=k, v=v)

        x = x + self.mlp(self.ln3(x))

        return x, context