# src/ensam3d_inference/core/pose_estimation/decoder/__init__.py
from typing import Callable
import torch.nn as nn
from timm.layers import LayerNormFp32
from .layers import TransformerDecoderLayer
from .types import ContextSequence
from ...feature_extraction.backbone.types import TokenSequence, BackboneOutput
from ....shared import config


class PromptableDecoder(nn.Module):
    """
    Cross-attention transformer decoder with iterative pose refinement.

    Applies a stack of TransformerDecoderLayer modules. After each layer
    calls token_to_pose_output_fn to produce a pose estimate and
    keypoint_token_update_fn to refresh the token embeddings before the
    next layer.

    Attributes
    ----------
    layers : nn.ModuleList
        Stack of TransformerDecoderLayer modules.
    norm_final : LayerNormFp32
        Final layer normalisation applied to token embeddings before each
        pose prediction call.
    """

    def __init__(self) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    token_dims=config.decoder_dim,
                    context_dims=config.decoder_context_dim,
                    num_heads=config.decoder_heads,
                    head_dims=config.decoder_head_dims,
                    mlp_dims=config.decoder_mlp_dims,
                    skip_first_pe=(i == 0),
                )
                for i in range(config.decoder_depth)
            ]
        )

        self.norm_final = LayerNormFp32(config.decoder_dim, eps=1e-6)

    def forward(
        self,
        token_embedding: TokenSequence,
        image_embedding: BackboneOutput,
        token_to_pose_output_fn: Callable,
        keypoint_token_update_fn: Callable,
        token_augment: TokenSequence,
        image_augment: ContextSequence,
    ) -> tuple[TokenSequence, list]:
        """
        Run the decoder stack with iterative pose refinement.

        Parameters
        ----------
        token_embedding : TokenSequence
            Pose token sequence, shape (B, N, C).
        image_embedding : BackboneOutput
            Image feature map from the feature extractor, shape (B, C, Hp, Wp).
        token_to_pose_output_fn : Callable
            Called after each layer with normalised tokens to produce a pose
            estimate.
        keypoint_token_update_fn : Callable
            Called after each intermediate pose estimate to refresh token
            embeddings.
        token_augment : TokenSequence
            Positional encoding for tokens, shape (B, N, C).
        image_augment : ContextSequence
            Positional encoding for image context, shape (B, Nc, Dc).

        Returns
        -------
        normed : TokenSequence
            Final normalised token sequence, shape (B, N, C).
        all_pose_outputs : list
            Pose estimates collected after every layer including the final one.
        """
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
        image_augment = image_augment.flatten(2).permute(0, 2, 1)

        all_pose_outputs = []

        for layer_idx, layer in enumerate(self.layers):
            token_embedding, image_embedding = layer(
                token_embedding,
                image_embedding,
                token_augment,
                image_augment,
            )

            normed = self.norm_final(token_embedding)
            curr_pose_output = token_to_pose_output_fn(
                normed,
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr_pose_output)

            if layer_idx < len(self.layers) - 1:
                token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                    token_embedding, token_augment, curr_pose_output, layer_idx
                )

        return normed, all_pose_outputs