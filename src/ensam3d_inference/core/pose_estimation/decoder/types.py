# src/ensam3d_inference/core/pose_estimation/decoder/types.py
import torch
from jaxtyping import Bool, Float


ContextSequence = Float[torch.Tensor, "B Nc Dc"]
"""
Image context sequence represented as a PyTorch tensor with shape (B, Nc, Dc) and floating-point
dtype (matching the model output), where B is the number of frames in the batch, Nc is the number
of spatial positions (flattened patch grid Hp * Wp), and Dc is the image context channel width
(context_dims) produced by the feature extractor and passed to each decoder layer as keys and
values in the cross-attention block.
"""

TokenMask = Bool[torch.Tensor, "B N"]
"""
Boolean validity mask for pose tokens represented as a PyTorch tensor with shape (B, N) and
boolean dtype, where B is the number of frames in the batch and N is the number of pose tokens;
True indicates a valid token that participates in self-attention, False marks a padding token
that is excluded via the attention mask.
"""

AttentionHeads = Float[torch.Tensor, "B H N D"]
"""
Multi-head attention tensor represented as a PyTorch tensor with shape (B, H, N, D) and
floating-point dtype (matching the model output), where B is the number of frames in the batch,
H is the number of attention heads, N is the sequence length, and D is the per-head dimension
(embed_dims // num_heads).
"""