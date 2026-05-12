# src/ensam3d_inference/core/pose_estimation/heads/types.py
import torch
from jaxtyping import Float


PoseToken = Float[torch.Tensor, "B C"]
"""
Pose token represented as a PyTorch tensor with shape (B, C) and floating-point dtype
(matching the model output), where B is the number of frames in the batch and C is the
decoder token width (dec_dim); extracted as the first token from the decoder output
sequence before being passed to the MHR and camera heads.
"""