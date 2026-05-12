# src/ensam3d_inference/core/pose_estimation/heads/perspective_head/types.py
import torch
from jaxtyping import Float


PerspectiveHeadOutput = Float[torch.Tensor, "B 3"]
"""
Weak-perspective camera parameters represented as a PyTorch tensor with shape (B, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
(one camera vector per frame) and 3 corresponds to (s, tx, ty): s is a scale factor relating
bounding box size to focal depth, and tx, ty are image-plane translations in normalised units.
"""