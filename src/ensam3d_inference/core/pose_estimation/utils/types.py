# src/ensam3d_inference/core/pose_estimation/utils/types.py
from typing import NamedTuple
import torch
from jaxtyping import Float
from ....preprocessor.types import CroppedSizeTensor, OriImgSizeTensor


FourierEncoding = Float[torch.Tensor, "... C"]
"""
Random Fourier positional encoding represented as a PyTorch tensor with shape (..., C) and
floating-point dtype (matching the model output), where the leading dimensions match the input
coordinate tensor and C is 2 * num_pos_feats (sin and cos components concatenated); each
position is encoded by projecting onto a frozen random Gaussian frequency matrix then applying
sin and cos with a 2*pi scale factor.
"""


NormalisedCoords = Float[torch.Tensor, "... 2"]
"""
Normalised spatial coordinates represented as a PyTorch tensor with shape (..., 2) and
floating-point dtype (matching the model output), where the leading dimensions are arbitrary
(batch, spatial grid, or both) and 2 corresponds to (x, y) coordinates normalised to the
range [0, 1] over the spatial extent of the input grid; used as input to the random Fourier
positional encoding projection.
"""


PatchPositionEncoding = Float[torch.Tensor, "1 C Hp Wp"]
"""
Dense positional encoding for the image context sequence represented as a PyTorch tensor with
shape (1, C, Hp, Wp) and floating-point dtype (matching the model output), where 1 is a
singleton batch dimension for broadcasting, C is the encoding width (embed_dim), and Hp, Wp
are the spatial patch grid dimensions; used as context_pe in each TransformerDecoderLayer
following the LaPE scheme.
"""


SpatialSize = tuple[int, int]
"""
Spatial grid dimensions represented as a tuple (height, width) in patch cells, where height
and width are the number of patch rows and columns respectively after convolutional projection
in PatchEmbed; used to generate dense positional encoding grids over the image context sequence.
"""


CLIFFConditionTensor = Float[torch.Tensor, "B 3"]
"""
CLIFF-style decoder conditioning vector represented as a PyTorch tensor with shape (B, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
and 3 corresponds to (cx/f, cy/f, b/f): the principal-point-centred bounding box centre
coordinates and box size each divided by the focal length, following the CLIFF parameterisation
for perspective-aware pose estimation.
"""


Points3DTensor = Float[torch.Tensor, "B N 3"]
"""
3D point cloud represented as a PyTorch tensor with shape (B, N, 3) and floating-point
dtype (matching the model output), where B is the number of frames in the batch, N is the
number of points (keypoints or mesh vertices depending on context), and 3 corresponds to
(x, y, z) coordinates in camera space.
"""


BBoxSizeTensor = Float[torch.Tensor, "B"]
"""
Bounding box size represented as a PyTorch tensor with shape (B,) and floating-point dtype
(matching the model output), where B is the number of frames in the batch (one scalar per
frame); the scalar is the side length in pixels of the square crop used to build the input
image for that frame.
"""


Points2DTensor = Float[torch.Tensor, "B N 2"]
"""
Projected 2D points represented as a PyTorch tensor with shape (B, N, 2) and floating-point
dtype (matching the model output), where B is the number of frames in the batch, N is the
number of points (keypoints or mesh vertices depending on context), and 2 corresponds to
(x, y) pixel coordinates in the original full-image coordinate space after perspective
projection via the camera intrinsic matrix.
"""


PointsDepthTensor = Float[torch.Tensor, "B N"]
"""
Depth (z-coordinate in camera space) for each projected point represented as a PyTorch
tensor with shape (B, N) and floating-point dtype (matching the model output), where B is
the number of frames in the batch and N is the number of points (keypoints or mesh vertices
depending on context); each value is the z-component of the corresponding 3D point after
camera translation is applied, before the division by z that produces the 2D projection.
"""


CamTranslationTensor = Float[torch.Tensor, "B 3"]
"""
Perspective camera translation vector represented as a PyTorch tensor with shape (B, 3) and
floating-point dtype (matching the model output), where B is the number of frames in the batch
(one translation per frame) and 3 corresponds to (tx, ty, tz) in camera space, derived from the
weak-perspective parameters (s, tx, ty) and the focal length; tz encodes depth as 2*f / (bbox_size * s).
"""


FocalLengthTensor = Float[torch.Tensor, "B"]
"""
Per-frame focal length represented as a PyTorch tensor with shape (B,) and floating-point
dtype (matching the model output), where B is the number of frames in the batch (one scalar
per frame); values are extracted from the (0, 0) entry of the camera intrinsic matrix and
are expressed in pixels in the original full-image coordinate space.
"""


class ProjectionOutput(NamedTuple):
    """
    Container for the output of perspective_projection.

    Attributes
    ----------
    pred_keypoints_2d : Points2DTensor
        Projected 2D points in full image space, shape (B, N, 2).
    pred_keypoints_2d_depth : PointsDepthTensor
        Depth (z-coordinate in camera space) for each projected point,
        shape (B, N).
    pred_cam_t : CamTranslationTensor or None
        Perspective camera translation (tx, ty, tz) in camera space,
        shape (B, 3); None when full_output is False.
    focal_length : FocalLengthTensor or None
        Per-frame focal length extracted from the intrinsic matrix,
        shape (B,); None when full_output is False.
    """

    pred_keypoints_2d: Points2DTensor
    pred_keypoints_2d_depth: PointsDepthTensor
    pred_cam_t: CamTranslationTensor | None
    focal_length: FocalLengthTensor | None


NormalisedCropCoords = Float[torch.Tensor, "B N 2"]
"""
Keypoint coordinates normalised to the crop space represented as a PyTorch tensor with shape
(B, N, 2) and floating-point dtype (matching the model output), where B is the number of
frames in the batch, N is the number of keypoints, and 2 corresponds to (x, y) coordinates
in the range [-0.5, 0.5] relative to the canonical crop dimensions; values outside this range
indicate keypoints that fall outside the crop boundary and are marked as invalid during token
update.
"""