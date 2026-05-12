# src/ensam3d_inference/preprocessor/utils/types.py
from typing import NamedTuple
import numpy as np
import torch
from jaxtyping import Float
from ..detector.types import RGBFrame


Vec2 = Float[np.ndarray, "2"]
"""
2D vector represented as a NumPy float32 array with shape (2,), where the two elements 
correspond to the spatial coordinates or dimensions of a quantity in 2D space. 
"""


AffineMatrix = Float[np.ndarray, "2 3"]
"""
Affine transformation matrix represented as a NumPy float32 array with shape (2, 3), 
where (2, 3) corresponds to a 2D affine transformation that maps points from the 
original image coordinate space to the warped crop space.
"""


class WarpCropInput(NamedTuple):
    """
    Container for the source image and bounding box geometry passed to the warper.

    Attributes
    ----------
    img : RGBFrame
        Source RGB image in uint8 format, shape (H, W, 3).
    bbox_center : Vec2
        Bounding box center (x, y) in full-image pixel coordinates.
    bbox_scale : Vec2
        Bounding box dimensions (w, h) in pixels after padding, 
        representing the initial scale before aspect-ratio alignment.
    """

    img: RGBFrame
    bbox_center: Vec2
    bbox_scale: Vec2


class WarpCropOutput(NamedTuple):
    """
    Container for a warped image crop and its associated metadata.

    Attributes
    ----------
    img : RGBFrame
        Warped image in RGB uint8 format, shape matches warp size.
    bbox_center : Vec2
        Original detection center (x, y), unaffected by
        aspect-ratio adjustments.
    bbox_scale : Vec2
        Final box dimensions (w, h) after all aspect-ratio
        transformations.
    affine_trans : AffineMatrix
        Affine transformation matrix of shape (2, 3) used by
        cv2.warpAffine for this frame.
    """

    img: RGBFrame
    bbox_center: Vec2
    bbox_scale: Vec2
    affine_trans: AffineMatrix


CroppedTensor = Float[torch.Tensor, "B 3 H W"]
"""
Batch of person-centered RGB crops in ImageNet normalization space represented as a PyTorch 
float32 tensor with shape (B, 3, H, W), where B is the number of frames in the batch (one crop per frame), not
the number of people in a single image; 3 is RGB in channel-first order, and H and W are 
the fixed spatial dimensions of the crop.
"""


CamIntrinsicsTensor = Float[torch.Tensor, "B 3 3"]
"""
Camera intrinsic matrix represented as a PyTorch float32 tensor with shape (B, 3, 3), where B
is the number of frames in the batch (one matrix per frame) and (3, 3) is the standard pinhole
camera intrinsic matrix with focal lengths on the diagonal and principal point in the last
column; kept in float32 rather than bfloat16 because bfloat16 has insufficient mantissa
precision for focal lengths derived from typical image resolutions (e.g. sqrt(1920^2 + 1080^2)).
"""
