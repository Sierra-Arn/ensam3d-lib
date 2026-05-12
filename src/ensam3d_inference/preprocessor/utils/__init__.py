# src/ensam3d_inference/preprocessor/utils/__init__.py
import numpy as np
import torch
from .types import (
    CroppedTensor,
    Vec2,
    CamIntrinsicsTensor
)
from ..detector.types import RGBFrame, BBoxTensor
from ...shared import config
from .geometry import warp_affine_crop


def decompose_bbox(bbox: BBoxTensor) -> tuple[Vec2, Vec2]:
    """
    Decompose a single axis-aligned bounding box into its geometric centre
    and padded scale vector.

    Accepts a bounding box in xyxy pixel format, computes the midpoint,
    and scales the width and height by the provided padding factor. The
    resulting scale vector is ready for downstream aspect-ratio alignment
    and affine warp matrix construction.

    Parameters
    ----------
    bbox : BBoxTensor
        Bounding box coordinates (x1, y1, x2, y2) in full-image pixels,
        shape (4,), with floating-point dtype.

    Returns
    -------
    center : Vec2
        Box centre (x, y) in full-image pixel coordinates, shape (2,).
    scale : Vec2
        Box width and height (w, h) after padding, shape (2,). Both values
        are in pixels and will be used to construct the canonical crop.
    """
    x1, y1, x2, y2 = bbox.cpu().numpy()

    center: Vec2 = np.array(
        [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
        dtype=config.numpy_dtype,
    )
    scale: Vec2 = np.array(
        [x2 - x1, y2 - y1],
        dtype=config.numpy_dtype,
    ) * config.scale_padding

    return center, scale


def transform_images(frames: list[RGBFrame]) -> CroppedTensor:
    """
    Transform a list of RGB frames into a normalized PyTorch tensor.

    Stacks frames along the batch dimension, permutes from HWC to CHW,
    scales pixel values to [0, 1], and applies ImageNet normalization.
    The resulting tensor remains on the CPU in float32; device placement
    and final dtype casting are deferred to a separate function.

    Parameters
    ----------
    frames : list of RGBFrame
        Input RGB frames in uint8 format, each with shape (H, W, 3).

    Returns
    -------
    CroppedRGBTensor
        Normalized image batch on CPU, shape (B, 3, H, W).
    """
    img_bhw3 = np.stack(frames, axis=0)
    img_tensor = torch.from_numpy(img_bhw3).permute(0, 3, 1, 2) / 255.0

    mean = torch.tensor(
        config.image_mean, 
        dtype=config.core_input_dtype
    ).view(1, 3, 1, 1)
    std = torch.tensor(
        config.image_std, 
        dtype=config.core_input_dtype
    ).view(1, 3, 1, 1)

    return (img_tensor - mean) / std


def build_intrinsics(ori_img_size: tuple[int, int]) -> CamIntrinsicsTensor:
    """
    Construct a default camera intrinsic matrix from frame dimensions.

    Uses a diagonal heuristic where the focal length equals the image diagonal
    sqrt(w^2 + h^2) and the principal point is at the image centre (w/2, h/2).

    Parameters
    ----------
    ori_img_size : tuple[int, int]
        Original image resolution (width, height) in pixels.

    Returns
    -------
    CamIntrinsicsTensor
        Camera intrinsic matrix as a float32 tensor, shape (1, 3, 3).
    """
    w, h = ori_img_size
    diag = np.sqrt(w * w + h * h)
    k = np.zeros(
        (3, 3), 
        dtype=config.numpy_dtype
    )
    k[0, 0] = diag
    k[1, 1] = diag
    k[0, 2] = w * 0.5
    k[1, 2] = h * 0.5
    k[2, 2] = 1.0
    return torch.from_numpy(k).unsqueeze(0)
