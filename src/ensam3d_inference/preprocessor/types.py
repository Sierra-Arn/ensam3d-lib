# src/ensam3d_inference/preprocessor/types.py
from typing import NamedTuple
import numpy as np
import torch
from jaxtyping import Float
from .detector.types import DetectorInput, DetectorOutput
from .utils.types import CamIntrinsicsTensor, CroppedTensor


CamIntrinsicsMatrix = Float[np.ndarray, "3 3"]
"""
Single camera intrinsic matrix represented as a NumPy array with shape (3, 3) and floating-point
dtype, where (3, 3) is the standard pinhole camera intrinsic matrix with focal lengths on the
diagonal (fx, fy) and principal point in the last column (cx, cy); shared across all frames in
the batch as it encodes static camera geometry for a single recording session, not per-frame
quantities; if not provided by the caller, a diagonal heuristic matrix is constructed from the
frame dimensions using focal length sqrt(w^2 + h^2) and principal point at (w/2, h/2).
"""


class PreprocessorInput(NamedTuple):
    """
    Container for the input to the preprocessor pipeline.

    Attributes
    ----------
    imgs : DetectorInput
        Batch of RGB frames to process, one entry per frame.
    cam_int : CamIntrinsicsMatrix or None, optional
        Shared camera intrinsic matrix, shape (3, 3) in pixels; if None,
        a diagonal matrix is constructed from the first valid frame size
        using focal length sqrt(w^2 + h^2) and principal point at
        (w/2, h/2). Default is None.
    """

    imgs: DetectorInput
    cam_int: CamIntrinsicsMatrix | None = None


BBoxCenterTensor = Float[torch.Tensor, "B 2"]
"""
Bounding box centre coordinates represented as a PyTorch float32 tensor with shape (B, 2), 
where B is the number of frames (one row per frame), not the number of people; 
each row is (x, y) in full-image pixel coordinates for that frame.
"""


BBoxScaleTensor = Float[torch.Tensor, "B 2"]
"""
Bounding box scale represented as a PyTorch float32 tensor with shape (B, 2), 
where B is the number of frames (one row per frame), not the number of people; 
each row is (w, h) in pixels after padding and aspect-ratio alignment, 
i.e. width and height of the box used to build the crop for that frame.
"""


AffineTransTensor = Float[torch.Tensor, "B 2 3"]
"""
Affine transforms from the full image to the warped crop represented as a PyTorch float32 
tensor with shape (B, 2, 3), where B is the number of frames (one 2x3 matrix per frame) 
not the number of people; each matrix matches the OpenCV cv2.warpAffine convention, 
mapping points from the original image coordinate space to the crop.
"""


CroppedSizeTensor = Float[torch.Tensor, "B 2"]
"""
Warped crop resolution represented as a PyTorch float32 tensor with shape (B, 2), 
where B is the number of frames in the batch (one row per frame) and 2 corresponds to 
(width, height) in pixels of the crop.
"""


OriImgSizeTensor = Float[torch.Tensor, "B 2"]
"""
Original full-image resolution represented as a PyTorch float32 tensor with shape (B, 2), 
where B is the number of frames in the batch (one row per frame) and 2 corresponds to 
(width, height) in pixels of the input frame before any warping or resizing.
"""


class PreprocessorOutput(NamedTuple):
    """
    Container for the output of the preprocessor pipeline.

    Attributes
    ----------
    detections : DetectorOutput
        Detection results aligned with the input frame sequence; each entry
        is either a DetectionResult if no person was detected in
        the corresponding frame.
    valid_indices : list of int
        Indices of input frames where detection succeeded, in ascending
        order; used to align model outputs back to the original frame sequence.
    img : CroppedRGBTensor
        Batched RGB crops as a PyTorch tensor, shape (B, 3, H, W).
    bbox_center : BBoxCenterTensor
        Bounding box centres in full-image pixel coordinates, shape (B, 2).
    bbox_scale : BBoxScaleTensor
        Final bounding box scales (w, h) in pixels, shape (B, 2).
    affine_trans : AffineTransTensor
        Affine transformation matrices mapping full-image coordinates to
        crop space, shape (B, 2, 3).
    ori_img_size : OriImgSizeTensor
        Original full-image resolution (width, height) in pixels, shape (B, 2);
        constant across all frames as all frames share the same resolution;
        None when valid_indices is empty.
    crop_img_size : CroppedSizeTensor
        Warped crop resolution (width, height) in pixels, shape (B, 2);
        constant across all frames as all crops share the same output.
    cam_int : CamIntrinsicsTensor
        Camera intrinsic matrix as a float32 tensor, shape (B, 3, 3); either
        converted from the matrix passed in PreprocessorInput or constructed
        from ori_img_size using a diagonal heuristic with focal length
        sqrt(w^2 + h^2) and principal point at (w/2, h/2).
    """

    detections: DetectorOutput
    valid_indices: list[int]
    img: CroppedTensor
    bbox_center: BBoxCenterTensor
    bbox_scale: BBoxScaleTensor
    affine_trans: AffineTransTensor
    ori_img_size: OriImgSizeTensor
    crop_img_size: CroppedSizeTensor
    cam_int: CamIntrinsicsTensor
