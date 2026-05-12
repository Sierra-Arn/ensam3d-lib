# src/ensam3d_inference/preprocessor/detector/types.py
from typing import NamedTuple
import numpy as np
import torch
from jaxtyping import UInt8, Float


RGBFrame = UInt8[np.ndarray, "H W 3"]
"""
RGB image frame represented as a NumPy array with shape (H, W, 3) and dtype uint8, 
where H is the image height, W is the image width, and 3 corresponds to the RGB color channels.
"""


DetectorInput = list[RGBFrame]
"""
Batch of RGB image frames represented as a list of NumPy arrays, where each element is an
RGBFrame of shape (H, W, 3) and dtype uint8; all frames in the batch must share the same
spatial dimensions and originate from the same camera, as the pipeline assumes a single
intrinsic matrix and a single original image resolution across the entire batch.
"""


BBoxTensor = Float[torch.Tensor, "4"]
"""
Bounding box represented as a PyTorch tensor with shape (4,) and floating-point dtype 
matching YOLO output, where 4 corresponds to the four coordinates (x1, y1, x2, y2), 
with (x1, y1) denoting the top-left corner in pixel coordinates and 
(x2, y2) denoting the bottom-right corner.
"""


ConfidenceTensor = Float[torch.Tensor, ""]
"""
Detection confidence score represented as a PyTorch scalar tensor with floating-point dtype 
matching YOLO output, where the value is in the range [0, 1], with higher values indicating 
greater certainty that the detected region contains a person.
"""


class Detection(NamedTuple):
    """
    Detection result corresponding to a single input frame.

    Attributes
    ----------
    coords : BBoxTensor
        Bounding box coordinates.
    confidence : ConfidenceTensor
        Detection confidence score.
    """

    coords: BBoxTensor
    confidence: ConfidenceTensor


DetectorOutput = list[Detection | None]
"""
Detection results for a batch of input frames represented as a list with one entry per input
frame aligned with the corresponding DetectorInput, where each entry is either a
Detection containing the highest-confidence bounding box and its score, or None if no
person was detected in the corresponding frame.
"""
