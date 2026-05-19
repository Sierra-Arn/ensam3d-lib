# src/ensam3d_inference/__init__.py
"""
Enhanced SAM 3D Body Inference.

High-performance, production-ready package for 3D human pose and mesh
estimation from RGB images. Combines YOLO-based person detection,
canonical cropping, and a ViT-HMR transformer backbone with iterative
pose refinement and MHR mesh reconstruction.

Quick Start
-----------
>>> from ensam3d_inference import Pipeline, PreprocessorInput, DeviceType
>>> import numpy as np

>>> # Initialize pipeline
>>> # Model weights are resolved in the following order:
>>> # 1. Local filesystem path (if exists)
>>> # 2. HuggingFace Hub repository ID
>>> pipeline = Pipeline(
...     model_path="sam-3d-body-vith",
...     model_device=DeviceType.CUDA,
...     detector_device=DeviceType.CUDA,
... )

>>> # A blank image contains no persons, so detection returns None
>>> dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
>>> results = pipeline(PreprocessorInput(imgs=[dummy_frame]))
>>> print(results[0] is None)
True

>>> # For real images with detected persons, access pose data like this:
>>> # if results[0] is not None:
... #     print(results[0].pose.pred_keypoints_2d.shape)
... #     # Expected output: (1, 70, 2)

For complete usage examples, including visualization, benchmarking, and profiling,
see the ensam3d_inference.examples subpackage:
- python -m ensam3d_inference.examples.visualization
- python -m ensam3d_inference.examples.benchmarking
- python -m ensam3d_inference.examples.profiling
"""

from .pipeline import Pipeline
from .preprocessor.types import PreprocessorInput
from .pipeline.types import FramePoseResult, PipelineOutput
from .shared.types import DeviceType

__all__ = [
    "Pipeline",
    "PreprocessorInput", 
    "FramePoseResult", 
    "PipelineOutput",
    "DeviceType",
]