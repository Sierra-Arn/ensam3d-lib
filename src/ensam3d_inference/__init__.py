# src/ensam3d_inference/__init__.py
from .pipeline import Pipeline, PreprocessorInput, PipelineOutput
from .shared import DeviceType
"""
Enhanced SAM 3D Body Inference.

High-performance, production-ready package for 3D human pose and mesh
estimation from RGB images. Combines YOLO-based person detection, cropping, 
and a ViT-HMR transformer backbone with iterative pose refinement 
and MHR mesh reconstruction.

Quick Start
-----------
>>> from ensam3d_inference import Pipeline, PreprocessorInput, DeviceType
>>> import cv2

>>> pipeline = Pipeline(
...     model_path="sam-3d-body-vith",
...     detector_device=DeviceType.CUDA,
...     model_device=DeviceType.CUDA,
... )

>>> img_bgr = cv2.imread("example.png")
>>> img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
>>> results = pipeline(PreprocessorInput(imgs=[img_rgb]))

>>> if results[0] is not None:
...     print("Detected keypoints shape:", results[0].pose.pred_keypoints_2d.shape)

Main Exports
------------
Pipeline : callable
    End-to-end inference orchestrator. Handles detection, cropping,
    model forward pass, and temporal alignment.
PreprocessorInput : NamedTuple
    Standard input container. Accepts a list of RGB frames (NumPy uint8
    HWC arrays) and optional shared camera intrinsics.
"""