# VII. **Dependencies Overview**

> *This document describes the runtime dependencies required by the Enhanced SAM 3D Body Inference package.*

## **Core Dependencies**

| Dependency | Purpose in This Project |
|------------|-------------------------|
| [PyTorch](https://github.com/pytorch/pytorch) | Core inference runtime used for tensor computation, neural network execution, GPU acceleration, and mixed-precision inference. |
| [NumPy](https://github.com/numpy/numpy) | CPU-side numerical processing, array manipulation, and interoperability between preprocessing stages. |
| [jaxtyping](https://github.com/patrick-kidger/jaxtyping) | Static tensor shape and dtype annotations used to improve readability, traceability, and IDE support across the inference graph. |
| [torchvision](https://github.com/pytorch/vision) | Image normalization and tensor transformation utilities used during preprocessing. |
| [timm](https://github.com/huggingface/pytorch-image-models) | Vision Transformer backbone implementations used for feature extraction. |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | YOLO-based person detection used during the preprocessing stage. |
| [OpenCV](https://github.com/opencv/opencv) | Deterministic geometric preprocessing, affine warping, and crop generation. |
| [RoMa](https://github.com/naver/roma) | Rotation representation conversion utilities used during 3D pose reconstruction. |

## **Optional Dependencies**

| Dependency | Purpose in This Project |
|------------|-------------------------|
| [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) | Enables GPU-accelerated inference execution through CUDA-enabled PyTorch builds. While CPU inference remains fully supported, CUDA-enabled inference is required to obtain practically acceptable runtimes. |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Optional remote checkpoint retrieval from the Hugging Face Hub. Not required when model weights are stored locally. |

## **Model Weights**

| Model | Source | Default Variant | Retrieval |
|-------|--------|-----------------|-----------|
| YOLO26 detector | [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26) | `yolo26n` | Automatically downloaded on first use. |
| SAM 3D Body | [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) | `vit-h` | Requires accepting Meta's license agreement before download. |