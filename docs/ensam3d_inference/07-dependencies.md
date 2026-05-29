# VII. **Dependencies Overview**

> *This document describes the runtime dependencies required by the Enhanced SAM 3D Body Inference package.*

## System Dependencies

| Dependency | Purpose in This Project | Why Not Alternatives? |
|------------|-------------------------|----------------------|
| [**Python**](https://www.python.org/) | Primary language for all project source code. | Python is the de facto standard for machine learning and computer vision. Its ecosystem is significantly more mature than those of other languages, providing a comprehensive suite of production-ready libraries. |
| [**CPython**](https://github.com/python/cpython)| Reference Python virtual machine used for executing the source code. | CPython is the de facto standard implementation of the Python language. That's why the scientific Python stack is tightly coupled to CPython’s C-API and native extension model. Alternative implementations (PyPy, GraalPy) lack full compatibility with compiled libraries, and offer no practical speedup since heavy computation already runs in optimized C/CUDA kernels. Experimental variants (freethreading, JIT) are excluded because the dependency graph is validated and locked to CPython 3.12, where these features aren't supported. |
| [**Pixi**](https://pixi.sh/latest/) | Package manager for creating reproducible environments. | 1. Unlike Uv, Pixi supports Conda packages, which are required for fully reproducible environments.<br>2. Unlike Miniconda/Micromamba, Pixi resolves mixed Conda/PyPI dependency graphs into a single deterministic lockfile. |
| **NVIDIA GPU + CUDA-compatible Driver + CUDA Toolkit** [^1] | Hardware acceleration backend for tensor computation and inference execution. | 1. CPU-only execution is impractical for production throughput. <br>2. AMD ROCm lacks full PyTorch feature parity and mature kernel optimization for transformer architectures. |

[^1]: While CPU inference remains fully supported, CUDA-enabled inference is required to obtain practically acceptable runtimes.

## Core Dependencies

| Dependency | Purpose in This Project | Why Not Alternatives? |
|------------|-------------------------|----------------------|
| [**PyTorch**](https://github.com/pytorch/pytorch) | Core inference runtime for tensor computation and neural network execution. | 1. The original SAM 3D Body architecture and pre-trained checkpoints are implemented in PyTorch; migrating to another framework would require non-trivial weight conversion and graph rewriting.<br>2. PyTorch is the de facto standard for deep learning research and production: it has the most mature ecosystem, active maintenance, and broadest hardware support (CUDA, mixed-precision, TorchScript) among ML frameworks. |
| [**torchvision**]((https://github.com/pytorch/vision)) | Image normalization and tensor transformation utilities for preprocessing. | Official PyTorch companion library: guaranteed API compatibility, synchronized release cycles, and shared CUDA backend. |
| [**timm**](https://github.com/huggingface/pytorch-image-models) | Pre-trained Vision Transformer backbones for feature extraction. | Curated collection of state-of-the-art PyTorch-native vision models with consistent APIs and pre-trained weights. |
| [**RoMa**](https://github.com/naver/roma) | Rotation representation conversion utilities for 3D pose reconstruction. | Solves a non-trivial task with a minimal, well-tested API; no comparable lightweight alternative exists in the PyTorch ecosystem. |
| [**Ultralytics**](https://github.com/ultralytics/ultralytics) | YOLO-based person detection for the preprocessing stage. | Official, actively maintained Python API for YOLO models: consistent model loading, preprocessing, and postprocessing out of the box. |
| [**jaxtyping**](https://github.com/patrick-kidger/jaxtyping) | Static tensor shape and dtype annotations for pipeline interfaces. | Actively maintained and the de facto standard for tensor type annotations in the PyTorch ecosystem. |
| [**OpenCV**](https://github.com/opencv/opencv) | Deterministic geometric preprocessing: affine warping and canonical crop generation via `cv2.warpAffine`. | 1. `cv2.warpAffine` is backed by a highly optimized C++ implementation and provides stable, deterministic pixel-level behavior.<br>2. While Kornia could reduce NumPy–Tensor conversion overhead, its real-world performance advantage is not guaranteed and would require additional benchmarking. Given that current latency requirements are already met, the added complexity and dependency cost of Kornia is not justified. |
| [**NumPy**](https://github.com/numpy/numpy) | CPU-side array manipulation and interoperability layer for preprocessing stages. | 1. OpenCV (`cv2`) operates exclusively on NumPy arrays; any alternative would require conversion overhead without functional benefit.<br>2. NumPy is the universal interchange format for the scientific Python stack; replacing it would add unnecessary abstraction layers and break compatibility with established ecosystem conventions. |

## Optional Dependencies

| Dependency | Purpose in This Project | Why Not Alternatives? |
|------------|-------------------------|----------------------|
| [**huggingface_hub**](https://github.com/huggingface/huggingface_hub) | Optional remote checkpoint retrieval with authenticated access to the Hugging Face Hub. | Official library for Hugging Face Hub: provides secure token-based authentication, license-gated model access, and resumable downloads out of the box. |

## Model Weights

| Model | Source | Default Variant | Retrieval |
|-------|--------|-----------------|-----------|
| **YOLO26 detector** | [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26) | `yolo26n` | Automatically downloaded on first use. |
| **SAM 3D Body** | [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) | `vit-h` | Requires accepting Meta's license agreement before download. |