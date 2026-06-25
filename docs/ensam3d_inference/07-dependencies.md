# VII. Dependencies Overview

> *This document describes the dependencies and model weights required by the Enhanced SAM 3D Body Inference package: what each one is, the role it plays in the pipeline, and why it was chosen over the alternatives.*

## System Dependencies

| Dependency | Repository | What it is | Role in the project |
|---|---|---|---|
| Python | [python.org](https://www.python.org/) | Programming language | Primary language for all project source code. |
| Pixi | [prefix‑dev/pixi](https://github.com/prefix-dev/pixi) | Package and environment manager | 1. Resolves mixed Conda/PyPI dependency graphs into a single deterministic lockfile, producing fully reproducible environments. <br>2. Manages the project's virtual environment. |
| NVIDIA GPU | — | GPU hardware | Processor whose thousands of parallel cores let it perform the massively data-parallel tensor arithmetic (matrix multiplications) at the core of neural-network inference far more efficiently than a CPU. |
| CUDA driver | — | Host GPU driver | Low-level host driver (`libcuda`) that gives user-space access to the GPU, making accelerated computation possible at all. |

## Pixi Dependencies

| Dependency | Repository | What it is | Role in the project |
|---|---|---|---|
| CPython | [python/cpython](https://github.com/python/cpython) | Python virtual machine | Executes all source code. |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | Deep learning framework | Core inference runtime for tensor computation and neural-network execution. |
| CUDA Toolkit | [nvidia.com](https://developer.nvidia.com/cuda/toolkit) | CUDA runtime and math libraries | User-space runtime and optimized GPU libraries through which PyTorch executes its tensor computations on the GPU instead of the CPU, where they run many times faster. |
| torchvision | [pytorch/vision](https://github.com/pytorch/vision) | PyTorch vision companion library | Image normalization and tensor transformation utilities for preprocessing. |
| timm | [huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) | Vision model collection | Pre-trained Vision Transformer backbones for feature extraction. |
| RoMa | [naver/roma](https://github.com/naver/roma) | Rotation representation library | Rotation-representation conversions for 3D pose reconstruction. |
| Ultralytics | [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) | YOLO model framework | YOLO-based person detection for the preprocessing stage. |
| jaxtyping | [patrick-kidger/jaxtyping](https://github.com/patrick-kidger/jaxtyping) | Tensor type-annotation library | Static tensor shape and dtype annotations for pipeline interfaces. |
| OpenCV | [opencv/opencv](https://github.com/opencv/opencv) | Computer vision library | Geometric preprocessing: affine warping and canonical crop generation via `cv2.warpAffine`. |
| NumPy | [numpy/numpy](https://github.com/numpy/numpy) | Array computing library | Array backend for OpenCV operations and CPU-side data interchange between preprocessing stages. |

## Optional Pixi Dependencies

| Dependency | Repository | What it is | Role in the project |
|---|---|---|---|
| huggingface_hub | [huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub) | Hugging Face Hub client | Optional remote checkpoint retrieval with authenticated access to the Hugging Face Hub. |

## Model Weights

| Model | Repository | Used by |
|-------|------------|---------|
| YOLO26 detector | [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26) | The `Detector` stage for person localization. |
| SAM 3D Body | [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) | The `Engine` for pose estimation and mesh reconstruction. |

## Rationale for Dependency Choices

| Dependency | Rationale |
|------------|-----------------|
| Python | The de facto standard for machine learning and computer vision. Its ecosystem is far more mature than those of other languages, offering a comprehensive suite of production-ready libraries. |
| Pixi | Unlike Uv, it supports Conda packages, which are required here for fully reproducible environments. Unlike Miniconda/Micromamba, it resolves mixed Conda/PyPI dependency graphs into a single deterministic lockfile. |
| CPython | The reference and de facto standard implementation of Python. The scientific stack is tightly coupled to its C-API and native extension model; alternative implementations (PyPy, GraalPy) lack full compatibility with compiled libraries and offer no practical speedup, since heavy computation already runs in optimized C/CUDA kernels. |
| PyTorch | The original SAM 3D Body codebase is built on PyTorch, and its pre-trained weights are made specifically for it. PyTorch is also widely adopted and actively developed, so there is no reason to port the entire codebase to another framework — doing so would require non-trivial weight conversion and graph rewriting for no real benefit. |
| NVIDIA GPU <br>+ CUDA driver <br>+ CUDA Toolkit | The model's computations are heavy enough that GPU acceleration is effectively mandatory: while CPU inference is fully supported and functional, its runtimes are impractical for any production use. AMD ROCm was not targeted because it lacks full PyTorch feature parity and mature kernel optimization for transformer architectures, leaving CUDA as the only practical acceleration backend. |
| torchvision <br>+ timm | torchvision is PyTorch's official companion library, with API compatibility guaranteed by the same maintainers — there is simply no reason to look for an alternative. timm provides ready-made, PyTorch-native model implementations; building equivalents by hand would be pointless when polished, well-tested ones already exist there. |
| RoMa | Already used in the original SAM 3D Body project, and there is no equivalent lightweight rotation-representation library in the PyTorch ecosystem. |
| Ultralytics | The official, actively maintained Python API for YOLO models, providing model loading, preprocessing, and postprocessing out of the box — so the detection stage needs no custom plumbing. |
| jaxtyping | The de facto standard for tensor shape and dtype annotations in the PyTorch ecosystem, and actively maintained. |
| OpenCV | Used in the original SAM 3D Body project, and provides a highly optimized, stable C++ implementation of the geometric operations needed for preprocessing (`cv2.warpAffine`). Switching to Kornia might reduce NumPy–Tensor conversion overhead, but verifying any real-world gain would require separate benchmarking — and since the current throughput is already sufficient, that migration effort is not justified. |
| NumPy | OpenCV operates exclusively on NumPy arrays, so it is required as the array backend regardless; it is also the universal interchange format for the scientific Python stack. |

## Rationale for Model Weights

| Model | Rationale |
|-------|-----------------|
| `yolo26n` | The detection stage needs only fast bounding-box localization of a single person, nothing more. The `yolo26n` (nano) variant is the smallest and fastest in the family, which is exactly the right trade-off when detection quality beyond a reliable bounding box is not required. |
| `sam3d‑body‑vith` | SAM 3D Body ships in two backbone families — ViT-H and a larger DINOv3-based variant. In line with the single-checkpoint decision, the package supports exactly one, and ViT-H was chosen as the lighter of the two (632M vs 840M parameters), which better fits the throughput and memory goals of a production inference pipeline. |