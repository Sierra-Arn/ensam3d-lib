# **Enhanced SAM 3D Body Library**

*Collection of production-oriented inference pipelines based on [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) for 3D human body pose and shape estimation.*

## **Project Structure**

```bash
ensam3d-lib/
├── src/                        # Standard src-layout for isolated package development
│   └── ensam3d_inference/      # Production inference pipeline for 3D human pose estimation
├── docs/                       # Conceptual documentation — mirrors src/ structure for traceability
│   └── ensam3d_inference/      # Package-specific docs for ensam3d_inference
├── pyproject.toml              # Unified project configuration: metadata, dependencies, and build backend
├── pixi.toml                   # Pixi project configuration: environments, dependencies, and platforms
└── pixi.lock                   # Locked dependency versions for reproducible environments
```

Each directory includes its own `README.md` with detailed information about its contents.

## **Dependencies Overview**

| Dependency | Description | Usage in This Project |
|------------|------------|------------------------|
| [PyTorch](https://github.com/pytorch/pytorch) | Deep learning framework for tensor computation and automatic differentiation with GPU acceleration | Core inference runtime for executing neural network models |
| [NumPy](https://github.com/numpy/numpy) | The fundamental package for scientific computing with Python | General-purpose array operations for CPU-side numerical processing and data handling |
| [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) | GPU computing platform for hardware-accelerated computation | Provides GPU acceleration for PyTorch inference computations [^1] |
| [jaxtyping](https://github.com/patrick-kidger/jaxtyping) | Type annotations and runtime checking for shape and dtype of NumPy/PyTorch arrays | Static shape and dtype annotations for tensors, used to improve code readability and traceability across the inference pipeline |
| [torchvision](https://github.com/pytorch/vision) | Popular datasets, model architectures, and common image transformations for computer vision | Image preprocessing stage before input to the neural network |
| [timm](https://github.com/huggingface/pytorch-image-models) | Сollection of PyTorch image encoders / backbones | Feature extraction in the model architecture |
| [Ultralytics](https://github.com/ultralytics/ultralytics) | Framework for YOLO-based object detection models | First stage in the inference pipeline for human detection |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Official Python client for the Hugging Face Hub | Model weights and checkpoints retrieval for inference initialization [^2] |
| [OpenCV](https://github.com/opencv/opencv) | Open Source Computer Vision Library  | Geometric preprocessing stage in the inference pipeline |
| [RoMa](https://github.com/naver/roma) | Lightweight library to deal with 3D rotations in PyTorch | Rotation representation conversion in 3D pose estimation |

[^1]: While PyTorch supports CPU execution, the models used in this project are so computationally intensive that CUDA is required for practical use.

[^2]: This dependency is only required if model weights are fetched remotely from the HuggingFace Hub. If all checkpoints are available locally, `huggingface_hub` is never invoked at runtime and can be omitted from the environment without affecting inference.

## **Model Weights**

To run inference, two model checkpoints are required.

| Model | Source | Default Variant | Download |
|-------|--------|-----------------|----------|
| YOLO26 | [Ultralytics/YOLO26](https://huggingface.co/Ultralytics/YOLO26) | `yolo26n` | Automatic on first run |
| SAM 3D Body | [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) | `vit-h` | Requires accepting Meta's license agreement on HuggingFace |

## **Installation as dependency**

Due to the complexity of compiled dependencies in the PyTorch + CUDA ecosystem, core dependencies must be installed manually prior to installing this project. The provided `pyproject.toml` is intended solely for editable development installation and does not resolve or install runtime ML dependencies. Therefore, installing the package in isolation will not result in a functional runtime environment.

Below is an example of `pixi.toml` file that defines an environment with this project and all of its required dependencies:

```toml
[workspace]
authors = ["Sierra Arn"]
channels = ["nvidia", "conda-forge"]
name = "project"
platforms = ["linux-64"]
version = "0.1.0"

[system-requirements]
cuda = "12.8"

[dependencies]
python = "*"
numpy = ">=1.26,<2"
cuda-toolkit = ">=12.8.0,<12.9"
pytorch-gpu = "*"
torchvision = "*"
ultralytics = "*"
timm = "*"
jaxtyping = "*"
huggingface_hub = "*"
opencv = "*"

[pypi-dependencies]
roma = "*"
ensam3d-lib = { git = "https://github.com/Sierra-Arn/ensam3d-lib.git" }
```

## **Installation for development**

### **I. Prerequisites**

- [Pixi](https://pixi.sh/latest/) package manager.

> **Platform note:**  
All development and testing were performed on `linux-64`. If you're using a different platform, you’ll need to update the `platforms` list in the `pixi.toml` accordingly. In some cases, certain packages may not be available for your platform. If that happens, you might need to adjust or replace those dependencies.

### **II. Initial Setup**

1. **Clone the repository**

    ```bash
    git clone git@github.com:Sierra-Arn/ensam3d-lib.git
    cd ensam3d-lib
    ```

2. **Install dependencies**

    ```bash
    pixi install
    ```

### **III. Developing**

Once the environment is ready, you can start developing and working with the codebase. Simply activate the isolated environment with `pixi shell` and begin running scripts, iterating on modules, or launching benchmarks — all dependencies and Python paths are preconfigured.

## **License**

This project is licensed under the [Apache License 2.0](LICENSE).

YOLO26 model weights are subject to the [Ultralytics AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE).

SAM 3D Body model weights are subject to the [Meta SAM License](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE).