# VIII. **Installation Guide**

> *This document describes how to install and set up the `ensam3d_inference` package for execution.*

## Overview

To run the `ensam3d_inference` package, the following setup steps are required:

1. Install all dependencies listed in the Dependencies Overview.
2. Install the `ensam3d-lib` package.
3. Download or obtain the pre-trained model weights for `sam-3d-body-vith`.

## Environment Setup (Pixi Configuration)

Below is an example of a `pixi.toml` configuration file for a GNU/Linux-based system on `x86_64` architecture with NVIDIA GPU support and an NVIDIA driver compatible with CUDA Toolkit `>= 12.8`, which installs all required dependencies for executing the `ensam3d_inference` package.

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

## Model Weights Access

To access the `sam-3d-body-vith` model weights, navigate to the following repository:

```
https://huggingface.co/facebook/sam-3d-body-vith
```

Access is restricted and requires authentication via a Hugging Face account. The user must request access to the repository and wait for approval.

After access is granted, the weights can be obtained in one of two ways:
- authenticate through the Hugging Face Hub CLI or the installed `huggingface_hub` package and load the model directly at runtime, 
- manually download the weights and specify the local filesystem path in the configuration or inference code.

## Usage Example

A minimal example of how to initialize and run the inference pipeline is provided in the [source code](../../src/ensam3d_inference/__init__.py).