# VIII. Installation Guide

> *This document explains how to install and set up the `ensam3d_inference` package for execution. It opens with a short overview of the required steps, after which each section walks through one step in detail.*

## Overview

Running the `ensam3d_inference` package requires three steps:

- **Step 1**  
Install all dependencies listed in the Dependencies Overview and the `ensam3d-lib` package itself.
- **Step 2**  
Obtain the pre-trained weights for both the YOLO detector and SAM 3D Body.
- **Step 3**  
Initialize and run the inference pipeline.

## Step 1 — Install Dependencies and the Package

Below is an example of a `pixi.toml` configuration file for a GNU/Linux-based system on `x86_64` architecture with NVIDIA GPU support and NVIDIA driver that supports CUDA `>= 12.8`, which installs all required dependencies for executing the `ensam3d_inference` package.

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
python = ">=3.12.13,<3.13"
numpy = ">=1.26.4,<2"
cuda-toolkit = ">=12.8.2,<13"
pytorch-gpu = ">=2.5.1,<3"
torchvision = ">=0.20.1,<0.21"
ultralytics = ">=8.4.46,<9"
timm = ">=1.0.26,<2"
opencv = ">=4.10.0,<5"
jaxtyping = ">=0.3.7,<0.4"
huggingface_hub = ">=1.13.0,<2"

[pypi-dependencies]
roma = ">=1.5.6, <2"
ensam3d-lib = { git = "https://github.com/Sierra-Arn/ensam3d-lib.git" }
```

## Step 2 — Obtain the Model Weights

The pipeline uses two sets of weights: one public, one access-gated.

### 1. YOLO detector

The YOLO detector weights are public and require no authentication. By default the `Detector` is configured with `yolo26n.pt`; if that file is not found locally, the Ultralytics client downloads it automatically on first use. No manual step is required unless you want to pin a specific local file, in which case pass its path as `model_path`.

### 2. SAM 3D Body

These weights are access-gated. To obtain them:

1. Navigate to the [facebook/sam-3d-body-vith](https://huggingface.co/facebook/sam-3d-body-vith) repository.
2. Sign in with a Hugging Face account, request access to the repository, and wait for approval.

Once access is granted, the weights can be obtained in one of two ways:
- authenticate through the Hugging Face Hub CLI or the installed `huggingface_hub` package and load the model directly at runtime, or
- manually download the weights and provide the local filesystem path in the configuration or inference code.

## Step 3 — Run the Pipeline

A minimal example of how to initialize and run the inference pipeline is provided in the [source code](../../src/ensam3d_inference/__init__.py).