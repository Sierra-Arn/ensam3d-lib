# **`ensam3d_inference/`**

## I. **Overview**

This package is a production-oriented, inference-only reimplementation of the original research-oriented SAM 3D Body architecture. It is designed for high-throughput, low-latency 3D human pose estimation in video streams, optimized for single-subject tracking scenarios.

The effectiveness of the package is reflected in the benchmark results below:

**Environment**
| | |
|------------------|------------------------------------------------------------------------------------------------------------------|
| Benchmark Video  | [Man with prosthetic leg jogging, Pexels](https://www.pexels.com/video/man-with-prosthetic-leg-jogging-8344814/) |
| Video Resolution | 3840 × 2160 (4K)                                                                                                 |
| Video FPS        | 25.00                                                                                                            |
| GPU              | NVIDIA GeForce RTX 3070 Laptop GPU                                                                               |
| PyTorch Version  | 2.5.1.post306                                                                                                    |
| CUDA Version     | 12.6                                                                                                             |
| Batch Size       | 30                                                                                                               |

**Results**
| | |
|------------------|-----------------|
| Processed Frames | 608             |
| Total Time       | 36.266 sec      |
| Latency          | 59.648 ms/frame |
| Throughput       | 16.765 FPS      |
| Peak VRAM Usage  | 2.82 GB         |

To reproduce the benchmark on your hardware, see [ensam3d_inference.examples.benchmarking](./examples/benchmarking.py).

## II. **Architectural Differences**

1. **Bounding Box-Based Person Localization**  
   Person localization is performed exclusively using bounding box detection. Mask-based localization is not implemented.

2. **YOLO-Based Detection Pipeline**  
   Person detection is performed exclusively using a lightweight `YOLO` detector. Original detection pipelines (`Detectron2` with `ViTDet` and `SAM 3 image encoder`) are not implemented.

3. **Single-Subject Assumption**  
   The pipeline assumes a single primary subject per frame. When multiple detections are present, only the highest-confidence detection is processed.

4. **Temporal Batch-Oriented Execution**  
   Batch size represents sequential video frames rather than multiple subjects within a single frame.

5. **Inference-Only Distribution**  
   The package contains only runtime inference logic and pretrained checkpoint execution. Training, fine-tuning, and evaluation are not implemented.

6. **Body-Only Reconstruction**  
   Independent hand prediction branches are removed. Only the body reconstruction branch is retained.

7. **Restricted Checkpoint Support**  
   Only the `sam-3d-body-vith` checkpoint is supported. Support for `sam-3d-body-dinov3` is not implemented.

## III. **Runtime Optimizations**

1. **Mixed-Precision Compute (`bfloat16`)**  
   Forward-pass activations in the ViT backbone and transformer decoder are executed under `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`. This routes matrix multiplications, convolutions, and self-attention through NVIDIA Tensor Cores, reducing arithmetic.

2. **Static Weight Precision Conversion**  
   All floating-point parameters and buffers are permanently cast to `bfloat16` at model initialization (excluding explicitly protected modules). This halves the VRAM footprint for stored weights and eliminates implicit on-the-fly dtype casting overhead during inference.

3. **FP32-Protected Geometry & Kinematics**  
   Numerically sensitive operations — including weak-perspective camera projection, CLIFF conditioning, and the MHR kinematic solver — are strictly constrained to `float32`. This is mandatory because the frozen TorchScript MHR solver relies on sparse CUDA kernels (`addmm_sparse_cuda`) that lack `bfloat16` support, and FP32 mantissa precision prevents sub-pixel drift during depth and translation recovery.

4. **Exclusion of `torch.compile`**  
   Despite its potential for kernel fusion, `torch.compile` was deliberately omitted after benchmarking. The pipeline's execution model is incompatible with static graph compilation for two reasons: 
   
      1. **Variable Batch Truncation** — temporal batching processes fixed-size chunks, but the final sequence remainder may yield a smaller batch (e.g., 100 valid detections extracted from 120 input frames with batch_size=30 yields sequential chunks of 30, 30, 30, 10). This shape mismatch invalidates cached CUDA graphs, triggering recompilation overhead that erases any fusion benefits. 
   
      2. **Precision Boundaries** — mandatory `.float()` casts in geometric projection and the TorchScript MHR solver fragment the compiled execution graph, causing frequent fallbacks to eager mode. Consequently, the native eager path with `autocast` currently delivers superior latency.

## IV. **Package Structure**

The codebase is organized by logical execution stages rather than technical artifact types. Files are grouped by *when and how they are used* in the inference chain, not by *what kind of component they are*. Each directory encapsulates a complete pipeline phase, co-locating everything required for that step. For example, all type definitions for the pose estimator reside strictly within `core/pose_estimation/`; there is no global `types/` or `models/` directory. 

This dramatically simplifies onboarding. Instead of tracing logic across scattered directories, developers can follow the data flow stage-by-stage, understanding each module in its full context. Every directory acts as a self-contained unit, eliminating the need to jump between files to piece together how the pipeline works.

### **Directory Layout**

```bash
ensam3d_inference/
├── shared/
├── preprocessor/
│   ├── detector/
│   └── utils/
├── core/
│   ├── feature_extraction/
│   │   ├── backbone/
│   │   └── camera_encoder/
│   └── pose_estimation/
│       ├── decoder/
│       ├── heads/
│       │   ├── mhr_head/
│       │   └── perspective_head/
│       └── utils/
├── pipeline/
└── examples/
```

### **Component Breakdown**

- **`shared/`**  
   Cross-cutting infrastructure used across all stages. Contains the centralized `PipelineConfig` and device enums, acting as the single source of truth for architectural constants.

- **`preprocessor/` — Stage 1: *Input Preparation***  
   Handles detection, canonical cropping, and tensor packing. `detector/` provides the class-based YOLO wrapper, while `utils/` holds stateless functions for affine warping, intrinsic generation, and normalization.

- **`core/` — Stage 2: *Neural Inference Execution***  
   Contains the complete PyTorch execution graph. `feature_extraction/` hosts the ViT-H backbone and Fourier-based camera conditioning. `pose_estimation/` contains the transformer decoder, regression heads (`mhr_head/`, `perspective_head/`), and stateless `utils/` for projection math and token refinement.

- **`pipeline/` — Stage 3: *Runtime Orchestration***  
   Glues preprocessing and inference, manages device routing, aligns batch outputs with original frame indices, and formats results into type-safe named tuples.

- **`examples/`**  
   Standalone scripts for benchmarking, profiling, and single-image visualization.

## V. **Examples**

| Script | Description |
|--------|-------------|
| [benchmarking.py](./examples/benchmarking.py) | Measure throughput, latency, and VRAM usage |
| [profiling.py](./examples/profiling.py) | PyTorch profiler with CPU/CUDA activity tracing — outputs top 20 CUDA operations by execution time or exports a Chrome trace file |
| [visualization.py](./examples/visualization.py) | Single-image pose estimation visualizer — overlays bounding box, keypoints, and skeleton links via matplotlib; supports interactive display or export to disk |