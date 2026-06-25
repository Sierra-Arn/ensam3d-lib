# VI. Package Structure

> *This document describes the logical organization of the codebase following the stage-based execution model defined in the system architecture.*

## Directory Layout

The previous section defined the runtime architecture as a sequence of explicit execution stages with clearly defined responsibilities and typed data contracts. This section maps that runtime architecture onto the repository structure.

The original SAM 3D Body research codebase organized modules by technical artifact type rather than execution stage, which made the runtime flow difficult to trace and increased cognitive overhead during code exploration.

In contrast, `ensam3d_inference` organizes the repository around the runtime stages defined in the system architecture. Related components are grouped by execution responsibility rather than implementation type, making the execution flow explicit, improving code navigation, and reducing cognitive overhead during development and maintenance.

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

## Component Overview

> **Note:** all modules are documented with docstrings, and inline comments explain any non-obvious or unusual logic where it occurs.

1. **`shared/`**

    This module centralizes runtime configuration, shared type definitions, and global device settings used across the pipeline.

2. **`preprocessor/`**, **`core/`**, **`pipeline/`**

    Implementations of the runtime modules defined in the system architecture, corresponding respectively to the `Preprocessor`, `Engine`, and `Pipeline`. 
    
    Their nested directories follow the same one-to-one mapping: `detector/`, `feature_extraction/`, `backbone/`, `camera_encoder/`, `pose_estimation/`,  `decoder/`, `mhr_head/`, and `perspective_head/` correspond to the `Detector`, `FeatureExtractor`, `Backbone`, `CameraEncoder`, `PoseEstimator`, `PromptableDecoder`, `MHRHead`, and `PerspectiveHead` modules respectively.

    All `utils/` directories contain auxiliary utilities scoped to their corresponding architectural module.

3. **`examples/`**

    Standalone utilities for benchmarking, profiling, and visualization.

    | Script | Description |
    |--------|-------------|
    | `benchmarking.py` | Measures throughput, latency, and VRAM usage. |
    | `profiling.py` | PyTorch profiler with CPU/CUDA tracing; outputs top-k CUDA operations or Chrome trace files. |
    | `visualization.py` | Single-image pose visualization with bounding boxes, keypoints, and skeleton overlays; supports interactive display and file export. |