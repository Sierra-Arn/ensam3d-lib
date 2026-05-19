# VI. **Package Structure**

> *This document describes the logical organization of the codebase following the stage-based execution model defined in the system architecture.*

## **Directory Layout**

As introduced in the engineering decisions section, the original research codebase grouped logic by technical artifact type rather than execution stage, making the runtime graph difficult to navigate and increasing cognitive overhead during code exploration.

In contrast, `ensam3d_inference` organizes the codebase around the logical runtime stages defined in the system architecture, grouping related components by execution responsibility rather than implementation type, thereby making the execution flow explicit, improving code navigation, and reducing cognitive overhead during development and maintenance.

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

## **Component Overview**

> **Note:** all files include inline comments describing backend-specific decisions and non-obvious implementation details.

1. **`shared/`**

    This module centralizes runtime configuration, shared type definitions, and global device settings used across the pipeline.

2. **`preprocessor/`**, **`core/`**, **`pipeline/`**

    Implementations of the runtime modules defined in the system architecture, corresponding respectively to the `Preprocessor`, `Engine`, and `Pipeline`.

    Nested directories such as `detector/`, `feature_extraction/`, `backbone/`, `camera_encoder/`, `pose_estimation/`, `decoder/`, `mhr_head/`, and `perspective_head/` are implementations of the runtime modules defined in the system architecture, corresponding respectively to the `Detector`, `FeatureExtractor`, `Backbone`, `CameraEncoder`, `PoseEstimator`, `PromptableDecoder`, `MHRHead`, and `PerspectiveHead`.

    All `utils/` directories contain auxiliary utilities scoped to their corresponding architectural module.

3. **`examples/`**

    Standalone utilities for benchmarking, profiling, and visualization.

    | Script | Description |
    |--------|-------------|
    | `benchmarking.py` | Measures throughput, latency, and VRAM usage. |
    | `profiling.py` | PyTorch profiler with CPU/CUDA tracing; outputs top-k CUDA operations or Chrome trace files. |
    | `visualization.py` | Single-image pose visualization with bounding boxes, keypoints, and skeleton overlays; supports interactive display and file export. |