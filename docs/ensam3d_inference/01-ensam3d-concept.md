# I. **Enhanced SAM 3D Body Inference: Conceptual Overview**

> *This document describes the high-level conceptual architecture of the Enhanced SAM 3D Body Inference package.*

## Task and First Decisions

When I started this project, I was given a concrete objective: build a production-grade inference pipeline for 3D human pose estimation from monocular video. Before selecting a neural architecture, defining the preprocessing strategy, or committing to any inference routing logic, I first needed to formalize exactly what the system must do — and, just as importantly, what falls outside its scope.

Stripping away all implementation details, I formalized the task as the following input–output contract:

| Component | Description |
| ----------|-------------|
| **Input** | A video stream represented as a sequence of RGB frames. |
| **Output** | A sequence of pose estimations. Each element is either: <br>- a structured object containing 3D keypoints and their corresponding 2D projections, or <br>- `None` if no person is detected. |
| **Contract** | 1. The output length must exactly match the input length. <br>(i.e. `estimation[i]` always corresponds to `frame[i]`). <br>2. At most one primary subject is assumed per frame. |

Based on that task definition, I made the following foundational design decisions:

| Decision | Rationale |
|----------|-----------|
| **Machine Learning approach** | This task requires learning a highly non-linear mapping from RGB images to 3D joint coordinates. Classical geometric methods typically lack the representational capacity to generalize across diverse poses, viewpoints, and occlusions in real-world video. |
| **Deep Learning approach** | Deep learning provides strong representation learning capabilities for this highly non-linear pixel-to-3D-pose mapping, and generally outperforms classical ML and geometric methods in accuracy, robustness to occlusion, and generalization across viewpoints. [^1] |
| **Stateless per-frame processing** | The task can be formulated as: *«estimate the pose for frame t; repeat for each frame»*. In other words, each frame is processed independently, and there is no explicit temporal dependency between frames. Therefore, recurrent neural networks or other temporal sequence models are not required. |
| **Modular Pipeline Decomposition** | The system is decomposed into the following stages: *localize person -> crop & normalize -> estimate 3D pose -> project coordinates back to the original image*. This separation of concerns provides key advantages:<br> - **Training efficiency**: An end-to-end model that maps raw full-frame RGB directly to 3D pose would need to implicitly learn background suppression, scale normalization, and translation invariance, which makes training significantly more difficult and less efficient.<br> - **Component flexibility**: The detector and pose estimator are decoupled. If future requirements introduce tracking, alternative detectors, or multi-person routing, only the preprocessing stage needs to change, while the core model remains intact. |

[^1]: The trade-off of deep learning models — higher computational cost and memory usage — is accepted here. Since there are no hard constraints on deployment hardware, the additional computational and memory overhead of deep learning models is considered acceptable

## Pipeline Blueprint

With the task constraints and foundational decisions established, I needed to translate them into a concrete execution graph. I organized the flow as a strict sequential pipeline, where each frame passes through person detection, canonical cropping, neural inference, and coordinate reprojection. Frames without valid detections bypass the model and are stored as `None`, ensuring strict 1:1 index alignment between input and output sequences.

```mermaid
flowchart TD
    subgraph Legend[" "]
        direction LR
        L1["Action"]
        L2(["Data"])
        L3{"Decision"}
    end

    Start([Sequence of RGB frames]) --> Fetch[Fetch next frame]
    Fetch --> Detect[Person detection]
    Detect --> Check{Person detected?}
    
    Check -->|No| Skip[Mark frame as None]
    Check -->|Yes| Prep[Crop & normalize frame]
    
    Prep --> Infer[Neural network inference]
    Infer --> Post[Reproject to original image coordinates]
    Post --> Store[Store result]
    Skip --> Store
    
    Store --> Loop{More frames available?}
    Loop -->|Yes| Fetch
    Loop -->|No| Output([Sequence of either pose estimations or None])
```

## Next Steps

With the task formulation and foundational decisions in place, I could have started building the model from scratch. But in production engineering, reinventing the wheel is rarely justified when high-quality, state-of-the-art components already exist. Building a competitive 3D pose estimator from zero would require months of architecture search, dataset curation, and representation learning — effort better spent on pipeline reliability, latency optimization, and maintainability. Rather than training a competitive 3D pose estimator from scratch, I chose to build on an existing pre-trained foundation and adapt it to the project’s requirements. This shifts the effort from model development toward building a reliable, deterministic inference pipeline around a proven model.