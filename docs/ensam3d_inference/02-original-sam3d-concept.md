# II. **Original SAM 3D Body: Conceptual Overview**

> *This document describes the high-level conceptual architecture of the original SAM 3D Body.*

## **Reference Paper**

> **SAM 3D Body: Robust Full-Body Human Mesh Recovery**  
> by Xitong Yang, Devansh Kukreja, Don Pinkus, Anushka Sagar, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Matt > Feiszli, Jitendra Malik, Piotr Dollár, and Kris Kitani
> 
> [arXiv:2602.15989](https://arxiv.org/abs/2602.15989?utm_source=chatgpt.com)

For implementation details, please refer to the original publication.

## **Architecture at a Glance**

```mermaid
flowchart TD
    subgraph Legend[" "]
        direction LR
        L1["Action"]
        L2(["Data"])
    end

    FullBodyImg([Full-body RGB Crop]) --> ImEnc[Image Encoder]
    HandImgs([Hand RGB Crops]) --> ImEnc

    Cam([Camera Intrinsics + Affine]) --> CamEnc[Camera Encoder]
    Prompt([2D Keypoints / Masks]) --> PromptEnc[Prompt Encoder]

    ImEnc --> FullFeats([Full-body Image Features])
    ImEnc --> HandFeats([Hand Image Features])

    CamEnc & PromptEnc --> TokenBuilder[Query Builder]
    TokenBuilder --> Tokens([Query Token Sequence])

    Tokens & FullFeats --> BodyDec[Body Decoder]
    Tokens & HandFeats --> HandDec[Hand Decoder]

    BodyDec --> MHRHead[MHR Head]
    BodyDec --> CamHead[Camera Head]
    HandDec --> HandRef([Enhanced Hand Pose])

    MHRHead --> BodyOut([Body MHR Parameters])
    CamHead --> CamParams([Weak-Perspective Camera])
    BodyOut & CamParams --> Proj[Perspective Projection]
    BodyOut & HandRef --> Merge[Prompt-guided Merge]
    Proj & Merge --> Final([Full-Body Mesh + 2D Keypoints])
```

## **Component Logic Breakdown**

| Module | Responsibility | Implementation | Key Properties |
| -------|----------------|----------------|----------------|
| **Image Encoder** | Extract dense visual features from full-body and hand crops | Shared Vision Transformer backbone producing patch-aligned embeddings | Single shared weights; outputs used by both decoders via cross-attention |
| **Camera Encoder** | Inject geometric camera context into visual features | Fourier encoding of intrinsics + 1x1 convolution fusion | Non-learnable positional encoding; enables explicit perspective reasoning |
| **Prompt Encoder** | Encode optional user guidance into model-compatible embeddings | Keypoints -> positional tokens; masks -> convolutional feature injection | Enables interactive / guided inference |
| **Query Builder** | Assemble decoder input token sequence | Concatenation of learnable tokens, prompt embeddings, and auxiliary tokens | Fully flexible token composition; controls modality participation |
| **Body Decoder** | Predict full-body 3D human representation | Multi-layer cross-attention transformer (6+ layers) | Iterative refinement; supports auxiliary keypoint supervision |
| **Hand Decoder** | Predict high-resolution hand articulation | Separate cross-attention transformer on hand crops | Independent training stream; avoids body-hand gradient interference |
| **MHR Head** | Regress human rig parameters | Linear regression head from primary token | Outputs structured rig (pose, shape, scale) |
| **Camera Head** | Estimate weak-perspective camera parameters | Linear projection head from pose token | Produces 2D projection parameters for rendering |
| **Perspective Projection** | Map 3D outputs to 2D image space | Deterministic geometric projection | Non-learnable; preserves differentiability for training |
| **Prompt-guided Merge** | Fuse hand and body predictions | Feedback loop from hand outputs into body decoder prompts | Improves wrist alignment; removes kinematic discontinuities |