# `ensam3d_inference/`

> *This directory holds the full technical documentation for the `ensam3d_inference` package. This file serves as its entry point: it gives a high-level overview of the package backed by a performance benchmark, then points to the file covering each layer of the system, from core concept to installation.*

## I. Package Overview

This package is a production-oriented, inference-only reimplementation of the original research-oriented SAM 3D Body architecture. It is designed for high-throughput 3D human pose estimation in video streams, where each frame is assumed to contain at most one primary person.

Performance characteristics of the pipeline are summarized below using a representative 4K video benchmark.

**Configuration**
| | |
|------------------|------------------------------------------------------------------------------------------------------------------|
| Benchmark Video  | [Man with prosthetic leg jogging, Pexels](https://www.pexels.com/video/man-with-prosthetic-leg-jogging-8344814/) |
| Video Resolution | 3840 × 2160 (4K)                                                                                                 |
| Video Duration   | 24.32 sec                                                                                                        |
| Video FPS        | 25.00                                                                                                            |
| Total Frames     | 608                                                                                                              |
| CPU              | AMD Ryzen 7 5800H with Radeon Graphics                                                                           |
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

To benchmark the pipeline on your own hardware, run [ensam3d_inference.examples.benchmarking](../../src/ensam3d_inference/examples/benchmarking.py).

## II. Documentation Overview

This section provides a structured guide to the documentation layout of the `ensam3d_inference` package. Each file focuses on a specific layer of abstraction, ranging from high-level conceptual design to low-level installation and runtime setup.

- To understand the core idea and task formulation of the system, see  
    [01-ensam3d-concept.md](01-ensam3d-concept.md).

- To review the original SAM 3D Body architecture used as the foundation of this project, see  
    [02-sam3d-concept.md](02-sam3d-concept.md).

- To understand the practical issues encountered when adapting the reference implementation for production inference, see  
    [03-sam3d-issues.md](03-sam3d-issues.md).

- To review the architectural and engineering changes introduced in the production pipeline, see  
    [04-enhancements.md](04-enhancements.md).

- To study the runtime architecture and execution graph, see  
    [05-system-architecture.md](05-system-architecture.md).

- To understand the repository layout and package structure, see  
    [06-package-structure.md](06-package-structure.md).

- To review the dependency stack and required external assets, see  
    [07-dependencies.md](07-dependencies.md).

- To install and run the project, see  
    [08-installation-guide.md](08-installation-guide.md).