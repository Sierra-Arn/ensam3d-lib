# **`ensam3d_inference/`**

> *This directory contains the full technical documentation for the `ensam3d_inference` package, covering system architecture, runtime pipeline, implementation decisions, dependencies, installation, and usage. The documentation is structured to provide a clear, reproducible, and engineering-level understanding of the inference flow.*

## I. **Package Overview**

This package is a production-oriented, inference-only reimplementation of the original research-oriented SAM 3D Body architecture. It is designed for high-throughput 3D human pose estimation in video streams, where each frame is assumed to contain at most one primary person.

Performance characteristics of the pipeline are summarized below using a representative 4K video benchmark.

**Configuration**
| | |
|------------------|------------------------------------------------------------------------------------------------------------------|
| Benchmark Video  | [Man with prosthetic leg jogging, Pexels](https://www.pexels.com/video/man-with-prosthetic-leg-jogging-8344814/) |
| Video Resolution | 3840 × 2160 (4K)                                                                                                 |
| Video Duration   | 24.32 sec (608 frames)                                                                                           |
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

To reproduce this benchmark on your hardware, refer to [ensam3d_inference.examples.benchmarking](../../src/ensam3d_inference/examples/benchmarking.py).

## II. **Documentation Overview**

This section provides a structured guide to the documentation layout of the `ensam3d_inference` package. Each file focuses on a specific layer of abstraction, ranging from high-level conceptual design to low-level installation and runtime setup.

- If you want to understand the **core idea and task formulation of the system**, refer to  
    [01-ensam3d-concept.md](01-ensam3d-concept.md)

- If you want to review the **original SAM 3D Body architecture that the project is based on**, refer to  
    [02-original-sam3d-concept.md](02-original-sam3d-concept.md)

- If you want to understand the **architectural modifications applied to the original model**, refer to  
    [03-architectural-changes.md](03-architectural-changes.md)

- If you want to understand the **engineering-level design decisions and trade-offs**, refer to  
    [04-engineering-decisions.md](04-engineering-decisions.md)

- If you want to study the **runtime system architecture and execution graph**, refer to  
    [05-system-architecture.md](05-system-architecture.md)

- If you want to understand the **codebase organization and package structure**, refer to  
    [06-package-structure.md](06-package-structure.md)

- If you want to review the **dependency stack and external requirements**, refer to  
    [07-dependencies.md](07-dependencies.md)

- If you want to **install, configure, and run the project**, refer to  
    [08-installation-guide.md](08-installation-guide.md)