# Enhanced SAM 3D Body Library

> *Collection of production-oriented inference pipelines based on [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body) for 3D human body pose and shape estimation.*

## Project Structure

```bash
ensam3d-lib/
├── src/                        # Standard src-layout; each subpackage is a self-contained, 
│                               # production-ready reimplementation of SAM 3D Body
│
├── docs/                       # Technical documentation — each subdirectory 
│                               # mirrors its src/ counterpart with subpackage overview, 
│                               # reference implementation analysis, production-oriented 
│                               # architectural changes, runtime pipeline design, 
│                               # package structure, dependencies, and installation.
│
├── pyproject.toml              # Standard Python project manifest defining 
│                               # build system and package metadata
│
├── pixi.toml                   # Pixi environment configuration for working with the repository
│                               # as a local development project rather than a consumed dependency
│
└── pixi.lock                   # Fully resolved and reproducible dependency lockfile
                                # for the local Pixi development environment
```

## Available Subpackages

### I. `ensam3d_inference`

Production-oriented, inference-only reimplementation of the original research-oriented SAM 3D Body architecture. Designed for high-throughput 3D human pose estimation in video streams, where each frame is assumed to contain at most one primary person.

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

> **Want more details?**  
> For the full technical documentation — including system concepts, reference implementation limitations, production-oriented engineering changes, runtime pipeline diagrams, data contracts, dependencies, and installation — see [the package-specific documentation](./docs/ensam3d_inference/README.md).

## Installation

You can install the package directly from the GitHub repository:

```bash
# Install via pip
pip install "ensam3d-lib @ git+https://github.com/Sierra-Arn/ensam3d-lib.git"

# Install via pixi
pixi add --pypi "ensam3d-lib @ git+https://github.com/Sierra-Arn/ensam3d-lib.git"
```

> **Note:**  
> This package is intentionally designed as a lightweight distribution containing only source code and package metadata [^1]. Runtime ML dependencies are not included and must be installed separately; the full dependency list is provided in the corresponding documentation section.

[^1]: Modern ML environments — especially those built around PyTorch, CUDA, Conda, and compiled libraries — are often highly sensitive to dependency resolution order and binary compatibility. Automatically enforcing installation of a predefined PyTorch/CUDA runtime via PyPI can easily destabilize an otherwise working environment by introducing incompatible CUDA builds, conflicting with Conda-managed binaries, pulling conflicting transitive dependencies, or causing ABI mismatches across compiled extensions. For this reason, the package does not ship with ML runtime dependencies and is designed to be safely integrated into existing environments without modifying their runtime stack.

## Installation for Development

### I. Prerequisites

- [Pixi](https://pixi.sh/latest/) package manager.
- GNU/Linux-based system on `x86_64` architecture.
- NVIDIA GPU with a driver that supports CUDA `>= 12.8`.

> **Note: on these prerequisites**  
> These are not strict requirements but describe the environment used for development. The package can be set up in alternative environments with different package managers, operating systems, or GPU configurations if needed.

> **Note: on CUDA versions**  
> The `>= 12.8` figure is a *driver* requirement, enforced through pixi's `system-requirements`. It was chosen because 12.8 was the default CUDA build shipped by PyTorch at the time development started (a plain `pip install torch` pulled the 12.8 build back then).

> **Note: the benchmarking and profiling scripts report**  
> `CUDA Version: 12.6`. This is expected and not a mismatch: that value comes from `torch.version.cuda`, i.e. the CUDA version the PyTorch binary was *compiled against*, which is independent of the newer CUDA toolkit resolved into the environment. CUDA minor-version compatibility lets a 12.6 build run on any 12.x driver with the same major version.

### II. Setup

1. **Clone the repository**

    ```bash
    git clone git@github.com:Sierra-Arn/ensam3d-lib.git
    cd ensam3d-lib
    ```

2. **Install dependencies**

    ```bash
    pixi install
    ```

3. **Activate environment**

    ```bash
    pixi shell
    ```

### III. Development workflow

Once the environment is activated, you can:
- run inference pipelines,
- modify or extend modules,
- execute benchmarks,
- iterate on experiments,

— all dependencies, Python paths, and environment variables are already configured by Pixi.

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).