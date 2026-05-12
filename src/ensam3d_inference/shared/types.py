# src/ensam3d_inference/shared/types.py
from enum import StrEnum


class DeviceType(StrEnum):
    """
    Computation device target for model inference and tensor operations.

    Attributes
    ----------
    CPU : DeviceType
        Executes computations on the central processing unit.
    CUDA : DeviceType
        Executes computations on an NVIDIA GPU using the CUDA runtime.
    """

    CPU = "cpu"
    CUDA = "cuda"
