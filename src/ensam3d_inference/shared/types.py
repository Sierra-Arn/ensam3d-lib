# Copyright (c) 2026 Ilya Snegov (aka Sierra Arn)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
