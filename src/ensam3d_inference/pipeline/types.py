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

# src/ensam3d_inference/pipeline/types.py
from typing import NamedTuple
from ..preprocessor.detector.types import Detection
from ..core.pose_estimation.types import PoseEstimatorOutput


class FramePoseResult(NamedTuple):
    """
    Container for the detector output with the corresponding pose 
    estimation result for a single successfully processed frame.

    Attributes
    ----------
    detection : Detection
        Detector output containing the selected person bounding box and
        its confidence score.
    pose : PoseEstimatorOutput
        Pose estimation outputs produced from the cropped detection
        region, including pose parameters, mesh predictions, and camera
        outputs.
    """

    detection: Detection
    pose: PoseEstimatorOutput


PipelineOutput = list[FramePoseResult | None]
"""
Inference results for a batch of input frames represented as a list with one entry per input
frame aligned with the corresponding PreprocessorInput, where each entry is either a
FramePoseResult containing the detector output and corresponding pose estimation result,
or None if no person was detected in the corresponding frame.
"""