# src/ensam3d_inference/pipeline/types.py
from typing import NamedTuple
from ..preprocessor.detector.types import Detection
from ..core.pose_estimation.types import PoseEstimatorOutput


class PipelineOutput(NamedTuple):
    """
    Container for the output of the end-to-end human pose pipeline.

    Combines the detector output with the corresponding pose estimation
    result for a single successfully processed frame.

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