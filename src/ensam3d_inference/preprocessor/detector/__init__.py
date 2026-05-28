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

# src/ensam3d_inference/preprocessor/detector/__init__.py
import torch
from ultralytics import YOLO
from .types import (
    DetectorInput, 
    BBoxTensor, 
    ConfidenceTensor, 
    Detection,
    DetectorOutput
)
from ...shared import DeviceType, config


class Detector:
    """
    Human detector based on a YOLO model.

    Runs inference on a batch of RGB frames and returns the single
    highest-confidence bounding box per frame. Frames where no person
    is detected above the confidence threshold produce a None entry in
    the output, preserving alignment with the input list.

    Parameters
    ----------
    model_path : str, optional
        Model name or local file path. If not found locally, the model
        weights are downloaded automatically. Default is "yolo26n.pt".
    device : DeviceType, optional
        Device used for inference. Default is DeviceType.CUDA.

    Attributes
    ----------
    detector : YOLO
        Underlying YOLO model instance.
    """

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        device: DeviceType = DeviceType.CUDA,
    ) -> None:
        self.detector = YOLO(model_path)
        self.detector.to(device)

    def __call__(self, imgs: DetectorInput) -> DetectorOutput:
        """
        Run human detection on a batch of images and return the highest-confidence
        bounding box per frame.

        Parameters
        ----------
        imgs : DetectorInput
            Batch of RGB frames to run detection on.

        Returns
        -------
        DetectorOutput
            One entry per input frame aligned with imgs; None where no person
            was detected above the confidence threshold.
        """
        results = self.detector.predict(
            imgs,
            conf=config.detector_conf_thr,
            iou=config.detector_nms_thr,
            classes=[config.detector_cat_id],
            verbose=False,
        )

        output: DetectorOutput = []
        for result in results:

            if len(result.boxes) == 0:
                output.append(None)
                continue

            coords = result.boxes.xyxy   # shape: [N, 4]
            confs = result.boxes.conf    # shape: [N]

            best_idx = torch.argmax(confs)

            best_coords: BBoxTensor = coords[best_idx]
            best_conf: ConfidenceTensor = confs[best_idx]

            output.append(Detection(
                coords=best_coords,
                confidence=best_conf,
            ))

        return output