# src/ensam3d_inference/preprocessor/__init__.py
import numpy as np
import torch
from .detector import Detector
from .types import PreprocessorInput, PreprocessorOutput
from .utils import decompose_bbox, warp_affine_crop, transform_images, build_intrinsics
from .utils.types import WarpCropInput
from ..shared import DeviceType, config


class Preprocessor:
    """
    Human detection followed by canonical cropping for a batch of RGB frames.

    Runs YOLO-based person detection on every input frame, filters frames
    with valid detections, and applies the canonical cropping pipeline to
    the surviving samples. The resulting tensors are moved to the target
    model device for downstream feature extraction or pose estimation.

    Frames where no person is detected are preserved as None entries in
    the detection output to maintain alignment with the original input
    frame sequence. Cropped tensors and geometric metadata are produced
    only for frames listed in valid_indices.

    Parameters
    ----------
    detector_device : DeviceType, optional
        Device used for YOLO inference. Default is DeviceType.CUDA.
    model_device : DeviceType, optional
        Target device for cropped tensors and geometric metadata returned
        by the preprocessing pipeline. Typically matches the downstream
        backbone or pose estimation model device.
        Default is DeviceType.CUDA.
    detector_model_path : str, optional
        YOLO model name or local file path. If the weights are not found
        locally, they are downloaded automatically.
        Default is "yolo26n.pt".

    Attributes
    ----------
    detector : Detector
        Human detector applied independently to each input frame.
    model_device : DeviceType
        Target device used for tensors returned by the preprocessing
        pipeline.
    """

    def __init__(
        self,
        detector_device: DeviceType = DeviceType.CUDA,
        model_device: DeviceType = DeviceType.CUDA,
        detector_model_path: str = "yolo26n.pt",
    ) -> None:
        self.detector = Detector(model_path=detector_model_path, device=detector_device)
        self.model_device = model_device

    def __call__(self, request: PreprocessorInput) -> PreprocessorOutput | None:
        """
        Detect persons, crop valid detections, and return tensors on the target device.

        Parameters
        ----------
        request : PreprocessorInput
            Batch of RGB frames and optional shared camera intrinsics.

        Returns
        -------
        PreprocessorOutput or None
            Packed crops and geometric metadata on model_device, or None
            if the detector found zero valid persons across the entire batch.
        """
        dets = self.detector(request.imgs)
        valid_indices = [i for i, d in enumerate(dets) if d is not None]

        if not valid_indices:
            return None

        cropped_frames = []
        centers = []
        scales = []
        affines = []

        for idx in valid_indices:
            frame = request.imgs[idx]
            bbox = dets[idx].coords

            center, scale = decompose_bbox(bbox)
            crop_result = warp_affine_crop(
                WarpCropInput(
                    img=frame,
                    bbox_center=center,
                    bbox_scale=scale
                )
            )

            cropped_frames.append(crop_result.img)
            centers.append(crop_result.bbox_center)
            scales.append(crop_result.bbox_scale)
            affines.append(crop_result.affine_trans)

        img = transform_images(cropped_frames)
        b = len(valid_indices)
        first_frame = request.imgs[valid_indices[0]]
        ori_wh = (first_frame.shape[1], first_frame.shape[0])

        bbox_center = torch.as_tensor(np.stack(centers), dtype=config.core_input_dtype)
        bbox_scale = torch.as_tensor(np.stack(scales), dtype=config.core_input_dtype)
        affine_trans = torch.as_tensor(np.stack(affines), dtype=config.core_input_dtype)

        ori_img_size = torch.tensor([ori_wh] * b, dtype=config.core_input_dtype)
        crop_img_size = torch.tensor(
            [config.backbone_input_size] * b, 
            dtype=config.core_input_dtype
        )

        cam_int = ( 
            torch.from_numpy(request.cam_int).unsqueeze(0).float()
            if request.cam_int is not None
            else build_intrinsics(ori_wh)
        ).expand(b, -1, -1).contiguous()

        return PreprocessorOutput(
            detections=dets,
            valid_indices=valid_indices,
            img=img.to(self.model_device),
            bbox_center=bbox_center.to(self.model_device),
            bbox_scale=bbox_scale.to(self.model_device),
            affine_trans=affine_trans.to(self.model_device),
            ori_img_size=ori_img_size.to(self.model_device),
            crop_img_size=crop_img_size.to(self.model_device),
            cam_int=cam_int.to(self.model_device),
        )