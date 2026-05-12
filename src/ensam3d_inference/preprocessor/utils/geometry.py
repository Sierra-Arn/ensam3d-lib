# src/ensam3d_inference/preprocessor/utils/geometry.py
import cv2
import numpy as np
from .types import AffineMatrix, WarpCropInput, WarpCropOutput, Vec2
from ...shared import config


def _fix_aspect_ratio(
    bbox_scale: Vec2,
    aspect_ratio: float,
) -> Vec2:
    """
    Reshape a scale vector to a fixed width-over-height ratio.

    Parameters
    ----------
    bbox_scale : Vec2
        Scale (w, h) in pixels.
    aspect_ratio : float
        Target ratio w / h.

    Returns
    -------
    Vec2
        Adjusted scale (w, h) in pixels.
    """
    row = bbox_scale[None, :]
    w, h = np.hsplit(row, [1])
    out = np.where(
        w > h * aspect_ratio,
        np.hstack([w, w / aspect_ratio]),
        np.hstack([h * aspect_ratio, h]),
    )
    return out[0]


def _get_warp_matrix(
    center: Vec2,
    scale: Vec2,
    rot: float = 0.0,
    shift: tuple[float, float] = (0.0, 0.0)
) -> AffineMatrix:
    """
    Compute an affine matrix mapping the source bbox region to the warp canvas.

    Parameters
    ----------
    center : Vec2
        Bounding box centre (x, y) in source image pixels.
    scale : Vec2
        Bounding box scale (w, h) in source image pixels.
    rot : float, optional
        Rotation angle in degrees. Default is 0.0.
    shift : tuple of float, optional
        Shift ratio relative to scale applied to the source centre.
        Default is (0.0, 0.0).

    Returns
    -------
    AffineMatrix
        Affine transform matrix, shape (2, 3), following the OpenCV
        cv2.warpAffine convention.
    """
    def _rotate_point(pt: Vec2, angle_rad: float) -> Vec2:
        sn, cs = np.sin(angle_rad), np.cos(angle_rad)
        return np.array([[cs, -sn], [sn, cs]]) @ pt

    def _get_3rd_point(a: Vec2, b: Vec2) -> Vec2:
        direction = a - b
        return b + np.r_[-direction[1], direction[0]]

    shift_arr = np.array(
        shift, 
        dtype=config.numpy_dtype
    )
    src_w = float(scale[0])
    dst_w, dst_h = config.backbone_input_size

    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(
        np.array(
            [0.0, src_w * -0.5], 
            dtype=config.numpy_dtype
        ), 
        rot_rad
    )
    dst_dir = np.array(
        [0.0, dst_w * -0.5], 
        dtype=config.numpy_dtype
    )

    src = np.zeros(
        (3, 2), 
        dtype=config.numpy_dtype
    )
    src[0] = center + scale * shift_arr
    src[1] = center + src_dir + scale * shift_arr
    src[2] = _get_3rd_point(src[0], src[1])

    dst = np.zeros(
        (3, 2), 
        dtype=config.numpy_dtype
    )
    dst[0] = [dst_w * 0.5, dst_h * 0.5]
    dst[1] = np.array(
        [dst_w * 0.5, dst_h * 0.5], 
        dtype=config.numpy_dtype
    ) + dst_dir
    dst[2] = _get_3rd_point(dst[0], dst[1])

    return cv2.getAffineTransform(src, dst)


def warp_affine_crop(
    request: WarpCropInput,
) -> WarpCropOutput:
    """
    Warp an image region around a bounding box to a canonical resolution.

    Accepts a structured input container, applies intermediate aspect ratio 
    alignment followed by a final adjustment to the target output size, then 
    computes the affine transformation and applies OpenCV warp. The output 
    includes the warped image and all geometric metadata required for keypoint 
    reprojection into the original image space.

    Parameters
    ----------
    request : WarpCropInput
        Structured container with source RGB image (uint8, HWC), bounding box 
        center (x, y) in full-image pixels, and padded bounding box scale (w, h).

    Returns
    -------
    WarpCropOutput
        Warped crop in RGB uint8 format and geometric metadata including the 
        original center, final adjusted scale (after aspect-ratio alignment), 
        and the affine transformation matrix used for warping.
    """
    mid_scale = _fix_aspect_ratio(
        bbox_scale=request.bbox_scale,
        aspect_ratio=config.intermediate_aspect_ratio,
    )
    adj_scale = _fix_aspect_ratio(
        bbox_scale=mid_scale,
        aspect_ratio=config.backbone_input_size[0] / config.backbone_input_size[1],
    )
    warp_mat = _get_warp_matrix(
        center=request.bbox_center,
        scale=adj_scale
    )
    img_warped = cv2.warpAffine(
        request.img, warp_mat, config.backbone_input_size, flags=cv2.INTER_LINEAR
    )
    return WarpCropOutput(
        img=img_warped,
        bbox_center=request.bbox_center,
        bbox_scale=adj_scale,
        affine_trans=warp_mat,
    )