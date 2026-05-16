# src/ensam3d_inference/examples/visualization.py
"""
Single-image pose estimation visualizer for the Enhanced SAM 3D Body inference pipeline.

Runs the full end-to-end pipeline on one image and overlays the requested 
annotations — bounding box, keypoints, and skeleton links — using matplotlib. 
If an export path is provided the annotated image is saved to disk; otherwise
it is displayed interactively.

Usage
-----
python -m ensam3d_inference.examples.visualization \
    --image_path example.png \
    --model_path sam-3d-body-vith \
    --show_bbox \
    --show_keypoints \
    --show_skeleton

python -m ensam3d_inference.examples.visualization \
    --image_path example.png \
    --model_path sam-3d-body-vith \
    --show_keypoints \
    --show_skeleton \
    --image_export_path result.png
"""
import argparse
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from ensam3d_inference import (
    Pipeline, 
    PreprocessorInput, 
    FramePoseResult,
    DeviceType
)
from ensam3d_inference.examples.keypoints import KEYPOINTS, SKELETON


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose estimation visualizer")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--show_bbox", action="store_true")
    parser.add_argument("--show_keypoints", action="store_true")
    parser.add_argument("--show_skeleton", action="store_true")
    parser.add_argument("--image_export_path", type=str, default=None)
    return parser.parse_args()


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        model_path=args.model_path,
        detector_device=DeviceType.CUDA,
        model_device=DeviceType.CUDA,
    )


def _read_image(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _draw_bbox(ax: plt.Axes, output: FramePoseResult) -> None:
    x1, y1, x2, y2 = output.detection.coords.cpu().numpy()
    w = x2 - x1
    h = y2 - y1
    rect = patches.Rectangle(
        (x1, y1), w, h,
        linewidth=2,
        edgecolor=(1.0, 1.0, 0.0),
        facecolor="none",
    )
    ax.add_patch(rect)


def _draw_keypoints(ax: plt.Axes, output: FramePoseResult) -> None:
    kps = output.pose.pred_keypoints_2d[0].cpu().numpy()
    for idx, (x, y) in enumerate(kps):
        color = tuple(c / 255.0 for c in KEYPOINTS[idx].color)
        ax.scatter(x, y, c=[color], s=20, zorder=3)


def _draw_skeleton(ax: plt.Axes, output: FramePoseResult) -> None:
    kps = output.pose.pred_keypoints_2d[0].cpu().numpy()
    for link in SKELETON:
        x_start, y_start = kps[link.start]
        x_end, y_end = kps[link.end]
        color = tuple(c / 255.0 for c in link.color)
        ax.plot(
            [x_start, x_end],
            [y_start, y_end],
            c=color,
            linewidth=1.5,
            zorder=2,
        )


def _visualize(
    image: np.ndarray,
    output: FramePoseResult,
    show_bbox: bool,
    show_keypoints: bool,
    show_skeleton: bool,
    image_export_path: str | None,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")

    if show_bbox:
        _draw_bbox(ax, output)
    if show_skeleton:
        _draw_skeleton(ax, output)
    if show_keypoints:
        _draw_keypoints(ax, output)

    plt.tight_layout()

    if image_export_path is not None:
        plt.savefig(image_export_path, bbox_inches="tight", dpi=150)
        print(f"✅ Saved to: {image_export_path}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    args = _parse_args()
    pipeline = _build_pipeline(args)
    image = _read_image(args.image_path)

    results = pipeline(PreprocessorInput(imgs=[image]))
    result = results[0]

    if result is None:
        print("No person detected in the image.")
        return

    _visualize(
        image=image,
        output=result,
        show_bbox=args.show_bbox,
        show_keypoints=args.show_keypoints,
        show_skeleton=args.show_skeleton,
        image_export_path=args.image_export_path,
    )


if __name__ == "__main__":
    main()