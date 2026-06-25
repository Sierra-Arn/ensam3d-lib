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
from pathlib import Path
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
    """
    Parse command-line arguments for the visualizer script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the image path, the model path, the three
        annotation toggles (bounding box, keypoints, skeleton), and the
        optional image export path.
    """
    parser = argparse.ArgumentParser(description="Pose estimation visualizer")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--show_bbox", action="store_true")
    parser.add_argument("--show_keypoints", action="store_true")
    parser.add_argument("--show_skeleton", action="store_true")
    parser.add_argument("--image_export_path", type=str, default=None)
    return parser.parse_args()


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    """
    Construct the inference pipeline configured to run on CUDA devices.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments providing the model path.

    Returns
    -------
    Pipeline
        A pipeline instance with both the detector and the model placed on
        the CUDA device.
    """
    return Pipeline(
        model_path=args.model_path,
        detector_device=DeviceType.CUDA,
        model_device=DeviceType.CUDA,
    )


def _read_image(image_path: str) -> np.ndarray:
    """
    Read an image from disk and convert it to RGB.

    The image is decoded with OpenCV, which returns frames in BGR order, and
    then converted to RGB so that the colors match matplotlib's expectations.

    Parameters
    ----------
    image_path : str
        Path to the image file to read.

    Returns
    -------
    numpy.ndarray
        The decoded image as an RGB array.

    Raises
    ------
    RuntimeError
        If the image cannot be read from the given path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _draw_bbox(ax: plt.Axes, output: FramePoseResult) -> None:
    """
    Draw the detection bounding box on the given axes.

    The box coordinates are read from the detection result as top-left and
    bottom-right corners, converted to a width-height rectangle, and drawn as
    an unfilled yellow outline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the bounding box on.
    output : FramePoseResult
        The per-frame pose result carrying the detection coordinates.
    """
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
    """
    Draw the predicted 2D keypoints as colored points on the given axes.

    Each keypoint is rendered as a scatter marker using the per-keypoint color
    defined in the KEYPOINTS table, with the color converted from 0-255 to the
    0-1 range matplotlib expects.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the keypoints on.
    output : FramePoseResult
        The per-frame pose result carrying the predicted 2D keypoints.
    """
    kps = output.pose.pred_keypoints_2d[0].cpu().numpy()
    for idx, (x, y) in enumerate(kps):
        color = tuple(c / 255.0 for c in KEYPOINTS[idx].color)
        ax.scatter(x, y, c=[color], s=20, zorder=3)


def _draw_skeleton(ax: plt.Axes, output: FramePoseResult) -> None:
    """
    Draw the skeleton links connecting keypoints on the given axes.

    Each link in the SKELETON table connects two keypoints and is drawn as a
    line segment using the link's own color, converted from 0-255 to the 0-1
    range matplotlib expects.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to draw the skeleton on.
    output : FramePoseResult
        The per-frame pose result carrying the predicted 2D keypoints.
    """
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
    """
    Render the image with the selected annotations, then save or display it.

    The annotations are drawn in a fixed order so that keypoints sit on top of
    the skeleton, which in turn sits on top of the bounding box. When an export
    path is given the figure is written to disk and its absolute location is
    printed; otherwise the figure is shown interactively.

    Parameters
    ----------
    image : numpy.ndarray
        The RGB image to use as the background.
    output : FramePoseResult
        The per-frame pose result providing detection and keypoint data.
    show_bbox : bool
        Whether to draw the detection bounding box.
    show_keypoints : bool
        Whether to draw the predicted keypoints.
    show_skeleton : bool
        Whether to draw the skeleton links.
    image_export_path : str or None
        Destination path for the annotated image, or None to display it
        interactively.
    """
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
        print(f"✅ Saved to: {Path(image_export_path).resolve()}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    """
    Entry point wiring together argument parsing, image reading, pipeline
    inference, and visualization.

    If the pipeline detects no person in the image, a message is printed and
    no figure is produced.
    """
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