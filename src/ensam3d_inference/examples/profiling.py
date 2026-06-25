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

# src/ensam3d_inference/examples/profiling.py
"""
CUDA and CPU profiling for the Enhanced SAM 3D Body inference pipeline.

Runs the pipeline on a single batch from the video and profiles CPU and CUDA
activity. If an export path is provided, saves the full Chrome trace for
analysis in chrome://tracing or Perfetto. Otherwise prints the top 20
operations sorted by total CUDA time.

Usage
-----
python -m ensam3d_inference.examples.profiling \
    --video_path example.mp4 \
    --model_path sam-3d-body-vith \
    --batch_size 30

python -m ensam3d_inference.examples.profiling \
    --video_path example.mp4 \
    --model_path sam-3d-body-vith \
    --batch_size 30 \
    --prof_export_path trace.json
"""
import argparse
from pathlib import Path
from typing import NamedTuple
import cv2
import torch
from ensam3d_inference import Pipeline, PreprocessorInput, DeviceType


class SystemInfo(NamedTuple):
    """
    Container for the hardware and software environment of the profiling run.

    Attributes
    ----------
    cpu_name : str
        Human-readable model name of the host CPU, or a fallback string when
        it cannot be determined.
    gpu_name : str
        Name of the active CUDA device.
    torch_version : str
        Version string of the installed PyTorch build.
    cuda_version : str
        Version string of the CUDA toolkit PyTorch was built against.
    """

    cpu_name: str
    gpu_name: str
    torch_version: str
    cuda_version: str


def _get_system_info() -> SystemInfo:
    """
    Collect hardware and software details of the current environment.

    The CPU model name is read from /proc/cpuinfo, which is available on
    Linux systems. When the file is missing or no model name entry is present,
    a fallback value is returned instead. The GPU name and CUDA version are
    queried through PyTorch and therefore describe the active CUDA device.

    Returns
    -------
    SystemInfo
        The CPU name, GPU name, PyTorch version, and CUDA version.
    """
    cpu_name = "Unknown CPU"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_name = line.split(":")[1].strip()
                    break
    except FileNotFoundError:
        pass

    return SystemInfo(
        cpu_name=cpu_name,
        gpu_name=torch.cuda.get_device_name(0),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the profiling script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the video path, the model path, the batch
        size, and the optional Chrome trace export path.
    """
    parser = argparse.ArgumentParser(description="Pipeline profiling")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--prof_export_path", type=str, default=None)
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


def _warmup(pipeline: Pipeline, video_path: str) -> None:
    """
    Run a few frames through the pipeline to eliminate one-time startup costs.

    Reads up to five frames from the video and executes a single forward pass.
    This triggers initial CUDA memory allocation, cuDNN kernel autotuning, and
    GPU clock ramp-up, so that the profiled run is not contaminated by
    initialization overhead.

    Parameters
    ----------
    pipeline : Pipeline
        The inference pipeline to warm up.
    video_path : str
        Path to the video file used to source warm-up frames.

    Raises
    ------
    RuntimeError
        If the video cannot be opened or contains no readable frames.
    """
    print("🔥 Warming up...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while len(frames) < 5:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames in video: {video_path}")

    pipeline(PreprocessorInput(imgs=frames))


def _read_batch(video_path: str, batch_size: int) -> list:
    """
    Read a single batch of RGB frames from the start of a video file.

    Frames are read sequentially from the beginning of the video, converted
    from BGR to RGB, and collected until batch_size frames are gathered or the
    video ends. The returned batch may be smaller than batch_size when the
    video contains fewer frames.

    Parameters
    ----------
    video_path : str
        Path to the video file to decode.
    batch_size : int
        Maximum number of frames to read into the batch.

    Returns
    -------
    list of numpy.ndarray
        The batch of RGB frames.

    Raises
    ------
    RuntimeError
        If the video cannot be opened or contains no readable frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    while len(frames) < batch_size:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames in video: {video_path}")

    return frames


def _print_config(
    video_path: str,
    batch_size: int,
    frames_in_batch: int,
) -> None:
    """
    Print the environment and video configuration of the profiling run.

    Reports the host environment, the source video resolution, and how many
    frames are actually profiled. Profiling is intentionally limited to a
    single batch, so the printed frame count reflects only that batch and not
    the full video.

    Parameters
    ----------
    video_path : str
        Path to the profiled video file.
    batch_size : int
        Requested maximum number of frames per batch.
    frames_in_batch : int
        Number of frames actually present in the profiled batch, which may be
        smaller than batch_size for short videos.
    """
    info = _get_system_info()
    file_name = Path(video_path).name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    print("\n📊 ===== Configuration =====")
    print(f"File Name:           {file_name}")
    print(f"Video Resolution:    {width} × {height}")
    print(f"CPU:                 {info.cpu_name}")
    print(f"GPU:                 {info.gpu_name}")
    print(f"PyTorch Version:     {info.torch_version}")
    print(f"CUDA Version:        {info.cuda_version}")
    print(f"Batch Size:          {batch_size}")
    print(f"Frames in Batch:     {frames_in_batch}")
    print("================================")
    print("ℹ️  Profiling runs on a single batch only.\n")


def _run_profiling(
    pipeline: Pipeline,
    batch: list,
    prof_export_path: str | None,
) -> None:
    """
    Profile a single forward pass of the pipeline over one batch.

    Records both CPU and CUDA activity for one invocation of the pipeline.
    When prof_export_path is provided, the full Chrome trace is written to that
    path for inspection in chrome://tracing or Perfetto. Otherwise the top 20
    operations by total CUDA time are printed to the console.

    Parameters
    ----------
    pipeline : Pipeline
        The inference pipeline to profile.
    batch : list of numpy.ndarray
        The batch of RGB frames to feed through the pipeline.
    prof_export_path : str or None
        Destination path for the Chrome trace, or None to print a summary
        table instead.
    """
    print("🔬 Profiling...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        pipeline(PreprocessorInput(imgs=batch))

    if prof_export_path is not None:
        prof.export_chrome_trace(prof_export_path)
        print(f"✅ Trace saved to: {prof_export_path}")
    else:
        print("\n🔍 ===== Top 20 CUDA Operations =====")
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        ))
        print("================================\n")


def main() -> None:
    """
    Entry point wiring together argument parsing, warm-up, batch reading,
    configuration reporting, and profiling.
    """
    args = _parse_args()
    pipeline = _build_pipeline(args)
    _warmup(pipeline=pipeline, video_path=args.video_path)
    batch = _read_batch(video_path=args.video_path, batch_size=args.batch_size)
    _print_config(
        video_path=args.video_path,
        batch_size=args.batch_size,
        frames_in_batch=len(batch),
    )
    _run_profiling(
        pipeline=pipeline,
        batch=batch,
        prof_export_path=args.prof_export_path,
    )


if __name__ == "__main__":
    main()