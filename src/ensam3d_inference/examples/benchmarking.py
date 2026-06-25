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

# src/ensam3d_inference/examples/benchmarking.py
"""
Throughput and latency benchmark for the Enhanced SAM 3D Body inference pipeline.

Reads a video file in batches, runs the full pipeline on each batch, and
reports total time, latency per frame, throughput in FPS, and peak GPU memory allocation.

Usage
-----
python -m ensam3d_inference.examples.benchmarking \
    --video_path example.mp4 \
    --model_path sam-3d-body-vith \
    --batch_size 30
"""
import argparse
import time
from pathlib import Path
from typing import NamedTuple
import cv2
import torch
from ensam3d_inference import Pipeline, PreprocessorInput, DeviceType


class SystemInfo(NamedTuple):
    """
    Container for the hardware and software environment of the benchmark run.

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
    Parse command-line arguments for the benchmark script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments containing the video path, the model path, and the
        batch size.
    """
    parser = argparse.ArgumentParser(description="Benchmarking pipeline")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=30)
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


def _warmup(
    pipeline: Pipeline,
    video_path: str,
) -> None:
    """
    Run a few frames through the pipeline to eliminate one-time startup costs.

    Reads up to five frames from the video and executes a single forward pass.
    This triggers initial CUDA memory allocation, cuDNN kernel autotuning, and
    GPU clock ramp-up, so that the subsequent timed benchmark loop is not
    skewed by initialization overhead.

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


def _read_batches(
    video_path: str,
    batch_size: int,
):
    """
    Yield successive batches of RGB frames decoded from a video file.

    Frames are read sequentially, converted from BGR to RGB, and grouped into
    lists of at most batch_size elements. The final batch may contain fewer
    frames than batch_size when the total frame count is not an exact multiple
    of the batch size.

    Parameters
    ----------
    video_path : str
        Path to the video file to decode.
    batch_size : int
        Maximum number of frames per yielded batch.

    Yields
    ------
    list of numpy.ndarray
        A batch of RGB frames decoded from the video.
    """
    cap = cv2.VideoCapture(video_path)

    batch = []
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        total_frames += 1

        if len(batch) == batch_size:
            yield batch
            batch = []

    cap.release()

    if batch:
        yield batch


def _run_benchmark(
    pipeline: Pipeline,
    video_path: str,
    batch_size: int,
) -> tuple[int, float]:
    """
    Run the timed benchmark loop over the entire video.

    Clears cached GPU memory and resets the peak memory tracker before timing,
    so that the measurement reflects only the benchmark loop and excludes model
    loading and warm-up. The GPU is synchronized both before starting and after
    finishing the timer to account for the asynchronous execution of PyTorch
    operations.

    Parameters
    ----------
    pipeline : Pipeline
        The inference pipeline to benchmark.
    video_path : str
        Path to the video file to process.
    batch_size : int
        Number of frames per batch fed to the pipeline.

    Returns
    -------
    tuple of (int, float)
        The total number of processed frames and the total elapsed wall-clock
        time in seconds.
    """
    print("🚀 Starting benchmark...")
    processed_frames = 0

    # Frees unused cached memory to ensure a clean VRAM state before measurement.
    torch.cuda.empty_cache()

    # Resets the peak memory tracker so max_memory_allocated()
    # only captures the benchmark loop, excluding model loading and warm-up.
    torch.cuda.reset_peak_memory_stats()

    # Wait for all pending GPU kernels to finish
    torch.cuda.synchronize()
    start_time = time.perf_counter()

    for batch in _read_batches(video_path, batch_size):
        pipeline(PreprocessorInput(imgs=batch))
        processed_frames += len(batch)

    # Ensure the last batch has fully completed on the GPU
    # before stopping the timer. PyTorch executes operations
    # asynchronously, so the CPU thread finishes the loop long
    # before the GPU actually renders the final frame.
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    return processed_frames, total_time


def _print_results(
    processed_frames: int,
    total_time: float,
    batch_size: int,
    video_path: str,
) -> None:
    """
    Format and print the benchmark configuration and results to the console.

    Parameters
    ----------
    processed_frames : int
        Total number of frames processed during the benchmark.
    total_time : float
        Total elapsed wall-clock time of the benchmark loop in seconds.
    batch_size : int
        Number of frames per batch used during the benchmark.
    video_path : str
        Path to the benchmarked video file.
    """
    info = _get_system_info()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    ms_per_frame = (total_time / processed_frames) * 1000
    fps = processed_frames / total_time

    file_name = Path(video_path).name

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    video_duration = frame_count / video_fps if video_fps > 0 else 0

    print("\n⚙️  ===== Configuration =====")
    print(f"File Name:           {file_name}")
    print(f"Video Resolution:    {width} × {height}")
    print(f"Video Duration:      {video_duration:.2f} sec")
    print(f"Video FPS:           {video_fps:.2f}")
    print(f"Total Frames:        {int(frame_count)}")
    print(f"CPU:                 {info.cpu_name}")
    print(f"GPU:                 {info.gpu_name}")
    print(f"PyTorch Version:     {info.torch_version}")
    print(f"CUDA Version:        {info.cuda_version}")
    print(f"Batch Size:          {batch_size}")
    print()
    print("📊 ===== Results =====")
    print(f"Processed Frames:    {processed_frames}")
    print(f"Total Time:          {total_time:.3f} sec")
    print(f"Latency:             {ms_per_frame:.3f} ms/frame")
    print(f"Throughput:          {fps:.3f} FPS")
    print(f"Peak VRAM Usage:     {peak_vram_gb:.2f} GB")
    print("=====================")


def main() -> None:
    """
    Entry point wiring together argument parsing, warm-up, benchmarking, and
    result reporting.
    """
    args = _parse_args()
    pipeline = _build_pipeline(args)

    # Warm-up is mandatory even without torch.compile: it triggers
    # initial CUDA memory allocation, cuDNN kernel autotuning, and
    # ensures the GPU clocks ramp up from idle state. Skipping this
    # would skew latency measurements with one-time initialization overhead.
    _warmup(
        pipeline=pipeline,
        video_path=args.video_path
    )
    processed_frames, total_time = _run_benchmark(
        pipeline=pipeline,
        video_path=args.video_path,
        batch_size=args.batch_size
    )
    _print_results(
        processed_frames=processed_frames,
        total_time=total_time,
        batch_size=args.batch_size,
        video_path=args.video_path,
    )


if __name__ == "__main__":
    main()