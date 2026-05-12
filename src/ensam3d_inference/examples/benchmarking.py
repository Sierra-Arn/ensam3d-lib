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
import cv2
import torch
from ensam3d_inference import Pipeline, PreprocessorInput, DeviceType


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmarking pipeline")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=30)
    return parser.parse_args()


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        model_path=args.model_path,
        detector_device=DeviceType.CUDA,
        model_device=DeviceType.CUDA,
    )


def _warmup(
    pipeline: Pipeline, 
    video_path: str
) -> None:
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
    batch_size: int
):
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
    batch_size: int
) -> tuple[int, float]:
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
    batch_size: int
) -> None:
    
    gpu_name = torch.cuda.get_device_name(0)
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    ms_per_frame = (total_time / processed_frames) * 1000
    fps = processed_frames / total_time

    print("\n📊 ===== Benchmark Results =====")
    print(f"GPU:                 {gpu_name}")
    print(f"PyTorch Version:     {torch_ver}")
    print(f"CUDA Version:        {cuda_ver}")
    print(f"Batch Size:          {batch_size}")
    print(f"Processed frames:    {processed_frames}")
    print(f"Total time:          {total_time:.3f} sec")
    print(f"Latency:             {ms_per_frame:.3f} ms/frame")
    print(f"Throughput:          {fps:.3f} FPS")
    print(f"Peak VRAM Usage:     {peak_vram_gb:.2f} GB")
    print("================================")


def main() -> None:
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
    _print_results(processed_frames, total_time, args.batch_size)


if __name__ == "__main__":
    main()