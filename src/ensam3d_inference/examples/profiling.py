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
import cv2
import torch
from ensam3d_inference import Pipeline, PreprocessorInput, DeviceType


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline profiling")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=30)
    parser.add_argument("--prof_export_path", type=str, default=None)
    return parser.parse_args()


def _build_pipeline(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        model_path=args.model_path,
        detector_device=DeviceType.CUDA,
        model_device=DeviceType.CUDA,
    )


def _warmup(pipeline: Pipeline, video_path: str) -> None:
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


def _run_profiling(
    pipeline: Pipeline,
    batch: list,
    batch_size: int,
    prof_export_path: str | None,
) -> None:
    print("🔬 Profiling...")

    gpu_name = torch.cuda.get_device_name(0)
    torch_ver = torch.__version__
    cuda_ver = torch.version.cuda

    print("\n📊 ===== Environment Info =====")
    print(f"GPU:                 {gpu_name}")
    print(f"PyTorch Version:     {torch_ver}")
    print(f"CUDA Version:        {cuda_ver}")
    print(f"Batch Size:          {batch_size}")
    print("================================\n")

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
    args = _parse_args()
    pipeline = _build_pipeline(args)
    _warmup(pipeline=pipeline, video_path=args.video_path)
    batch = _read_batch(video_path=args.video_path, batch_size=args.batch_size)
    _run_profiling(
        pipeline=pipeline,
        batch=batch,
        batch_size=args.batch_size,
        prof_export_path=args.prof_export_path,
    )


if __name__ == "__main__":
    main()