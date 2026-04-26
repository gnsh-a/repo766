"""Test script for Depth Anything 3 video inference."""

import argparse
import logging
import os
import time

import cv2
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_frames(video_path, fps=1.0):
    """Extract frames from video at given FPS."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps
    frame_interval = max(1, int(video_fps / fps))
    actual_fps = video_fps / frame_interval

    logger.info(f"Video: {video_path}")
    logger.info(f"  Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    logger.info(f"  FPS: {video_fps:.1f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
    logger.info(f"  Extracting at {actual_fps:.1f} FPS (every {frame_interval} frame(s))")

    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1

    cap.release()
    logger.info(f"  Extracted {len(frames)} frames")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Test Depth Anything 3 on video")
    parser.add_argument("--model", default="depth-anything/DA3-BASE",
                        help="HuggingFace model ID")
    parser.add_argument("--video", default="data/test-video.mp4",
                        help="Path to input video")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Frame sampling rate from video")
    parser.add_argument("--output-dir", default="outputs/video/")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=20,
                        help="Max frames per inference batch (to avoid OOM)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract frames
    frames = extract_frames(args.video, fps=args.fps)
    if not frames:
        logger.info("No frames extracted.")
        return

    # Load model
    logger.info(f"Loading model: {args.model}")
    t0 = time.time()
    from depth_anything_3.api import DepthAnything3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DepthAnything3.from_pretrained(args.model).to(device)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s on {device}")

    # Run batched inference
    n = len(frames)
    bs = args.batch_size
    num_batches = (n + bs - 1) // bs
    logger.info(f"Running inference on {n} frames in {num_batches} batches of {bs} (process_res={args.process_res})")

    all_depths = []
    t0 = time.time()
    for b in range(num_batches):
        batch = frames[b * bs : (b + 1) * bs]
        logger.info(f"  Batch {b + 1}/{num_batches} ({len(batch)} frames)")
        pred = model.inference(batch, process_res=args.process_res)
        all_depths.append(pred.depth)
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    depth_all = np.concatenate(all_depths, axis=0)
    logger.info(f"Inference done in {elapsed:.2f}s ({elapsed / n:.2f}s/frame)")
    logger.info(f"  depth shape: {depth_all.shape}")

    # Save raw depth
    npz_path = os.path.join(args.output_dir, "depth_raw.npz")
    np.savez_compressed(npz_path, depth=depth_all)
    logger.info(f"Saved raw depth to {npz_path}")

    # Build side-by-side video: input | depth
    depth_min = depth_all.min()
    depth_max = depth_all.max()
    depth_norm = ((depth_all - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    h, w = frames[0].shape[:2]
    video_path = os.path.join(args.output_dir, "depth_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(video_path, fourcc, args.fps, (w * 2, h))

    for i in range(n):
        depth_color = cv2.applyColorMap(depth_norm[i], cv2.COLORMAP_INFERNO)
        depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        combined = np.concatenate([frame_bgr, depth_color], axis=1)
        out_video.write(combined)

    out_video.release()
    logger.info(f"Saved video to {video_path}")


if __name__ == "__main__":
    start = time.time()
    logger.info("Video depth inference started")
    main()
    logger.info(f"Total time: {time.time() - start:.2f}s")
