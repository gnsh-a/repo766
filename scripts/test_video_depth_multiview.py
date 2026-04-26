"""DA3 multi-view inference on a video.

Unlike `test_video_depth.py`, which processes frames in independent mini-batches,
this script feeds *all* sampled frames to DA3 in a single inference call so the
model can reason across views: cross-frame attention produces consistent depth
plus reconstructed per-frame extrinsics/intrinsics, and DA3 can export a 3D
scene (GLB) of the whole clip.

Multi-view attention is roughly O(N^2) in frames, so cap with --max-frames.
"""

import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def extract_frames(video_path, fps, max_frames):
    """Extract up to `max_frames` RGB frames from a video, sampled at `fps`.

    If the requested sampling produces more than `max_frames`, uniformly subsample
    the result so the temporal coverage is preserved.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / video_fps
    stride = max(1, int(video_fps / fps))
    logger.info(
        f"Video: {video_path}  ({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, {video_fps:.1f} fps, {duration:.1f}s)"
    )
    logger.info(f"Sampling every {stride}th frame ({video_fps / stride:.2f} fps)")

    frames = []
    i = 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if i % stride == 0:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()

    if len(frames) > max_frames:
        # Uniform subsample to fit memory budget for multi-view attention.
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        logger.warning(
            f"Sampled {len(frames)} frames exceeds --max-frames={max_frames}; "
            f"keeping {max_frames} uniformly spaced frames."
        )
        frames = [frames[k] for k in idx]

    logger.info(f"Using {len(frames)} frames for multi-view inference")
    return frames


def save_side_by_side(frames, depth, fps, out_path):
    """Write [input | colorized depth] mp4 with a global depth normalization."""
    dmin, dmax = float(depth.min()), float(depth.max())
    norm = ((depth - dmin) / max(dmax - dmin, 1e-8) * 255).astype(np.uint8)

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w * 2, h))
    for i, frame in enumerate(frames):
        d = cv2.applyColorMap(norm[i], cv2.COLORMAP_INFERNO)
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        writer.write(np.concatenate([cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), d], axis=1))
    writer.release()


def main():
    p = argparse.ArgumentParser(description="Multi-view DA3 inference on a video")
    p.add_argument("--video", default="data/test-video.mp4")
    p.add_argument("--model", default="depth-anything/DA3-BASE")
    p.add_argument("--fps", type=float, default=2.0, help="Sampling FPS")
    p.add_argument(
        "--max-frames", type=int, default=32,
        help="Hard cap; multi-view attention is ~O(N^2) so be conservative",
    )
    p.add_argument("--process-res", type=int, default=504)
    p.add_argument("--output-dir", default="outputs/video_multiview/")
    p.add_argument(
        "--export-format", default="glb",
        choices=["glb", "ply", "npz", "mini_npz"],
        help="DA3 scene export format",
    )
    p.add_argument("--no-export", action="store_true",
                   help="Skip GLB/PLY scene export (depth + cameras still saved)")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frames = extract_frames(args.video, args.fps, args.max_frames)
    if not frames:
        logger.error("No frames extracted")
        return

    from depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading {args.model} on {device}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(args.model).to(device)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    logger.info(
        f"Running single multi-view inference: {len(frames)} frames, "
        f"process_res={args.process_res}, export={'off' if args.no_export else args.export_format}"
    )
    t0 = time.time()
    pred = model.inference(
        frames,
        process_res=args.process_res,
        export_dir=None if args.no_export else args.output_dir,
        export_format=args.export_format,
    )
    elapsed = time.time() - t0
    logger.info(f"Inference done in {elapsed:.2f}s ({elapsed / len(frames):.2f}s/frame)")

    depth = np.asarray(pred.depth)
    extrinsics = np.asarray(pred.extrinsics)
    intrinsics = np.asarray(pred.intrinsics)
    logger.info(f"depth={depth.shape}  ext={extrinsics.shape}  int={intrinsics.shape}")

    npz_path = os.path.join(args.output_dir, "depth_raw.npz")
    np.savez_compressed(npz_path, depth=depth, extrinsics=extrinsics, intrinsics=intrinsics)
    logger.info(f"Saved {npz_path}")

    cams_path = os.path.join(args.output_dir, "cameras.json")
    with open(cams_path, "w") as f:
        json.dump(
            {"extrinsics": extrinsics.tolist(), "intrinsics": intrinsics.tolist()},
            f, indent=2,
        )
    logger.info(f"Saved {cams_path}")

    video_path = os.path.join(args.output_dir, "depth_video.mp4")
    save_side_by_side(frames, depth, args.fps, video_path)
    logger.info(f"Saved {video_path}")


if __name__ == "__main__":
    start = time.time()
    logger.info("Multi-view video inference started")
    main()
    logger.info(f"Total time: {time.time() - start:.2f}s")
