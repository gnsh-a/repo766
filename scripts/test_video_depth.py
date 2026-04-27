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

from sklearn.linear_model import RANSACRegressor


def detect_cliff(depth, fx=500, fy=500, cx=None, cy=None,
                 min_inlier_ratio=0.2, cliff_threshold=0.15):
    """
    Fully geometric cliff detection using ground plane estimation.
    Returns binary mask of cliff regions.
    """

    h, w = depth.shape

    if cx is None: cx = w / 2
    if cy is None: cy = h / 2

    # ------------------------------------------------------------
    # 1. Back-project depth → 3D points
    # ------------------------------------------------------------
    i, j = np.indices((h, w))
    z = depth.reshape(-1)

    x = (j.reshape(-1) - cx) * z / fx
    y = (i.reshape(-1) - cy) * z / fy

    pts = np.stack([x, y, z], axis=1)

    # ------------------------------------------------------------
    # 2. Use lower image region as ground candidates only
    # ------------------------------------------------------------
    mask_ground = i.reshape(-1) > (0.5 * h)
    pts_ground = pts[mask_ground]

    X = pts_ground[:, [0, 2]]  # x, z
    Y = pts_ground[:, 1]       # height (y)

    # ------------------------------------------------------------
    # 3. Robust plane fit (RANSAC)
    # y = ax + bz + c
    # ------------------------------------------------------------
    ransac = RANSACRegressor(residual_threshold=0.05)
    ransac.fit(X, Y)

    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_

    # ------------------------------------------------------------
    # 4. Compute distance to ground plane for all points
    # ------------------------------------------------------------
    y_pred = a * pts[:, 0] + b * pts[:, 2] + c
    residual = pts[:, 1] - y_pred

    residual_img = residual.reshape(h, w)

    # ------------------------------------------------------------
    # 5. Cliff = loss of support (positive height jump)
    # ------------------------------------------------------------
    cliff_mask = residual_img > cliff_threshold

    # ------------------------------------------------------------
    # 6. Remove tiny noise components
    # ------------------------------------------------------------
    cliff_mask = cliff_mask.astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats(cliff_mask)
    min_area = h * w * 0.001

    out = np.zeros_like(cliff_mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > min_area:
            out[labels == i] = 255

    return out

def overlay_cliff(frame_bgr, cliff_mask, depth_norm_frame):
    """Overlay cliff detection on a BGR frame"""
    overlay = frame_bgr.copy()
    overlay[cliff_mask > 0] = (0, 0, 255)
    result = cv2.addWeighted(frame_bgr, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(cliff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 0, 255), 2)
    if contours:
        cv2.putText(result, "CLIFF", (contours[0][0][0][0], contours[0][0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return result


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
    parser.add_argument("--skip-inference", action="store_true",
                    help="Load depth from existing depth_raw.npz instead of running model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Extract frames
    frames = extract_frames(args.video, fps=args.fps)
    if not frames:
        logger.info("No frames extracted.")
        return

    npz_path = os.path.join(args.output_dir, "depth_raw.npz")

    if args.skip_inference:
        depth_all = np.load(npz_path)["depth"]
        n = len(frames)
        logger.info(f"Frames extracted: {len(frames)}, depth frames cached: {len(depth_all)}")
        if len(frames) != len(depth_all):
            logger.error(f"MISMATCH: {len(frames)} video frames vs {len(depth_all)} depth frames. "
                         f"Re-run without --skip-inference or match --fps to original run.")
            return
    else:
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

        np.savez_compressed(npz_path, depth=depth_all)
        logger.info(f"Saved raw depth to {npz_path}")

    # Build side-by-side video: input | depth
    depth_min = depth_all.min()
    depth_max = depth_all.max()
    depth_norm = ((depth_all - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    h, w = frames[0].shape[:2]
    video_path = os.path.join(args.output_dir, "depth_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out_video = cv2.VideoWriter(video_path, fourcc, max(args.fps, 8.0), (w * 2, h))
    if not out_video.isOpened():
        # fallback
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video = cv2.VideoWriter(video_path, fourcc, max(args.fps, 8.0), (w * 2, h))

    for i in range(n):
        d = depth_all[i].astype(np.float32)
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)

        if i == 0:
            dy = d_norm[:-1, :] - d_norm[1:, :]
            dx = d_norm[:, :-1] - d_norm[:, 1:]

            dy = np.pad(dy, ((0,1),(0,0)))
            dx = np.pad(dx, ((0,0),(0,1)))

            raw_edges = ((dy > 0.02) | (dx > 0.02)).astype(np.uint8) * 255
            cv2.imwrite("debug_edges.png", raw_edges)

            logger.info(f"DEBUG dy max: {dy.max():.4f}, dx max: {dx.max():.4f}")

        cliff_mask = detect_cliff(d)

         # Diagnose depth quality
        if i == 0:
            logger.info(f"Depth frame 0 stats: min={d.min():.4f} max={d.max():.4f} "
                    f"mean={d.mean():.4f} std={d.std():.4f} "
                    f"normalized_std={d_norm.std():.4f}")

        if cliff_mask.any():
            logger.info(f"Frame {i}: cliff detected, {cliff_mask.sum()} pixels")
        else:
            logger.info(f"Frame {i}: no cliff detected")

        depth_color = cv2.applyColorMap(depth_norm[i], cv2.COLORMAP_INFERNO)
        depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_LINEAR)
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)

        # Resize cliff mask to match frame size
        cliff_resized = cv2.resize(cliff_mask, (w, h), interpolation=cv2.INTER_NEAREST)

        frame_annotated = overlay_cliff(frame_bgr, cliff_resized, d_norm)
        combined = np.concatenate([frame_annotated, depth_color], axis=1)
        out_video.write(combined)

    out_video.release()
    logger.info(f"Saved video to {video_path}")


if __name__ == "__main__":
    start = time.time()
    logger.info("Video depth inference started")
    main()
    logger.info(f"Total time: {time.time() - start:.2f}s")
