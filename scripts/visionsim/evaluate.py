"""Run DA3 on a produced dataset and compute depth + cliff metrics.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    conda run -n da3 python scripts/visionsim/evaluate.py \\
        --dataset data/sim_dataset/tabletop_cliff/ \\
        --output-dir outputs/sim_eval/tabletop_cliff/
"""

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies():
    """Consumer needs DA3; nothing Blender/VisionSIM."""
    missing = []
    if importlib.util.find_spec("depth_anything_3") is None:
        missing.append(
            "depth_anything_3 not importable. "
            "Install via the submodule: `cd Depth-Anything-3 && pip install -e .`"
        )
    if missing:
        for m in missing:
            logger.error(m)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Dataset auto-produce + loading
# ---------------------------------------------------------------------------
def ensure_dataset(dataset_dir):
    """If the dataset is missing, run produce_dataset.py with this --output-dir.

    The producer needs Blender + VisionSIM. If those aren't installed, the
    subprocess fails and we exit with a message telling the user to either
    install the producer deps or drop a pre-rendered dataset at this path.
    """
    transforms_path = os.path.join(dataset_dir, "transforms.json")
    depth_path = os.path.join(dataset_dir, "depth_gt_float32.npz")
    if os.path.exists(transforms_path) and os.path.exists(depth_path):
        logger.info(f"Using existing dataset at {dataset_dir}")
        return
    logger.info(f"Dataset not found at {dataset_dir}; running produce_dataset.py")
    producer = os.path.join(os.path.dirname(os.path.abspath(__file__)), "produce_dataset.py")
    if not os.path.exists(producer):
        logger.error(f"Producer script not found at {producer}. Cannot auto-produce.")
        sys.exit(1)
    cmd = [sys.executable, producer, "--output-dir", dataset_dir]
    logger.info(f"  $ {' '.join(cmd)}")
    if subprocess.run(cmd).returncode != 0:
        logger.error(
            "Producer failed. To run inference only, drop a pre-rendered VisionSIM "
            f"dataset at {dataset_dir} (transforms.json + depth_gt_float32.npz + RGB [+ scene/scene.meta.json])."
        )
        sys.exit(1)
    if not (os.path.exists(transforms_path) and os.path.exists(depth_path)):
        logger.error("Producer ran but transforms.json or depth_gt_float32.npz is still missing. Aborting.")
        sys.exit(1)


def load_dataset(dataset_dir, rgb_source="mp4"):
    """Load dataset from disk.

    rgb_source ('mp4' or 'png') picks which RGB representation to read.
    'mp4' falls back to PNG with a warning if rgb.mp4 is absent; 'png'
    requires per-frame PNGs and errors otherwise.
    Ground-truth depth is required as one compressed FP32 NPZ in meters.
    """
    transforms_path = os.path.join(dataset_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"No transforms.json in {dataset_dir}")
    with open(transforms_path) as f:
        transforms = json.load(f)

    meta_path = os.path.join(dataset_dir, "scene", "scene.meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            scene_meta = json.load(f)
    else:
        logger.warning(f"No scene/scene.meta.json in {dataset_dir}; cliff metrics will be skipped.")
        scene_meta = None

    frame_infos = transforms["frames"]
    depth_gt_file = transforms["depth_gt_file"]
    depth_gt_key = transforms["depth_gt_key"]
    depth_path = os.path.join(dataset_dir, depth_gt_file)
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Missing required GT depth NPZ: {depth_path}")
    with np.load(depth_path) as depth_npz:
        if depth_gt_key not in depth_npz:
            raise KeyError(f"{depth_path} has no `{depth_gt_key}` array")
        depth_gt = depth_npz[depth_gt_key].astype(np.float32, copy=False)
    if depth_gt.ndim != 3:
        raise ValueError(f"GT depth must have shape (N, H, W), got {depth_gt.shape}")
    if len(depth_gt) != len(frame_infos):
        raise ValueError(
            f"GT depth frame count ({len(depth_gt)}) does not match transforms ({len(frame_infos)})"
        )

    rgb_iter = _rgb_frame_iter(dataset_dir, frame_infos, rgb_source)

    frames_rgb, frames_depth, frame_names = [], [], []
    for i, frame_info in enumerate(frame_infos):
        rgb = next(rgb_iter, None)
        if rgb is None:
            logger.warning(f"RGB stream ran out at frame {i} (transforms expected {len(frame_infos)})")
            break
        frames_rgb.append(rgb)
        frames_depth.append(depth_gt[i])
        frame_names.append(os.path.basename(frame_info["file_path"]))

    logger.info(f"Loaded {len(frames_rgb)} frames and GT depth from {depth_path}")
    return frames_rgb, frames_depth, frame_names, scene_meta


def _rgb_frame_iter(dataset_dir, frame_infos, rgb_source):
    """Yield RGB frames in transforms.json order.

    rgb_source='mp4' uses rgb.mp4, falling back to PNG with a warning if absent.
    rgb_source='png' uses per-frame PNGs and raises if any are missing.
    """
    mp4_path = os.path.join(dataset_dir, "rgb.mp4")

    if rgb_source == "mp4" and os.path.exists(mp4_path):
        logger.info(f"Reading RGB from {mp4_path}")
        cap = cv2.VideoCapture(mp4_path)
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    return
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()
        return

    if rgb_source == "mp4":
        logger.warning(f"--rgb-source=mp4 but {mp4_path} missing; falling back to PNGs")

    logger.info("Reading RGB from per-frame PNGs")
    for frame_info in frame_infos:
        rgb_path = os.path.join(dataset_dir, frame_info["file_path"])
        if not os.path.exists(rgb_path):
            raise FileNotFoundError(f"Missing PNG: {rgb_path}")
        yield cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# DA3 inference
# ---------------------------------------------------------------------------
def run_da3(frames_rgb, model_name, process_res, batch_size):
    from depth_anything_3.api import DepthAnything3
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading DA3 model: {model_name} on {device}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_name).to(device)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    n = len(frames_rgb)
    logger.info(f"Running DA3 on {n} frames (batch_size={batch_size}, process_res={process_res})")
    t0 = time.time()
    all_depths = []
    for i in range(0, n, batch_size):
        batch = frames_rgb[i : i + batch_size]
        pred = model.inference(batch, process_res=process_res)
        all_depths.append(pred.depth)
        torch.cuda.empty_cache()
    depth_pred = np.concatenate(all_depths, axis=0)
    elapsed = time.time() - t0
    logger.info(f"DA3 inference done in {elapsed:.2f}s ({elapsed / n:.2f}s/frame)")
    return depth_pred


# ---------------------------------------------------------------------------
# Depth alignment + standard metrics
# ---------------------------------------------------------------------------
def align_depth(pred, gt, valid_mask):
    """Least-squares scale + shift: min ||a * pred + b - gt||^2 over valid pixels."""
    p = pred[valid_mask].flatten()
    g = gt[valid_mask].flatten()
    A = np.stack([p, np.ones_like(p)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, g, rcond=None)
    return a * pred + b, a, b


def compute_depth_metrics(pred, gt, valid_mask):
    p = np.clip(pred[valid_mask], 1e-6, None)
    g = np.clip(gt[valid_mask], 1e-6, None)
    abs_rel = float(np.mean(np.abs(p - g) / g))
    rmse = float(np.sqrt(np.mean((p - g) ** 2)))
    ratio = np.maximum(p / g, g / p)
    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "delta_1.25": float(np.mean(ratio < 1.25) * 100),
        "delta_1.25^2": float(np.mean(ratio < 1.25 ** 2) * 100),
        "delta_1.25^3": float(np.mean(ratio < 1.25 ** 3) * 100),
    }


# ---------------------------------------------------------------------------
# Per-frame cliff annotation (just the analytic GT distance for now)
# ---------------------------------------------------------------------------
def compute_cliff_metrics(frame_idx, scene_meta):
    """Analytic ground-truth distance from camera to cliff edge.

    Derived from the trajectory in scene_meta (no depth involved); positive
    means the cliff is still ahead. All other cliff metrics removed pending
    a smarter detection algorithm.
    """
    fps = scene_meta.get("fps", 25)
    speed = scene_meta.get("trajectory_params", {}).get("speed_mps", 0.3)
    distance_to_edge = scene_meta.get("trajectory_params", {}).get("distance_to_edge", 1.5)
    step_per_frame = speed / fps
    return {"gt_distance_to_cliff_m": float(distance_to_edge - frame_idx * step_per_frame)}


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def save_comparison(rgb, gt, pred_aligned, depth_metrics, output_path, frame_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title("Input (VisionSIM render)")
    axes[0, 0].axis("off")

    valid = (gt > 0) & (gt < 1e4)
    vmin, vmax = np.percentile(gt[valid], [2, 98])
    axes[0, 1].imshow(gt, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Ground Truth Depth")
    axes[0, 1].axis("off")
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].imshow(pred_aligned, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("DA3 Predicted (aligned)")
    axes[1, 0].axis("off")
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046, pad=0.04)

    error = np.abs(pred_aligned - gt)
    error[~valid] = 0
    axes[1, 1].imshow(error, cmap="hot", vmin=0, vmax=np.percentile(error[valid], 95))
    axes[1, 1].set_title("Absolute Error")
    axes[1, 1].axis("off")
    plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1], fraction=0.046, pad=0.04)

    metrics_str = "  ".join(f"{k}: {v:.4f}" for k, v in depth_metrics.items())
    fig.suptitle(f"{frame_name}\n{metrics_str}", fontsize=11, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_comparison_video(frames_rgb, frames_gt, aligned_preds, output_path, fps=25):
    """3-panel vertical mp4: RGB / GT depth / DA3 (aligned), stacked top-to-bottom.

    Depth panels share a global vmin/vmax (computed across all GT frames,
    valid pixels only) so you can see depth changing across the trajectory
    without per-frame colormap drift.
    """
    h, w = frames_rgb[0].shape[:2]
    panel_w = w
    panel_h = h
    n = len(frames_rgb)

    valid_concat = np.concatenate([gt[(gt > 0) & (gt < 1e4) & np.isfinite(gt)] for gt in frames_gt])
    vmin, vmax = np.percentile(valid_concat, [2, 98])

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(output_path, fourcc, max(fps, 8.0), (panel_w, panel_h * 3))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, max(fps, 8.0), (panel_w, panel_h * 3))

    def colorize(d):
        d = np.clip(d, vmin, vmax)
        norm = ((d - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

    def label(img, text):
        cv2.putText(img, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    for i in range(n):
        rgb_bgr = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2BGR)
        gt_bgr = colorize(frames_gt[i])
        pred_bgr = colorize(aligned_preds[i])

        # Match panel sizes (DA3 native is at process_res; aligned_pred resized in loop)
        if gt_bgr.shape[:2] != (panel_h, panel_w):
            gt_bgr = cv2.resize(gt_bgr, (panel_w, panel_h))
        if pred_bgr.shape[:2] != (panel_h, panel_w):
            pred_bgr = cv2.resize(pred_bgr, (panel_w, panel_h))

        label(rgb_bgr, "Input")
        label(gt_bgr, "Ground Truth")
        label(pred_bgr, "DA3 (aligned)")
        cv2.putText(rgb_bgr, f"#{i:03d}", (10, panel_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        combined = np.concatenate([rgb_bgr, gt_bgr, pred_bgr], axis=0)
        writer.write(combined)

    writer.release()


def save_cliff_summary(cliff_records, output_path):
    """Plot the analytic GT distance-to-cliff over frames (sanity for the trajectory)."""
    frames = [c["frame"] for c in cliff_records]
    gt_dist = [c["gt_distance_to_cliff_m"] for c in cliff_records]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(frames, gt_dist, color="black")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5, label="cliff edge")
    ax.set_xlabel("Frame")
    ax.set_ylabel("GT distance to cliff (m)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="DA3 inference + depth & cliff metrics on a produced dataset")
    parser.add_argument("--dataset", default="data/sim_dataset/tabletop_cliff/",
                        help="Dataset directory produced by produce_dataset.py")
    parser.add_argument("--output-dir", default="outputs/sim_eval/tabletop_cliff/",
                        help="Where to write predictions, plots, metrics.json")
    parser.add_argument("--model", default="depth-anything/DA3-BASE")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save 4-panel comparison plot every Nth frame to keep IO small")
    parser.add_argument("--rgb-source", default="mp4", choices=["mp4", "png"],
                        help="Read RGB from rgb.mp4 (default; falls back to PNG if missing) or "
                             "frames/*.png (lossless; required for canonical eval).")
    args = parser.parse_args()

    check_dependencies()
    os.makedirs(args.output_dir, exist_ok=True)

    ensure_dataset(args.dataset)
    frames_rgb, frames_gt, frame_names, scene_meta = load_dataset(args.dataset, rgb_source=args.rgb_source)
    if not frames_rgb:
        logger.error("No frames loaded. Exiting.")
        return

    depth_pred = run_da3(frames_rgb, args.model, args.process_res, args.batch_size)
    npz_path = os.path.join(args.output_dir, "da3_depth_pred.npz")
    np.savez_compressed(npz_path, depth=depth_pred)
    logger.info(f"Saved raw DA3 depth to {npz_path}  shape={depth_pred.shape}")

    frames_subdir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_subdir, exist_ok=True)

    depth_records = []
    cliff_records = []
    aligned_preds = []
    accepted_rgb = []
    accepted_gt = []
    for i, gt in enumerate(frames_gt):
        if gt is None:
            continue
        pred_i = depth_pred[i]
        if pred_i.shape != gt.shape:
            pred_i = cv2.resize(pred_i, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        valid = (gt > 0) & (gt < 1e4) & np.isfinite(gt) & np.isfinite(pred_i)
        if valid.sum() < 100:
            logger.warning(f"Frame {i}: too few valid pixels, skipping")
            continue

        pred_aligned, scale, shift = align_depth(pred_i, gt, valid)
        aligned_preds.append(pred_aligned)
        accepted_rgb.append(frames_rgb[i])
        accepted_gt.append(gt)
        d_metrics = compute_depth_metrics(pred_aligned, gt, valid)
        d_metrics.update({"frame": i, "scale": float(scale), "shift": float(shift)})
        depth_records.append(d_metrics)

        if scene_meta is not None:
            c_metrics = compute_cliff_metrics(i, scene_meta)
            c_metrics["frame"] = i
            cliff_records.append(c_metrics)

        if i % args.save_every == 0:
            out_path = os.path.join(frames_subdir, f"frame_{i:03d}_4panel.png")
            save_comparison(frames_rgb[i], gt, pred_aligned, d_metrics, out_path, frame_names[i])

    # Aggregate depth metrics
    if depth_records:
        logger.info("\n--- Depth metrics (aggregate) ---")
        for key in ("AbsRel", "RMSE", "delta_1.25", "delta_1.25^2", "delta_1.25^3"):
            vals = [r[key] for r in depth_records]
            logger.info(f"  {key:>15s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # Cliff: just the GT-distance trajectory plot (no derived metrics yet)
    if cliff_records:
        save_cliff_summary(cliff_records, os.path.join(args.output_dir, "cliff_summary.png"))

    # 3-panel side-by-side video (RGB | GT depth | DA3 aligned)
    fps = scene_meta.get("fps", 25) if scene_meta else 25
    video_path = os.path.join(args.output_dir, "eval_video_3panel.mp4")
    save_comparison_video(accepted_rgb, accepted_gt, aligned_preds, video_path, fps=fps)
    logger.info(f"Saved comparison video to {video_path}")

    # Dump all metrics
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({"depth": depth_records, "cliff": cliff_records}, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    start = time.time()
    logger.info("Evaluation started")
    main()
    logger.info(f"Total time: {time.time() - start:.2f}s")
