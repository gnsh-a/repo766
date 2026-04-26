"""Integration test: VisionSIM ground-truth depth vs Depth Anything 3.

Creates a Blender scene, renders RGB + ground-truth depth with VisionSIM,
runs DA3 inference, aligns predictions to ground truth via least-squares
(scale + shift), and saves AbsRel / RMSE / delta metrics with 4-panel
comparison plots.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    conda run -n da3 python scripts/visionsim/test_visionsim.py [--skip-render]
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BLENDER_EXEC = "/home/ganesh/Packages/blender-4.0.1-linux-x64/blender"


def check_dependencies(skip_render):
    """Fail fast with clear messages if external deps are missing."""
    missing = []
    if not skip_render:
        if not os.path.exists(BLENDER_EXEC):
            missing.append(
                f"Blender not found at {BLENDER_EXEC}. "
                "Install Blender 4.0.1 or update BLENDER_EXEC at the top of this script."
            )
        if shutil.which("visionsim") is None:
            missing.append(
                "VisionSIM CLI not found on PATH. "
                "Activate the `da3` conda env or run `pip install visionsim`."
            )
    if importlib.util.find_spec("depth_anything_3") is None:
        missing.append(
            "depth_anything_3 not importable. "
            "Install via the submodule: `cd Depth-Anything-3 && pip install -e .`"
        )
    if importlib.util.find_spec("imageio") is None:
        missing.append("imageio not installed. `pip install imageio` (needed to read ground-truth depth EXRs).")
    if missing:
        for m in missing:
            logger.error(m)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Step 1: Create .blend scene
# ---------------------------------------------------------------------------
def create_blend_file(blend_path):
    """Run the Blender scene creation script."""
    script = os.path.join(os.path.dirname(__file__), "create_simple_scene.py")
    os.makedirs(os.path.dirname(blend_path), exist_ok=True)
    cmd = [BLENDER_EXEC, "--background", "--python", script, "--", os.path.abspath(blend_path)]
    logger.info(f"Creating scene: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"Scene saved to {blend_path}")


# ---------------------------------------------------------------------------
# Step 2: Render with VisionSIM
# ---------------------------------------------------------------------------
def render_with_visionsim(blend_path, output_dir, device="cuda"):
    """Use VisionSIM CLI to render RGB + ground-truth depth."""
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "visionsim", "blender.render-animation",
        os.path.abspath(blend_path),
        os.path.abspath(output_dir),
        "--render-config.executable", BLENDER_EXEC,
        "--render-config.width", "640",
        "--render-config.height", "480",
        "--render-config.depths",
        "--render-config.debug",
        "--render-config.max-samples", "64",
        "--render-config.device-type", device,
        "--render-config.use-denoising",
        "--render-config.log-dir", "outputs/logs/",
    ]
    logger.info(f"Rendering with VisionSIM: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"VisionSIM stderr:\n{result.stderr}")
        raise RuntimeError("VisionSIM render failed")
    logger.info("VisionSIM render complete")


# ---------------------------------------------------------------------------
# Step 3: Load rendered data
# ---------------------------------------------------------------------------
def load_visionsim_dataset(output_dir):
    """Load RGB frames and ground-truth depth from VisionSIM output."""
    transforms_path = os.path.join(output_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"No transforms.json found in {output_dir}")

    with open(transforms_path) as f:
        transforms = json.load(f)

    frames_rgb = []
    frames_depth = []
    frame_names = []

    for frame_info in transforms["frames"]:
        # Load RGB
        rgb_path = os.path.join(output_dir, frame_info["file_path"])
        if not os.path.exists(rgb_path):
            logger.warning(f"Skipping missing frame: {rgb_path}")
            continue
        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)

        # Load ground-truth depth (EXR)
        depth_path = os.path.join(output_dir, frame_info["depth_file_path"])
        if os.path.exists(depth_path):
            import imageio
            depth_gt = imageio.imread(depth_path)
            if depth_gt.ndim == 3:
                depth_gt = depth_gt[:, :, 0]  # Take first channel
            frames_depth.append(depth_gt)
        else:
            logger.warning(f"No depth file: {depth_path}")
            frames_depth.append(None)

        frame_names.append(os.path.basename(frame_info["file_path"]))

    logger.info(f"Loaded {len(frames_rgb)} frames from VisionSIM output")
    return frames_rgb, frames_depth, frame_names, transforms


# ---------------------------------------------------------------------------
# Step 4: Run DA3 inference
# ---------------------------------------------------------------------------
def run_da3(frames_rgb, model_name, process_res, batch_size):
    """Run Depth Anything 3 on a list of RGB frames."""
    from depth_anything_3.api import DepthAnything3

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading DA3 model: {model_name} on {device}")
    t0 = time.time()
    model = DepthAnything3.from_pretrained(model_name).to(device)
    logger.info(f"Model loaded in {time.time() - t0:.1f}s")

    n = len(frames_rgb)
    bs = batch_size
    all_depths = []

    logger.info(f"Running DA3 on {n} frames (batch_size={bs}, process_res={process_res})")
    t0 = time.time()
    for i in range(0, n, bs):
        batch = frames_rgb[i : i + bs]
        pred = model.inference(batch, process_res=process_res)
        all_depths.append(pred.depth)
        torch.cuda.empty_cache()

    depth_pred = np.concatenate(all_depths, axis=0)
    logger.info(f"DA3 inference done in {time.time() - t0:.2f}s ({(time.time() - t0) / n:.2f}s/frame)")
    return depth_pred


# ---------------------------------------------------------------------------
# Step 5: Depth alignment and metrics
# ---------------------------------------------------------------------------
def align_depth(pred, gt, valid_mask):
    """Align predicted depth to ground truth via least-squares (scale + shift).

    Solves: min ||a * pred + b - gt||^2 over valid pixels.
    Returns aligned prediction.
    """
    p = pred[valid_mask].flatten()
    g = gt[valid_mask].flatten()

    # Least squares: [p, 1] @ [a, b]^T = g
    A = np.stack([p, np.ones_like(p)], axis=1)
    result = np.linalg.lstsq(A, g, rcond=None)
    a, b = result[0]

    aligned = a * pred + b
    return aligned, a, b


def compute_metrics(pred, gt, valid_mask):
    """Compute standard depth estimation metrics on valid pixels."""
    p = pred[valid_mask]
    g = gt[valid_mask]

    # Clamp to avoid division by zero
    p = np.clip(p, 1e-6, None)
    g = np.clip(g, 1e-6, None)

    # AbsRel
    abs_rel = np.mean(np.abs(p - g) / g)

    # RMSE
    rmse = np.sqrt(np.mean((p - g) ** 2))

    # delta < threshold
    ratio = np.maximum(p / g, g / p)
    delta1 = np.mean(ratio < 1.25) * 100
    delta2 = np.mean(ratio < 1.25 ** 2) * 100
    delta3 = np.mean(ratio < 1.25 ** 3) * 100

    return {
        "AbsRel": abs_rel,
        "RMSE": rmse,
        "delta_1.25": delta1,
        "delta_1.25^2": delta2,
        "delta_1.25^3": delta3,
    }


# ---------------------------------------------------------------------------
# Step 6: Visualization
# ---------------------------------------------------------------------------
def save_comparison(frame_rgb, depth_gt, depth_pred_aligned, metrics, output_path, frame_name):
    """Save a side-by-side comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Input RGB
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title("Input (VisionSIM render)")
    axes[0, 0].axis("off")

    # Ground-truth depth
    valid = (depth_gt > 0) & (depth_gt < 1e4)
    vmin, vmax = np.percentile(depth_gt[valid], [2, 98])
    axes[0, 1].imshow(depth_gt, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("Ground Truth Depth")
    axes[0, 1].axis("off")
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, pad=0.04)

    # DA3 predicted (aligned)
    axes[1, 0].imshow(depth_pred_aligned, cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("DA3 Predicted (aligned)")
    axes[1, 0].axis("off")
    plt.colorbar(axes[1, 0].images[0], ax=axes[1, 0], fraction=0.046, pad=0.04)

    # Error map
    error = np.abs(depth_pred_aligned - depth_gt)
    error[~valid] = 0
    axes[1, 1].imshow(error, cmap="hot", vmin=0, vmax=np.percentile(error[valid], 95))
    axes[1, 1].set_title("Absolute Error")
    axes[1, 1].axis("off")
    plt.colorbar(axes[1, 1].images[0], ax=axes[1, 1], fraction=0.046, pad=0.04)

    # Metrics text
    metrics_str = "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    fig.suptitle(f"{frame_name}\n{metrics_str}", fontsize=11, y=0.98)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="VisionSIM + DA3 integration test")
    parser.add_argument("--blend-file", default="data/simple_scene.blend",
                        help="Path to .blend scene file (created if missing)")
    parser.add_argument("--render-dir", default="outputs/visionsim_render/",
                        help="VisionSIM render output directory")
    parser.add_argument("--output-dir", default="outputs/visionsim_da3/",
                        help="Comparison output directory")
    parser.add_argument("--model", default="depth-anything/DA3-BASE",
                        help="DA3 HuggingFace model ID")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--render-device", default="cuda",
                        choices=["cuda", "cpu", "optix"],
                        help="Blender render device")
    parser.add_argument("--skip-render", action="store_true",
                        help="Skip rendering (use existing VisionSIM output)")
    args = parser.parse_args()

    check_dependencies(skip_render=args.skip_render)
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1 & 2: Create scene and render
    if not args.skip_render:
        if not os.path.exists(args.blend_file):
            create_blend_file(args.blend_file)
        render_with_visionsim(args.blend_file, args.render_dir, device=args.render_device)
    else:
        logger.info("Skipping render (--skip-render)")

    # Step 3: Load rendered data
    frames_rgb, frames_depth_gt, frame_names, transforms = load_visionsim_dataset(args.render_dir)
    if not frames_rgb:
        logger.error("No frames loaded. Exiting.")
        return

    # Step 4: Run DA3
    depth_pred = run_da3(frames_rgb, args.model, args.process_res, args.batch_size)

    # Save raw DA3 depth
    npz_path = os.path.join(args.output_dir, "da3_depth_raw.npz")
    np.savez_compressed(npz_path, depth=depth_pred)
    logger.info(f"Saved raw DA3 depth to {npz_path}  shape={depth_pred.shape}")

    # Step 5 & 6: Align, evaluate, and visualize per frame
    all_metrics = []
    for i in range(len(frames_rgb)):
        gt = frames_depth_gt[i]
        if gt is None:
            logger.warning(f"No ground-truth depth for frame {i}, skipping")
            continue

        # Resize prediction to match ground-truth resolution
        pred_i = depth_pred[i]
        if pred_i.shape != gt.shape:
            pred_i = cv2.resize(pred_i, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Valid mask: positive, finite, not sky (depth < large threshold)
        valid = (gt > 0) & (gt < 1e4) & np.isfinite(gt) & np.isfinite(pred_i)

        if valid.sum() < 100:
            logger.warning(f"Frame {i}: too few valid pixels ({valid.sum()}), skipping")
            continue

        # Align
        pred_aligned, scale, shift = align_depth(pred_i, gt, valid)
        logger.info(f"Frame {i}: alignment scale={scale:.4f}, shift={shift:.4f}")

        # Metrics
        metrics = compute_metrics(pred_aligned, gt, valid)
        metrics["frame"] = i
        all_metrics.append(metrics)
        logger.info(f"Frame {i}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items() if k != "frame"))

        # Save visualization
        out_path = os.path.join(args.output_dir, f"comparison_{i:03d}.png")
        save_comparison(frames_rgb[i], gt, pred_aligned, metrics, out_path, frame_names[i])

    # Summary
    if all_metrics:
        logger.info("\n--- Aggregate Metrics ---")
        for key in ["AbsRel", "RMSE", "delta_1.25", "delta_1.25^2", "delta_1.25^3"]:
            values = [m[key] for m in all_metrics]
            logger.info(f"  {key:>15s}: mean={np.mean(values):.4f}  std={np.std(values):.4f}")

        # Save metrics to JSON
        metrics_path = os.path.join(args.output_dir, "metrics.json")
        # Convert numpy floats to native Python floats for JSON
        serializable = [
            {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in m.items()}
            for m in all_metrics
        ]
        with open(metrics_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    else:
        logger.warning("No valid frames for evaluation.")


if __name__ == "__main__":
    start_time = time.time()
    logger.info("VisionSIM + DA3 integration test started")
    main()
    logger.info(f"Total time: {time.time() - start_time:.2f}s")
