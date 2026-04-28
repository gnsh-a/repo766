"""Build a Blender scene and render it with VisionSIM into a dataset directory.

Layout: scene authoring artifacts (.blend + meta + topdown) live under
<output-dir>/scene/, sitting next to the rendered transforms.json + rgb.mp4
+ depth_gt_float32.npz so a single dir is self-contained.

Author: Ganesh Arivoli <arivoli@wisc.edu>

Usage:
    conda run -n da3 python scripts/visionsim/produce_dataset.py \\
        --scene-script scripts/visionsim/scenes/tabletop_cliff.py \\
        --blend-file data/sim_dataset/tabletop_cliff/scene/scene.blend \\
        --output-dir data/sim_dataset/tabletop_cliff/
"""

import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BLENDER_EXEC = "/home/ganesh/Packages/blender-4.0.1-linux-x64/blender"
DEPTH_GT_FILE = "depth_gt_float32.npz"
DEPTH_GT_KEY = "depth"
INVALID_DEPTH_THRESHOLD = 1e4


def check_dependencies(need_ffmpeg=True):
    """Producer needs Blender + VisionSIM + imageio; ffmpeg if encoding mp4."""
    missing = []
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
    if importlib.util.find_spec("imageio") is None:
        missing.append("imageio not installed. `pip install imageio` (needed to pack GT depth EXRs).")
    if need_ffmpeg and shutil.which("ffmpeg") is None:
        missing.append("ffmpeg not on PATH (needed to encode rgb.mp4). Install via apt/conda.")
    if missing:
        for m in missing:
            logger.error(m)
        sys.exit(1)


def build_blend(scene_script, blend_path):
    """Run a Blender scene-builder script (a Python file using bpy)."""
    os.makedirs(os.path.dirname(blend_path), exist_ok=True)
    cmd = [BLENDER_EXEC, "--background", "--python", scene_script, "--", os.path.abspath(blend_path)]
    logger.info(f"Building scene: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    logger.info(f"Scene saved to {blend_path}")


def render_with_visionsim(blend_path, output_dir, device="cuda"):
    """Use VisionSIM to render RGB + ground-truth depth into output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    scene_name = os.path.basename(os.path.normpath(output_dir))
    log_dir = os.path.join("outputs/logs", f"produce_{scene_name}")
    os.makedirs(log_dir, exist_ok=True)
    cmd = [
        "visionsim", "blender.render-animation",
        os.path.abspath(blend_path),
        os.path.abspath(output_dir),
        "--render-config.executable", BLENDER_EXEC,
        "--render-config.width", "640",
        "--render-config.height", "480",
        "--render-config.depths",
        "--render-config.no-debug",
        "--render-config.max-samples", "64",
        "--render-config.device-type", device,
        "--render-config.use-denoising",
        "--render-config.no-allow-skips",
        "--render-config.log-dir", log_dir,
    ]
    logger.info(f"Rendering with VisionSIM: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"VisionSIM stderr:\n{result.stderr}")
        raise RuntimeError("VisionSIM render failed")
    logger.info("VisionSIM render complete")


def pack_depth_gt_npz(output_dir):
    """Pack VisionSIM EXR depth frames into one compressed float32 NPZ.

    VisionSIM writes per-frame EXRs under depths/. The project dataset contract
    stores metric ground-truth depth as one FP32 array in meters, keyed by
    DEPTH_GT_KEY, then removes the per-frame depth directory.
    """
    import imageio.v2 as imageio

    transforms_path = os.path.join(output_dir, "transforms.json")
    if not os.path.exists(transforms_path):
        raise FileNotFoundError(f"VisionSIM did not write {transforms_path}")
    with open(transforms_path) as f:
        transforms = json.load(f)

    depths = []
    for i, frame_info in enumerate(transforms["frames"]):
        depth_rel = frame_info.get("depth_file_path")
        if not depth_rel:
            raise KeyError(f"Frame {i} is missing depth_file_path in transforms.json")
        depth_path = os.path.join(output_dir, depth_rel)
        if not os.path.exists(depth_path):
            raise FileNotFoundError(f"Missing VisionSIM depth EXR: {depth_path}")
        depth = imageio.imread(depth_path)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depths.append(depth.astype(np.float32, copy=False))
        del frame_info["depth_file_path"]

    depth_stack = np.stack(depths, axis=0)
    depth_rel = DEPTH_GT_FILE
    depth_path = os.path.join(output_dir, depth_rel)
    np.savez_compressed(depth_path, **{DEPTH_GT_KEY: depth_stack})

    transforms["depth_gt_file"] = depth_rel
    transforms["depth_gt_key"] = DEPTH_GT_KEY
    transforms["depth_units"] = "meters"
    transforms["invalid_depth_threshold"] = INVALID_DEPTH_THRESHOLD
    with open(transforms_path, "w") as f:
        json.dump(transforms, f, indent=2)

    size_mb = os.path.getsize(depth_path) / 1e6
    logger.info(f"Packed GT depth to {depth_path} shape={depth_stack.shape} size={size_mb:.2f} MB")

    depths_dir = os.path.join(output_dir, "depths")
    shutil.rmtree(depths_dir)
    logger.info(f"Removed {depths_dir} (GT depth is stored in {DEPTH_GT_FILE})")


def encode_rgb_mp4(output_dir, fps=25, crf=18):
    """Encode frames/*.png -> rgb.mp4 (H.264, yuv420p, CRF 18 = visually lossless)."""
    frames_dir = os.path.join(output_dir, "frames")
    mp4_path = os.path.join(output_dir, "rgb.mp4")
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-pattern_type", "glob", "-i", os.path.join(frames_dir, "*.png"),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        "-loglevel", "error", mp4_path,
    ]
    logger.info(f"Encoding RGB to MP4 (CRF {crf}): {mp4_path}")
    subprocess.run(cmd, check=True)
    size_mb = os.path.getsize(mp4_path) / 1e6
    logger.info(f"  rgb.mp4 = {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Produce a VisionSIM dataset for DA3 cliff-detection eval")
    parser.add_argument("--scene-script", default="scripts/visionsim/scenes/tabletop_cliff.py",
                        help="Blender scene-builder script (uses bpy)")
    parser.add_argument("--blend-file", default="data/sim_dataset/tabletop_cliff/scene/scene.blend",
                        help="Path to .blend file (created if missing). Sidecar meta is read from "
                             "<dirname>/scene.meta.json next to it. Conventionally lives at "
                             "<output-dir>/scene/scene.blend so each scene is self-contained.")
    parser.add_argument("--output-dir", default="data/sim_dataset/tabletop_cliff/",
                        help="Dataset output directory. Logs land in outputs/logs/produce_<scene>/.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu", "optix"],
                        help="Blender render device")
    parser.add_argument("--rgb-format", default="mp4", choices=["mp4", "png", "both"],
                        help="What to keep on disk for RGB. mp4 (default): only rgb.mp4 (~232x smaller, "
                             "~3%% AbsRel cost). png: only frames/*.png (lossless). both: keep both.")
    parser.add_argument("--mp4-crf", type=int, default=18,
                        help="H.264 CRF for rgb.mp4 (lower = higher quality). 18 is visually lossless.")
    args = parser.parse_args()

    need_ffmpeg = args.rgb_format in ("mp4", "both")
    check_dependencies(need_ffmpeg=need_ffmpeg)
    if not os.path.exists(args.blend_file):
        build_blend(args.scene_script, args.blend_file)
    render_with_visionsim(args.blend_file, args.output_dir, device=args.device)
    pack_depth_gt_npz(args.output_dir)

    if args.rgb_format in ("mp4", "both"):
        encode_rgb_mp4(args.output_dir, crf=args.mp4_crf)
    if args.rgb_format == "mp4":
        frames_dir = os.path.join(args.output_dir, "frames")
        shutil.rmtree(frames_dir)
        logger.info(f"Removed {frames_dir} (rgb.mp4 is the canonical RGB; pass --rgb-format both to keep PNGs)")

    logger.info(f"Dataset ready at {args.output_dir}")


if __name__ == "__main__":
    main()
