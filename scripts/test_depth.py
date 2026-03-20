"""Test script for Depth Anything 3 inference."""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description="Test Depth Anything 3")
    parser.add_argument("--model", default="depth-anything/DA3-SMALL",
                        help="HuggingFace model ID (e.g. depth-anything/DA3-BASE)")
    parser.add_argument("--images", nargs="+", default=[
        "data/",
    ])
    parser.add_argument("--output-dir", default="outputs/")
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()

    # If a directory is passed, expand to all image files in it
    expanded = []
    for path in args.images:
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    expanded.append(os.path.join(path, fname))
        else:
            expanded.append(path)
    args.images = expanded

    if not args.images:
        print("No images found. Put test images in data/ or pass --images <path>.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"\nLoading model: {args.model}")
    t0 = time.time()
    from depth_anything_3.api import DepthAnything3
    model = DepthAnything3.from_pretrained(args.model)
    model = model.to("cuda")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # Run inference
    print(f"\nRunning inference on {len(args.images)} images (process_res={args.process_res})")
    t0 = time.time()
    prediction = model.inference(args.images, process_res=args.process_res)
    elapsed = time.time() - t0
    print(f"Inference done in {elapsed:.2f}s")

    # Print prediction info
    print(f"\nPrediction shapes:")
    print(f"  depth:      {prediction.depth.shape}")
    if prediction.conf is not None:
        print(f"  conf:       {prediction.conf.shape}")
    if prediction.extrinsics is not None:
        print(f"  extrinsics: {prediction.extrinsics.shape}")
    if prediction.intrinsics is not None:
        print(f"  intrinsics: {prediction.intrinsics.shape}")
    print(f"  is_metric:  {prediction.is_metric}")

    # Save raw depth as NPZ
    npz_path = os.path.join(args.output_dir, "depth_raw.npz")
    np.savez_compressed(npz_path, depth=prediction.depth,
                        extrinsics=prediction.extrinsics,
                        intrinsics=prediction.intrinsics)
    print(f"\nSaved raw depth to {npz_path}")

    # Save side-by-side visualizations
    from PIL import Image
    for i, img_path in enumerate(args.images):
        img = np.array(Image.open(img_path))
        depth = prediction.depth[i]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(img)
        axes[0].set_title("Input")
        axes[0].axis("off")

        im = axes[1].imshow(depth, cmap="inferno")
        axes[1].set_title("Depth")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        out_path = os.path.join(args.output_dir, f"depth_{i:03d}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved visualization: {out_path}")

    print(f"\nDone! Outputs in {args.output_dir}/")


if __name__ == "__main__":
    main()
