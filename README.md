# repo766

Repository for CS 766 Group Project (Spring 2026)

## Setup

Clone and setup [Depth-Anything-3](https://github.com/ByteDance-Seed/depth-anything-3) inside this repo.

Put test images in `data/`, then run:

```
python scripts/test_depth.py --images data/my_image.png
```

## Running video cliff detection

If you have not generated your depth_raw.npz, run

```
KMP_DUPLICATE_LIB_OK=TRUE python3 ./scripts/test_video_depth.py \
  --video data/video_sample.MOV \
  --fps 2.0 \
  --process-res 518 \
  --model depth-anything/DA3-LARGE
```

DA3-LARGE produces the best results but can take up to 40 minutes to generate the inference using those parameters for a 1 minute video. Once you have depth_raw.npz, you can skip this step and just run

```
KMP_DUPLICATE_LIB_OK=TRUE python3 ./scripts/test_video_depth.py \
 --video data/video_sample.MOV \
 --fps 2.0 \
 --skip-inference
```

which should run in less than 20s



### Some notes/summary stuff from ChatGPT about this algorithm we can use:

This pipeline performs monocular depth-based geometric scene understanding to detect “cliffs,” defined as sudden drops in supporting surface geometry relative to an estimated ground plane. The system begins by applying a pretrained monocular depth estimator (Depth Anything 3) to each RGB frame, producing a dense depth map. Because monocular depth is inherently scale-ambiguous and not metrically calibrated, all subsequent reasoning is performed in relative geometry rather than absolute distance.

The core geometric step is lifting the depth image into 3D space using the pinhole camera model. Each pixel is back-projected into a 3D point cloud using its depth value and camera intrinsics. This transforms the problem from 2D image analysis into 3D surface modeling, where geometric structure becomes more explicit.

A key assumption is that the lower portion of the image corresponds primarily to traversable ground. Points from this region are extracted and used to robustly estimate a ground plane. The algorithm uses RANSAC regression to fit a linear plane model of the form:

y = ax + bz + c

RANSAC is essential because depth maps contain many outliers such as trees, fences, and object boundaries. Unlike least squares, RANSAC is resilient to these outliers and estimates a plane that best represents the dominant ground structure.

After estimating the ground plane, the algorithm computes per-pixel residuals between observed 3D points and the predicted ground surface. These residuals represent vertical deviation from expected support. A cliff is defined as a region where the observed surface lies significantly above the ground plane, indicating a loss of supporting terrain rather than a raised obstacle.

The final binary cliff mask is produced by thresholding these residuals and applying connected component filtering to remove small noisy detections. This enforces spatial coherence and reduces false positives caused by local depth noise.

---

Key challenges include monocular depth noise, semantic bias in learned depth models, and ambiguity between vertical structures (walls, fences, trees) and true drop-offs. These issues are mitigated using RANSAC-based plane fitting, spatial priors on ground location, and residual-based geometric reasoning rather than raw depth thresholds.

---

Tunable parameters include camera intrinsics (fx, fy), the RANSAC residual threshold, the cliff height threshold, and the assumed ground region (bottom image half). These parameters control sensitivity and robustness but do not fundamentally change the geometric formulation.

---

Limitations include the strong planar ground assumption, sensitivity to camera orientation, and failure cases where vertical structures mimic depth discontinuities. The system also lacks temporal consistency across frames and does not explicitly model uncertainty in depth predictions.

---

Future improvements include multi-frame temporal smoothing of the ground plane, semantic segmentation to distinguish walkable surfaces from obstacles, multi-plane geometric modeling for non-flat terrain, and hybrid learning-based classifiers operating on residual geometry. Integration with SLAM systems would further improve stability and real-world applicability.