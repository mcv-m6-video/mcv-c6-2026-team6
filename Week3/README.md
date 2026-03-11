# C6

Multi-Target Single-Camera (MTSC) tracking with optical flow. Includes optical flow evaluation (Task 1.1), object tracking (Task 1.2), and MTSC evaluation (Task 2).

---

## Project Structure

```
Week3/
├── configs/           # Configuration files
├── scripts/           # Entry-point scripts
├── src/
│   ├── evaluation/    # Tracking and flow metrics
│   ├── optical_flow/  # Flow methods and wrappers
│   ├── tracking/      # Tracker implementations
│   └── utils/         # KITTI / AI City utilities
└── ...
```

---

## Configs

| File | Description |
|------|-------------|
| `configs/optical_flow.yaml` | Task 1.1 — KITTI paths, method params (PyFlow, Farneback, RAFT, UniMatch, SEA-RAFT, FlowFormer), evaluation metrics, ablation settings |
| `configs/tracker.yaml` | Task 1.2 + Task 2 — detector config, optical flow backend, tracker HPs (IoU baseline, Kalman, OF tracker), hyperparameter sweep ranges |

---

## Scripts

| Script | Description |
|--------|-------------|
| `scripts/precompute_flow.py` | Pre-compute optical flow for camera sequences. Saves forward/backward `.npy` flows for fast loading. Supports RAFT, Farneback, SEA-RAFT, UniMatch, FlowFormer. |
| `scripts/run_hp_sweep.py` | Hyperparameter sweep for trackers (iou_threshold, max_age, min_hits). Compares Hungarian vs Greedy matching. Writes best HPs to JSON for `run_tracking.py`. |
| `scripts/run_mtsc.py` | Task 2 — Run best tracker on all cameras of SEQ01, SEQ03, SEQ04. Produces IDF1/HOTA/MOTA and optional video output. |
| `scripts/run_optical_flow.py` | Task 1.1 — Evaluate optical flow methods on KITTI Seq 45. Ablations: Farneback winsize, RAFT iters, PyFlow alpha. |
| `scripts/run_tracking.py` | Task 1.2 — Run IoU, Kalman, OF, Adaptive OF trackers on a camera sequence. Computes IDF1/MOTA, ablation, ID-switch analysis, side-by-side comparison. |
| `scripts/viz_tracking.py` | Render predicted and GT bounding boxes on video frames. Export side-by-side or overlay GIFs via FFmpeg. |

---

## Source Modules

### Evaluation

| Module | Description |
|--------|-------------|
| `src/evaluation/tracking_metrics.py` | MOT metrics: IDF1, HOTA, MOTA, MOTP, ID switches, fragmentation. Uses motmetrics. `compute_mot_metrics`, `write_mot_results`, `compute_id_switches`. |

### Optical Flow

| Module | Description |
|--------|-------------|
| `src/optical_flow/adaptive_flow.py` | Adaptive flow aggregation per bounding box: small bbox → trimmed-mean, high variance → mode, FB-unreliable → skip. Original contribution for robust OF warping. |
| `src/optical_flow/unimatch_wrapper.py` | UniMatch (CVPR 2023) wrapper. Transformer-based global matching. Best on KITTI F1. |

### Tracking

| Module | Description |
|--------|-------------|
| `src/tracking/adaptive_of_tracker.py` | OF tracker with per-bbox adaptive aggregation. Uses `adaptive_flow` and FB-consistency. |
| `src/tracking/iou_tracker.py` | Baseline IoU-only tracker. No motion model. |
| `src/tracking/kalman_tracker.py` | SORT-style Kalman filter + IoU association (Bewley et al., ICIP 2016). |
| `src/tracking/of_tracker.py` | Optical flow tracker: median/mean flow aggregation, FB consistency, occlusion recovery, motion filter. |
| `src/tracking/track.py` | Track state machine (TENTATIVE, CONFIRMED, LOST, DELETED). `SimpleTrack`, `KalmanBoxTracker`, `bbox_to_state`, `state_to_bbox`. |

### Utils

| Module | Description |
|--------|-------------|
| `src/utils/kitti_utils.py` | KITTI and AI City helpers: `load_detections_aicity` (NMS, ROI, conf filter), `load_gt_aicity` (skip active==0), `read_kitti_flow_gt`, `write_flo_file`. |

---

## Quick Start

```bash
cd Week3

# 1. Pre-compute flow for a camera
python scripts/precompute_flow.py --seq_dir data/aicity/S03/c010 --method raft

# 2. Run optical flow evaluation (Task 1.1)
python scripts/run_optical_flow.py --config configs/optical_flow.yaml

# 3. Run HP sweep, then tracking (Task 1.2)
python scripts/run_hp_sweep.py --seq_dir data/aicity/S01/c010
python scripts/run_tracking.py --seq_dir data/aicity/S03/c010 \
    --hp_file results/tracking/hp_sweep/best_hyperparameters.json

# 4. MTSC on all cameras (Task 2)
python scripts/run_mtsc.py --aicity_dir data/aicity --seqs S01 S03 S04

# 5. Visualize results
python scripts/viz_tracking.py --video data/aicity/S01/c010/img1 \
    --pred results/tracking/s01/results_c010.txt \
    --gt data/aicity/S01/c010/gt/gt.txt \
    --output output/c010_compare.gif --mode side
```
