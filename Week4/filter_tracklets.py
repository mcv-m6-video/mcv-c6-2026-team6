"""
filter_tracklets.py
===================
Filter YOLO tracking predictions to match the criteria the GT evaluator uses:

  1. active==0 / occluded boxes  → drop boxes below min_area or min_conf
  2. Single-camera vehicles      → drop tracklets that only appear in 1 camera
                                   (the MTMC evaluator ignores these)
  3. ROI mask                    → drop detections outside the camera ROI
  4. Per-frame NMS               → remove duplicate detections
  5. Min-length tracklets        → drop tracklets shorter than min_track_len frames
                                   (parked / briefly visible vehicles)

The filtered output is saved in the SAME MOTChallenge format as the input,
one .txt per camera, ready to be consumed by cityflow_preprocessing.py.

Usage
-----
# Filter a single camera
python -m src.data.filter_tracklets \
    --det-file  data/.../c001/det/det.txt \
    --out-file  data/.../c001/det/det_filtered.txt \
    --roi-file  data/.../c001/roi.jpg \
    --min-conf  0.4 --min-area 400 --min-track-len 3

# Filter an entire sequence (all cameras at once) and cross-camera filter
python -m src.data.filter_tracklets \
    --seq-dir   data/AIC22_Track1_MTMC_Tracking/train/S03 \
    --out-dir   data/AIC22_Track1_MTMC_Tracking/train/S03_filtered \
    --min-conf  0.4 --min-area 400 --min-track-len 3 \
    --multi-cam-only   # <-- drop vehicles seen in only 1 camera
"""

from __future__ import annotations
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# NMS (reused from utils.py logic, self-contained here)
# ---------------------------------------------------------------------------

def _nms_boxes(boxes: List[List], iou_thresh: float = 0.5) -> List[List]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda r: r[4], reverse=True)
    kept = []
    for cand in boxes:
        cx1, cy1, cx2, cy2 = cand[0], cand[1], cand[2], cand[3]
        suppress = False
        for k in kept:
            ix1 = max(cx1, k[0]); iy1 = max(cy1, k[1])
            ix2 = min(cx2, k[2]); iy2 = min(cy2, k[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area_c = (cx2 - cx1) * (cy2 - cy1)
            area_k = (k[2] - k[0]) * (k[3] - k[1])
            union  = area_c + area_k - inter
            if union > 0 and inter / union >= iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(cand)
    return kept


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_mot_file(path: str) -> Dict[int, List[List]]:
    """
    Read MOTChallenge-format tracking file.
    Handles both comma-separated (YOLO tracker output) and
    space-separated formats automatically.
    Row: frame, track_id, x, y, w, h, conf, -1, -1, -1
    Returns {track_id: [[frame, x1, y1, x2, y2, conf], ...]}

    Special case: if all track_ids are -1 (raw detection format, not tracked),
    assigns pseudo-track IDs by grouping detections with frame gaps < 5 frames
    that also have overlapping bounding boxes (simple greedy IoU tracker).
    """
    raw: Dict[int, List] = defaultdict(list)
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',') if ',' in line else line.split()
            if len(parts) < 6:
                continue
            try:
                frame  = int(float(parts[0]))
                tid    = int(float(parts[1]))
                x, y   = float(parts[2]), float(parts[3])
                w, h   = float(parts[4]), float(parts[5])
                conf   = float(parts[6]) if len(parts) > 6 else 1.0
            except (ValueError, IndexError):
                continue
            raw[tid].append([frame, x, y, x + w, y + h, conf])

    # If all tids are -1 (detection format), run simple greedy IoU tracker
    if set(raw.keys()) == {-1}:
        return _assign_track_ids(raw[-1])

    return dict(raw)


def _assign_track_ids(dets: List[List]) -> Dict[int, List[List]]:
    """
    Greedy IoU tracker for detection-format files (all tid=-1).
    Groups detections into tracklets: same track if IoU>0.3 in consecutive frames.
    Max frame gap allowed: 5 frames.
    """
    if not dets:
        return {}

    dets_sorted = sorted(dets, key=lambda d: d[0])   # sort by frame

    def _iou(a, b):
        ix1 = max(a[1], b[1]); iy1 = max(a[2], b[2])
        ix2 = min(a[3], b[3]); iy2 = min(a[4], b[4])
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        ua = (a[3]-a[1])*(a[4]-a[2]) + (b[3]-b[1])*(b[4]-b[2]) - inter
        return inter / ua if ua > 0 else 0

    MAX_GAP = 5
    IOU_THR = 0.3

    tracks: Dict[int, List] = {}     # tid → [dets]
    active: List             = []    # [(tid, last_frame, last_det)]
    next_tid                 = 1

    for det in dets_sorted:
        frame = det[0]
        best_tid   = -1
        best_iou   = IOU_THR
        best_idx   = -1

        for idx, (tid, last_frame, last_det) in enumerate(active):
            if frame - last_frame > MAX_GAP:
                continue
            v = _iou(det, last_det)
            if v > best_iou:
                best_iou, best_tid, best_idx = v, tid, idx

        if best_tid == -1:
            # New tracklet
            best_tid = next_tid
            next_tid += 1
            tracks[best_tid] = []
            active.append([best_tid, frame, det])
        else:
            active[best_idx] = [best_tid, frame, det]

        tracks[best_tid].append(det)

    # Remove stale active entries (cleanup)
    return tracks


def load_roi_mask(roi_file: Optional[str]):
    if not roi_file:
        return None
    roi_img = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
    if roi_img is None:
        return None
    return roi_img > 127


# ---------------------------------------------------------------------------
# Per-camera filter
# ---------------------------------------------------------------------------

def filter_camera_tracklets(
    tracklets:     Dict[int, List[List]],
    roi_mask:      Optional[np.ndarray] = None,
    min_conf:      float = 0.3,
    min_area:      float = 400.0,
    min_track_len: int   = 3,
    nms_iou:       float = 0.5,
) -> Dict[int, List[List]]:
    """
    Apply per-detection and per-tracklet filters to one camera's tracklets.

    Filters applied in order:
      1. Confidence threshold
      2. Minimum bounding-box area
      3. ROI mask (centre-point must be inside ROI)
      4. Per-frame NMS across all detections in a frame
      5. Minimum tracklet length (frames with surviving detections)

    Returns filtered {track_id: [[frame, x1, y1, x2, y2, conf], ...]}
    """
    # Step 1–3: per-detection filters
    per_frame: Dict[int, List] = defaultdict(list)   # frame → [x1,y1,x2,y2,conf,tid]
    for tid, dets in tracklets.items():
        for det in dets:
            frame, x1, y1, x2, y2, conf = det
            if conf < min_conf:
                continue
            if (x2 - x1) * (y2 - y1) < min_area:
                continue
            if roi_mask is not None:
                cx = int(np.clip((x1 + x2) / 2, 0, roi_mask.shape[1] - 1))
                cy = int(np.clip((y1 + y2) / 2, 0, roi_mask.shape[0] - 1))
                if not roi_mask[cy, cx]:
                    continue
            per_frame[frame].append([x1, y1, x2, y2, conf, tid])

    # Step 4: per-frame NMS (operates on [x1,y1,x2,y2,conf] slice)
    kept_by_tid: Dict[int, List] = defaultdict(list)
    for frame, dets in per_frame.items():
        boxes_for_nms = [[d[0], d[1], d[2], d[3], d[4]] for d in dets]
        tids_for_nms  = [d[5] for d in dets]

        # Build NMS result → map back to track IDs
        kept_boxes = _nms_boxes(boxes_for_nms, nms_iou)
        kept_set   = {tuple(b[:4]) for b in kept_boxes}

        for d in dets:
            box_key = tuple(d[:4])
            if box_key in kept_set:
                tid = d[5]
                kept_by_tid[tid].append([frame, d[0], d[1], d[2], d[3], d[4]])

    # Step 5: minimum tracklet length
    filtered = {
        tid: sorted(dets, key=lambda x: x[0])
        for tid, dets in kept_by_tid.items()
        if len(dets) >= min_track_len
    }
    return filtered


# ---------------------------------------------------------------------------
# Cross-camera filter: keep only vehicles seen in >= min_cameras cameras
# ---------------------------------------------------------------------------

def filter_multi_camera(
    all_cam_tracklets: Dict[str, Dict[int, List[List]]],
    gt_mtmc_path:      Optional[str] = None,
    min_cameras:       int = 2,
) -> Dict[str, Dict[int, List[List]]]:
    """
    Remove tracklets whose vehicle does not appear in at least min_cameras cameras.

    Two modes:
      - gt_mtmc_path provided: use GT global IDs to identify cross-camera vehicles.
        Keeps only track_ids that map to a global_id seen in >= min_cameras cams.
      - gt_mtmc_path=None (test mode): use appearance-based heuristic —
        keep all tracklets (can't know which are multi-camera without GT).
        In this mode the filter is a no-op and a warning is printed.

    Returns filtered version of all_cam_tracklets.
    """
    if gt_mtmc_path is None or not Path(gt_mtmc_path).exists():
        print(
            "  [filter_multi_camera] No gt_mtmc.txt found — "
            "skipping cross-camera filter (test/inference mode)."
        )
        return all_cam_tracklets

    # Parse gt_mtmc.txt: <cam_id> <global_vehicle_id> <frame_id> <x> <y> <w> <h>
    # In CityFlow, global_vehicle_id (parts[1]) == local track ID in gt/gt.txt
    global_id_map: Dict[Tuple[str, int], int] = {}
    gid_cameras: Dict[int, set] = defaultdict(set)
    with open(gt_mtmc_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            cam_raw   = parts[0]
            global_id = int(parts[1])
            cam_name  = cam_raw if cam_raw.startswith('c') else f'c{int(cam_raw):03d}'
            global_id_map[(cam_name, global_id)] = global_id
            gid_cameras[global_id].add(cam_name)

    # Keep only global IDs seen in >= min_cameras
    valid_gids = {gid for gid, cams in gid_cameras.items() if len(cams) >= min_cameras}

    # Filter
    filtered: Dict[str, Dict[int, List[List]]] = {}
    for cam_name, cam_tracklets in all_cam_tracklets.items():
        kept = {}
        for tid, dets in cam_tracklets.items():
            gid = global_id_map.get((cam_name, tid))
            if gid is not None and gid in valid_gids:
                kept[tid] = dets
            elif gid is None:
                # Track ID not in GT → keep anyway (predicted track, not GT-matched)
                kept[tid] = dets
        filtered[cam_name] = kept

    total_before = sum(len(v) for v in all_cam_tracklets.values())
    total_after  = sum(len(v) for v in filtered.values())
    print(
        f"  [filter_multi_camera] Kept {total_after}/{total_before} tracklets "
        f"(min_cameras={min_cameras})"
    )
    return filtered


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_mot_file(
    tracklets: Dict[int, List[List]],
    out_path:  str,
):
    """Write filtered tracklets back to MOTChallenge format."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for tid, dets in tracklets.items():
        for det in dets:
            frame, x1, y1, x2, y2, conf = det
            w = x2 - x1
            h = y2 - y1
            rows.append((frame, tid, x1, y1, w, h, conf))
    rows.sort(key=lambda r: (r[0], r[1]))
    with open(out_path, 'w') as f:
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.4f},-1,-1,-1\n")
    print(f"  Wrote {len(rows)} rows → {out_path}")


# ---------------------------------------------------------------------------
# Sequence-level runner
# ---------------------------------------------------------------------------

def filter_sequence(
    seq_dir:        str,
    out_dir:        str,
    det_subpath:    str   = 'det/det.txt',
    roi_subpath:    str   = 'roi.jpg',
    min_conf:       float = 0.3,
    min_area:       float = 400.0,
    min_track_len:  int   = 3,
    nms_iou:        float = 0.5,
    multi_cam_only: bool  = False,
    gt_mtmc_path:   Optional[str] = None,
):
    """
    Filter all cameras in a sequence and optionally apply cross-camera filter.

    seq_dir layout:
      c001/det/det.txt  (or whatever det_subpath you specify)
      c001/roi.jpg
      c002/...
    """
    seq_dir = Path(seq_dir)
    out_dir = Path(out_dir)

    cameras = sorted([d for d in seq_dir.iterdir()
                      if d.is_dir() and d.name.startswith('c')])

    all_cam_tracklets: Dict[str, Dict] = {}

    for cam_dir in cameras:
        cam_name = cam_dir.name
        det_file = cam_dir / det_subpath
        roi_file = cam_dir / roi_subpath

        if not det_file.exists():
            print(f"  [skip] {cam_name}: no det file at {det_file}")
            continue

        tracklets = load_mot_file(str(det_file))
        roi_mask  = load_roi_mask(str(roi_file) if roi_file.exists() else None)

        n_before = len(tracklets)
        filtered = filter_camera_tracklets(
            tracklets,
            roi_mask      = roi_mask,
            min_conf      = min_conf,
            min_area      = min_area,
            min_track_len = min_track_len,
            nms_iou       = nms_iou,
        )
        n_after = len(filtered)
        print(f"  {cam_name}: {n_before} → {n_after} tracklets after per-cam filter")

        all_cam_tracklets[cam_name] = filtered

    # Cross-camera filter
    if multi_cam_only:
        all_cam_tracklets = filter_multi_camera(
            all_cam_tracklets,
            gt_mtmc_path = gt_mtmc_path,
            min_cameras  = 2,
        )

    # Write output
    for cam_name, cam_tracklets in all_cam_tracklets.items():
        out_path = out_dir / cam_name / det_subpath
        write_mot_file(cam_tracklets, str(out_path))

    print(f"\nDone. Filtered detections saved to: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description='Filter YOLO tracklets for MTMC')

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--det-file',  help='Single camera det.txt input')
    mode.add_argument('--seq-dir',   help='Sequence root (all cameras)')

    p.add_argument('--out-file',    help='Output path (single-camera mode)')
    p.add_argument('--out-dir',     help='Output root (sequence mode)')
    p.add_argument('--roi-file',    default=None, help='ROI mask image')
    p.add_argument('--det-subpath', default='det/det.txt',
                   help='Relative path from camera dir to det file')

    # Filter params
    p.add_argument('--min-conf',       type=float, default=0.3)
    p.add_argument('--min-area',       type=float, default=400.0,
                   help='Min bbox area in pixels (drops tiny/distant vehicles)')
    p.add_argument('--min-track-len',  type=int,   default=3,
                   help='Min frames a tracklet must survive')
    p.add_argument('--nms-iou',        type=float, default=0.5)
    p.add_argument('--multi-cam-only', action='store_true',
                   help='Drop tracklets not seen in >= 2 cameras (needs GT MTMC)')
    p.add_argument('--gt-mtmc',        default=None,
                   help='Path to gt_mtmc.txt for cross-camera filter')

    args = p.parse_args()

    if args.det_file:
        # Single camera mode
        tracklets = load_mot_file(args.det_file)
        roi_mask  = load_roi_mask(args.roi_file)
        filtered  = filter_camera_tracklets(
            tracklets,
            roi_mask      = roi_mask,
            min_conf      = args.min_conf,
            min_area      = args.min_area,
            min_track_len = args.min_track_len,
            nms_iou       = args.nms_iou,
        )
        out = args.out_file or args.det_file.replace('.txt', '_filtered.txt')
        write_mot_file(filtered, out)
    else:
        # Sequence mode
        out_dir = args.out_dir or str(Path(args.seq_dir).parent /
                                      (Path(args.seq_dir).name + '_filtered'))
        filter_sequence(
            seq_dir        = args.seq_dir,
            out_dir        = out_dir,
            det_subpath    = args.det_subpath,
            roi_subpath    = 'roi.jpg',
            min_conf       = args.min_conf,
            min_area       = args.min_area,
            min_track_len  = args.min_track_len,
            nms_iou        = args.nms_iou,
            multi_cam_only = args.multi_cam_only,
            gt_mtmc_path   = args.gt_mtmc,
        )


if __name__ == '__main__':
    main()