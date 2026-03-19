"""
run_yolo_tracking.py
====================
Run YOLO tracking on every camera of every CityFlow sequence and save
per-camera MOTChallenge-format .txt files ready for filter_tracklets.py.

Data split used:
  S01, S04  →  train  (fine-tuned weights, leakage-safe if not from S03)
  S03       →  test   (BASE weights only — keeps test set clean)

Output structure (mirrors what filter_tracklets.py expects):
  {seq_root}/
    c001/
      det/
        det.txt        ← MOTChallenge: frame,tid,x,y,w,h,conf,-1,-1,-1
    c002/
      det/
        det.txt
    ...

Usage
-----
# Basic — auto-discovers all sequences and cameras
python -m scripts.run_yolo_tracking \
    --data-root  data/AIC22_Track1_MTMC_Tracking/train \
    --ft-weights yolo26l_ft.pt \
    --base-weights yolo26l.pt

# Override which sequences use which weights
python -m scripts.run_yolo_tracking \
    --data-root  data/AIC22_Track1_MTMC_Tracking/train \
    --ft-weights   yolo26l_ft.pt   --ft-seqs   S01 S04 \
    --base-weights yolo26l.pt      --base-seqs S03

# Dry-run: print what would be run without executing
python -m scripts.run_yolo_tracking --data-root ... --dry-run

# Skip already-done cameras (safe to re-run after interruption)
python -m scripts.run_yolo_tracking --data-root ... --skip-existing
"""

from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict


# ---------------------------------------------------------------------------
# YOLO tracker wrapper
# ---------------------------------------------------------------------------

def run_tracker_on_camera(
    img_dir:     str,
    out_txt:     str,
    weights:     str,
    conf:        float = 0.25,
    iou:         float = 0.45,
    imgsz:       int   = 1280,
    tracker:     str   = 'bytetrack.yaml',
    device:      str   = '0',
    classes:     Optional[List[int]] = None,  # None = all, [2,5,7] = car/bus/truck
    skip_existing: bool = False,
    dry_run:     bool  = False,
) -> bool:
    """
    Run YOLO tracking on a single camera image directory.

    Saves results to out_txt in MOTChallenge format:
        frame, track_id, x, y, w, h, conf, -1, -1, -1

    Returns True on success, False on failure.
    """
    out_path = Path(out_txt)

    if skip_existing and out_path.exists() and out_path.stat().st_size > 0:
        print(f"    [skip] already exists: {out_path}")
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"    [dry-run] would track: {img_dir}")
        print(f"              → {out_txt}")
        print(f"              weights={weights} conf={conf} iou={iou} imgsz={imgsz}")
        return True

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    model = YOLO(weights)

    # Build source glob — YOLO accepts a directory of images directly
    img_path = Path(img_dir)
    if not img_path.exists():
        print(f"    [warn] img dir not found: {img_dir}")
        return False

    print(f"    tracking: {img_dir}")
    print(f"    → {out_txt}")

    results = model.track(
        source   = str(img_path),
        conf     = conf,
        iou      = iou,
        imgsz    = imgsz,
        tracker  = tracker,
        device   = device,
        classes  = classes,
        persist  = True,
        stream   = True,
        verbose  = False,
        agnostic_nms = True,   # class-agnostic NMS helps when classes differ
    )

    rows: List[str] = []
    frame_no = 1

    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            frame_no += 1
            continue

        boxes = result.boxes.xyxy.cpu().numpy()   # x1,y1,x2,y2
        confs = result.boxes.conf.cpu().numpy()

        # ids can be None if tracker lost all objects — fall back to det index
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            # No track IDs: assign sequential IDs per frame
            # (detection-only mode — happens when conf is near threshold)
            ids = [frame_no * 10000 + i for i in range(len(boxes))]

        for box, tid, cf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            # MOTChallenge format: frame,id,x,y,w,h,conf,-1,-1,-1
            rows.append(
                f"{frame_no},{tid},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{cf:.4f},-1,-1,-1"
            )

        frame_no += 1

    with open(out_txt, 'w') as f:
        f.write('\n'.join(rows))
        if rows:
            f.write('\n')

    print(f"    wrote {len(rows)} detections across {frame_no-1} frames")
    if len(rows) == 0:
        print(f"    [warn] 0 detections — possible causes:")
        print(f"           1. Model not suited for this camera (wrong training domain)")
        print(f"           2. Try lower --conf (e.g. 0.10 or 0.05)")
        print(f"           3. Run: python diagnose_yolo.py  to debug")
        print(f"           The pipeline will use MTSC tracklets for this camera instead.")
    return True


# ---------------------------------------------------------------------------
# Sequence runner
# ---------------------------------------------------------------------------

def run_sequence(
    seq_path:      Path,
    weights:       str,
    conf:          float,
    iou:           float,
    imgsz:         int,
    tracker:       str,
    device:        str,
    classes:       Optional[List[int]],
    skip_existing: bool,
    dry_run:       bool,
) -> Dict[str, bool]:
    """Run tracking on all cameras in one sequence. Returns {cam_name: success}."""
    cameras = sorted([d for d in seq_path.iterdir()
                      if d.is_dir() and d.name.startswith('c')])

    if not cameras:
        print(f"  [warn] no camera dirs found in {seq_path}")
        return {}

    results = {}
    for cam_dir in cameras:
        cam_name = cam_dir.name
        out_txt  = cam_dir / 'det' / 'det.txt'

        print(f"\n  [{seq_path.name}/{cam_name}]")

        # Find frame directory — CityFlow uses img1/ but some sequences use vdo/
        img_dir = None
        for candidate in ['img1', 'vdo', 'frames', 'images']:
            d = cam_dir / candidate
            if d.exists() and d.is_dir():
                img_dir = d
                break

        if img_dir is None:
            # Check for a video file directly in cam_dir
            video_files = list(cam_dir.glob('*.mp4')) + list(cam_dir.glob('*.avi')) + list(cam_dir.glob('*.mov'))
            if video_files:
                img_dir = video_files[0]  # YOLO can track video files too
                print(f"    [info] using video file: {img_dir.name}")
            else:
                print(f"    [warn] no frame dir found in {cam_dir} (tried img1/, vdo/, frames/)")
                results[cam_name] = False
                continue

        ok = run_tracker_on_camera(
            img_dir       = str(img_dir),
            out_txt       = str(out_txt),
            weights       = weights,
            conf          = conf,
            iou           = iou,
            imgsz         = imgsz,
            tracker       = tracker,
            device        = device,
            classes       = classes,
            skip_existing = skip_existing,
            dry_run       = dry_run,
        )
        results[cam_name] = ok

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    data_root = Path(args.data_root)
    if not data_root.exists():
        print(f"ERROR: data-root not found: {data_root}")
        sys.exit(1)

    # Fine-tuned models typically use class 0 = vehicle (single-class)
    # Base YOLO uses COCO classes 2=car, 3=motorcycle, 5=bus, 7=truck
    # Default: None = no class filter, let the model output what it detects
    # Override with --classes 2 3 5 7 if using base weights on a clean COCO model
    classes = args.classes if args.classes else None

    # If --seqs is given, run only those sequences with ft-weights
    if args.seqs:
        ft_seqs   = set(args.seqs)
        base_seqs = set()
    else:
        ft_seqs   = set(args.ft_seqs)
        base_seqs = set(args.base_seqs)

    # Discover all sequences in data_root
    # Match only bare sequence dirs like S01, S03, S04 — not S01_filtered etc.
    import re
    all_seqs = sorted([d for d in data_root.iterdir()
                       if d.is_dir() and re.fullmatch(r'S\d+', d.name)])
    if not all_seqs:
        print(f"ERROR: no S0* sequence dirs found in {data_root}")
        sys.exit(1)

    print(f"\nFound sequences: {[s.name for s in all_seqs]}")
    print(f"Fine-tuned weights ({args.ft_weights}): {sorted(ft_seqs)}")
    print(f"Base weights      ({args.base_weights}): {sorted(base_seqs)}")
    print(f"Classes tracked: {classes if classes else 'all (no filter)'}")
    print(f"Tracker: {args.tracker}  conf={args.conf}  iou={args.iou}  imgsz={args.imgsz}")
    if args.dry_run:
        print("\n*** DRY RUN — no files will be written ***\n")

    start = time.time()
    all_results: Dict[str, Dict] = {}

    for seq_dir in all_seqs:
        seq_name = seq_dir.name

        # Determine which weights to use
        if seq_name in base_seqs:
            weights = args.base_weights
            tag     = 'base'
        elif seq_name in ft_seqs:
            weights = args.ft_weights
            tag     = 'fine-tuned'
        else:
            # Default: S03 → base, everything else → fine-tuned
            if seq_name == 'S03':
                weights = args.base_weights
                tag     = 'base (default for test set)'
            else:
                weights = args.ft_weights
                tag     = 'fine-tuned (default)'

        if not Path(weights).exists():
            print(f"\n[warn] weights not found: {weights} — skipping {seq_name}")
            continue

        print(f"\n{'='*55}")
        print(f"  Sequence: {seq_name}  |  weights: {tag}")
        print(f"{'='*55}")

        seq_results = run_sequence(
            seq_path      = seq_dir,
            weights       = weights,
            conf          = args.conf,
            iou           = args.iou,
            imgsz         = args.imgsz,
            tracker       = args.tracker,
            device        = args.device,
            classes       = classes,
            skip_existing = args.skip_existing,
            dry_run       = args.dry_run,
        )
        all_results[seq_name] = seq_results

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'='*55}")
    print(f"  YOLO tracking complete  ({elapsed/60:.1f} min)")
    print(f"{'='*55}")

    total_ok   = 0
    total_fail = 0
    for seq_name, cam_results in all_results.items():
        ok   = sum(v for v in cam_results.values())
        fail = len(cam_results) - ok
        total_ok   += ok
        total_fail += fail
        status = '✓' if fail == 0 else f'✗ {fail} failed'
        print(f"  {seq_name}: {ok}/{len(cam_results)} cameras  {status}")

    print(f"\n  Total: {total_ok} cameras OK, {total_fail} failed")

    if total_fail > 0:
        print("\n  Failed cameras:")
        for seq_name, cam_results in all_results.items():
            for cam, ok in cam_results.items():
                if not ok:
                    print(f"    {seq_name}/{cam}")

    print(f"\nNext step: run filter_tracklets.py")
    print(f"  python -m src.data.filter_tracklets \\")
    print(f"      --seq-dir {data_root}/S03 \\")
    print(f"      --out-dir {data_root}/S03_filtered \\")
    print(f"      --min-conf 0.4 --min-area 400 --min-track-len 3 --multi-cam-only \\")
    print(f"      --gt-mtmc {data_root}/S03/gt_mtmc.txt")


def parse_args():
    p = argparse.ArgumentParser(
        description='Run YOLO tracking on CityFlow sequences',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    p.add_argument('--data-root',    required=True,
                   help='Root containing S01/, S03/, S04/ dirs')
    p.add_argument('--ft-weights',   default='yolo26l_ft.pt',
                   help='Fine-tuned YOLO weights (used for train sequences)')
    p.add_argument('--base-weights', default='yolo26l.pt',
                   help='Base YOLO weights (used for test sequence S03)')

    # Which sequences get which weights
    p.add_argument('--ft-seqs',   nargs='+', default=['S01', 'S04'],
                   help='Sequences to run with fine-tuned weights')
    p.add_argument('--base-seqs', nargs='+', default=['S03'],
                   help='Sequences to run with base weights')
    p.add_argument('--seqs', nargs='+', default=None,
                   help='Run only these sequences (uses ft-weights for all of them). '
                        'Overrides --ft-seqs and --base-seqs when set.')

    # Detection params
    p.add_argument('--conf',    type=float, default=0.25,
                   help='Detection confidence threshold')
    p.add_argument('--iou',     type=float, default=0.45,
                   help='NMS IoU threshold inside YOLO')
    p.add_argument('--imgsz',   type=int,   default=1280,
                   help='Inference image size (pixels)')
    p.add_argument('--classes', type=int,   nargs='+', default=None,
                   help='Class IDs to track. Default: None (all classes). '
                        'Use --classes 2 3 5 7 for base COCO model vehicle-only tracking.')
    p.add_argument('--tracker', type=str,   default='bytetrack.yaml',
                   help='Tracker config: bytetrack.yaml or botsort.yaml')

    # Hardware
    p.add_argument('--device', type=str, default='0',
                   help='CUDA device id, or "cpu"')

    # Behaviour
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip cameras that already have a det.txt')
    p.add_argument('--dry-run', action='store_true',
                   help='Print what would be run without executing')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())