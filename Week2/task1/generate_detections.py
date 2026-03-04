#!/usr/bin/env python3
"""
regen_finetuned_detections.py

Run from week2_project_C6/ BEFORE task2_1 and task2_2.

Problems this fixes:
  1. yolo26l_finetuned_detections.txt was written with sequential fids (1-N)
     but GT frame IDs start at 218. Detections were evaluated against the
     wrong frames → artificially low IDF1.

  2. The fine-tuned model (mAP@50=0.9661) should dramatically outperform
     the off-the-shelf yolo26l (mAP@50=0.5468). If finetuned IDF1 < yolo26l
     IDF1, the detection file is corrupt.

This script:
  - Loads best_config.json from task1_2 to find weights + conf threshold
  - Runs inference frame-by-frame using cap.set() to seek by GT frame ID
  - Saves results/task1_2/yolo26l_finetuned_detections.txt with correct fids
  - Prints a before/after detection count so you can verify the fix worked
"""

import os, sys, json, cv2
import numpy as np
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_BASE = os.environ.get('DATA_ROOT', 'data')
if os.path.basename(_DATA_BASE.rstrip('/')) == 'c010':
    C010_ROOT = _DATA_BASE
else:
    C010_ROOT = os.path.join(_DATA_BASE, 'AICity_data', 'train', 'S03', 'c010')

ANN_PATH   = os.path.join(C010_ROOT, 'annotations.xml')
GT_FALLBACK= os.path.join(C010_ROOT, 'gt', 'gt.txt')
VIDEO_PATH = os.environ.get('VIDEO_PATH', os.path.join(C010_ROOT, 'vdo.avi'))

T12_JSON   = 'results/task1_2/best_config.json'
OUT_DIR    = 'results/task1_2'
os.makedirs(OUT_DIR, exist_ok=True)

ALLOWED = {'car', 'bicycle', 'motorcycle', 'truck', 'bus'}


# ── GT loader (CVAT XML or MOT txt) ──────────────────────────────────────────
def load_gt():
    import xml.etree.ElementTree as ET

    def _cvat(path):
        tree = ET.parse(path); root = tree.getroot()
        gt = {}
        for track in root.findall('track'):
            if track.attrib.get('label','').lower() not in ('car','vehicle'):
                continue
            for box in track.findall('box'):
                if box.attrib.get('outside','0') == '1': continue
                fid = int(box.attrib['frame']) + 1
                gt.setdefault(fid, []).append([
                    float(box.attrib['xtl']), float(box.attrib['ytl']),
                    float(box.attrib['xbr']), float(box.attrib['ybr']),
                ])
        return gt

    def _mot(path):
        gt = {}
        with open(path) as f:
            for line in f:
                p = line.strip().split(',')
                if len(p) < 6: continue
                fid = int(p[0])
                x,y,w,h = float(p[2]),float(p[3]),float(p[4]),float(p[5])
                conf = float(p[6]) if len(p)>6 else 1.0
                if conf == 0: continue
                gt.setdefault(fid,[]).append([x,y,x+w,y+h])
        return gt

    if os.path.exists(ANN_PATH):
        print(f"  [GT] CVAT XML: {ANN_PATH}")
        return _cvat(ANN_PATH)
    if os.path.exists(GT_FALLBACK):
        print(f"  [GT] MOT txt: {GT_FALLBACK}")
        return _mot(GT_FALLBACK)
    raise FileNotFoundError(f"No GT at {ANN_PATH} or {GT_FALLBACK}")


# ── Detection file parser ─────────────────────────────────────────────────────
def parse_det_mot(path):
    dets = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) < 6: continue
            fid = int(p[0])
            x,y,w,h = float(p[2]),float(p[3]),float(p[4]),float(p[5])
            score = float(p[6]) if len(p)>6 else 1.0
            dets.setdefault(fid,[]).append([x,y,x+w,y+h,score])
    return dets


def save_det_mot(det_dict, out_path):
    with open(out_path, 'w') as f:
        for fid in sorted(det_dict.keys()):
            for b in det_dict[fid]:
                x1,y1,x2,y2 = b[:4]
                score = b[4] if len(b)>4 else 1.0
                f.write(f"{fid},-1,{x1:.2f},{y1:.2f},"
                        f"{x2-x1:.2f},{y2-y1:.2f},{score:.4f},-1,-1,-1\n")


# ── Diagnose existing detection file ─────────────────────────────────────────
def diagnose(det_path, gt_fids, label):
    if not os.path.exists(det_path):
        print(f"  [{label}] file not found: {det_path}")
        return
    dets = parse_det_mot(det_path)
    det_fids = sorted(dets.keys())
    overlap  = len(set(det_fids) & set(gt_fids))
    total_boxes = sum(len(v) for v in dets.values())
    gt_frame_boxes = sum(len(dets.get(f,[])) for f in gt_fids)
    print(f"  [{label}]")
    print(f"    det fid range : {min(det_fids)} – {max(det_fids)}  ({len(det_fids)} frames)")
    print(f"    GT  fid range : {min(gt_fids)} – {max(gt_fids)}   ({len(gt_fids)} frames)")
    print(f"    overlap       : {overlap} / {len(gt_fids)} GT frames  ← {'OK' if overlap > 100 else 'BAD — fid mismatch!'}")
    print(f"    total boxes   : {total_boxes}")
    print(f"    boxes on GT frames: {gt_frame_boxes}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Regenerate fine-tuned detections with correct frame IDs")
    print("=" * 60)

    # Load task1_2 config
    if not os.path.exists(T12_JSON):
        print(f"[ERROR] {T12_JSON} not found — run task1_2 first")
        sys.exit(1)
    t12 = json.load(open(T12_JSON))

    ft_weights = t12.get('best_weights', '')
    ft_model   = t12.get('model_name', 'yolo26l')
    ft_name    = f'{ft_model}_finetuned'
    # Use the conf threshold from task1_2 best config, default 0.25
    best_cfg   = t12.get('best_config', {})
    if isinstance(best_cfg, dict):
        conf_thr = float(best_cfg.get('conf', 0.25))
    else:
        conf_thr = 0.25

    out_path   = os.path.join(OUT_DIR, f'{ft_name}_detections.txt')

    print(f"\n  model    : {ft_model}")
    print(f"  weights  : {ft_weights}")
    print(f"  conf_thr : {conf_thr}")
    print(f"  output   : {out_path}")

    if not ft_weights or not os.path.exists(ft_weights):
        print(f"\n[ERROR] Weights not found: {ft_weights}")
        sys.exit(1)

    # Load GT to get the correct frame IDs
    print("\nLoading GT...")
    gt_dict  = load_gt()
    gt_fids  = sorted(gt_dict.keys())
    print(f"  GT frames: {len(gt_fids)}  ({gt_fids[0]} – {gt_fids[-1]})")

    # Diagnose existing file
    print("\nDiagnosing existing detection file...")
    diagnose(out_path, gt_fids, "CURRENT")

    # Re-run inference with correct frame IDs
    print(f"\nRunning inference on {len(gt_fids)} GT frames...")
    print(f"  video: {VIDEO_PATH}")

    from ultralytics import YOLO
    model = YOLO(ft_weights)

    cap     = cv2.VideoCapture(VIDEO_PATH)
    ft_det  = {}
    n_boxes = 0

    for i, fid in enumerate(gt_fids):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)   # seek by 1-based GT fid
        ret, frame = cap.read()
        if not ret:
            ft_det[fid] = []
            continue

        res   = model(frame, conf=conf_thr, verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            for xyxy, score, cls_id in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy()):
                cls_name = model.names.get(int(cls_id), '').lower()
                if cls_name and cls_name not in ALLOWED:
                    continue
                x1,y1,x2,y2 = xyxy
                boxes.append([float(x1),float(y1),float(x2),float(y2),float(score)])
        ft_det[fid] = boxes
        n_boxes += len(boxes)

        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(gt_fids)} frames  ({n_boxes} boxes so far)")

    cap.release()

    # Back up old file
    if os.path.exists(out_path):
        bak = out_path + '.old'
        os.rename(out_path, bak)
        print(f"\n  Backed up old file → {bak}")

    save_det_mot(ft_det, out_path)
    print(f"  Saved {len(ft_det)} frames, {n_boxes} boxes → {out_path}")

    # Diagnose new file
    print("\nDiagnosing new detection file...")
    diagnose(out_path, gt_fids, "NEW")

    # Quick per-frame stats
    frames_with_dets = sum(1 for v in ft_det.values() if v)
    avg_boxes = n_boxes / max(len(gt_fids), 1)
    print(f"\n  frames with detections : {frames_with_dets} / {len(gt_fids)}")
    print(f"  avg boxes per frame    : {avg_boxes:.1f}")
    print(f"  GT boxes per frame     : {sum(len(v) for v in gt_dict.values())/len(gt_fids):.1f}")

    if frames_with_dets < len(gt_fids) * 0.5:
        print("\n  [WARN] Less than 50% of frames have detections.")
        print("         Check that VIDEO_PATH points to the right file.")
        print(f"         VIDEO_PATH = {VIDEO_PATH}")
    else:
        print("\n  ✓ Detection file looks healthy.")
        print("    Now rerun task2_1 and task2_2 — yolo26l_finetuned IDF1 should")
        print("    be significantly higher than yolo26l.")

    print("\n✓ Done.")


if __name__ == '__main__':
    main()