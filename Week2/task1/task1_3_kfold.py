"""
task1/task1_3_kfold.py  (fixed)

Fixes applied:
  1. Replaced parse_annotations_mot(GT_PATH) with inline parse_cvat_xml()
     GT_PATH had a doubled path (DATA_ROOT already ends in c010).
     CVAT XML is the actual annotation format.
  2. Fixed best_cfg['lr'] → best_cfg.get('lr0', best_cfg.get('lr', 0.01))
     task1_2 saves the key as 'lr0', not 'lr' — was causing lr=0 → zero mAP.
  3. Changed vdo.mp4 → vdo.avi (both module-level and the shadowed local).
  4. Inlined all utils.* dependencies:
       parse_annotations_mot, parse_detections_mot, save_detections_mot,
       strategy_a_split, strategy_b_kfold, strategy_c_random_kfold,
       VideoFrameLoader, evaluate_detections, compute_ap, iou
  5. Removed save_detections_mot call that referenced undefined import.
"""

import os
import sys
import json
import cv2
import random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

DATA_ROOT   = os.environ.get('DATA_ROOT', 'data/AICity_data/train/S03/c010')
ANN_PATH    = os.path.join(DATA_ROOT, 'annotations.xml')
GT_FALLBACK = os.path.join(DATA_ROOT, 'AICity_data/train/S03/c010/gt', 'gt.txt')   # likely absent
DET_DIR     = os.path.join(DATA_ROOT, 'det')
VIDEO_PATH = os.environ.get('VIDEO_PATH',
             os.path.join(DATA_ROOT, 'AICity_data/train/S03/c010/vdo.avi'))
RESULTS_DIR = 'results/task1_3'
PLOTS_DIR   = 'plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

ALLOWED_CLASS_NAMES = {'car', 'bicycle', 'motorcycle'}


# ═══════════════════════════════════════════════════════════════════
# Inlined utilities
# ═══════════════════════════════════════════════════════════════════

# ── GT parsers ──────────────────────────────────────────────────────

def parse_cvat_xml(xml_path):
    """
    Parse CVAT XML → {frame_id_1based: [[x1,y1,x2,y2], ...]}
    CVAT frame attribute is 0-based; we convert to 1-based for MOT compatibility.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    gt = {}
    for track in root.findall('track'):
        label = track.attrib.get('label', '').lower()
        if label not in ('car', 'vehicle'):
            continue
        for box in track.findall('box'):
            if box.attrib.get('outside', '0') == '1':
                continue
            fid = int(box.attrib['frame']) + 1   # 0-based → 1-based
            x1  = float(box.attrib['xtl'])
            y1  = float(box.attrib['ytl'])
            x2  = float(box.attrib['xbr'])
            y2  = float(box.attrib['ybr'])
            gt.setdefault(fid, []).append([x1, y1, x2, y2])
    return gt


def parse_annotations_mot(gt_path):
    """Parse MOT-format gt.txt → {frame_id: [[x1,y1,x2,y2], ...]}"""
    gt = {}
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            fid  = int(parts[0])
            x, y = float(parts[2]), float(parts[3])
            w, h = float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0:
                continue
            gt.setdefault(fid, []).append([x, y, x + w, y + h])
    return gt


def load_gt():
    """Load GT: prefer CVAT XML, fall back to MOT txt."""
    if os.path.exists(ANN_PATH):
        print(f"  [GT] Parsing CVAT XML: {ANN_PATH}")
        return parse_cvat_xml(ANN_PATH)
    if os.path.exists(GT_FALLBACK):
        print(f"  [GT] Parsing MOT txt: {GT_FALLBACK}")
        return parse_annotations_mot(GT_FALLBACK)
    raise FileNotFoundError(
        f"No GT found at:\n  {ANN_PATH}\n  {GT_FALLBACK}")


# ── Detection parser / saver ────────────────────────────────────────

def parse_detections_mot(det_path):
    """Parse MOT-format detection file → {frame_id: [[x1,y1,x2,y2,score], ...]}"""
    dets = {}
    with open(det_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            fid   = int(parts[0])
            x, y  = float(parts[2]), float(parts[3])
            w, h  = float(parts[4]), float(parts[5])
            score = float(parts[6]) if len(parts) > 6 else 1.0
            dets.setdefault(fid, []).append([x, y, x + w, y + h, score])
    return dets


def save_detections_mot(det_dict, out_path):
    """Save detection dict to MOT-format txt file."""
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w') as f:
        for fid in sorted(det_dict.keys()):
            for b in det_dict[fid]:
                x1, y1, x2, y2 = b[:4]
                score = b[4] if len(b) > 4 else 1.0
                w, h  = x2 - x1, y2 - y1
                f.write(f"{fid},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},"
                        f"{score:.4f},-1,-1,-1\n")


# ── Split strategies ─────────────────────────────────────────────────

def strategy_a_split(frame_ids, train_ratio=0.25):
    """First train_ratio frames for train, rest for test."""
    n     = len(frame_ids)
    n_tr  = max(1, int(n * train_ratio))
    ids   = sorted(frame_ids)
    return ids[:n_tr], ids[n_tr:]


def strategy_b_kfold(frame_ids, k=4):
    """Fixed consecutive K-fold: each fold is a contiguous block."""
    ids    = sorted(frame_ids)
    n      = len(ids)
    size   = n // k
    folds  = []
    for i in range(k):
        start    = i * size
        end      = start + size if i < k - 1 else n
        test_ids = ids[start:end]
        train_ids = ids[:start] + ids[end:]
        folds.append((train_ids, test_ids))
    return folds


def strategy_c_random_kfold(frame_ids, k=4, seed=42):
    """Random K-fold: frames randomly shuffled then split."""
    ids = sorted(frame_ids)
    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n    = len(shuffled)
    size = n // k
    folds = []
    for i in range(k):
        start     = i * size
        end       = start + size if i < k - 1 else n
        test_ids  = shuffled[start:end]
        train_ids = shuffled[:start] + shuffled[end:]
        folds.append((sorted(train_ids), sorted(test_ids)))
    return folds


# ── IoU and AP ──────────────────────────────────────────────────────

def iou(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    xi1 = max(box_a[0], box_b[0]); yi1 = max(box_a[1], box_b[1])
    xi2 = min(box_a[2], box_b[2]); yi2 = min(box_a[3], box_b[3])
    inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2]-box_a[0]) * (box_a[3]-box_a[1])
    area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])
    return inter / (area_a + area_b - inter + 1e-9)


def compute_ap(precisions, recalls):
    """11-point interpolated AP."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += max(p_at_r) if p_at_r else 0.0
    return ap / 11.0


def evaluate_detections(gt_dict, det_dict, iou_threshold=0.5):
    """
    Compute AP, precision, recall, F1, mIoU over the provided frame dicts.
    Returns dict with keys: AP, precision, recall, F1, mIoU
    """
    all_scores, all_tp, all_fp = [], [], []
    n_gt_total = 0
    iou_sum    = 0.0
    iou_count  = 0

    for fid in sorted(set(gt_dict) | set(det_dict)):
        gt_boxes  = gt_dict.get(fid, [])
        det_boxes = det_dict.get(fid, [])
        n_gt_total += len(gt_boxes)

        if not det_boxes:
            continue

        # Sort detections by confidence descending
        det_sorted = sorted(det_boxes,
                            key=lambda d: d[4] if len(d) > 4 else 1.0,
                            reverse=True)
        matched_gt = set()
        for det in det_sorted:
            score = det[4] if len(det) > 4 else 1.0
            all_scores.append(score)
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                v = iou(det[:4], gt[:4])
                if v > best_iou:
                    best_iou, best_j = v, j
            if best_iou >= iou_threshold and best_j >= 0:
                all_tp.append(1); all_fp.append(0)
                matched_gt.add(best_j)
                iou_sum   += best_iou
                iou_count += 1
            else:
                all_tp.append(0); all_fp.append(1)

    if not all_scores:
        return {'AP': 0.0, 'precision': 0.0, 'recall': 0.0,
                'F1': 0.0, 'mIoU': 0.0}

    # Sort all detections by score for PR curve
    order = np.argsort(-np.array(all_scores))
    tp_arr = np.array(all_tp)[order]
    fp_arr = np.array(all_fp)[order]

    cum_tp = np.cumsum(tp_arr)
    cum_fp = np.cumsum(fp_arr)

    recalls    = cum_tp / (n_gt_total + 1e-9)
    precisions = cum_tp / (cum_tp + cum_fp + 1e-9)

    ap        = compute_ap(precisions.tolist(), recalls.tolist())
    final_p   = float(precisions[-1]) if len(precisions) else 0.0
    final_r   = float(recalls[-1])    if len(recalls)    else 0.0
    f1        = (2 * final_p * final_r / (final_p + final_r + 1e-9)
                 if (final_p + final_r) > 0 else 0.0)
    miou      = iou_sum / iou_count if iou_count > 0 else 0.0

    return {'AP': ap, 'precision': final_p, 'recall': final_r,
            'F1': f1, 'mIoU': miou}


# ═══════════════════════════════════════════════════════════════════
# Experiment runners
# ═══════════════════════════════════════════════════════════════════

def evaluate_on_split(gt_dict, det_dict, frame_ids, iou_threshold=0.5):
    gt_sub  = {fid: gt_dict[fid] for fid in frame_ids if fid in gt_dict}
    det_sub = {fid: det_dict.get(fid, []) for fid in frame_ids}
    return evaluate_detections(gt_sub, det_sub, iou_threshold=iou_threshold)


def run_kfold_experiment(gt_dict, det_dict, strategy_name, folds, iou_threshold=0.5):
    fold_results = []
    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        m = evaluate_on_split(gt_dict, det_dict, test_ids, iou_threshold)
        fold_results.append({
            'strategy':  strategy_name,
            'fold':      fold_idx + 1,
            'n_train':   len(train_ids),
            'n_test':    len(test_ids),
            'mAP@50':    round(m['AP'],        4),
            'mIoU':      round(m['mIoU'],      4),
            'precision': round(m['precision'], 4),
            'recall':    round(m['recall'],    4),
            'F1':        round(m['F1'],        4),
        })
        print(f"  {strategy_name} Fold {fold_idx+1}/{len(folds)}: "
              f"mAP@50={m['AP']:.4f}  mIoU={m['mIoU']:.4f}  F1={m['F1']:.4f}")
    return fold_results


# ── YOLO per-fold retraining ─────────────────────────────────────────

def _yolo_kfold_inference(weights_path, video_path, frame_ids):
    """
    Run YOLO inference on specific frame IDs.
    Uses cap.set(CAP_PROP_POS_FRAMES) to seek directly — avoids the
    sequential-read mismatch when GT frame IDs don't start at frame 1.
    """
    from ultralytics import YOLO
    model    = YOLO(weights_path)
    det_dict = {}

    cap = cv2.VideoCapture(video_path)
    for fid in sorted(frame_ids):          # fid is 1-based GT frame ID
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)   # seek to 0-based index
        ret, frame = cap.read()
        if not ret:
            det_dict[fid] = []
            continue
        res   = model(frame, conf=0.25, verbose=False)[0]
        boxes = []
        if res.boxes is not None:
            for xyxy, score, cls_id in zip(
                res.boxes.xyxy.cpu().numpy(),
                res.boxes.conf.cpu().numpy(),
                res.boxes.cls.cpu().numpy(),
            ):
                cls_name = model.names.get(int(cls_id), '').lower()
                if cls_name and cls_name not in ALLOWED_CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = xyxy
                boxes.append([float(x1), float(y1),
                               float(x2), float(y2), float(score)])
        det_dict[fid] = boxes
    cap.release()
    for f in frame_ids:
        det_dict.setdefault(f, [])
    return det_dict


def _write_yolo_fold_dataset(gt_dict, train_ids, video_path, out_dir):
    """
    Write JPEG frames + YOLO labels for one fold's train split.

    IMPORTANT: GT frame IDs (1-based from CVAT/MOT) do NOT start at 1 —
    they start at whatever frame the annotation begins (e.g. 218).
    We must seek directly to each frame with CAP_PROP_POS_FRAMES (0-based)
    rather than counting cap.read() calls, otherwise fid never matches target.
    """
    img_dir = os.path.join(out_dir, 'images')
    lbl_dir = os.path.join(out_dir, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cap     = cv2.VideoCapture(video_path)
    written = 0

    for fid in sorted(train_ids):          # fid is 1-based GT frame ID
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)   # seek to 0-based index
        ret, frame = cap.read()
        if not ret:
            continue
        h, w = frame.shape[:2]
        cv2.imwrite(os.path.join(img_dir, f'frame_{fid:06d}.jpg'), frame)
        with open(os.path.join(lbl_dir, f'frame_{fid:06d}.txt'), 'w') as f:
            for ann in gt_dict.get(fid, []):
                x1, y1, x2, y2 = ann[:4]
                cx = ((x1+x2)/2)/w;  cy = ((y1+y2)/2)/h
                bw = (x2-x1)/w;      bh = (y2-y1)/h
                if bw > 0 and bh > 0:
                    f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
        written += 1

    cap.release()

    yaml_path = os.path.join(out_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write("train: images\nval: images\nnc: 1\nnames: ['car']\n")
    return yaml_path, written


def run_yolo_kfold(gt_dict, base_model_name, best_cfg,
                   strategy_name, folds, fold_cache_dir):
    """
    Proper k-fold: retrain YOLO from COCO init on each fold's train split,
    evaluate on test split.
    """
    from ultralytics import YOLO

    fold_results = []
    for fold_idx, (train_ids, test_ids) in enumerate(folds):
        print(f"  {strategy_name} Fold {fold_idx+1}/{len(folds)}: "
              f"train={len(train_ids)} frames, test={len(test_ids)} frames ...")

        fold_dir     = os.path.join(fold_cache_dir,
                                    f'strategy_{strategy_name}_fold_{fold_idx+1}')
        data_dir     = os.path.join(fold_dir, 'dataset')
        cached_wts   = os.path.join(fold_dir, 'weights', 'best.pt')

        # ── Train (skip if cached) ───────────────────────────────
        if not os.path.exists(cached_wts):
            yaml_path, n_written = _write_yolo_fold_dataset(
                gt_dict, train_ids, VIDEO_PATH, data_dir)
            if n_written == 0:
                print(f"    [warn] No frames written — skipping fold")
                _append_zero(fold_results, strategy_name, fold_idx+1,
                             train_ids, test_ids)
                continue

            model = YOLO(f'{base_model_name}.pt')
            try:
                train_result = model.train(
                    data=yaml_path,
                    epochs=best_cfg['epochs'],
                    lr0=best_cfg['lr0'],
                    imgsz=640,
                    batch=4,
                    project=fold_dir,
                    name='weights',
                    exist_ok=True,
                    verbose=False,
                )
                # Resolve actual save_dir from results object
                try:
                    save_dir   = str(train_result.save_dir)
                    wts_path   = os.path.join(save_dir, 'best.pt')
                    if not os.path.exists(wts_path):
                        wts_path = os.path.join(save_dir, 'last.pt')
                    # Cache to canonical location
                    os.makedirs(os.path.dirname(cached_wts), exist_ok=True)
                    if os.path.exists(wts_path):
                        import shutil
                        shutil.copy2(wts_path, cached_wts)
                except Exception as we:
                    print(f"    [warn] Could not resolve save_dir: {we}")
            except Exception as e:
                print(f"    [warn] Training failed: {e}")
                _append_zero(fold_results, strategy_name, fold_idx+1,
                             train_ids, test_ids)
                continue
        else:
            print(f"    [cache] Using existing weights: {cached_wts}")

        # ── Inference on test split ──────────────────────────────
        if not os.path.exists(cached_wts):
            print(f"    [warn] Weights not found after training: {cached_wts}")
            _append_zero(fold_results, strategy_name, fold_idx+1,
                         train_ids, test_ids)
            continue

        det_dict = _yolo_kfold_inference(cached_wts, VIDEO_PATH, test_ids)
        m        = evaluate_on_split(gt_dict, det_dict, test_ids)
        ap, miou = m['AP'], m['mIoU']
        prec, rec, f1 = m['precision'], m['recall'], m['F1']

        print(f"    mAP@50={ap:.4f}  mIoU={miou:.4f}  F1={f1:.4f}")
        fold_results.append({
            'strategy':  strategy_name,
            'fold':      fold_idx + 1,
            'n_train':   len(train_ids),
            'n_test':    len(test_ids),
            'mAP@50':    round(ap,   4),
            'mIoU':      round(miou, 4),
            'precision': round(prec, 4),
            'recall':    round(rec,  4),
            'F1':        round(f1,   4),
        })

    return fold_results


def _append_zero(results, strategy, fold, train_ids, test_ids):
    results.append({
        'strategy': strategy, 'fold': fold,
        'n_train': len(train_ids), 'n_test': len(test_ids),
        'mAP@50': 0.0, 'mIoU': 0.0,
        'precision': 0.0, 'recall': 0.0, 'F1': 0.0,
    })


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TASK 1.3: K-Fold Cross-Validation")
    print("=" * 60)

    # ── Ground truth ──────────────────────────────────────────────
    gt_dict       = load_gt()
    all_frame_ids = sorted(gt_dict.keys())
    print(f"  GT frames: {len(all_frame_ids)}  "
          f"({all_frame_ids[0]} – {all_frame_ids[-1]})")

    # ── Pre-computed detectors ────────────────────────────────────
    det_dicts = {}
    for name, fname in [
        ('MaskRCNN_provided', 'det_mask_rcnn.txt'),
        ('SSD512_provided',   'det_ssd512.txt'),
        ('YOLOv3_provided',   'det_yolo3.txt'),
    ]:
        p = os.path.join(DET_DIR, fname)
        if os.path.exists(p):
            det_dicts[name] = parse_detections_mot(p)
            print(f"Loaded: {name}")

    # ── Best off-the-shelf from task1_1 ───────────────────────────
    t11_path = 'results/task1_1/best_model.json'
    if os.path.exists(t11_path):
        t11       = json.load(open(t11_path))
        best_name = t11.get('best_model_name', t11.get('model_name', ''))
        det_txt   = t11.get('det_txt', '')
        if not det_txt or not os.path.exists(det_txt):
            det_txt = os.path.join('results', 'task1_1', 'detections',
                                   f'{best_name}.txt')
        if best_name and os.path.exists(det_txt):
            det_dicts[best_name] = parse_detections_mot(det_txt)
            print(f"Loaded task1_1 best: {best_name}")

    # ── Fine-tuned from task1_2 ───────────────────────────────────
    t12_path = 'results/task1_2/best_config.json'
    t12      = None
    if os.path.exists(t12_path):
        t12 = json.load(open(t12_path))
        ft_weights = t12.get('best_weights', '')
        ft_name    = 'FineTuned_' + t12.get('best_config', 'best')
        ft_det_txt = os.path.join('results', 'task1_2',
                                   f'{ft_name}_detections.txt')

        if os.path.exists(ft_det_txt):
            det_dicts[ft_name] = parse_detections_mot(ft_det_txt)
            print(f"Loaded task1_2 fine-tuned detections: {ft_name}")
        elif ft_weights and os.path.exists(ft_weights):
            print(f"Running inference with fine-tuned weights: {ft_name}")
            try:
                from ultralytics import YOLO
                model  = YOLO(ft_weights)
                cap    = cv2.VideoCapture(VIDEO_PATH)   # module-level VIDEO_PATH
                ft_det = {}
                # Seek directly to each annotated frame so the saved fid keys
                # match GT frame IDs (which start at e.g. 218, not 1).
                for fid in all_frame_ids:              # sorted 1-based GT fids
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fid - 1)
                    ret, frame = cap.read()
                    if not ret:
                        ft_det[fid] = []
                        continue
                    res   = model(frame, conf=0.25, verbose=False)[0]
                    boxes = []
                    if res.boxes is not None:
                        for xyxy, score, cls_id in zip(
                            res.boxes.xyxy.cpu().numpy(),
                            res.boxes.conf.cpu().numpy(),
                            res.boxes.cls.cpu().numpy()):
                            cls_name = model.names.get(int(cls_id), '').lower()
                            if cls_name and cls_name not in ALLOWED_CLASS_NAMES:
                                continue
                            x1, y1, x2, y2 = xyxy
                            boxes.append([float(x1), float(y1),
                                          float(x2), float(y2), float(score)])
                    ft_det[fid] = boxes
                cap.release()
                save_detections_mot(ft_det, ft_det_txt)
                det_dicts[ft_name] = ft_det
                print(f"  Inference done: {len(ft_det)} frames → {ft_det_txt}")
            except Exception as e:
                print(f"  [warn] Fine-tuned inference failed: {e}")

    # ── Yolo26l detections from task1_2 best weights (direct eval) ─
    # Also add the task1_2 fine-tuned weights as a named entry
    # so strategy A/B/C evaluation can use them via run_kfold_experiment
    if t12 and t12.get('best_weights') and os.path.exists(t12['best_weights']):
        yolo_ft_name = t12.get('model_name', 'yolo26l') + '_ft'
        if yolo_ft_name not in det_dicts:
            # run full-video inference and cache
            ft_full_txt = os.path.join('results', 'task1_2',
                                        f'{yolo_ft_name}_full_detections.txt')
            if os.path.exists(ft_full_txt):
                det_dicts[yolo_ft_name] = parse_detections_mot(ft_full_txt)
                print(f"Loaded task1_2 full-video detections: {yolo_ft_name}")
            else:
                print(f"[info] Full-video det cache not found for {yolo_ft_name} — skipping")

    if not det_dicts:
        print("[ERROR] No detections loaded. Check DET_DIR and task1 outputs.")
        return

    # ── Splits ────────────────────────────────────────────────────
    train_a, test_a = strategy_a_split(all_frame_ids)
    folds_a         = [(train_a, test_a)]
    folds_b         = strategy_b_kfold(all_frame_ids, k=4)
    folds_c         = strategy_c_random_kfold(all_frame_ids, k=4, seed=42)

    print(f"\nStrategy A: train={len(train_a)}, test={len(test_a)}")
    print(f"Strategy B: K=4 consecutive, ~{len(all_frame_ids)//4} per fold")
    print(f"Strategy C: K=4 random, ~{len(all_frame_ids)//4} per fold")

    all_fold_results = []
    summary_rows     = []

    # ── Evaluate pre-computed detectors ───────────────────────────
    for det_name, det_dict in det_dicts.items():
        print(f"\n--- Model: {det_name} ---")
        for strat_name, folds in [('A', folds_a), ('B', folds_b), ('C', folds_c)]:
            print(f"Strategy {strat_name}:")
            res = run_kfold_experiment(gt_dict, det_dict, strat_name, folds)
            all_fold_results.extend([{**r, 'model': det_name} for r in res])
            maps = [r['mAP@50'] for r in res]
            summary_rows.append({
                'model':        det_name,
                'strategy':     strat_name,
                'mean_mAP@50':  round(np.mean(maps), 4),
                'std_mAP@50':   round(np.std(maps),  4),
                'min_mAP@50':   round(np.min(maps),  4),
                'max_mAP@50':   round(np.max(maps),  4),
                'n_folds':      len(res),
            })

    # ── YOLO proper k-fold (retrain per fold) ─────────────────────
    if t12 is not None:
        base_model = t12.get('model_name', 'yolov8m')

        # FIX: task1_2 saves 'lr0', not 'lr' — was the cause of zeros
        lr0 = float(t12.get('lr0', t12.get('lr', 0.01)))

        best_cfg = {
            'lr0':    lr0,
            'epochs': int(t12.get('epochs', 10)),
        }
        print(f"\n  [kfold cfg] base={base_model}  "
              f"lr0={best_cfg['lr0']}  epochs={best_cfg['epochs']}")

        yolo_kfold_name = f'{base_model}_kfold'
        fold_cache      = os.path.join(RESULTS_DIR, 'yolo_kfold_cache')
        os.makedirs(fold_cache, exist_ok=True)

        print(f"\n--- Model: {yolo_kfold_name} (proper k-fold retraining) ---")
        for strat_name, folds in [('A', folds_a), ('B', folds_b), ('C', folds_c)]:
            print(f"Strategy {strat_name} (retrain per fold):")
            res = run_yolo_kfold(gt_dict, base_model, best_cfg,
                                 strat_name, folds, fold_cache)
            all_fold_results.extend([{**r, 'model': yolo_kfold_name} for r in res])
            maps = [r['mAP@50'] for r in res]
            summary_rows.append({
                'model':        yolo_kfold_name,
                'strategy':     strat_name,
                'mean_mAP@50':  round(np.mean(maps), 4),
                'std_mAP@50':   round(np.std(maps),  4),
                'min_mAP@50':   round(np.min(maps),  4),
                'max_mAP@50':   round(np.max(maps),  4),
                'n_folds':      len(res),
            })
    else:
        print("\n[info] results/task1_2/best_config.json not found — "
              "skipping YOLO proper k-fold.")

    # ── Save ──────────────────────────────────────────────────────
    fold_df    = pd.DataFrame(all_fold_results)
    summary_df = pd.DataFrame(summary_rows)

    fold_df.to_csv(   os.path.join(RESULTS_DIR, 'per_fold_metrics.csv'), index=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'kfold_summary.csv'),   index=False)
    print(f"\nSaved → {RESULTS_DIR}/")
    print(summary_df.to_string(index=False))

    _save_plots(fold_df, summary_df)
    print("\n✓ Task 1.3 complete.")


# ═══════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════

def _save_plots(fold_df, summary_df):
    models     = summary_df['model'].unique()
    strategies = ['A', 'B', 'C']

    # 1) Grouped bar: mean mAP per strategy per model
    fig, ax = plt.subplots(figsize=(11, 5))
    x     = np.arange(len(models))
    width = 0.25
    colors = {'A': 'steelblue', 'B': 'darkorange', 'C': 'seagreen'}
    for i, strat in enumerate(strategies):
        sub  = summary_df[summary_df['strategy'] == strat]
        vals = [sub[sub['model'] == m]['mean_mAP@50'].values[0]
                if m in sub['model'].values else 0 for m in models]
        errs = [sub[sub['model'] == m]['std_mAP@50'].values[0]
                if m in sub['model'].values else 0 for m in models]
        ax.bar(x + i * width - width, vals, width,
               label=f'Strategy {strat}', color=colors[strat],
               yerr=errs, capsize=4, edgecolor='black', linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_provided', '') for m in models],
                       rotation=20, ha='right')
    ax.set_ylabel('Mean mAP@50')
    ax.set_title('K-Fold Cross-Validation — mAP@50 per Strategy', fontsize=12)
    ax.legend(); ax.set_ylim(0, 1.0); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_3_kfold_map.png'), dpi=150)
    plt.close()

    # 2) Per-fold bars for strategies B and C
    for model in models:
        mdf = fold_df[fold_df['model'] == model]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, strat, title in zip(
            axes, ['B', 'C'],
            ['Fixed K-Fold (B)', 'Random K-Fold (C)']
        ):
            sub = mdf[mdf['strategy'] == strat].sort_values('fold')
            if sub.empty:
                ax.set_visible(False)
                continue
            clrs = plt.cm.tab10(np.linspace(0, 0.8, len(sub)))
            bars = ax.bar(sub['fold'].astype(str), sub['mAP@50'],
                          color=clrs, edgecolor='black')
            ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=9)
            ax.axhline(sub['mAP@50'].mean(), color='red', ls='--',
                       label=f'Mean={sub["mAP@50"].mean():.3f}')
            ax.set_title(f'{model}\n{title}', fontsize=10)
            ax.set_xlabel('Fold'); ax.set_ylabel('mAP@50')
            ax.set_ylim(0, 1.0); ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')
        plt.suptitle(f'Per-Fold mAP@50 — {model}', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR,
            f'task1_3_per_fold_{model.replace("/","_")}.png'), dpi=150)
        plt.close()

    # 3) Heatmap: model × strategy
    try:
        pivot = summary_df.pivot(index='model', columns='strategy',
                                 values='mean_mAP@50')
        fig, ax = plt.subplots(figsize=(7, max(3, len(pivot) * 0.7)))
        im = ax.imshow(pivot.values, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'Strategy {c}' for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([m.replace('_provided', '') for m in pivot.index])
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                        fontsize=11, color='black')
        ax.set_title('Mean mAP@50 (Model × CV Strategy)')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task1_3_heatmap.png'), dpi=150)
        plt.close()
    except Exception as e:
        print(f"  [warn] heatmap skipped: {e}")

    # 4) Variance scatter
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, model in enumerate(models):
        sub = summary_df[summary_df['model'] == model]
        x_vals = [strategies.index(s) + i * 0.2
                  for s in sub['strategy'] if s in strategies]
        y_vals = [sub[sub['strategy'] == s]['std_mAP@50'].values[0]
                  for s in sub['strategy'] if s in strategies]
        ax.scatter(x_vals, y_vals,
                   label=model.replace('_provided', ''), s=80, zorder=3)
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels([f'Strategy {s}' for s in strategies])
    ax.set_ylabel('Std of mAP@50 across folds')
    ax.set_title('Variance Analysis: Stability of CV Strategies')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_3_variance.png'), dpi=150)
    plt.close()

    print(f"Plots saved to {PLOTS_DIR}/")


if __name__ == '__main__':
    main()