"""
task2/task2_1_overlap_tracker.py  (upgraded)

Upgrades over previous version:
  1. Grid search over iou_threshold × max_age × min_confidence
     → saves best config per detector, used downstream
  2. Hungarian matching alternative to greedy (scipy linear_sum_assignment)
  3. Confidence threshold filtering (min_confidence sweep)
  4. Runs ALL detectors including provided ones (MaskRCNN, SSD, YOLOv3)
     so task2_2 and task2_3 have SORT/IoU track files for all of them
  5. Saves both _iou_tracker.txt (greedy best) and _iou_hungarian_tracker.txt
  6. Ablation plots: IoU threshold, max_age, min_confidence, greedy vs hungarian
"""

import os
import sys
import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from itertools import product as iterproduct
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

_DATA_BASE  = os.environ.get('DATA_ROOT', 'data')
if os.path.basename(_DATA_BASE.rstrip('/')) == 'c010':
    C010_ROOT = _DATA_BASE
else:
    C010_ROOT = os.path.join(_DATA_BASE, 'AICity_data', 'train', 'S03', 'c010')
DATA_ROOT   = C010_ROOT
ANN_PATH    = os.path.join(C010_ROOT, 'annotations.xml')
GT_FALLBACK = os.path.join(C010_ROOT, 'gt', 'gt.txt')
DET_DIR     = os.path.join(C010_ROOT, 'det')
VIDEO_PATH  = os.environ.get('VIDEO_PATH', os.path.join(C010_ROOT, 'vdo.avi'))
RESULTS_DIR = 'results/task2_1'
PLOTS_DIR   = 'plots/try2'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,   exist_ok=True)
os.makedirs('qualitative', exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Inlined utilities
# ═══════════════════════════════════════════════════════════════════

def parse_cvat_xml(xml_path):
    tree = ET.parse(xml_path); root = tree.getroot(); gt = {}
    for track in root.findall('track'):
        if track.attrib.get('label','').lower() not in ('car','vehicle'): continue
        tid = int(track.attrib.get('id', 0))
        for box in track.findall('box'):
            if box.attrib.get('outside','0') == '1': continue
            fid = int(box.attrib['frame']) + 1
            gt.setdefault(fid,[]).append([float(box.attrib['xtl']),float(box.attrib['ytl']),
                float(box.attrib['xbr']),float(box.attrib['ybr']),tid])
    return gt

def load_gt():
    if os.path.exists(ANN_PATH):
        print(f"  [GT] Parsing CVAT XML: {ANN_PATH}")
        return parse_cvat_xml(ANN_PATH)
    if os.path.exists(GT_FALLBACK):
        print(f"  [GT] Parsing MOT txt: {GT_FALLBACK}")
        return parse_annotations_mot(GT_FALLBACK)
    raise FileNotFoundError(f"No GT at {ANN_PATH} or {GT_FALLBACK}")

def parse_annotations_mot(gt_path):
    """Parse MOT gt.txt → {fid: [[x1,y1,x2,y2,track_id], ...]}
    MOT format: frame, track_id, x, y, w, h, conf, class, visibility
    """
    gt = {}
    with open(gt_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) < 6:
                continue
            fid = int(parts[0])
            tid = int(parts[1])          # track_id is parts[1] — was missing!
            x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            if conf == 0:
                continue
            gt.setdefault(fid, []).append([x, y, x+w, y+h, tid])
    return gt


def parse_detections_mot(det_path):
    dets = {}
    with open(det_path) as f:
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
            dets.setdefault(fid, []).append([x, y, x+w, y+h, score])
    return dets


def save_tracks_mot(tracks_dict, out_path):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    with open(out_path, 'w') as f:
        for fid in sorted(tracks_dict.keys()):
            for t in tracks_dict[fid]:
                x1, y1, x2, y2, tid = t[0], t[1], t[2], t[3], int(t[4])
                f.write(f"{fid},{tid},{x1:.2f},{y1:.2f},"
                        f"{x2-x1:.2f},{y2-y1:.2f},1,-1,-1,-1\n")


def compute_iou_matrix(boxes_a, boxes_b):
    boxes_a = np.array(boxes_a, dtype=np.float32)
    boxes_b = np.array(boxes_b, dtype=np.float32)
    N, M = len(boxes_a), len(boxes_b)
    if N == 0 or M == 0:
        return np.zeros((N, M), dtype=np.float32)
    a = boxes_a[:, np.newaxis, :]
    b = boxes_b[np.newaxis, :, :]
    xi1 = np.maximum(a[..., 0], b[..., 0])
    yi1 = np.maximum(a[..., 1], b[..., 1])
    xi2 = np.minimum(a[..., 2], b[..., 2])
    yi2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0., xi2-xi1) * np.maximum(0., yi2-yi1)
    area_a = (boxes_a[:,2]-boxes_a[:,0]) * (boxes_a[:,3]-boxes_a[:,1])
    area_b = (boxes_b[:,2]-boxes_b[:,0]) * (boxes_b[:,3]-boxes_b[:,1])
    union  = area_a[:, np.newaxis] + area_b[np.newaxis, :] - inter + 1e-6
    return inter / union


class VideoFrameLoader:
    def __init__(self, video_path=VIDEO_PATH):
        self.video_path = video_path
        self._cap = None

    def _open(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)

    def get_frame(self, frame_idx_0based):
        self._open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_0based)
        ret, frame = self._cap.read()
        return frame if ret else None

    def __del__(self):
        if self._cap is not None:
            self._cap.release()


_COLORS = [
    (255,85,0),(0,170,255),(170,0,255),(0,255,85),(255,0,170),
    (85,255,0),(0,85,255),(255,170,0),(0,255,170),(170,255,0),
]

def _track_color(tid):
    return _COLORS[int(tid) % len(_COLORS)]


def draw_tracks(frame, tracks, trails=None, gt_boxes=None, frame_id=None):
    img = frame.copy()
    if gt_boxes:
        for box in gt_boxes:
            cv2.rectangle(img, (int(box[0]),int(box[1])),
                          (int(box[2]),int(box[3])), (0,200,0), 1)
    if trails:
        for tid, pts in trails.items():
            col = _track_color(tid)
            for k in range(1, len(pts)):
                cv2.line(img, pts[k-1], pts[k], col, 1)
    for trk in tracks:
        x1,y1,x2,y2 = int(trk[0]),int(trk[1]),int(trk[2]),int(trk[3])
        tid = int(trk[4])
        col = _track_color(tid)
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
        cv2.putText(img,f"ID:{tid}",(x1,max(y1-4,10)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.45,col,1)
    if frame_id is not None:
        cv2.putText(img,f"f{frame_id}",(6,18),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    return img


def save_qualitative_grid(panels, out_path, nrows=2, ncols=3):
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3.2))
    axes = np.array(axes).flatten()
    for i, ax in enumerate(axes):
        if i < len(panels):
            img, title = panels[i]
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Track data structure
# ═══════════════════════════════════════════════════════════════════

class Track:
    _id_counter = 0

    def __init__(self, box, score=1.0):
        Track._id_counter += 1
        self.id      = Track._id_counter
        self.box     = np.array(box[:4], dtype=float)
        self.score   = score
        self.age     = 0
        self.hits    = 1
        self.history = [self.box.copy()]

    def update(self, box, score=1.0):
        self.box   = np.array(box[:4], dtype=float)
        self.score = score
        self.age   = 0
        self.hits += 1
        self.history.append(self.box.copy())

    def predict(self):
        if len(self.history) >= 2:
            delta    = self.history[-1] - self.history[-2]
            self.box = self.box + delta
        self.age += 1

    def to_mot(self):
        x1, y1, x2, y2 = self.box
        return [x1, y1, x2, y2, self.id]


# ═══════════════════════════════════════════════════════════════════
# Trackers
# ═══════════════════════════════════════════════════════════════════

class OverlapTracker:
    """IoU-based tracker with greedy or Hungarian matching."""

    def __init__(self, iou_threshold=0.3, max_age=5, min_hits=1,
                 use_hungarian=False, use_prediction=False):
        self.iou_threshold  = iou_threshold
        self.max_age        = max_age
        self.min_hits       = min_hits
        self.use_hungarian  = use_hungarian
        self.use_prediction = use_prediction
        self.tracks = []
        Track._id_counter = 0

    def reset(self):
        self.tracks = []
        Track._id_counter = 0

    def update(self, detections):
        if self.use_prediction:
            for t in self.tracks:
                t.predict()

        if not detections:
            for t in self.tracks:
                if not self.use_prediction:
                    t.age += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return []

        det_arr = np.array([d[:4] for d in detections])
        trk_arr = np.array([t.box for t in self.tracks]) if self.tracks else np.empty((0,4))

        if self.use_hungarian:
            matched, unmatched_dets, unmatched_trks = self._hungarian(det_arr, trk_arr)
        else:
            matched, unmatched_dets, unmatched_trks = self._greedy(det_arr, trk_arr)

        for di, ti in matched:
            self.tracks[ti].update(detections[di],
                                   score=detections[di][4] if len(detections[di])>4 else 1.0)

        for ti in unmatched_trks:
            self.tracks[ti].age += 1

        for di in unmatched_dets:
            d = detections[di]
            self.tracks.append(Track(d[:4], score=d[4] if len(d)>4 else 1.0))

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]

        return [t.to_mot() for t in self.tracks
                if t.hits >= self.min_hits and t.age == 0]

    def _greedy(self, det_arr, trk_arr):
        if len(trk_arr) == 0:
            return [], list(range(len(det_arr))), []
        if len(det_arr) == 0:
            return [], [], list(range(len(trk_arr)))
        iou_mat = compute_iou_matrix(det_arr, trk_arr)
        matched_dets, matched_trks, pairs = set(), set(), []
        for flat_idx in np.argsort(-iou_mat, axis=None):
            di, ti = np.unravel_index(flat_idx, iou_mat.shape)
            if di in matched_dets or ti in matched_trks:
                continue
            if iou_mat[di, ti] < self.iou_threshold:
                break
            pairs.append((di, ti))
            matched_dets.add(di); matched_trks.add(ti)
        return pairs, [i for i in range(len(det_arr)) if i not in matched_dets], \
                      [i for i in range(len(trk_arr)) if i not in matched_trks]

    def _hungarian(self, det_arr, trk_arr):
        if len(trk_arr) == 0:
            return [], list(range(len(det_arr))), []
        if len(det_arr) == 0:
            return [], [], list(range(len(trk_arr)))
        iou_mat = compute_iou_matrix(det_arr, trk_arr)
        row_ind, col_ind = linear_sum_assignment(-iou_mat)
        matched_dets, matched_trks, pairs = set(), set(), []
        for di, ti in zip(row_ind, col_ind):
            if iou_mat[di, ti] >= self.iou_threshold:
                pairs.append((int(di), int(ti)))
                matched_dets.add(di); matched_trks.add(ti)
        return pairs, [i for i in range(len(det_arr)) if i not in matched_dets], \
                      [i for i in range(len(trk_arr)) if i not in matched_trks]


def _nms(dets, nms_iou=0.3):
    """Tight NMS: iou=0.3 removes near-duplicates aggressively."""
    if len(dets) <= 1:
        return dets
    dets_s = sorted(dets, key=lambda d: d[4] if len(d)>4 else 1.0, reverse=True)
    keep = []
    for d in dets_s:
        suppressed = False
        for k in keep:
            xi1=max(d[0],k[0]); yi1=max(d[1],k[1])
            xi2=min(d[2],k[2]); yi2=min(d[3],k[3])
            inter=max(0,xi2-xi1)*max(0,yi2-yi1)
            if inter == 0: continue
            iou=inter/((d[2]-d[0])*(d[3]-d[1])+(k[2]-k[0])*(k[3]-k[1])-inter+1e-6)
            if iou >= nms_iou:
                suppressed = True; break
        if not suppressed:
            keep.append(d)
    return keep


def _motion_filter(det_dict, min_movement=8.0, window=3):
    """
    Remove detections that stay static across `window` consecutive frames.
    Static = box centre moves < min_movement pixels over the window.
    Parked cars and background FPs are static; moving vehicles are not.
    """
    fids = sorted(det_dict.keys())
    filtered = {}
    for i, fid in enumerate(fids):
        kept = []
        for box in det_dict[fid]:
            cx = (box[0]+box[2])/2; cy = (box[1]+box[3])/2
            # Check if a similar box exists in surrounding frames (static = FP)
            static_count = 0
            check_fids = [fids[j] for j in range(max(0,i-window), min(len(fids),i+window+1)) if fids[j]!=fid]
            for cfid in check_fids:
                for cbox in det_dict.get(cfid, []):
                    ccx=(cbox[0]+cbox[2])/2; ccy=(cbox[1]+cbox[3])/2
                    # same region IoU check
                    xi1=max(box[0],cbox[0]); yi1=max(box[1],cbox[1])
                    xi2=min(box[2],cbox[2]); yi2=min(box[3],cbox[3])
                    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
                    if inter > 0:
                        iou=inter/((box[2]-box[0])*(box[3]-box[1])+(cbox[2]-cbox[0])*(cbox[3]-cbox[1])-inter+1e-6)
                        movement = ((cx-ccx)**2+(cy-ccy)**2)**0.5
                        if iou > 0.7 and movement < min_movement:
                            static_count += 1
            # If static in more than half the window → likely parked car / FP
            if static_count < window:
                kept.append(box)
        filtered[fid] = kept
    return filtered


def run_tracker(det_dict, tracker,
                min_confidence=0.0,
                nms_iou=0.3,
                min_area=0,
                use_motion_filter=True):

    tracker.reset()

    # Pre-filter: confidence + optional motion filter
    if use_motion_filter:
        conf_filtered = {}
        for fid, dets in det_dict.items():
            filtered = []
            for d in dets:
                if len(d) >= 5 and d[4] < min_confidence:
                    continue
                area = (d[2] - d[0]) * (d[3] - d[1])
                if area < min_area:
                    continue
                filtered.append(d)
            conf_filtered[fid] = filtered

        conf_filtered = _motion_filter(conf_filtered)
        det_dict_use = conf_filtered
    else:
        det_dict_use = {}
        for fid, dets in det_dict.items():
            filtered = []
            for d in dets:
                if len(d) >= 5 and d[4] < min_confidence:
                    continue
                area = (d[2] - d[0]) * (d[3] - d[1])
                if area < min_area:
                    continue
                filtered.append(d)
            det_dict_use[fid] = filtered

    tracks_dict = {}

    for fid in sorted(det_dict_use.keys()):
        dets = det_dict_use.get(fid, [])
        dets = _nms(dets, nms_iou=nms_iou)
        tracks_dict[fid] = tracker.update(dets)

    return tracks_dict



# ═══════════════════════════════════════════════════════════════════
# Quick IDF1 proxy for grid search (fast, no full eval)
# ═══════════════════════════════════════════════════════════════════

def align_det_to_gt(det_dict, gt_dict):
    """
    Align detection frame IDs to GT frame IDs.

    GT uses absolute video frame IDs (218-2141).
    Provided det_*.txt files and some task1 outputs use sequential
    1-based IDs (1-N). Detects the mismatch and remaps positionally.

    Returns a new det_dict with GT-aligned frame IDs.
    """
    gt_fids  = sorted(gt_dict.keys())
    det_fids = sorted(det_dict.keys())
    if not gt_fids or not det_fids:
        return det_dict

    overlap = len(set(gt_fids) & set(det_fids))
    # If >10% overlap already, assume IDs are aligned
    if overlap > 0.1 * min(len(gt_fids), len(det_fids)):
        return det_dict

    # Positional remap: det frame i  →  gt frame i
    n = min(len(det_fids), len(gt_fids))
    remap = {det_fids[i]: gt_fids[i] for i in range(n)}
    return {remap.get(fid, fid): boxes for fid, boxes in det_dict.items()}


def quick_idf1(gt_dict, tracks_dict, iou_thr=0.5):
    """
    Fast IDF1 proxy using greedy per-frame matching.
    Assumes tracks_dict frame IDs are already aligned to GT (via align_det_to_gt).
    Only counts predictions on GT-annotated frames to avoid IDFP inflation.
    """
    from collections import Counter, defaultdict

    gt_to_pred = defaultdict(list)
    total_gt   = sum(len(v) for v in gt_dict.values())
    # Only count preds on annotated GT frames — prevents inflating IDFP
    # with predictions on frames where GT has no boxes
    total_pred = sum(len(tracks_dict.get(fid, [])) for fid in gt_dict)

    for fid in sorted(set(gt_dict) | set(tracks_dict)):
        gts   = gt_dict.get(fid, [])
        preds = tracks_dict.get(fid, [])
        if not gts or not preds:
            continue
        gt_boxes   = np.array([b[:4] for b in gts])
        pred_boxes = np.array([b[:4] for b in preds])
        gt_ids     = [int(b[4]) if len(b) > 4 else 0 for b in gts]
        pred_ids   = [int(b[4]) for b in preds]

        iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
        matched_p, matched_g = set(), set()
        for flat_idx in np.argsort(-iou_mat, axis=None):
            pi, gi = np.unravel_index(flat_idx, iou_mat.shape)
            if pi in matched_p or gi in matched_g:
                continue
            if iou_mat[pi, gi] < iou_thr:
                break
            matched_p.add(pi); matched_g.add(gi)
            gt_to_pred[gt_ids[gi]].append(pred_ids[pi])

    idtp = sum(Counter(v).most_common(1)[0][1] for v in gt_to_pred.values() if v)
    idfp = total_pred - idtp
    idfn = total_gt   - idtp
    return (2 * idtp) / (2 * idtp + idfp + idfn + 1e-9)


# ═══════════════════════════════════════════════════════════════════
# Grid search
# ═══════════════════════════════════════════════════════════════════

def grid_search(gt_dict, det_dict, det_name):
    """
    Grid search over iou_threshold, max_age, min_confidence, matcher.
    Returns best config dict and full results DataFrame.
    """
    iou_thresholds  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    max_ages        = [1, 3, 5, 10, 15]
    min_confs       = [0.0, 0.25, 0.4, 0.5, 0.6, 0.7]
    min_areas       = [0, 5000, 15000, 30000, 50000]  # filter small FP boxes
    matchers        = ['greedy', 'hungarian']

    rows     = []
    best_idf1 = -1
    best_cfg  = {}

    total = len(iou_thresholds) * len(max_ages) * len(min_confs) * len(min_areas) * len(matchers)
    print(f"  Grid search: {total} configs ...")

    for iou_thr, max_age, min_conf, min_area, matcher in iterproduct(
            iou_thresholds, max_ages, min_confs, min_areas, matchers):

        tracker = OverlapTracker(
            iou_threshold  = iou_thr,
            max_age        = max_age,
            min_hits       = 1,
            use_hungarian  = (matcher == 'hungarian'),
            use_prediction = False,
        )
        tracks = run_tracker(det_dict, tracker, min_confidence=min_conf, nms_iou=0.5, min_area=min_area)
        idf1   = quick_idf1(gt_dict, tracks)
        n_ids  = len({int(t[4]) for fid in tracks for t in tracks[fid]})

        rows.append({
            'iou_threshold': iou_thr,
            'max_age':       max_age,
            'min_conf':      min_conf,
            'min_area':      min_area,
            'matcher':       matcher,
            'IDF1':          round(idf1, 4),
            'n_ids':         n_ids,
        })

        if idf1 > best_idf1:
            best_idf1 = idf1
            best_cfg  = {
                'iou_threshold': iou_thr,
                'max_age':       max_age,
                'min_conf':      min_conf,
                'min_area':      min_area,
                'matcher':       matcher,
                'IDF1':          round(idf1, 4),
            }

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, f'{det_name}_grid_search.csv'), index=False)
    print(f"  Best: matcher={best_cfg['matcher']}  "
          f"iou={best_cfg['iou_threshold']}  "
          f"max_age={best_cfg['max_age']}  "
          f"min_conf={best_cfg['min_conf']}  "
          f"min_area={best_cfg.get('min_area',0)}  "
          f"IDF1={best_cfg['IDF1']:.4f}")
    return best_cfg, df


# ═══════════════════════════════════════════════════════════════════
# Ablations (for reporting/plots)
# ═══════════════════════════════════════════════════════════════════

def ablation_iou_threshold(gt_dict, det_dict, max_age=5, min_conf=0.0):
    rows = []
    for thr in np.arange(0.1, 0.8, 0.1):
        tracker = OverlapTracker(iou_threshold=float(thr), max_age=max_age)
        tracks  = run_tracker(det_dict, tracker, min_confidence=min_conf)
        n_ids   = len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1    = quick_idf1(gt_dict, tracks)
        rows.append({'iou_threshold': round(float(thr),2),
                     'n_unique_ids': n_ids, 'IDF1': round(idf1,4)})
    return pd.DataFrame(rows)


def ablation_max_age(gt_dict, det_dict, iou_thr=0.3, min_conf=0.0):
    rows = []
    for age in [1, 2, 3, 5, 7, 10, 15, 20]:
        tracker = OverlapTracker(iou_threshold=iou_thr, max_age=age)
        tracks  = run_tracker(det_dict, tracker, min_confidence=min_conf)
        n_ids   = len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1    = quick_idf1(gt_dict, tracks)
        rows.append({'max_age': age, 'n_unique_ids': n_ids, 'IDF1': round(idf1,4)})
    return pd.DataFrame(rows)


def ablation_min_conf(gt_dict, det_dict, iou_thr=0.3, max_age=5):
    rows = []
    for conf in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        tracker = OverlapTracker(iou_threshold=iou_thr, max_age=max_age)
        tracks  = run_tracker(det_dict, tracker, min_confidence=conf)
        n_ids   = len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1    = quick_idf1(gt_dict, tracks)
        rows.append({'min_conf': conf, 'n_unique_ids': n_ids, 'IDF1': round(idf1,4)})
    return pd.DataFrame(rows)


def compare_matchers(gt_dict, det_dict,
                     iou_thr=0.3,
                     max_age=5,
                     min_conf=0.0,
                     min_area=0):

    rows = []

    for matcher in ['greedy', 'hungarian']:
        tracker = OverlapTracker(
            iou_threshold=iou_thr,
            max_age=max_age,
            use_hungarian=(matcher == 'hungarian')
        )

        tracks = run_tracker(
            det_dict,
            tracker,
            min_confidence=min_conf,
            nms_iou=0.5,
            min_area=min_area
        )

        n_ids = len({int(t[4]) for fid in tracks for t in tracks[fid]})
        idf1  = quick_idf1(gt_dict, tracks)

        rows.append({
            'matcher': matcher,
            'n_unique_ids': n_ids,
            'IDF1': round(idf1, 4)
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# Qualitative helpers
# ═══════════════════════════════════════════════════════════════════

def build_trails(tracks_dict, trail_length=20):
    trails = {}
    for fid in sorted(tracks_dict.keys()):
        for trk in tracks_dict[fid]:
            tid = int(trk[4])
            cx  = int((trk[0]+trk[2])/2)
            cy  = int((trk[1]+trk[3])/2)
            trails.setdefault(tid, []).append((fid,(cx,cy)))
    return trails


def get_trail_at_frame(trails_full, fid, trail_length=20):
    active = {}
    for tid, pts in trails_full.items():
        pts_before = [(f,p) for f,p in pts if f <= fid]
        if pts_before:
            active[tid] = [p for _,p in pts_before[-trail_length:]]
    return active


def save_qualitative(loader, tracks_dict, gt_dict, det_name, suffix=''):
    all_fids = sorted(tracks_dict.keys())
    if not all_fids:
        return
    sample      = all_fids[::max(1, len(all_fids)//6)][:6]
    trails_full = build_trails(tracks_dict)
    panels      = []
    for fid in sample:
        frame = loader.get_frame(fid - 1)
        if frame is None:
            continue
        trail_at = get_trail_at_frame(trails_full, fid)
        trks     = tracks_dict.get(fid, [])
        ann      = draw_tracks(frame, trks, trails=trail_at,
                               gt_boxes=gt_dict.get(fid,[]), frame_id=fid)
        panels.append((ann, f"Frame {fid}"))
    save_qualitative_grid(panels,
        f'qualitative/task2_1_{det_name}{suffix}.png', nrows=2, ncols=3)


# ═══════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════

def save_ablation_plots(det_name, iou_df, age_df, conf_df, matcher_df, grid_df):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # IoU threshold vs IDF1
    ax = axes[0,0]
    ax.plot(iou_df['iou_threshold'], iou_df['IDF1'], 'b-o', ms=5, label='IDF1')
    ax2 = ax.twinx()
    ax2.plot(iou_df['iou_threshold'], iou_df['n_unique_ids'], 'r--s', ms=4, label='N IDs')
    ax2.set_ylabel('# Unique IDs', color='red')
    ax.set_xlabel('IoU Threshold'); ax.set_ylabel('IDF1')
    ax.set_title(f'IoU Threshold\n{det_name}'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0,1)

    # Max age vs IDF1
    ax = axes[0,1]
    ax.plot(age_df['max_age'], age_df['IDF1'], 'g-^', ms=5, label='IDF1')
    ax2 = ax.twinx()
    ax2.plot(age_df['max_age'], age_df['n_unique_ids'], 'r--s', ms=4)
    ax2.set_ylabel('# Unique IDs', color='red')
    ax.set_xlabel('Max Age (frames)'); ax.set_ylabel('IDF1')
    ax.set_title(f'Max Age\n{det_name}'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0,1)

    # Min confidence vs IDF1
    ax = axes[0,2]
    ax.plot(conf_df['min_conf'], conf_df['IDF1'], 'm-D', ms=5, label='IDF1')
    ax2 = ax.twinx()
    ax2.plot(conf_df['min_conf'], conf_df['n_unique_ids'], 'r--s', ms=4)
    ax2.set_ylabel('# Unique IDs', color='red')
    ax.set_xlabel('Min Confidence'); ax.set_ylabel('IDF1')
    ax.set_title(f'Confidence Threshold\n{det_name}'); ax.grid(True, alpha=0.3)
    ax.set_ylim(0,1)

    # Greedy vs Hungarian
    ax = axes[1,0]
    colors = ['steelblue', 'darkorange']
    bars = ax.bar(matcher_df['matcher'], matcher_df['IDF1'], color=colors, edgecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=3)
    ax.set_ylabel('IDF1'); ax.set_ylim(0,1)
    ax.set_title(f'Greedy vs Hungarian\n{det_name}'); ax.grid(True, alpha=0.3, axis='y')

    # Grid search heatmap: iou_threshold × max_age for best matcher
    ax = axes[1,1]
    best_matcher = grid_df.groupby('matcher')['IDF1'].mean().idxmax()
    sub = grid_df[(grid_df['matcher']==best_matcher) & (grid_df['min_conf']==0.0)]
    if not sub.empty:
        pivot = sub.pivot_table(index='max_age', columns='iou_threshold',
                                values='IDF1', aggfunc='mean')
        im = ax.imshow(pivot.values, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.1f}' for v in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel('IoU Threshold'); ax.set_ylabel('Max Age')
        ax.set_title(f'IDF1 Heatmap ({best_matcher})\n{det_name}')
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f'{pivot.values[i,j]:.2f}',
                        ha='center', va='center', fontsize=7)

    # Grid search: top-10 configs
    ax = axes[1,2]
    top10 = grid_df.nlargest(10, 'IDF1')
    labels = [f"{r['matcher'][:1].upper()} iou={r['iou_threshold']} age={r['max_age']} c={r['min_conf']}"
              for _, r in top10.iterrows()]
    bars = ax.barh(labels[::-1], top10['IDF1'].values[::-1],
                   color='steelblue', edgecolor='black')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.set_xlabel('IDF1'); ax.set_xlim(0,1)
    ax.set_title(f'Top-10 Configs\n{det_name}'); ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle(f'Task 2.1 Overlap Tracker Analysis — {det_name}', fontsize=13)
    plt.tight_layout()
    fname = f'task2_1_analysis_{det_name.replace("/","_")}.png'
    plt.savefig(os.path.join(PLOTS_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved: {fname}")


def save_summary_plot(summary_rows):
    """Grouped bar: IDF1 for greedy best vs hungarian best per detector."""
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return
    detectors = df['detector'].unique()
    x = np.arange(len(detectors))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(detectors)*2), 5))
    greedy_vals = [df[(df['detector']==d)&(df['tracker']=='greedy_best')]['IDF1'].values
                   for d in detectors]
    hung_vals   = [df[(df['detector']==d)&(df['tracker']=='hungarian_best')]['IDF1'].values
                   for d in detectors]
    greedy_vals = [v[0] if len(v) else 0 for v in greedy_vals]
    hung_vals   = [v[0] if len(v) else 0 for v in hung_vals]
    b1 = ax.bar(x - width/2, greedy_vals, width, label='Greedy (best)',
                color='steelblue', edgecolor='black')
    b2 = ax.bar(x + width/2, hung_vals,   width, label='Hungarian (best)',
                color='darkorange', edgecolor='black')
    ax.bar_label(b1, fmt='%.3f', padding=3, fontsize=8)
    ax.bar_label(b2, fmt='%.3f', padding=3, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(detectors, rotation=20, ha='right')
    ax.set_ylabel('IDF1 (grid-search best)'); ax.set_ylim(0, 1.1)
    ax.set_title('Task 2.1 — Best IDF1: Greedy vs Hungarian per Detector')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task2_1_summary.png'), dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TASK 2.1: Tracking by Maximum Overlap")
    print("=" * 60)

    # ── GT ────────────────────────────────────────────────────────
    try:
        gt_dict = load_gt()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}"); return
    print(f"  GT frames: {len(gt_dict)}  "
          f"({min(gt_dict)} – {max(gt_dict)})")

    loader = VideoFrameLoader(video_path=VIDEO_PATH)

    # ── Detection sources ─────────────────────────────────────────
    det_files = {}
    for name, fname in [
        ('MaskRCNN_provided', 'det_mask_rcnn.txt'),
        ('SSD512_provided',   'det_ssd512.txt'),
        ('YOLOv3_provided',   'det_yolo3.txt'),
    ]:
        p = os.path.join(DET_DIR, fname)
        if os.path.exists(p):
            det_files[name] = p

    # task1_1 best off-the-shelf
    t11_path = 'results/task1_1/best_model.json'
    if os.path.exists(t11_path):
        t11       = json.load(open(t11_path))
        best_name = t11.get('best_model_name', t11.get('model_name', ''))
        det_txt   = t11.get('det_txt', '')
        if not det_txt or not os.path.exists(det_txt):
            det_txt = os.path.join('results','task1_1','detections',f'{best_name}.txt')
        if best_name and os.path.exists(det_txt):
            det_files[best_name] = det_txt
            print(f"  [task1_1] Loaded best: {best_name}")

    # task1_2 fine-tuned
    t12_path = 'results/task1_2/best_config.json'
    if os.path.exists(t12_path):
        t12       = json.load(open(t12_path))
        ft_model  = t12.get('model_name', '')
        ft_name   = f'{ft_model}_finetuned'
        ft_det    = os.path.join('results','task1_2',f'{ft_name}_detections.txt')
        if os.path.exists(ft_det):
            det_files[ft_name] = ft_det
            print(f"  [task1_2] Loaded fine-tuned: {ft_name}")

    if not det_files:
        print("[ERROR] No detection files found.")
        return

    # ── Per-detector processing ───────────────────────────────────
    all_best_configs = {}
    summary_rows     = []
    metrics_rows     = []

    for det_name, det_path in det_files.items():
        if not os.path.exists(det_path):
            continue
        print(f"\n{'='*50}\nDetector: {det_name}\n{'='*50}")
        det_dict = parse_detections_mot(det_path)
        pre_overlap  = len(set(det_dict.keys()) & set(gt_dict.keys()))
        det_dict     = align_det_to_gt(det_dict, gt_dict)
        post_overlap = len(set(det_dict.keys()) & set(gt_dict.keys()))
        print(f"  Frame alignment: overlap {pre_overlap}→{post_overlap} "
              f"/ {len(gt_dict)} GT frames")

        # 1. Grid search → best config
        best_cfg, grid_df = grid_search(gt_dict, det_dict, det_name)
        all_best_configs[det_name] = best_cfg

        # 2. Run best greedy config → save as primary iou_tracker
        greedy_tracker = OverlapTracker(
            iou_threshold = best_cfg['iou_threshold'] if best_cfg['matcher']=='greedy'
                            else grid_df[grid_df['matcher']=='greedy']['IDF1'].idxmax(),
            max_age       = best_cfg['max_age'],
            use_hungarian = False,
        )
        # simpler: run best config regardless of matcher type for primary file,
        # and also save a greedy-specific best
        best_greedy = grid_df[grid_df['matcher']=='greedy'].nlargest(1,'IDF1').iloc[0]
        best_hung   = grid_df[grid_df['matcher']=='hungarian'].nlargest(1,'IDF1').iloc[0]

        for cfg_row, suffix, label in [
            (best_greedy, '_iou_tracker',    'greedy_best'),
            (best_hung,   '_iou_hungarian_tracker', 'hungarian_best'),
        ]:
            tracker = OverlapTracker(
                iou_threshold = cfg_row['iou_threshold'],
                max_age       = cfg_row['max_age'],
                use_hungarian = (cfg_row['matcher'] == 'hungarian'),
            )
            tracks = run_tracker(det_dict, tracker,
                                 min_confidence=cfg_row['min_conf'],
                                 nms_iou=0.5, min_area=int(cfg_row.get('min_area',0)))
            out_path = os.path.join(RESULTS_DIR, f'{det_name}{suffix}.txt')
            save_tracks_mot(tracks, out_path)
            n_ids = len({int(t[4]) for fid in tracks for t in tracks[fid]})
            idf1  = quick_idf1(gt_dict, tracks)
            print(f"  [{label}] {n_ids} IDs  IDF1={idf1:.4f}  → {out_path}")
            metrics_rows.append({'detector': det_name, 'tracker': label,
                                 'n_ids': n_ids, 'IDF1': round(idf1,4)})
            summary_rows.append({'detector': det_name, 'tracker': label,
                                 'IDF1': round(idf1,4)})

        # 3. Ablation curves using best_greedy config as baseline
        print("  Running ablation curves ...")
        iou_df  = ablation_iou_threshold(gt_dict, det_dict,
                      max_age=int(best_greedy['max_age']),
                      min_conf=float(best_greedy['min_conf']))
        age_df  = ablation_max_age(gt_dict, det_dict,
                      iou_thr=float(best_greedy['iou_threshold']),
                      min_conf=float(best_greedy['min_conf']))
        conf_df = ablation_min_conf(gt_dict, det_dict,
                      iou_thr=float(best_greedy['iou_threshold']),
                      max_age=int(best_greedy['max_age']))
        match_df = compare_matchers(gt_dict, det_dict,
                        iou_thr=float(best_greedy['iou_threshold']),
                        max_age=int(best_greedy['max_age']),
                        min_conf=float(best_greedy['min_conf']),
                        min_area=int(best_greedy.get('min_area', 0)))

        for df_abl, fname in [
            (iou_df,  f'{det_name}_iou_abl.csv'),
            (age_df,  f'{det_name}_age_abl.csv'),
            (conf_df, f'{det_name}_conf_abl.csv'),
            (match_df,f'{det_name}_matcher_abl.csv'),
        ]:
            df_abl.to_csv(os.path.join(RESULTS_DIR, fname), index=False)

        # 4. Ablation plots
        save_ablation_plots(det_name, iou_df, age_df, conf_df, match_df, grid_df)

        # 5. Qualitative on best config
        best_tracker = OverlapTracker(
            iou_threshold = float(best_cfg['iou_threshold']),
            max_age       = int(best_cfg['max_age']),
            use_hungarian = (best_cfg['matcher'] == 'hungarian'),
        )
        best_tracks = run_tracker(det_dict, best_tracker,
                                  min_confidence=float(best_cfg['min_conf']),
                                  nms_iou=0.5, min_area=int(best_cfg.get('min_area',0)))
        save_qualitative(loader, best_tracks, gt_dict, det_name, suffix='_best')

    # ── Save best configs and summary ─────────────────────────────
    with open(os.path.join(RESULTS_DIR, 'best_configs.json'), 'w') as f:
        json.dump(all_best_configs, f, indent=2)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(os.path.join(RESULTS_DIR, 'summary.csv'), index=False)
    print(f"\nSaved → {RESULTS_DIR}/")
    print(metrics_df.to_string(index=False))

    save_summary_plot(summary_rows)
    print(f"\nPlots saved to {PLOTS_DIR}/")
    print("\n✓ Task 2.1 complete.")


if __name__ == '__main__':
    main()