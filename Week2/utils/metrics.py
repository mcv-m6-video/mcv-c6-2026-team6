"""
utils/metrics.py
IoU computation, mAP@50 / mAP@75 for object detection evaluation.
"""

import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes [x1,y1,x2,y2].
    """
    ix1 = max(box1[0], box2[0])
    iy1 = max(box1[1], box2[1])
    ix2 = min(box1[2], box2[2])
    iy2 = min(box1[3], box2[3])

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter   = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def compute_iou_matrix(boxes1, boxes2):
    """
    Vectorised IoU between all pairs.
    boxes1: (N,4), boxes2: (M,4) -> returns (N,M)
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))

    boxes1 = np.array(boxes1, dtype=float)  # (M, 4)
    boxes2 = np.array(boxes2, dtype=float)  # (N, 4)

    # Expand dims for broadcasting: (M,1,4) vs (1,N,4) → (M,N)
    b1 = boxes1[:, None, :]  # (M, 1, 4)
    b2 = boxes2[None, :, :]  # (1, N, 4)

    ix1 = np.maximum(b1[:, :, 0], b2[:, :, 0])
    iy1 = np.maximum(b1[:, :, 1], b2[:, :, 1])
    ix2 = np.minimum(b1[:, :, 2], b2[:, :, 2])
    iy2 = np.minimum(b1[:, :, 3], b2[:, :, 3])

    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)  # (M, N)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])  # (M,)
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])  # (N,)

    union = area1[:, None] + area2[None, :] - inter  # (M, N)
    iou = np.where(union > 0, inter / union, 0.0)
    return iou


# ---------------------------------------------------------------------------
# Mean Average Precision (mAP)
# ---------------------------------------------------------------------------

def compute_ap(recalls, precisions):
    """Compute AP using the 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        mask = recalls >= t
        p = precisions[mask].max() if mask.any() else 0.0
        ap += p / 11.0
    return ap


def evaluate_detections(
    gt_dict,
    det_dict,
    iou_threshold=0.5,
    score_threshold=0.0
):
    """
    Evaluate detection results.
    gt_dict:  {frame_id: [[x1,y1,x2,y2,track_id], ...]}
    det_dict: {frame_id: [[x1,y1,x2,y2,score], ...]}

    Returns dict with keys: AP, mIoU, precision, recall, F1, 
                             all_ious (per matched pair), 
                             iou_over_time {frame_id: mean_iou}
    """
    all_tp, all_fp, all_fn = 0, 0, 0
    all_scores = []
    all_labels = []  # 1=TP, 0=FP
    all_ious_matched = []
    iou_over_time = {}

    all_frame_ids = sorted(set(list(gt_dict.keys()) + list(det_dict.keys())))

    for fid in all_frame_ids:
        gt_boxes  = [b[:4] for b in gt_dict.get(fid, [])]
        det_boxes_full = det_dict.get(fid, [])

        # Filter by score
        det_boxes_full = [d for d in det_boxes_full if d[4] >= score_threshold]
        # Sort by descending confidence
        det_boxes_full = sorted(det_boxes_full, key=lambda x: -x[4])
        det_boxes = [d[:4] for d in det_boxes_full]
        det_scores = [d[4] for d in det_boxes_full]

        matched_gt = set()
        frame_ious = []

        for di, db in enumerate(det_boxes):
            best_iou = 0.0
            best_gi  = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(db, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_gi  = gi

            if best_iou >= iou_threshold and best_gi >= 0:
                matched_gt.add(best_gi)
                all_tp += 1
                all_labels.append(1)
                all_ious_matched.append(best_iou)
                frame_ious.append(best_iou)
            else:
                all_fp += 1
                all_labels.append(0)

            all_scores.append(det_scores[di])

        fn = len(gt_boxes) - len(matched_gt)
        all_fn += fn
        iou_over_time[fid] = float(np.mean(frame_ious)) if frame_ious else 0.0

    # Precision / Recall
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    # AP curve
    sorted_idx = np.argsort(-np.array(all_scores))
    labels_sorted = np.array(all_labels)[sorted_idx]
    cum_tp = np.cumsum(labels_sorted)
    cum_fp = np.cumsum(1 - labels_sorted)
    total_gt = all_tp + all_fn
    precisions_curve = cum_tp / (cum_tp + cum_fp + 1e-9)
    recalls_curve    = cum_tp / (total_gt + 1e-9)
    AP = compute_ap(recalls_curve, precisions_curve)

    mIoU = float(np.mean(all_ious_matched)) if all_ious_matched else 0.0

    return {
        'AP':    AP,
        'mAP50': AP,       # alias
        'mIoU':  mIoU,
        'precision': precision,
        'recall':    recall,
        'F1':        f1,
        'TP':   all_tp,
        'FP':   all_fp,
        'FN':   all_fn,
        'iou_over_time': iou_over_time,
        'pr_curve': (recalls_curve, precisions_curve),
        'all_ious': all_ious_matched,
    }


def evaluate_across_thresholds(gt_dict, det_dict, thresholds=None):
    """
    Evaluate at multiple IoU thresholds and return per-threshold metrics.
    Default: 0.5 and 0.75.
    """
    if thresholds is None:
        thresholds = [0.5, 0.75]
    results = {}
    for thr in thresholds:
        results[thr] = evaluate_detections(gt_dict, det_dict, iou_threshold=thr)
    return results