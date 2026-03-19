"""
utils.py  (from Week 3)
KITTI / AI City utility functions integrated into Week 4 pipeline.

Key functions used by cityflow_preprocessing.py:
  load_gt_aicity       — GT loader with active==0 skip (more robust than _read_gt_file
                         for direct use, but _read_gt_file now has visibility filter too)
  load_detections_aicity — YOLO det loader with NMS+conf+area+ROI in one pass
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple


def _nms_boxes(boxes_confs: List[List], iou_thresh: float = 0.5) -> List[List]:
    if not boxes_confs:
        return []
    sorted_b = sorted(boxes_confs, key=lambda r: r[4], reverse=True)
    kept = []
    for cand in sorted_b:
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
                suppress = True; break
        if not suppress:
            kept.append(cand)
    return kept


def load_detections_aicity(det_file: str,
                            min_conf: float = 0.0,
                            min_area: float = 0.0,
                            roi_file: Optional[str] = None,
                            nms_iou:  float = 0.5) -> Dict[int, List]:
    """
    Load AI City YOLO detections (MOTChallenge format).
    Applies: conf filter → area filter → ROI mask → per-frame NMS.
    Returns: {frame_id: [[x1, y1, x2, y2, conf], ...]}
    """
    roi_mask = None
    if roi_file and Path(roi_file).exists():
        roi_img = cv2.imread(roi_file, cv2.IMREAD_GRAYSCALE)
        if roi_img is not None:
            roi_mask = roi_img > 127

    raw: Dict[int, List] = {}
    with open(det_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame = int(parts[0])
            x, y  = float(parts[2]), float(parts[3])
            w, h  = float(parts[4]), float(parts[5])
            conf  = float(parts[6])
            if conf < min_conf:
                continue
            if w * h < min_area:
                continue
            x2, y2 = x + w, y + h
            if roi_mask is not None:
                cx = min(max(int((x + x2) / 2), 0), roi_mask.shape[1] - 1)
                cy = min(max(int((y + y2) / 2), 0), roi_mask.shape[0] - 1)
                if not roi_mask[cy, cx]:
                    continue
            raw.setdefault(frame, []).append([x, y, x2, y2, conf])

    if nms_iou > 0:
        return {fid: _nms_boxes(boxes, nms_iou) for fid, boxes in raw.items()}
    return raw


def load_gt_aicity(gt_file: str, min_visibility: float = 0.2) -> Dict[int, List]:
    """
    Load AI City GT. Skips active==0 rows and low-visibility detections.
    Returns: {frame_id: [[tid, x1, y1, x2, y2], ...]}
    """
    gt: Dict[int, List] = {}
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 7:
                continue
            frame  = int(parts[0])
            tid    = int(parts[1])
            x, y   = float(parts[2]), float(parts[3])
            w, h   = float(parts[4]), float(parts[5])
            active = int(float(parts[6]))
            if active == 0:
                continue
            if len(parts) >= 9:
                vis = float(parts[8])
                if vis < min_visibility:
                    continue
            gt.setdefault(frame, []).append([tid, x, y, x + w, y + h])
    return gt


def read_flo_file(filepath: str) -> np.ndarray:
    with open(filepath, 'rb') as f:
        magic = np.frombuffer(f.read(4), dtype=np.float32)[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo magic: {filepath}")
        w = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        h = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
        flow = np.frombuffer(f.read(h * w * 8), dtype=np.float32).reshape((h, w, 2))
    return flow.copy()


def write_flo_file(flow: np.ndarray, filepath: str):
    H, W = flow.shape[:2]
    with open(filepath, 'wb') as f:
        np.array([202021.25], dtype=np.float32).tofile(f)
        np.array([W, H], dtype=np.int32).tofile(f)
        flow.astype(np.float32).tofile(f)


def read_kitti_flow_gt(flow_png_path: str) -> Tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(flow_png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {flow_png_path}")
    if img.dtype != np.uint16:
        raise ValueError(f"Expected uint16 PNG, got dtype={img.dtype}")
    flow_u = (img[:, :, 2].astype(np.float32) - 2**15) / 64.0
    flow_v = (img[:, :, 1].astype(np.float32) - 2**15) / 64.0
    valid  = img[:, :, 0].astype(bool)
    return np.stack([flow_u, flow_v], axis=-1), valid


def load_kitti_noc_mask(occ_png_path: str) -> np.ndarray:
    img = cv2.imread(occ_png_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load: {occ_png_path}")
    return img[:, :, 0].astype(bool)