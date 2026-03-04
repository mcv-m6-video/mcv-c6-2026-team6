"""
utils/data_utils.py
Utilities for loading AICity / MOT-format annotations and video frames.
"""

import os
import cv2
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# MOT / AICity annotation format helpers
# ---------------------------------------------------------------------------

def parse_annotations_mot(gt_path):
    """
    Parse MOT-format ground truth file.
    Format: frame, id, left, top, width, height, conf, class, visibility
    Returns dict: {frame_id: [ [x1,y1,x2,y2,track_id], ... ] }
    Only keeps class==1 (car in AICity).
    """
    annotations = {}
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            frame_id = int(parts[0])
            track_id = int(parts[1])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w  = float(parts[4])
            h  = float(parts[5])
            conf = float(parts[6])
            cls  = int(parts[7]) if len(parts) > 7 else 1
            x2, y2 = x1 + w, y1 + h

            # Filter: only keep cars (class 1) and valid annotations
            if conf == 0:
                continue

            if frame_id not in annotations:
                annotations[frame_id] = []
            annotations[frame_id].append([x1, y1, x2, y2, track_id])
    return annotations


def parse_detections_mot(det_path, score_threshold=0.0):
    """
    Parse detections in MOT format (no track_id).
    Format: frame, -1, left, top, width, height, score, -1, -1, -1
    Returns dict: {frame_id: [ [x1,y1,x2,y2,score], ... ] }
    """
    detections = {}
    with open(det_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            frame_id = int(parts[0])
            x1 = float(parts[2])
            y1 = float(parts[3])
            w  = float(parts[4])
            h  = float(parts[5])
            score = float(parts[6])
            if score < score_threshold:
                continue
            x2, y2 = x1 + w, y1 + h
            if frame_id not in detections:
                detections[frame_id] = []
            detections[frame_id].append([x1, y1, x2, y2, score])
    return detections


def save_detections_mot(detections_dict, save_path):
    """
    Save detections to MOT format file.
    detections_dict: {frame_id: [[x1,y1,x2,y2,score], ...]}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    for frame_id in sorted(detections_dict.keys()):
        for det in detections_dict[frame_id]:
            x1, y1, x2, y2 = det[:4]
            score = det[4] if len(det) > 4 else -1
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{score:.4f},-1,-1,-1\n")
    with open(save_path, 'w') as f:
        f.writelines(lines)
    print(f"Saved detections → {save_path}")


def save_tracks_mot(tracks_dict, save_path):
    """
    Save tracks to MOT format file.
    tracks_dict: {frame_id: [[x1,y1,x2,y2,track_id], ...]}
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    lines = []
    for frame_id in sorted(tracks_dict.keys()):
        for trk in tracks_dict[frame_id]:
            x1, y1, x2, y2 = trk[:4]
            track_id = int(trk[4])
            w, h = x2 - x1, y2 - y1
            lines.append(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")
    with open(save_path, 'w') as f:
        f.writelines(lines)
    print(f"Saved tracks → {save_path}")


# ---------------------------------------------------------------------------
# Video / frame utilities
# ---------------------------------------------------------------------------

class VideoFrameLoader:
    """Lazy frame loader from video file or image directory."""

    def __init__(self, video_path=None, frames_dir=None):
        self.video_path = video_path
        self.frames_dir = frames_dir
        self._cap = None
        self._frame_files = None

        if frames_dir and os.path.isdir(frames_dir):
            exts = ('.jpg', '.jpeg', '.png')
            self._frame_files = sorted(
                [f for f in Path(frames_dir).iterdir() if f.suffix.lower() in exts]
            )
        elif video_path:
            self._cap = cv2.VideoCapture(video_path)

    @property
    def total_frames(self):
        if self._frame_files:
            return len(self._frame_files)
        if self._cap:
            return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return 0

    def get_frame(self, frame_idx):
        """Return BGR frame (0-indexed)."""
        if self._frame_files:
            img = cv2.imread(str(self._frame_files[frame_idx]))
            return img
        if self._cap:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self._cap.read()
            return frame if ret else None
        return None

    def iter_frames(self, start=0, end=None, step=1):
        """Iterate (frame_idx, frame) pairs."""
        end = end or self.total_frames
        for idx in range(start, end, step):
            frame = self.get_frame(idx)
            if frame is not None:
                yield idx + 1, frame  # 1-indexed frame_id for MOT

    def __del__(self):
        if self._cap:
            self._cap.release()


# ---------------------------------------------------------------------------
# Data split utilities
# ---------------------------------------------------------------------------

def get_frame_ids(annotations):
    return sorted(annotations.keys())


def strategy_a_split(frame_ids):
    """First 25% train, rest test."""
    n = len(frame_ids)
    split = int(n * 0.25)
    return frame_ids[:split], frame_ids[split:]


def strategy_b_kfold(frame_ids, k=4):
    """Fixed K-Fold: split into K equal consecutive chunks."""
    n = len(frame_ids)
    fold_size = n // k
    folds = []
    for i in range(k):
        test_ids  = frame_ids[i * fold_size: (i + 1) * fold_size]
        train_ids = frame_ids[:i * fold_size] + frame_ids[(i + 1) * fold_size:]
        folds.append((train_ids, test_ids))
    return folds


def strategy_c_random_kfold(frame_ids, k=4, seed=42):
    """Random K-Fold: shuffle then split."""
    import random
    rng = random.Random(seed)
    ids = list(frame_ids)
    rng.shuffle(ids)
    n = len(ids)
    fold_size = n // k
    folds = []
    for i in range(k):
        test_ids  = ids[i * fold_size: (i + 1) * fold_size]
        train_ids = ids[:i * fold_size] + ids[(i + 1) * fold_size:]
        folds.append((train_ids, test_ids))
    return folds
