"""
utils/visualization.py
Qualitative visualizations: draw detections, tracking trails, comparison figures.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import random


# ---------------------------------------------------------------------------
# Color palette (consistent across frames for track IDs)
# ---------------------------------------------------------------------------

_COLOR_CACHE = {}

def id_to_color(track_id):
    if track_id not in _COLOR_CACHE:
        random.seed(track_id * 7 + 13)
        _COLOR_CACHE[track_id] = tuple(random.randint(50, 255) for _ in range(3))
    return _COLOR_CACHE[track_id]


# ---------------------------------------------------------------------------
# Frame annotation drawing
# ---------------------------------------------------------------------------

def draw_boxes(frame, boxes, color=(0, 255, 0), labels=None, thickness=2):
    """Draw bounding boxes on a copy of the frame."""
    out = frame.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels and i < len(labels):
            cv2.putText(out, str(labels[i]), (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return out


def draw_gt_and_det(frame, gt_boxes, det_boxes, det_scores=None):
    """
    Draw GT (blue) and detections (red) on the same frame.
    Returns annotated frame.
    """
    out = frame.copy()
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 0, 0), 2)   # blue = GT

    for i, box in enumerate(det_boxes):
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)   # red = Det
        if det_scores is not None:
            label = f"{det_scores[i]:.2f}"
            cv2.putText(out, label, (x1, max(y1 - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 200), 1)

    # Legend
    cv2.putText(out, "GT", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(out, "Det", (55, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out


def draw_tracks(frame, tracks, trails=None, frame_id=None):
    """
    Draw track boxes with IDs and optional trails.
    tracks: [[x1,y1,x2,y2,track_id], ...]
    trails: dict {track_id: [(cx,cy), ...]}
    """
    out = frame.copy()
    for trk in tracks:
        x1, y1, x2, y2, tid = [int(v) for v in trk[:5]]
        color = id_to_color(tid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"#{tid}", (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        if trails and tid in trails:
            pts = trails[tid]
            for j in range(1, len(pts)):
                cv2.line(out, pts[j-1], pts[j], color, 2)

    if frame_id is not None:
        cv2.putText(out, f"Frame {frame_id}", (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
# Save qualitative grids
# ---------------------------------------------------------------------------

def save_qualitative_grid(frames_and_titles, save_path, nrows=2, ncols=3, figsize=(18, 10)):
    """
    Save a grid of annotated frames as a matplotlib figure.
    frames_and_titles: list of (bgr_frame, title_str)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    if not frames_and_titles:
        plt.close()
        return

    for i, (frame, title) in enumerate(frames_and_titles[:nrows * ncols]):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis('off')

    n_used = min(len(frames_and_titles), nrows * ncols)
    for j in range(n_used, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved qualitative grid → {save_path}")


def save_comparison_grid(frame, gt_boxes, det_dicts, save_path):
    """
    Save a multi-panel figure comparing different detectors on the same frame.
    det_dicts: [(label, boxes, scores), ...]
    """
    panels = [frame.copy()]
    panel_titles = ["Input Frame"]

    gt_frame = draw_gt_and_det(frame, gt_boxes, [])
    panels.append(gt_frame)
    panel_titles.append("Ground Truth")

    for label, boxes, scores in det_dicts:
        ann = draw_gt_and_det(frame, gt_boxes, boxes, scores)
        panels.append(ann)
        panel_titles.append(label)

    n = len(panels)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if nrows == 1:
        axes = [axes] if ncols == 1 else list(axes)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (panel, title) in enumerate(zip(panels, panel_titles)):
        rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)
        axes[i].set_title(title, fontsize=11, fontweight='bold')
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Detector Comparison — S03/C010", fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison grid → {save_path}")


def save_tracking_montage(frames_annot, save_path, fps=5):
    """Save a video of tracking results."""
    if not frames_annot:
        return
    h, w = frames_annot[0].shape[:2]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames_annot:
        out.write(f)
    out.release()
    print(f"Saved tracking video → {save_path}")