"""
training_utils.py
=================
Loss functions, metrics (IDF1, F1, AUC), early stopping, result I/O,
and correct CityFlow Track-1 submission formatter.

Submission format (space-delimited, one row per detection per frame):
  <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>

Key constraints from the evaluation description:
  - camera_id  : global numeric ID from folder name (c001=1, c025=25, etc.)
  - obj_id     : positive integer, consistent across cameras for the same vehicle
  - frame_id   : 1-based frame count within the video
  - xmin,ymin,width,height : integers, top-left origin
  - xworld,yworld : world coords from homography bottom-centre projection
                    (set to -1 if calibration not available)
  - Output file must be named track1.txt
"""

from __future__ import annotations
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def get_device(gpu: int = -1) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_crossentropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return F.cross_entropy(logits, labels, weight=weight)


def compute_focal_loss(
    logits:     torch.Tensor,
    labels:     torch.Tensor,
    gamma:      float = 2.0,
    pos_weight: float = 10.0,
) -> torch.Tensor:
    """
    Focal loss with per-sample class weighting.
    Uses pos_weight as a direct multiplier on positive-class CE loss,
    with focal modulation (1-pt)^gamma to down-weight easy negatives.
    Avoids the alpha collapse where predicting all-positive is trivially optimal.
    """
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0:
        return F.cross_entropy(logits, labels)

    # Per-sample weights: positives get pos_weight, negatives get 1.0
    sample_w = torch.where(
        labels == 1,
        torch.full_like(labels, pos_weight, dtype=torch.float),
        torch.ones_like(labels, dtype=torch.float),
    )
    ce  = F.cross_entropy(logits, labels, reduction='none')
    pt  = torch.exp(-ce)
    focal_factor = (1 - pt) ** gamma
    return (sample_w * focal_factor * ce).sum() / sample_w.sum()


def compute_weighted_loss(
    cam_logits:      torch.Tensor,
    vehicle_logits:  torch.Tensor,
    cam_labels:      torch.Tensor,
    edge_labels:     torch.Tensor,
    lambda_cam:      float = 0.3,
    lambda_vehicle:  float = 0.7,
    focal:           bool  = True,
    pos_edge_weight: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cam_loss = compute_crossentropy_loss(cam_logits, cam_labels)

    n_pos = (edge_labels == 1).sum().item()
    if focal:
        vehicle_loss = compute_focal_loss(
            vehicle_logits, edge_labels, pos_weight=pos_edge_weight)
    else:
        w = torch.tensor(
            [1.0, pos_edge_weight if n_pos > 0 else 1.0],
            device=vehicle_logits.device,
        )
        vehicle_loss = compute_crossentropy_loss(vehicle_logits, edge_labels, weight=w)

    total = lambda_cam * cam_loss + lambda_vehicle * vehicle_loss
    return total, cam_loss, vehicle_loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def get_f1(logits, labels):
    from sklearn.metrics import f1_score
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    lbls  = labels.cpu().numpy()
    return (f1_score(lbls, preds, average='macro', zero_division=0),
            f1_score(lbls, preds, average='micro', zero_division=0))


def get_binary_f1(logits, labels):
    from sklearn.metrics import precision_score, recall_score, f1_score
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    lbls  = labels.cpu().numpy()
    return (precision_score(lbls, preds, pos_label=1, zero_division=0),
            recall_score(   lbls, preds, pos_label=1, zero_division=0),
            f1_score(       lbls, preds, pos_label=1, zero_division=0))


def compute_auc(logits, labels):
    from sklearn.metrics import average_precision_score
    probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    lbls  = labels.cpu().numpy()
    if len(np.unique(lbls)) < 2:
        return 0.0
    return float(average_precision_score(lbls, probs))


# ---------------------------------------------------------------------------
# IDF1
# ---------------------------------------------------------------------------

def compute_idf1(
    node_to_gid:   Dict[int, int],
    gt_global_ids: List[int],
    tracklet_ids:  List[str],
) -> float:
    """
    Compute IDF1 = 2*IDTP / (2*IDTP + IDFP + IDFN).
    Operates at the tracklet level (one node = one tracklet).
    """
    if not node_to_gid:
        return 0.0

    pred_clusters: Dict[int, List[int]] = {}
    for node, gid in node_to_gid.items():
        pred_clusters.setdefault(gid, []).append(node)

    gt_clusters: Dict[int, List[int]] = {}
    for node, gid in enumerate(gt_global_ids):
        if gid != -1:
            gt_clusters.setdefault(gid, []).append(node)

    total_idtp = total_idfn = total_idfp = 0
    matched_pred = set()

    for gt_gid, gt_nodes in gt_clusters.items():
        gt_set   = set(gt_nodes)
        best_tp  = 0
        best_pid = -1
        for pred_gid, pred_nodes in pred_clusters.items():
            if pred_gid in matched_pred:
                continue
            tp = len(gt_set & set(pred_nodes))
            if tp > best_tp:
                best_tp, best_pid = tp, pred_gid

        total_idtp += best_tp
        total_idfn += len(gt_set) - best_tp
        if best_pid != -1:
            matched_pred.add(best_pid)
            total_idfp += len(pred_clusters[best_pid]) - best_tp

    for pred_gid, pred_nodes in pred_clusters.items():
        if pred_gid not in matched_pred:
            total_idfp += len(pred_nodes)

    denom = 2 * total_idtp + total_idfp + total_idfn
    return (2 * total_idtp / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    def __init__(self, model, name, save_dir, metric='idf1', patience=50):
        self.model      = model
        self.name       = name
        self.save_dir   = Path(save_dir)
        self.metric     = metric
        self.patience   = patience
        self.counter    = 0
        self.best_score = None
        self.best_path  = None
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def step(self, value: float) -> str:
        better = (
            self.best_score is None or
            (self.metric == 'loss' and value < self.best_score) or
            (self.metric != 'loss' and value > self.best_score)
        )
        if better:
            self.best_score = value
            self.counter    = 0
            self.best_path  = self.save_dir / f'{self.name}.pt'
            torch.save(self.model.state_dict(), self.best_path)
            return 'improved'
        self.counter += 1
        return 'stop' if self.counter >= self.patience else 'continue'


# ---------------------------------------------------------------------------
# Result I/O
# ---------------------------------------------------------------------------

def save_results(name: str, results: dict, results_dir: Path):
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f'{name}.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results → {out}")


# ---------------------------------------------------------------------------
# Submission writer — correct CityFlow Track-1 format
# ---------------------------------------------------------------------------

def cam_name_to_id(cam_name: str) -> int:
    """
    Convert camera folder name to global numeric camera ID.
    'c001' → 1,  'c025' → 25,  'c046' → 46
    Handles both 'c001' and 'c01' style names.
    """
    digits = ''.join(ch for ch in cam_name if ch.isdigit())
    return int(digits) if digits else 0


def _project_to_world(
    bbox: List[float],
    H:    Optional[np.ndarray],
) -> Tuple[float, float]:
    """
    Project the bottom-centre of a bbox to world coordinates using homography H.
    Returns (-1, -1) if H is not available.
    Submission description says xworld/yworld are "not currently used in evaluation"
    but recommends including them.
    """
    if H is None:
        return -1.0, -1.0
    u  = (bbox[0] + bbox[2]) / 2.0   # horizontal centre
    v  = bbox[3]                       # bottom edge (ground contact point)
    pt = np.array([u, v, 1.0])
    w  = H @ pt
    w /= w[2]
    return float(w[0]), float(w[1])


def write_track1_submission(
    node_to_gid:   Dict[int, int],
    features:      Dict,
    output_path:   str,
    calibrations:  Optional[Dict[str, np.ndarray]] = None,
) -> None:
    """
    Write the official CityFlow Track-1 submission file.

    Format (space-delimited, one row per detection per frame):
      <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>

    Parameters
    ----------
    node_to_gid   : {node_idx: global_vehicle_id} from edges_to_global_ids()
    features      : dict from build_graph(), must contain:
                      'cam_names'   : List[str]   e.g. ['c010', 'c010', 'c011', ...]
                      'all_frames'  : List[List[(frame_no, [x0,y0,x1,y1])]]
    output_path   : should end in 'track1.txt'
    calibrations  : {cam_name: H_matrix_3x3} for world coord projection (optional)
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure output file is named track1.txt per submission spec
    if out_path.name != 'track1.txt':
        out_path = out_path.parent / 'track1.txt'

    rows: List[str] = []

    for node_idx, global_gid in node_to_gid.items():
        obj_id   = global_gid + 1    # must be positive integer
        cam_name = features['cam_names'][node_idx]
        cam_id   = cam_name_to_id(cam_name)
        H        = calibrations.get(cam_name) if calibrations else None

        # all_frames: [(frame_no, [x0,y0,x1,y1]), ...]
        all_frames = features['all_frames'][node_idx]

        for frame_no, bbox in all_frames:
            xmin   = int(round(bbox[0]))
            ymin   = int(round(bbox[1]))
            width  = int(round(bbox[2] - bbox[0]))
            height = int(round(bbox[3] - bbox[1]))

            # Clamp to valid pixel range
            xmin   = max(0, xmin)
            ymin   = max(0, ymin)
            width  = max(1, width)
            height = max(1, height)

            xworld, yworld = _project_to_world(bbox, H)

            rows.append(
                f"{cam_id} {obj_id} {frame_no} "
                f"{xmin} {ymin} {width} {height} "
                f"{xworld:.4f} {yworld:.4f}"
            )

    # Sort by camera, then frame for clean output
    rows.sort(key=lambda r: (int(r.split()[0]), int(r.split()[2])))

    with open(out_path, 'w') as f:
        f.write('\n'.join(rows))
        if rows:
            f.write('\n')

    print(f"  Submission → {out_path}  ({len(rows)} rows, "
          f"{len(node_to_gid)} tracklets)")


def load_calibrations_for_sequence(seq_path: str) -> Dict[str, np.ndarray]:
    """
    Load all calibration.txt files in a sequence and return
    {cam_name: H_matrix} for use in write_track1_submission.
    """
    seq_path     = Path(seq_path)
    calibrations = {}
    for cam_dir in sorted(seq_path.iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith('c'):
            continue
        cal_file = cam_dir / 'calibration.txt'
        if cal_file.exists():
            try:
                H = _parse_calibration(cal_file)
                calibrations[cam_dir.name] = H
            except Exception as e:
                print(f"  [warn] calibration load failed for {cam_dir.name}: {e}")
    return calibrations


def _parse_calibration(path: Path) -> np.ndarray:
    with open(path) as f:
        content = f.read()
    for line in content.split('\n'):
        if 'Homography' in line:
            mat_str = line.split(': ')[1].strip()
            rows    = mat_str.split(';')
            return np.array([[float(v) for v in r.split()] for r in rows])
    raise ValueError(f"No homography found in {path}")


# Legacy alias — kept so existing code that calls write_mtmc_output doesn't break
# but it now delegates to the correct implementation
def write_mtmc_output(
    node_to_gid:   Dict[int, int],
    features:      Dict,
    output_path:   str,
    seq_name:      str = '',
):
    """
    Deprecated alias for write_track1_submission.
    Tries to load calibrations automatically from features['seq_path'] if present.
    """
    calibrations = None
    if 'seq_path' in features:
        calibrations = load_calibrations_for_sequence(features['seq_path'])
    write_track1_submission(node_to_gid, features, output_path, calibrations)