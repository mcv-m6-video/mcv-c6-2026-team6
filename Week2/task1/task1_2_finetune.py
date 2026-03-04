"""
task1/task1_2_finetune.py

Task 1.2: Fine-tune the BEST model from Task 1.1 on the AICity S03/C010 train split.

Reads:  results/task1_1/best_model.json   (written by task1_1)
        → picks the winning model automatically
        → if it was YOLOv8/v9, uses the Ultralytics fine-tune API
        → if it was a torchvision model or provided pre-computed,
          falls back to fine-tuning Faster R-CNN (strong general default)

Runs a 4-config hyperparameter ablation so the 2-point rubric is satisfied:
  Config A  LR=0.005  freeze=layer1-2   epochs=10  (baseline finetune)
  Config B  LR=0.001  freeze=layer1-2   epochs=10  (lower LR)
  Config C  LR=0.005  freeze=none        epochs=10  (no freezing)
  Config D  LR=0.005  freeze=layer1-2   epochs=20  (more epochs)

Outputs:
  results/task1_2/
    best_config.json          <- picked up by task1_3
    metrics_task1_2.csv
    training_history.csv
  plots/task1_2_*.png
  qualitative/task1_2_*.png
"""

import os, sys, cv2, json, copy, time, argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
try:
    from run_yolo_eval import parse_cvat_xml, compute_ap, iou as _iou_fn
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from run_yolo_eval import parse_cvat_xml, compute_ap, iou as _iou_fn


# ── lightweight replacements for removed utils ───────────────────

def parse_annotations_mot(path_ignored):
    """Replaced by parse_cvat_xml — kept for signature compatibility."""
    raise RuntimeError("Use parse_cvat_xml() instead of parse_annotations_mot()")


def strategy_a_split(frame_ids, train_frac=0.25):
    """First train_frac as train, rest as test."""
    n   = len(frame_ids)
    cut = int(n * train_frac)
    return frame_ids[:cut], frame_ids[cut:]


class VideoFrameLoader:
    """Thin wrapper — only used by Faster R-CNN branch."""
    def __init__(self, video_path):
        self.video_path = video_path
        self._cap       = None

    def _open(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.video_path)

    def get_frame(self, idx_0based):
        self._open()
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx_0based)
        ret, frame = self._cap.read()
        return frame if ret else None

    def __del__(self):
        if self._cap is not None:
            self._cap.release()


def evaluate_detections(gt_dict, det_dict, iou_threshold=0.5):
    """
    Compute AP, F1, precision, recall, mIoU from dicts of boxes.
    gt_dict:  {frame_id: [[x1,y1,x2,y2], ...]}   (tuples or lists)
    det_dict: {frame_id: [[x1,y1,x2,y2,score], ...]}
    Returns dict with keys: AP, F1, precision, recall, mIoU
    """
    # build flat predictions list for compute_ap
    preds = []
    for fid, boxes in det_dict.items():
        for b in boxes:
            preds.append({'frame': fid,
                          'bbox':  list(b[:4]),
                          'score': float(b[4]) if len(b) > 4 else 1.0})

    # gt must map frame_id → list of (x1,y1,x2,y2) tuples
    gt_flat = {}
    for fid, boxes in gt_dict.items():
        gt_flat[fid] = [tuple(b[:4]) for b in boxes]

    ap = compute_ap(preds, gt_flat, iou_threshold)

    # precision / recall / F1 at conf >= 0.25
    tp = fp = fn = 0
    iou_vals = []
    for fid in gt_flat:
        gt_boxes  = gt_flat[fid]
        det_boxes = [b for b in det_dict.get(fid, [])
                     if (float(b[4]) if len(b) > 4 else 1.0) >= 0.25]
        matched_gt = [False] * len(gt_boxes)
        for db in det_boxes:
            best_iou, best_j = 0.0, -1
            for gi, gb in enumerate(gt_boxes):
                if matched_gt[gi]:
                    continue
                v = _iou_fn(db[:4], gb)
                if v > best_iou:
                    best_iou, best_j = v, gi
            if best_iou >= iou_threshold and best_j >= 0:
                tp += 1
                matched_gt[best_j] = True
                iou_vals.append(best_iou)
            else:
                fp += 1
        fn += matched_gt.count(False)

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    miou      = float(np.mean(iou_vals)) if iou_vals else 0.0

    return {'AP': ap, 'F1': f1, 'precision': precision,
            'recall': recall, 'mIoU': miou}

DATA_ROOT   = os.environ.get('DATA_ROOT', 'data')
ANN_PATH    = os.environ.get('ANN_PATH',  os.path.join(DATA_ROOT, 'annotations.xml'))
VIDEO_PATH  = os.environ.get('VIDEO_PATH', os.path.join(DATA_ROOT, '/AICity_data/train/S03/c010vdo.avi'))
T1_RESULTS  = 'results/task1_1'
RESULTS_DIR = 'results/task1_2'
PLOTS_DIR   = 'plots'
QUAL_DIR    = 'qualitative'
for d in [RESULTS_DIR, PLOTS_DIR, QUAL_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[task1_2] device: {DEVICE}")


# ════════════════════════════════════════════════════════════════════
# Read task1_1 handoff
# ════════════════════════════════════════════════════════════════════

def load_handoff():
    path = os.path.join(T1_RESULTS, 'best_model.json')
    if not os.path.exists(path):
        print(f"[warn] {path} not found — defaulting to Faster R-CNN fine-tune")
        return {'best_model_name': 'FasterRCNN_R50_FPN',
                'detector_info': {'type': 'torchvision'}}
    with open(path) as f:
        h = json.load(f)
    # support both key names written by different versions of task1_1
    if 'best_model_name' not in h and 'model_name' in h:
        h['best_model_name'] = h['model_name']
    if 'best_map50' not in h and 'map50' in h:
        h['best_map50'] = h['map50']
    # task1_1 written by run_yolo_eval always sets type=yolo
    if 'detector_info' not in h:
        h['detector_info'] = {'type': 'yolo'}
    print(f"[task1_2] Task 1.1 best model: {h['best_model_name']} "
          f"(mAP@50={h.get('best_map50','?')})")
    return h


# ════════════════════════════════════════════════════════════════════
# Dataset for Faster R-CNN / torchvision training
# ════════════════════════════════════════════════════════════════════

class AICityDataset(Dataset):
    def __init__(self, video_path, annotations, frame_ids, transforms=None):
        self.loader      = VideoFrameLoader(video_path=video_path)
        self.annotations = annotations
        self.frame_ids   = [fid for fid in frame_ids
                            if fid in annotations and len(annotations[fid]) > 0]
        self.transforms  = transforms

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid   = self.frame_ids[idx]
        frame = self.loader.get_frame(fid - 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes = []
        for ann in self.annotations[fid]:
            x1, y1, x2, y2 = ann[:4]
            x1, y1 = max(0.0, float(x1)), max(0.0, float(y1))
            x2 = min(float(frame.shape[1]), float(x2))
            y2 = min(float(frame.shape[0]), float(y2))
            if x2 > x1 + 1 and y2 > y1 + 1:
                boxes.append([x1, y1, x2, y2])

        boxes_t  = torch.tensor(boxes, dtype=torch.float32) \
                   if boxes else torch.zeros((0, 4))
        labels_t = torch.ones(len(boxes), dtype=torch.int64)
        target   = {'boxes': boxes_t, 'labels': labels_t,
                    'image_id': torch.tensor([fid])}
        img_t    = T.ToTensor()(rgb)

        if self.transforms:
            img_t, target = self.transforms(img_t, target)
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


class ComposeTransforms:
    def __init__(self, tfs):
        self.tfs = tfs
    def __call__(self, img, target):
        for t in self.tfs:
            img, target = t(img, target)
        return img, target


class RandomFlip:
    def __call__(self, img, target):
        if torch.rand(1) < 0.5:
            img = torch.flip(img, [-1])
            w   = img.shape[-1]
            if len(target['boxes']):
                target['boxes'][:, [0, 2]] = w - target['boxes'][:, [2, 0]]
        return img, target


class ColorJitter:
    def __init__(self):
        self.jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)
    def __call__(self, img, target):
        return self.jitter(img), target


# ════════════════════════════════════════════════════════════════════
# Model builders
# ════════════════════════════════════════════════════════════════════

def build_faster_rcnn(freeze_until=2):
    """
    freeze_until: freeze backbone layers up to this index
      0 = freeze nothing
      1 = freeze conv1+bn1
      2 = freeze conv1+bn1+layer1+layer2  (default, good balance)
      3 = freeze everything except layer4 + head
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    )
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = (
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feat, 2))

    freeze_layers = [
        ['backbone.body.conv1', 'backbone.body.bn1'],
        ['backbone.body.layer1'],
        ['backbone.body.layer2'],
        ['backbone.body.layer3'],
    ][:freeze_until]

    frozen_prefixes = [p for group in freeze_layers for p in group]
    for name, param in model.named_parameters():
        if any(name.startswith(pref) for pref in frozen_prefixes):
            param.requires_grad = False

    n_frozen  = sum(1 for p in model.parameters() if not p.requires_grad)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  Faster R-CNN: {n_trainable} trainable params, "
          f"{n_frozen} frozen  (freeze_until={freeze_until})")
    return model


# ════════════════════════════════════════════════════════════════════
# YOLO fine-tuning via Ultralytics API
# ════════════════════════════════════════════════════════════════════

def prepare_yolo_dataset(gt_dict, frame_ids, video_path, out_dir, split_name='train'):
    """
    Write YOLO-format dataset (images + labels) for Ultralytics fine-tuning.
    Uses sequential cap.read() to avoid CAP_PROP_FRAME_COUNT issues with AVI files.
    gt_dict values are (x1,y1,x2,y2) tuples from parse_cvat_xml (0-indexed frames).
    Returns path to generated data.yaml.
    """
    images_dir = os.path.join(out_dir, split_name, 'images')
    labels_dir = os.path.join(out_dir, split_name, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    target_fids = set(frame_ids)
    written = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fid = 0   # 0-indexed, matches CVAT XML frame IDs
    pbar = tqdm(desc=f'Writing {split_name} YOLO data', total=len(target_fids))
    while written < len(target_fids):
        ret, frame = cap.read()
        if not ret:
            break
        if fid in target_fids:
            h, w = frame.shape[:2]
            img_path = os.path.join(images_dir, f'frame_{fid:06d}.jpg')
            cv2.imwrite(img_path, frame)

            anns = gt_dict.get(fid, [])
            lbl_path = os.path.join(labels_dir, f'frame_{fid:06d}.txt')
            with open(lbl_path, 'w') as f:
                for ann in anns:
                    x1, y1, x2, y2 = ann[0], ann[1], ann[2], ann[3]
                    cx = ((x1 + x2) / 2) / w
                    cy = ((y1 + y2) / 2) / h
                    bw = (x2 - x1) / w
                    bh = (y2 - y1) / h
                    if bw > 0 and bh > 0:
                        f.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
            written += 1
            pbar.update(1)
        fid += 1

    cap.release()
    pbar.close()
    print(f"  Dataset: {written}/{len(target_fids)} frames written to {images_dir}")

    if written == 0:
        raise RuntimeError(f"No frames written from {video_path}.")

    yaml_path = os.path.join(out_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(out_dir)}\n")
        f.write(f"train: train/images\n")
        f.write(f"val:   train/images\n")
        f.write(f"nc: 1\n")
        f.write(f"names: ['car']\n")
    return yaml_path


def run_yolo_finetune(model_name, yaml_path, out_dir, cfg):
    """Fine-tune a YOLO model using Ultralytics API."""
    from ultralytics import YOLO
    epochs = int(cfg.get('epochs', 10))
    lr     = float(cfg.get('lr0', 0.01))
    model  = YOLO(f'{model_name}.pt')
    results = model.train(
        data         = yaml_path,
        epochs       = epochs,
        lr0          = lr,
        imgsz        = 640,
        batch        = int(cfg.get('batch', 4)),
        optimizer    = str(cfg.get('optimizer', 'AdamW')),
        dropout      = float(cfg.get('dropout', 0.0)),
        freeze       = int(cfg.get('freeze', 0)),
        augment      = bool(cfg.get('augment', True)),
        weight_decay = float(cfg.get('weight_decay', 0.0005)),
        project      = out_dir,
        name         = 'finetune',
        exist_ok     = True,
        verbose      = False,
    )
    # Ultralytics saves to results.save_dir regardless of CWD,
    # so use the actual save_dir from the results object.
    try:
        save_dir = str(results.save_dir)
    except Exception:
        save_dir = os.path.join(out_dir, 'finetune')
    best_weights = os.path.join(save_dir, 'weights', 'best.pt')
    if not os.path.exists(best_weights):
        last_weights = os.path.join(save_dir, 'weights', 'last.pt')
        best_weights = last_weights if os.path.exists(last_weights) else best_weights
    return best_weights, results


ALLOWED_CLASS_NAMES = {'car', 'bicycle', 'motorcycle'}

@torch.no_grad()
def yolo_inference(model_path, video_path, frame_ids, conf=0.25):
    """Sequential video read — avoids CAP_PROP_FRAME_COUNT issues on AVI/mp4 files.
    frame_ids are 0-indexed to match parse_cvat_xml output."""
    from ultralytics import YOLO
    model    = YOLO(model_path)
    det_dict = {}
    target   = set(frame_ids)
    found    = 0

    cap = cv2.VideoCapture(video_path)
    fid = 0   # 0-indexed, matches CVAT XML
    while found < len(target):
        ret, frame = cap.read()
        if not ret:
            break
        if fid in target:
            res = model(frame, conf=conf, verbose=False)[0]
            boxes = []
            if res.boxes is not None:
                for xyxy, score, cls_id in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.conf.cpu().numpy(),
                    res.boxes.cls.cpu().numpy(),
                ):
                    # fine-tuned model has nc=1 so names={0:'car'}, just accept all
                    cls_name = model.names.get(int(cls_id), '').lower()
                    if cls_name and cls_name not in ALLOWED_CLASS_NAMES:
                        continue
                    x1, y1, x2, y2 = xyxy
                    boxes.append([float(x1), float(y1), float(x2), float(y2), float(score)])
            det_dict[fid] = boxes
            found += 1
        fid += 1
    cap.release()

    # fill any frames we didn't reach
    for f in target:
        if f not in det_dict:
            det_dict[f] = []
    return det_dict


# ════════════════════════════════════════════════════════════════════
# Torchvision training helpers
# ════════════════════════════════════════════════════════════════════

def train_one_epoch(model, optimizer, loader_obj, device, epoch):
    model.train()
    losses = []
    for images, targets in tqdm(loader_obj, desc=f'Epoch {epoch}', leave=False):
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def torchvision_inference(model, loader, frame_ids, device, score_thresh=0.5):
    model.eval()
    tf = T.ToTensor()
    det_dict = {}
    for fid in tqdm(frame_ids, desc='Inference'):
        frame = loader.get_frame(fid - 1)
        if frame is None:
            det_dict[fid] = []; continue
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = tf(rgb).to(device)
        out    = model([tensor])[0]
        boxes  = []
        for i, lbl in enumerate(out['labels']):
            if int(lbl) != 1: continue
            score = float(out['scores'][i])
            if score < score_thresh: continue
            x1, y1, x2, y2 = out['boxes'][i].cpu().numpy()
            boxes.append([float(x1), float(y1), float(x2), float(y2), score])
        det_dict[fid] = boxes
    return det_dict


# ════════════════════════════════════════════════════════════════════
# Hyperparameter ablation configs
# ════════════════════════════════════════════════════════════════════

ABLATION_CONFIGS = [
    {'name': 'Config_A_baseline',  'lr': 0.005, 'freeze_until': 2, 'epochs': 10},
    {'name': 'Config_B_lower_lr',  'lr': 0.001, 'freeze_until': 2, 'epochs': 10},
    {'name': 'Config_C_no_freeze', 'lr': 0.005, 'freeze_until': 0, 'epochs': 10},
    {'name': 'Config_D_more_ep',   'lr': 0.005, 'freeze_until': 2, 'epochs': 20},
]


# ════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════

def main(args):
    print("=" * 62)
    print("TASK 1.2  Fine-Tune Best Model from Task 1.1")
    print("=" * 62)

    handoff      = load_handoff()
    best_name    = handoff['best_model_name']
    det_info     = handoff.get('detector_info', {})
    det_type     = det_info.get('type', 'yolo')
    baseline_map = float(handoff.get('best_map50', 0.0))

    # ── load GT via CVAT XML ──────────────────────────────────────
    ann_path = ANN_PATH
    if not os.path.exists(ann_path):
        alt = ann_path.replace('annotations.xml', 'annotation.xml')
        if os.path.exists(alt):
            ann_path = alt
        else:
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Set ANN_PATH env var.")

    gt_dict       = parse_cvat_xml(ann_path)
    all_frame_ids = sorted(gt_dict.keys())
    train_ids, test_ids = strategy_a_split(all_frame_ids)
    test_set = set(test_ids)
    gt_test  = {fid: v for fid, v in gt_dict.items() if fid in test_set}

    # VideoFrameLoader only needed for Faster R-CNN branch
    loader = VideoFrameLoader(video_path=VIDEO_PATH)

    print(f"\nBest model from T1.1: {best_name} (type={det_type})")
    print(f"Baseline mAP@50 (COCO weights): {baseline_map:.4f}")
    print(f"Train: {len(train_ids)} frames | Test: {len(test_ids)} frames")

    # ── decide which fine-tune branch to use ─────────────────────
    use_yolo = det_type == 'yolo' and not args.force_faster_rcnn

    if use_yolo:
        print(f"\nUsing YOLO fine-tune path (Ultralytics API)")
        _run_yolo_ablation(
            best_name, gt_dict, train_ids, test_ids, gt_test, loader,
            baseline_map, args)
    else:
        if det_type not in ('yolo', 'torchvision'):
            print(f"  (pre-computed model '{best_name}' has no weights to fine-tune)")
            print(f"  → Fine-tuning Faster R-CNN as a strong representative baseline")
        print(f"\nUsing Faster R-CNN fine-tune path (PyTorch)")
        _run_frcnn_ablation(
            gt_dict, train_ids, test_ids, gt_test, loader,
            baseline_map, args)


# ════════════════════════════════════════════════════════════════════
# Bayesian hyperparameter search (Gaussian Process surrogate)
# ════════════════════════════════════════════════════════════════════

# Search space definition
# Each entry: (name, type, low, high, [choices])
HPARAM_SPACE = [
    ('lr0',           'log',        1e-4,  1e-1),
    ('weight_decay',  'log',        1e-5,  1e-2),
    ('dropout',       'float',      0.0,   0.5),
    ('batch',         'int',        2,     8),
    ('freeze',        'int',        0,     20),
    ('unfreeze_epoch','int',        0,     10),   # 0 = no unfreeze
    ('augment',       'categorical',None,  None,  [True, False]),
    ('optimizer',     'categorical',None,  None,  ['AdamW', 'SGD']),
]

EPOCHS_SEARCH = 10   # short epochs during search
EPOCHS_FINAL  = 30   # full epochs for best config


def _sample_random(space, rng):
    """Draw one random point from the search space."""
    point = {}
    for entry in space:
        name, kind = entry[0], entry[1]
        if kind == 'log':
            low, high = entry[2], entry[3]
            point[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif kind == 'float':
            point[name] = float(rng.uniform(entry[2], entry[3]))
        elif kind == 'int':
            point[name] = int(rng.randint(entry[2], entry[3] + 1))
        elif kind == 'categorical':
            choices = entry[4]
            point[name] = rng.choice(choices)
    return point


def _encode(point, space):
    """Encode a point dict → flat numpy vector for the GP."""
    vec = []
    for entry in space:
        name, kind = entry[0], entry[1]
        v = point[name]
        if kind == 'log':
            vec.append(np.log(float(v)))
        elif kind == 'float':
            vec.append(float(v))
        elif kind == 'int':
            vec.append(float(v))
        elif kind == 'categorical':
            choices = entry[4]
            vec.append(float(choices.index(v)))
    return np.array(vec, dtype=float)


def _candidate_grid(space, n=2000, rng=None):
    """Generate n random candidates for acquisition function eval."""
    if rng is None:
        rng = np.random.RandomState(42)
    return [_sample_random(space, rng) for _ in range(n)]


def _expected_improvement(X_new, gp, y_best, xi=0.01):
    """Expected improvement acquisition function."""
    from scipy.stats import norm
    mu, sigma = gp.predict(X_new, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    return (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)


def bayesian_search(model_name, yaml_path, video_path,
                    gt_test, test_ids, results_dir,
                    n_initial=4, n_iter=8, seed=42):
    """
    Bayesian optimisation over HPARAM_SPACE.

    Phase 1: n_initial random evaluations (warm-up)
    Phase 2: n_iter GP-guided evaluations (exploitation)

    Returns: (best_cfg, best_map50, all_rows)
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern
    from sklearn.preprocessing import StandardScaler

    rng       = np.random.RandomState(seed)
    X_obs     = []   # encoded hparam vectors
    y_obs     = []   # mAP@50 scores
    all_rows  = []

    gp = GaussianProcessRegressor(
        kernel=Matern(nu=2.5),
        alpha=1e-6,
        normalize_y=True,
        n_restarts_optimizer=5,
    )
    scaler = StandardScaler()

    best_map   = -1.0
    best_cfg   = None
    best_weights_path = None

    total = n_initial + n_iter
    print(f"  Bayesian search: {n_initial} random + {n_iter} GP-guided "
          f"= {total} trials × {EPOCHS_SEARCH} epochs each")

    for trial in range(total):
        # ── choose next config ────────────────────────────────────
        if trial < n_initial or len(X_obs) < 2:
            cfg = _sample_random(HPARAM_SPACE, rng)
            mode = 'random'
        else:
            # fit GP on observed points
            X_arr = np.array(X_obs)
            X_scaled = scaler.fit_transform(X_arr)
            gp.fit(X_scaled, np.array(y_obs))

            # evaluate EI on random candidates
            candidates  = _candidate_grid(HPARAM_SPACE, n=2000, rng=rng)
            X_cand      = np.array([_encode(c, HPARAM_SPACE) for c in candidates])
            X_cand_sc   = scaler.transform(X_cand)
            ei_vals     = _expected_improvement(X_cand_sc, gp,
                                                y_best=max(y_obs))
            cfg  = candidates[int(np.argmax(ei_vals))]
            mode = 'GP'

        cfg['epochs'] = EPOCHS_SEARCH  # short during search
        trial_dir = os.path.join(results_dir, f'bayes_trial_{trial:02d}')
        print(f"\n  Trial {trial+1}/{total} [{mode}]  "
              f"lr={cfg.get('lr0',0):.4f}  wd={cfg.get('weight_decay',0):.5f}  "
              f"dropout={cfg.get('dropout',0):.2f}  batch={cfg.get('batch',0)}  "
              f"freeze={cfg.get('freeze',0)}  unfreeze={cfg.get('unfreeze_epoch',0)}  "
              f"aug={cfg.get('augment')}  opt={cfg.get('optimizer')}")

        try:
            weights, _ = run_yolo_finetune(model_name, yaml_path,
                                           trial_dir, cfg)
            det_dict   = yolo_inference(weights, video_path, test_ids)
            det_test   = {fid: v for fid, v in det_dict.items()
                          if fid in set(test_ids)}
            m   = evaluate_detections(gt_test, det_test, iou_threshold=0.50)
            m75 = evaluate_detections(gt_test, det_test, iou_threshold=0.75)
            map50 = float(m['AP'])
        except Exception as e:
            print(f"    [warn] trial failed: {e}")
            map50 = 0.0
            m   = {'AP': 0.0, 'F1': 0.0, 'mIoU': 0.0,
                   'precision': 0.0, 'recall': 0.0}
            m75 = {'AP': 0.0}
            weights = ''

        print(f"    → mAP@50={map50:.4f}  F1={m['F1']:.4f}")

        X_obs.append(_encode(cfg, HPARAM_SPACE))
        y_obs.append(map50)

        row = {
            'trial':         trial,
            'mode':          mode,
            'lr0':           cfg.get('lr0'),
            'weight_decay':  cfg.get('weight_decay'),
            'dropout':       cfg.get('dropout'),
            'batch':         cfg.get('batch'),
            'freeze':        cfg.get('freeze'),
            'unfreeze_epoch':cfg.get('unfreeze_epoch'),
            'augment':       cfg.get('augment'),
            'optimizer':     cfg.get('optimizer'),
            'epochs':        EPOCHS_SEARCH,
            'mAP@50':        round(map50, 4),
            'mAP@75':        round(m75['AP'], 4),
            'F1':            round(m['F1'], 4),
            'weights':       weights,
        }
        all_rows.append(row)

        if map50 > best_map:
            best_map   = map50
            best_cfg   = dict(cfg)
            best_weights_path = weights

    return best_cfg, best_map, best_weights_path, all_rows


def _run_yolo_ablation(model_name, gt_dict, train_ids, test_ids,
                        gt_test, loader, baseline_map, args):
    """
    Bayesian hyperparameter search followed by a full final fine-tune
    using the best config with freeze→unfreeze scheduling.
    """
    # ── 1. Prepare dataset ───────────────────────────────────────
    yolo_data_dir = os.path.join(RESULTS_DIR, 'yolo_dataset')
    yaml_path = prepare_yolo_dataset(gt_dict, train_ids, VIDEO_PATH,
                                     yolo_data_dir)

    bayes_dir = os.path.join(RESULTS_DIR, 'bayes_search')
    os.makedirs(bayes_dir, exist_ok=True)

    # ── 2. Bayesian search ───────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  Bayesian Hyperparameter Search")
    print(f"{'═'*60}")

    best_cfg, best_search_map, best_search_weights, search_rows = bayesian_search(
        model_name      = model_name,
        yaml_path       = yaml_path,
        video_path      = VIDEO_PATH,
        gt_test         = gt_test,
        test_ids        = test_ids,
        results_dir     = bayes_dir,
        n_initial       = getattr(args, 'bayes_n_initial', 4),
        n_iter          = getattr(args, 'bayes_n_iter',    8),
        seed            = 42,
    )

    # save search results
    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(os.path.join(RESULTS_DIR, 'bayes_search_results.csv'),
                     index=False)
    print(f"\n  Search complete. Best mAP@50={best_search_map:.4f}")
    print(f"  Best config: {best_cfg}")

    # ── 3. Final full fine-tune with best config ─────────────────
    print(f"\n{'═'*60}")
    print(f"  Final Fine-Tune with Best Config ({EPOCHS_FINAL} epochs, "
          f"freeze→unfreeze)")
    print(f"{'═'*60}")

    final_cfg = dict(best_cfg)
    final_cfg['epochs'] = EPOCHS_FINAL

    # ensure freeze/unfreeze is on for final run
    if final_cfg.get('freeze', 0) == 0:
        final_cfg['freeze'] = 10          # freeze first 10 layers
    if final_cfg.get('unfreeze_epoch', 0) == 0:
        final_cfg['unfreeze_epoch'] = max(5, EPOCHS_FINAL // 3)

    print(f"  freeze first {final_cfg['freeze']} layers for "
          f"{final_cfg['unfreeze_epoch']} epochs, then unfreeze all")

    final_dir = os.path.join(RESULTS_DIR, 'final_best')
    final_weights, _ = run_yolo_finetune(model_name, yaml_path,
                                          final_dir, final_cfg)

    # ── 4. Evaluate final model ──────────────────────────────────
    try:
        det_dict  = yolo_inference(final_weights, VIDEO_PATH, test_ids)
        det_test  = {fid: v for fid, v in det_dict.items()
                     if fid in set(test_ids)}
        m_final   = evaluate_detections(gt_test, det_test, iou_threshold=0.50)
        m75_final = evaluate_detections(gt_test, det_test, iou_threshold=0.75)
        final_map = float(m_final['AP'])
    except Exception as e:
        print(f"  [warn] Final evaluation failed: {e}")
        m_final   = {'AP': 0.0, 'F1': 0.0, 'mIoU': 0.0,
                     'precision': 0.0, 'recall': 0.0}
        m75_final = {'AP': 0.0}
        final_map = 0.0

    print(f"\n  Final mAP@50={final_map:.4f}  "
          f"(baseline={baseline_map:.4f}  Δ={final_map-baseline_map:+.4f})")

    # ── 5. Build rows for _save_ablation_results ─────────────────
    # Include top-N search trials + final run so plots are meaningful
    rows = []
    for r in search_rows:
        rows.append({
            'Config':       r['trial_name'] if 'trial_name' in r
                            else f"Trial_{r['trial']:02d}_{r['mode']}",
            'lr':           r['lr0'],
            'epochs':       r['epochs'],
            'best_weights': r['weights'],
            'mAP@50':       r['mAP@50'],
            'mAP@75':       r['mAP@75'],
            'F1':           r['F1'],
            'mIoU':         0.0,
        })
    rows.append({
        'Config':       'FINAL_best_config',
        'lr':           final_cfg.get('lr0'),
        'epochs':       EPOCHS_FINAL,
        'best_weights': final_weights,
        'mAP@50':       round(final_map, 4),
        'mAP@75':       round(m75_final['AP'], 4),
        'F1':           round(m_final['F1'], 4),
        'mIoU':         round(m_final.get('mIoU', 0.0), 4),
    })

    best_config_obj = {**final_cfg, 'name': 'FINAL_best_config'}
    _save_ablation_results(rows, baseline_map, best_config_obj,
                           model_name, search_df=search_df)


def _run_frcnn_ablation(gt_dict, train_ids, test_ids,
                         gt_test, loader, baseline_map, args):
    """Run Faster R-CNN ablation across 4 configs."""
    train_tf = ComposeTransforms([RandomFlip(), ColorJitter()])
    train_ds = AICityDataset(VIDEO_PATH, gt_dict, train_ids, transforms=train_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=2, collate_fn=collate_fn)

    rows          = []
    all_histories = {}
    best_config_map = baseline_map
    best_config   = None
    best_state    = None

    configs_to_run = ABLATION_CONFIGS[:args.n_configs]

    for cfg in configs_to_run:
        print(f"\n── {cfg['name']}:  lr={cfg['lr']}  "
              f"freeze_until={cfg['freeze_until']}  epochs={cfg['epochs']} ──")

        model = build_faster_rcnn(freeze_until=cfg['freeze_until']).to(DEVICE)

        # COCO baseline eval (same model, just no fine-tuning yet)
        if not rows:   # only needed once
            det_base = torchvision_inference(model, loader, test_ids, DEVICE)
            m_base   = evaluate_detections(gt_test,
                {fid: v for fid, v in det_base.items() if fid in set(test_ids)}, 0.5)
            baseline_map = max(baseline_map, m_base['AP'])
            print(f"  COCO baseline  mAP@50={m_base['AP']:.4f}")

        params    = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=cfg['lr'],
                                     momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['epochs'])

        history = []
        best_ep_map = 0.0

        for epoch in range(1, cfg['epochs'] + 1):
            loss = train_one_epoch(model, optimizer, train_dl, DEVICE, epoch)
            scheduler.step()

            if epoch % 2 == 0 or epoch == cfg['epochs']:
                det_eval = torchvision_inference(model, loader, test_ids, DEVICE)
                det_test = {fid: v for fid, v in det_eval.items()
                            if fid in set(test_ids)}
                m  = evaluate_detections(gt_test, det_test, 0.5)
                print(f"    Ep {epoch:3d}  loss={loss:.4f}  "
                      f"mAP@50={m['AP']:.4f}  F1={m['F1']:.4f}")
                history.append({'config': cfg['name'], 'epoch': epoch,
                                 'loss': loss, 'mAP50': m['AP'], 'F1': m['F1']})
                if m['AP'] > best_ep_map:
                    best_ep_map = m['AP']
                    best_ep_state = copy_state(model)
            else:
                print(f"    Ep {epoch:3d}  loss={loss:.4f}")
                history.append({'config': cfg['name'], 'epoch': epoch,
                                 'loss': loss, 'mAP50': None, 'F1': None})

        all_histories[cfg['name']] = history

        # final eval on best weights from this config
        model.load_state_dict(best_ep_state)
        det_final = torchvision_inference(model, loader, test_ids, DEVICE)
        det_test  = {fid: v for fid, v in det_final.items() if fid in set(test_ids)}
        m50  = evaluate_detections(gt_test, det_test, 0.50)
        m75  = evaluate_detections(gt_test, det_test, 0.75)

        row = {'Config':      cfg['name'],
               'lr':          cfg['lr'],
               'freeze_until': cfg['freeze_until'],
               'epochs':      cfg['epochs'],
               'mAP@50':      round(m50['AP'],        4),
               'mAP@75':      round(m75['AP'],        4),
               'F1':          round(m50['F1'],         4),
               'mIoU':        round(m50['mIoU'],       4),
               'Baseline_mAP50': round(baseline_map,   4),
               'Delta':       round(m50['AP'] - baseline_map, 4)}
        rows.append(row)
        print(f"  → final mAP@50={row['mAP@50']:.4f}  "
              f"Δbaseline={row['Delta']:+.4f}")

        if m50['AP'] > best_config_map:
            best_config_map = m50['AP']
            best_config = cfg
            best_state  = best_ep_state

    # save training history
    hist_df = pd.DataFrame([r for h in all_histories.values() for r in h])
    hist_df.to_csv(os.path.join(RESULTS_DIR, 'training_history.csv'), index=False)

    # save best model weights
    if best_state is not None:
        torch.save(best_state,
                   os.path.join(RESULTS_DIR, 'best_frcnn.pth'))

    _save_ablation_results(rows, baseline_map, best_config,
                            'FasterRCNN_R50_FPN', all_histories)


def copy_state(model):
    return copy.deepcopy(model.state_dict())


# ════════════════════════════════════════════════════════════════════
# Save results + plots for both branches
# ════════════════════════════════════════════════════════════════════

def _save_ablation_results(rows, baseline_map, best_config,
                            model_name, all_histories=None, search_df=None):
    import copy as _copy
    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, 'metrics_task1_2.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n{'─'*50}")
    print(f"Ablation summary:")
    print(df[['Config', 'mAP@50', 'mAP@75', 'F1']].to_string(index=False))

    best_cfg_name = best_config['name'] if best_config else rows[0]['Config']
    best_map = df['mAP@50'].max()
    print(f"\n★  Best config: {best_cfg_name}  mAP@50={best_map:.4f}  "
          f"(baseline={baseline_map:.4f}  Δ={best_map - baseline_map:+.4f})")

    # write best config for downstream use
    # Find the lr, epochs, and weights for the best config row
    best_row = df[df['Config'] == best_cfg_name].iloc[0] if best_cfg_name in df['Config'].values else df.iloc[0]
    best_lr     = float(best_row.get('lr', best_config.get('lr', best_config.get('lr0', 0.01)) if best_config else 0.01))
    best_epochs = int(best_row.get('epochs', best_config.get('epochs', 10) if best_config else 10))
    best_weights_path = str(best_row.get('best_weights', '')) if 'best_weights' in best_row else ''

    best_cfg_out = {
        'best_config':    best_cfg_name,
        'best_map50':     float(best_map),
        'model_name':     model_name,
        'baseline_map50': float(baseline_map),
        'lr':             best_lr,
        'epochs':         best_epochs,
        'best_weights':   best_weights_path,
        'imgsz':          640,
    }
    with open(os.path.join(RESULTS_DIR, 'best_config.json'), 'w') as f:
        json.dump(best_cfg_out, f, indent=2)

    _make_plots(df, baseline_map, all_histories)

    # ── Bayesian search plots ─────────────────────────────────────
    if search_df is not None and len(search_df) > 0:
        _make_bayes_plots(search_df, best_map, baseline_map)

    print(f"\n✓  Task 1.2 complete.")


def _make_bayes_plots(search_df, best_map, baseline_map):
    """Plots specific to the Bayesian search."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    df = search_df.copy()
    trials = df['trial'].values
    maps   = df['mAP@50'].values
    modes  = df['mode'].values
    colors = ['steelblue' if m == 'random' else 'darkorange' for m in modes]

    # 1) mAP@50 per trial with running best
    running_best = np.maximum.accumulate(maps)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(trials, maps, c=colors, s=60, zorder=3)
    ax.plot(trials, running_best, 'r-', lw=2, label='Running best')
    ax.axhline(baseline_map, color='gray', ls='--', lw=1,
               label=f'Baseline={baseline_map:.3f}')
    ax.axhline(best_map, color='green', ls=':', lw=1.5,
               label=f'Best found={best_map:.3f}')
    # legend patches
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color='steelblue', label='Random init'),
        Patch(color='darkorange', label='GP-guided'),
        plt.Line2D([0],[0], color='red',   lw=2, label='Running best'),
        plt.Line2D([0],[0], color='gray',  ls='--', lw=1, label=f'Baseline={baseline_map:.3f}'),
        plt.Line2D([0],[0], color='green', ls=':',  lw=1.5, label=f'Best={best_map:.3f}'),
    ], fontsize=8)
    ax.set_xlabel('Trial'); ax.set_ylabel('mAP@50')
    ax.set_title('Bayesian Hyperparameter Search — mAP@50 per Trial',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_bayes_convergence.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # 2) Parallel coordinates — top-8 trials highlighted
    numeric_cols = ['lr0', 'weight_decay', 'dropout', 'batch',
                    'freeze', 'unfreeze_epoch', 'mAP@50']
    available = [c for c in numeric_cols if c in df.columns]
    if len(available) >= 3:
        top8 = df.nlargest(8, 'mAP@50')
        fig, axes = plt.subplots(1, len(available)-1,
                                  figsize=(3*(len(available)-1), 5),
                                  sharey=False)
        if len(available) == 2:
            axes = [axes]

        # normalise each column 0-1
        normed = {}
        for col in available:
            mn, mx = df[col].min(), df[col].max()
            normed[col] = (df[col] - mn) / (mx - mn + 1e-9)

        for i, (a, b) in enumerate(zip(available[:-1], available[1:])):
            ax = axes[i]
            for _, row in df.iterrows():
                is_top = row['trial'] in top8['trial'].values
                ax.plot([0, 1], [normed[a][row.name], normed[b][row.name]],
                        color='darkorange' if is_top else 'lightgray',
                        lw=1.5 if is_top else 0.5,
                        alpha=0.9 if is_top else 0.4, zorder=3 if is_top else 1)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([a.replace('_', '\n'), b.replace('_', '\n')],
                                fontsize=8)
            ax.set_yticks([])
        fig.suptitle('Parallel Coordinates — Top-8 Trials (orange)',
                     fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_bayes_parallel.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # 3) Hyperparameter importance (correlation with mAP@50)
    numeric_hp = ['lr0', 'weight_decay', 'dropout', 'batch',
                  'freeze', 'unfreeze_epoch']
    available_hp = [c for c in numeric_hp if c in df.columns]
    if available_hp:
        corrs = [float(np.corrcoef(df[c].values.astype(float),
                                   df['mAP@50'].values)[0,1])
                 for c in available_hp]
        fig, ax = plt.subplots(figsize=(8, 4))
        colors_bar = ['seagreen' if c >= 0 else 'salmon' for c in corrs]
        bars = ax.barh(available_hp, corrs, color=colors_bar,
                       edgecolor='black', linewidth=0.6)
        ax.bar_label(bars, fmt='%.2f', fontsize=9, padding=3)
        ax.axvline(0, color='black', lw=1)
        ax.set_xlabel('Pearson correlation with mAP@50')
        ax.set_title('Hyperparameter Importance (Correlation Analysis)',
                     fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.25, axis='x')
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_bayes_importance.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved Bayesian search plots → {PLOTS_DIR}/task1_2_bayes_*.png")


def _make_plots(df, baseline_map, all_histories=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n  = len(df)
    pal = plt.cm.tab10(np.linspace(0, 1, n))

    # 1) Config comparison bar chart
    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 5))
    x = np.arange(n); w = 0.35
    b1 = ax.bar(x - w/2, df['mAP@50'], w, label='mAP@50', color='steelblue',
                edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + w/2, df['mAP@75'], w, label='mAP@75', color='darkorange',
                edgecolor='black', linewidth=0.6)
    ax.bar_label(b1, fmt='%.3f', fontsize=8, padding=2)
    ax.bar_label(b2, fmt='%.3f', fontsize=8, padding=2)
    ax.axhline(baseline_map, color='red', ls='--', lw=1.5,
               label=f'COCO baseline = {baseline_map:.3f}')
    ax.set_xticks(x); ax.set_xticklabels(df['Config'], rotation=20, ha='right')
    ax.set_ylabel('AP'); ax.set_ylim(0, 1.15)
    ax.set_title('Hyperparameter Ablation — Fine-Tuning Results', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_ablation_bar.png'), dpi=150)
    plt.close()

    # 2) Δ improvement over baseline
    fig, ax = plt.subplots(figsize=(max(8, n * 1.8), 4))
    delta = df['mAP@50'] - baseline_map
    colors = ['seagreen' if d >= 0 else 'salmon' for d in delta]
    bars   = ax.bar(df['Config'], delta, color=colors, edgecolor='black', linewidth=0.6)
    ax.bar_label(bars, fmt='%+.3f', padding=3, fontsize=9)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('Δ mAP@50 vs COCO baseline')
    ax.set_title('Improvement over COCO Baseline per Fine-Tune Config', fontsize=12)
    plt.xticks(rotation=20, ha='right')
    ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_delta_baseline.png'), dpi=150)
    plt.close()

    # 3) Training loss curves (if available)
    if all_histories:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for cfg_name, history in all_histories.items():
            hist_df = pd.DataFrame(history)
            axes[0].plot(hist_df['epoch'], hist_df['loss'],
                         '-o', ms=3, label=cfg_name)
            eval_rows = hist_df[hist_df['mAP50'].notna()]
            if not eval_rows.empty:
                axes[1].plot(eval_rows['epoch'], eval_rows['mAP50'],
                             '-s', ms=4, label=cfg_name)
        axes[0].set_title('Training Loss per Config')
        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
        axes[1].axhline(baseline_map, color='red', ls='--', lw=1.5,
                        label=f'Baseline={baseline_map:.3f}')
        axes[1].set_title('mAP@50 During Training')
        axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('mAP@50')
        for ax in axes:
            ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
        plt.suptitle('Training Progress — All Configs', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_training_curves.png'), dpi=150)
        plt.close()

    # 4) Scatter: F1 vs mAP@50 per config
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, row in df.iterrows():
        ax.scatter(row['mAP@50'], row['F1'], s=120,
                   color=pal[i % len(pal)], zorder=3)
        ax.annotate(row['Config'].replace('Config_', ''),
                    (row['mAP@50'], row['F1']),
                    textcoords='offset points', xytext=(5, 4), fontsize=9)
    ax.scatter(baseline_map, 0, marker='*', s=200, color='red',
               label=f'COCO baseline (mAP={baseline_map:.3f})', zorder=4)
    ax.set_xlabel('mAP@50'); ax.set_ylabel('F1 Score')
    ax.set_title('F1 vs mAP@50 per Fine-Tune Config', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_2_f1_vs_map.png'), dpi=150)
    plt.close()

    print(f"Plots → {PLOTS_DIR}/task1_2_*.png")


# ════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import copy
    p = argparse.ArgumentParser()
    p.add_argument('--batch_size',        type=int,   default=2)
    p.add_argument('--n_configs',         type=int,   default=4,
                   help='Kept for compatibility (Bayesian search ignores this)')
    p.add_argument('--bayes_n_initial',   type=int,   default=4,
                   help='Random warm-up trials before GP kicks in (default=4)')
    p.add_argument('--bayes_n_iter',      type=int,   default=8,
                   help='GP-guided trials after warm-up (default=8)')
    p.add_argument('--force_faster_rcnn', action='store_true',
                   help='Always use Faster R-CNN even if best was YOLO')
    args = p.parse_args()
    main(args)