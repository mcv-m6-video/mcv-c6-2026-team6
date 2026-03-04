"""
task1/task1_1_off_the_shelf.py

Task 1.1: Evaluate multiple off-the-shelf detectors on S03/C010 (test split).

Models:
  Pre-computed (from dataset): MaskRCNN_provided, SSD512_provided, YOLOv3_provided
  Live (--yolo_sizes n s m l x): YOLOv8 family
  Live (--yolov9):                YOLOv9c
  Live (--faster_rcnn):           Faster R-CNN ResNet-50 FPN
  Live (--mask_rcnn):             Mask R-CNN ResNet-50 FPN
  Live (--ssd):                   SSD300 VGG16

Writes results/task1_1/best_model.json  <- read by task1_2 automatically
"""

import os, sys, cv2, json, time, argparse
import numpy as np
import pandas as pd
import torch
import torchvision
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.data_utils import (
    parse_annotations_mot, parse_detections_mot, save_detections_mot,
    VideoFrameLoader, strategy_a_split,
)
from utils.metrics import evaluate_detections
from utils.visualization import draw_gt_and_det, save_qualitative_grid

DATA_ROOT   = os.environ.get('DATA_ROOT', 'data/AICity_data/train/S03/c010')
GT_PATH     = os.path.join(DATA_ROOT, 'gt/gt.txt')
DET_DIR     = os.path.join(DATA_ROOT, 'det')
VIDEO_PATH  = os.path.join(DATA_ROOT, 'vdo.avi')
RESULTS_DIR = 'results/task1_1'
PLOTS_DIR   = 'plots'
QUAL_DIR    = 'qualitative'
for d in [RESULTS_DIR + '/detections', PLOTS_DIR, QUAL_DIR]:
    os.makedirs(d, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[task1_1] device: {DEVICE}")
COCO_CAR_ID = 3


# ── detector wrappers ────────────────────────────────────────────

ALLOWED_CLASS_NAMES = {"car", "bicycle", "motorcycle"}

class YOLODetector:
    def __init__(self, model_id='yolov8m', conf=0.25, imgsz=1280):
        from ultralytics import YOLO
        self.model = YOLO(f'{model_id}.pt')
        self.conf  = conf
        self.imgsz = imgsz
        self.name  = model_id
        self._type = 'yolo'

    def detect(self, frame):
        res = self.model(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)
        boxes = []
        for r in res:
            if r.boxes is None:
                continue
            for xyxy, score, cls_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy(),
            ):
                cls_name = self.model.names.get(int(cls_id), "").lower()
                if cls_name not in ALLOWED_CLASS_NAMES:
                    continue
                x1, y1, x2, y2 = xyxy
                boxes.append([float(x1), float(y1), float(x2), float(y2), float(score)])
        return boxes

    @property
    def info(self):
        return {'name': self.name, 'type': self._type,
                'weights': f'{self.name}.pt'}


class TorchvisionDetector:
    def __init__(self, model_fn, weights, name, conf=0.5):
        self.model = model_fn(weights=weights).to(DEVICE).eval()
        self.conf  = conf
        self.name  = name
        self._type = 'torchvision'
        self._tf   = torchvision.transforms.ToTensor()

    @torch.no_grad()
    def detect(self, frame):
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = self._tf(rgb).to(DEVICE)
        out    = self.model([tensor])[0]
        boxes  = []
        for i, lbl in enumerate(out['labels']):
            if int(lbl) != COCO_CAR_ID:
                continue
            score = float(out['scores'][i])
            if score < self.conf:
                continue
            x1, y1, x2, y2 = out['boxes'][i].cpu().numpy()
            boxes.append([float(x1), float(y1), float(x2), float(y2), score])
        return boxes

    @property
    def info(self):
        return {'name': self.name, 'type': self._type}


def build_live_detectors(args):
    detectors = []
    imgsz = getattr(args, 'imgsz', 1280)
    if args.yolo_sizes:
        for size in args.yolo_sizes:
            try:
                detectors.append(YOLODetector(f'yolov8{size}', conf=0.25, imgsz=imgsz))
            except Exception as e:
                print(f"  [warn] YOLOv8{size}: {e}")
    if getattr(args, 'yolo26', None):
        for size in args.yolo26:
            try:
                detectors.append(YOLODetector(f'yolo26{size}', conf=0.25, imgsz=imgsz))
            except Exception as e:
                print(f"  [warn] YOLO26{size}: {e}")
    if args.yolov9:
        try:
            detectors.append(YOLODetector('yolov9c', conf=0.25, imgsz=imgsz))
        except Exception as e:
            print(f"  [warn] YOLOv9c: {e}")
    if args.faster_rcnn:
        detectors.append(TorchvisionDetector(
            torchvision.models.detection.fasterrcnn_resnet50_fpn,
            torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            'FasterRCNN_R50_FPN', conf=0.5))
    if args.mask_rcnn:
        detectors.append(TorchvisionDetector(
            torchvision.models.detection.maskrcnn_resnet50_fpn,
            torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
            'MaskRCNN_R50_FPN', conf=0.5))
    if args.ssd:
        detectors.append(TorchvisionDetector(
            torchvision.models.detection.ssd300_vgg16,
            torchvision.models.detection.SSD300_VGG16_Weights.COCO_V1,
            'SSD300_VGG16', conf=0.4))
    return detectors


def _iter_video_sequential(video_path):
    """
    Iterate frames using plain sequential cap.read().
    Avoids relying on CAP_PROP_FRAME_COUNT which is unreliable for
    msmpeg4v2/AVI files and returns 0, causing range(0,0) = no frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [error] Cannot open video: {video_path}")
        return
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        yield frame_id, frame
    cap.release()
    print(f"  [video] Read {frame_id} frames from {video_path}")


def run_detector_on_video(detector, video_path):
    det_dict = {}
    t0 = time.time()
    count = 0
    for frame_id, frame in tqdm(_iter_video_sequential(video_path),
                                 desc=detector.name):
        det_dict[frame_id] = detector.detect(frame)
        count += 1
    elapsed = max(time.time() - t0, 1e-3)
    fps = count / elapsed
    print(f"  {detector.name}: {count} frames  {fps:.1f} FPS")
    return det_dict


def conf_sweep(gt_dict, det_dict, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.10, 0.92, 0.05)
    rows = []
    for thr in thresholds:
        m = evaluate_detections(gt_dict, det_dict,
                                iou_threshold=0.5, score_threshold=float(thr))
        rows.append({'conf_thr': round(float(thr), 2), 'AP': m['AP'],
                     'F1': m['F1'], 'precision': m['precision'],
                     'recall': m['recall']})
    return pd.DataFrame(rows)


def main(args):
    print("=" * 62)
    print("TASK 1.1  Off-the-Shelf Object Detection")
    print("=" * 62)

    # ── diagnostics: confirm paths exist before doing anything ──────
    print(f"  GT_PATH    : {GT_PATH}  exists={os.path.exists(GT_PATH)}")
    print(f"  VIDEO_PATH : {VIDEO_PATH}  exists={os.path.exists(VIDEO_PATH)}")
    print(f"  DET_DIR    : {DET_DIR}  exists={os.path.exists(DET_DIR)}")

    if not os.path.exists(GT_PATH):
        print(f"[ERROR] Ground truth file not found: {GT_PATH}")
        print("       Set DATA_ROOT env var to the correct path.")
        return

    gt_dict       = parse_annotations_mot(GT_PATH)
    all_frame_ids = sorted(gt_dict.keys())
    print(f"  GT frames  : {len(all_frame_ids)}  (first={all_frame_ids[0] if all_frame_ids else 'none'}, last={all_frame_ids[-1] if all_frame_ids else 'none'})")

    if not all_frame_ids:
        print("[ERROR] GT file parsed but contains no frames. Check format.")
        return

    train_ids, test_ids = strategy_a_split(all_frame_ids)
    test_set = set(test_ids)
    gt_test  = {fid: v for fid, v in gt_dict.items() if fid in test_set}
    print(f"Train: {len(train_ids)} frames  |  Test: {len(test_ids)} frames")

    all_det_dicts       = {}
    live_detector_info  = {}

    # pre-computed detections
    PRECOMP = [('det_mask_rcnn.txt', 'MaskRCNN_provided'),
               ('det_ssd512.txt',   'SSD512_provided'),
               ('det_yolo3.txt',    'YOLOv3_provided')]
    for fname, name in PRECOMP:
        fp = os.path.join(DET_DIR, fname)
        if os.path.exists(fp):
            all_det_dicts[name] = parse_detections_mot(fp)
            print(f"  Loaded pre-computed: {name}")

    # live inference
    if not args.precomputed_only:
        for det in build_live_detectors(args):
            dd = run_detector_on_video(det, VIDEO_PATH)
            all_det_dicts[det.name] = dd
            live_detector_info[det.name] = det.info
            save_detections_mot(dd,
                os.path.join(RESULTS_DIR, 'detections', f'{det.name}.txt'))

    if not all_det_dicts:
        print("No detections. Use --precomputed_only or add --yolo_sizes m x")
        return

    # evaluate
    print("\n── Evaluation (test split, IoU@0.5) ──")
    rows = {}
    sweep_results = {}
    for name, det_dict in all_det_dicts.items():
        det_test = {fid: v for fid, v in det_dict.items() if fid in test_set}
        m50 = evaluate_detections(gt_test, det_test, iou_threshold=0.50)
        m75 = evaluate_detections(gt_test, det_test, iou_threshold=0.75)
        rows[name] = {
            'Model':     name,
            'mAP@50':    round(m50['AP'],        4),
            'mAP@75':    round(m75['AP'],        4),
            'mIoU':      round(m50['mIoU'],      4),
            'Precision': round(m50['precision'], 4),
            'Recall':    round(m50['recall'],    4),
            'F1':        round(m50['F1'],        4),
        }
        sweep_results[name] = conf_sweep(gt_test, det_test)
        sweep_results[name].to_csv(
            os.path.join(RESULTS_DIR, f'conf_sweep_{name}.csv'), index=False)
        print(f"  {name:28s}  mAP@50={rows[name]['mAP@50']:.4f}  "
              f"mAP@75={rows[name]['mAP@75']:.4f}  F1={rows[name]['F1']:.4f}")

    df = pd.DataFrame(list(rows.values())).sort_values('mAP@50', ascending=False)
    df.to_csv(os.path.join(RESULTS_DIR, 'metrics_task1_1.csv'), index=False)

    best_name = df.iloc[0]['Model']
    best_map  = float(df.iloc[0]['mAP@50'])
    print(f"\n★  Best: {best_name}  mAP@50={best_map:.4f}")

    # ── handoff file for task1_2 ──────────────────────────────────
    precomp_map = {'MaskRCNN_provided': 'det_mask_rcnn.txt',
                   'SSD512_provided':   'det_ssd512.txt',
                   'YOLOv3_provided':   'det_yolo3.txt'}
    if best_name in live_detector_info:
        det_txt = os.path.join(RESULTS_DIR, 'detections', f'{best_name}.txt')
    elif best_name in precomp_map:
        det_txt = os.path.join(DET_DIR, precomp_map[best_name])
    else:
        det_txt = ''

    handoff = {
        'best_model_name': best_name,
        'best_map50':      best_map,
        'detector_info':   live_detector_info.get(best_name, {}),
        'det_txt':         det_txt,
        'all_models':      df.to_dict(orient='records'),
    }
    handoff_path = os.path.join(RESULTS_DIR, 'best_model.json')
    with open(handoff_path, 'w') as f:
        json.dump(handoff, f, indent=2)
    print(f"Handoff written → {handoff_path}")

    _save_plots(df, gt_test, all_det_dicts, test_set, sweep_results, best_name)
    _save_qualitative(gt_dict, all_det_dicts, test_ids, best_name)
    print("\n✓  Task 1.1 complete.")
    return df, handoff


def _save_plots(df, gt_test, all_det_dicts, test_set, sweep_results, best_name):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n  = len(df)
    pal = plt.cm.tab20(np.linspace(0, 1, n))

    # 1) mAP@50 bar
    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
    colors = ['gold' if m == best_name else c
              for m, c in zip(df['Model'], pal)]
    bars = ax.bar(df['Model'], df['mAP@50'], color=colors,
                  edgecolor='black', linewidth=0.7)
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_title('Off-the-Shelf Detectors — mAP@50  (test split)', fontsize=13)
    ax.set_ylabel('mAP@50'); ax.set_ylim(0, 1.1)
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.axhline(df['mAP@50'].max(), color='red', ls=':', lw=1,
               label=f'Best = {df["mAP@50"].max():.3f}')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_map50_bar.png'), dpi=150)
    plt.close()

    # 2) grouped mAP@50 vs mAP@75
    x = np.arange(n); w = 0.38
    fig, ax = plt.subplots(figsize=(max(10, n * 1.4), 5))
    b1 = ax.bar(x - w/2, df['mAP@50'], w, label='mAP@50',
                color='steelblue', edgecolor='black', linewidth=0.6)
    b2 = ax.bar(x + w/2, df['mAP@75'], w, label='mAP@75',
                color='darkorange', edgecolor='black', linewidth=0.6)
    ax.bar_label(b1, fmt='%.3f', fontsize=7, padding=2)
    ax.bar_label(b2, fmt='%.3f', fontsize=7, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('AP'); ax.set_ylim(0, 1.15)
    ax.set_title('mAP@50 vs mAP@75 — All Detectors', fontsize=13)
    ax.legend(); ax.grid(True, alpha=0.25, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_map50_vs_75.png'), dpi=150)
    plt.close()

    # 3) PR curves
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (name, det_dict) in enumerate(all_det_dicts.items()):
        det_test = {fid: v for fid, v in det_dict.items() if fid in test_set}
        m = evaluate_detections(gt_test, det_test)
        r, p = m['pr_curve']
        lw = 2.5 if name == best_name else 1.2
        ax.plot(r, p, lw=lw,
                color=plt.cm.tab20(i / max(len(all_det_dicts)-1, 1)),
                label=f'{name}  AP={m["AP"]:.3f}',
                zorder=3 if name == best_name else 2)
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curves', fontsize=12)
    ax.legend(fontsize=7); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_pr_curves.png'), dpi=150)
    plt.close()

    # 4) IoU over time (top 3)
    top3 = df['Model'].head(3).tolist()
    fig, ax = plt.subplots(figsize=(13, 4))
    for name in top3:
        det_test = {fid: v for fid, v in all_det_dicts[name].items()
                    if fid in test_set}
        m = evaluate_detections(gt_test, det_test)
        fids = sorted(m['iou_over_time'].keys())
        ious = [m['iou_over_time'][f] for f in fids]
        ax.plot(fids, ious, label=name, alpha=0.85,
                lw=2 if name == best_name else 1.2)
    ax.set_xlabel('Frame ID'); ax.set_ylabel('Mean IoU')
    ax.set_title('IoU over Time — Top 3 Detectors', fontsize=12)
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_iou_over_time.png'), dpi=150)
    plt.close()

    # 5) confidence sweep (top 3)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in top3:
        sw = sweep_results[name]
        axes[0].plot(sw['conf_thr'], sw['AP'], marker='o', ms=4, label=name)
        axes[1].plot(sw['conf_thr'], sw['F1'], marker='s', ms=4, label=name)
    for ax, ylabel in zip(axes, ['AP', 'F1']):
        ax.set_xlabel('Confidence Threshold'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.25)
    axes[0].set_title('AP vs Confidence Threshold')
    axes[1].set_title('F1 vs Confidence Threshold')
    plt.suptitle('Confidence Sweep (IoU@0.5)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_conf_sweep.png'), dpi=150)
    plt.close()

    # 6) YOLO model-size scaling (if multiple yolov8 present)
    yolo_rows = df[df['Model'].str.startswith('yolov8')].copy()
    param_map = {'yolov8n': 3.2, 'yolov8s': 11.2,
                 'yolov8m': 25.9, 'yolov8l': 43.7, 'yolov8x': 68.2}
    if len(yolo_rows) >= 2:
        yolo_rows['params_M'] = yolo_rows['Model'].map(
            lambda nm: param_map.get(nm, np.nan))
        yolo_rows = yolo_rows.dropna(subset=['params_M'])
        if len(yolo_rows) >= 2:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.plot(yolo_rows['params_M'], yolo_rows['mAP@50'],
                    'b-o', ms=8, lw=2, label='mAP@50')
            ax.plot(yolo_rows['params_M'], yolo_rows['mAP@75'],
                    'r--s', ms=6, lw=1.5, label='mAP@75')
            for _, row in yolo_rows.iterrows():
                ax.annotate(row['Model'],
                            (row['params_M'], row['mAP@50']),
                            textcoords='offset points', xytext=(4, 4), fontsize=9)
            ax.set_xlabel('Parameters (M)'); ax.set_ylabel('AP')
            ax.set_title('YOLOv8 Model-Size Scaling', fontsize=12)
            ax.legend(); ax.grid(True, alpha=0.25)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_yolo_scaling.png'), dpi=150)
            plt.close()

    # 7) IoU histogram (top 3)
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in top3:
        det_test = {fid: v for fid, v in all_det_dicts[name].items()
                    if fid in test_set}
        m = evaluate_detections(gt_test, det_test)
        if m['all_ious']:
            ax.hist(m['all_ious'], bins=40, alpha=0.55, density=True, label=name)
    ax.set_xlabel('IoU'); ax.set_ylabel('Density')
    ax.set_title('IoU Distribution of Matched Detections', fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'task1_1_iou_dist.png'), dpi=150)
    plt.close()

    print(f"Plots → {PLOTS_DIR}/task1_1_*.png")


def _load_frame_from_video(video_path, frame_id):
    """Load a single frame by sequential scan (safe for msmpeg4v2 AVI)."""
    cap = cv2.VideoCapture(video_path)
    frame_out = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx == frame_id:
            frame_out = frame
            break
    cap.release()
    return frame_out


def _save_qualitative(gt_dict, all_det_dicts, test_ids, best_name):
    if not test_ids:
        print("  [warn] test_ids is empty — skipping qualitative.")
        return

    sample = test_ids[::max(1, len(test_ids) // 6)][:6]

    # A) best model across 6 frames
    panels = []
    for fid in sample:
        frame = _load_frame_from_video(VIDEO_PATH, fid)
        if frame is None:
            continue
        gt_b  = [b[:4] for b in gt_dict.get(fid, [])]
        dets  = all_det_dicts.get(best_name, {}).get(fid, [])
        ann   = draw_gt_and_det(frame, gt_b,
                                [d[:4] for d in dets], [d[4] for d in dets])
        panels.append((ann, f'{best_name} | Frame {fid}'))

    if panels:
        save_qualitative_grid(panels,
            os.path.join(QUAL_DIR, f'task1_1_best_{best_name}.png'), nrows=2, ncols=3)
    else:
        print("  [warn] Could not decode any frames for qualitative grid.")

    # B) all detectors on one mid-sequence frame
    fid_cmp = test_ids[len(test_ids) // 2]
    frame   = _load_frame_from_video(VIDEO_PATH, fid_cmp)
    if frame is not None:
        gt_b = [b[:4] for b in gt_dict.get(fid_cmp, [])]
        panels_cmp = [(frame.copy(), f'Input (Frame {fid_cmp})'),
                      (draw_gt_and_det(frame, gt_b, []), 'Ground Truth')]
        for name, dd in list(all_det_dicts.items())[:5]:
            dets = dd.get(fid_cmp, [])
            ann  = draw_gt_and_det(frame, gt_b,
                                   [d[:4] for d in dets], [d[4] for d in dets])
            panels_cmp.append((ann, name))
        if panels_cmp:
            ncols = min(4, len(panels_cmp))
            nrows = (len(panels_cmp) + ncols - 1) // ncols
            save_qualitative_grid(panels_cmp,
                os.path.join(QUAL_DIR, 'task1_1_detector_comparison.png'),
                nrows=nrows, ncols=ncols)

    print(f"Qualitative -> {QUAL_DIR}/task1_1_*.png")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--precomputed_only', action='store_true')
    p.add_argument('--yolo_sizes', nargs='+', default=[],
                   help='YOLOv8 sizes, e.g. n s m l x')
    p.add_argument('--yolo26', nargs='+', default=[],
                   help='YOLO26 sizes, e.g. n s m l')
    p.add_argument('--imgsz', type=int, default=1280,
                   help='Inference image size for YOLO (default: 1280)')
    p.add_argument('--yolov9',      action='store_true')
    p.add_argument('--faster_rcnn', action='store_true')
    p.add_argument('--mask_rcnn',   action='store_true')
    p.add_argument('--ssd',         action='store_true')
    args = p.parse_args()
    if args.precomputed_only:
        args.yolo_sizes = []; args.yolo26 = []; args.yolov9 = False
        args.faster_rcnn = args.mask_rcnn = args.ssd = False
    main(args)