import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt



def parse_cvat_xml(xml_path: str) -> Dict[int, List[Tuple[float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations: Dict[int, List[Tuple[float, float, float, float]]] = {}

    def add_box(frame: int, box):
        xtl = float(box.attrib.get('xtl', 0))
        ytl = float(box.attrib.get('ytl', 0))
        xbr = float(box.attrib.get('xbr', 0))
        ybr = float(box.attrib.get('ybr', 0))

        # skip parked
        for attr in box.findall('attribute'):
            if attr.attrib.get('name') == 'parked' and (attr.text or '').strip().lower() == 'true':
                return

        annotations.setdefault(frame, []).append((xtl, ytl, xbr, ybr))

    # Track-based annotations
    for track in root.findall('track'):
        for box in track.findall('box'):
            frame = int(box.attrib.get('frame', 0))
            add_box(frame, box)

    # Image-based annotations
    for image in root.findall('image'):
        if image.attrib.get('id') is not None:
            frame = int(image.attrib['id'])
        else:
            name = image.attrib.get('name', '')
            digits = ''.join(c for c in name if c.isdigit())
            frame = int(digits) if digits else 0

        for box in image.findall('box'):
            add_box(frame, box)

    return annotations


def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0.0, (boxA[2] - boxA[0])) * max(0.0, (boxA[3] - boxA[1]))
    boxBArea = max(0.0, (boxB[2] - boxB[0])) * max(0.0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


def compute_ap(predictions: List[dict], ground_truths: Dict[int, List[Tuple[float, float, float, float]]], iou_thresh: float = 0.5) -> float:
    if len(predictions) > 0:
        min_pred_frame = min(int(p['frame']) for p in predictions)
        gt_filtered = {f: boxes for f, boxes in ground_truths.items() if f >= min_pred_frame}
    else:
        gt_filtered = ground_truths
    total_gts = sum(len(boxes) for boxes in gt_filtered.values())
    preds = sorted(predictions, key=lambda x: x.get('score', 0.0), reverse=True)
    tp = np.zeros(len(preds), dtype=np.int32)
    fp = np.zeros(len(preds), dtype=np.int32)
    matched = {f: np.zeros(len(boxes), dtype=bool) for f, boxes in gt_filtered.items()}
    for i, p in enumerate(preds):
        frame = int(p['frame'])
        pb = p['bbox']
        best_iou = 0.0
        best_j = -1
        gts = gt_filtered.get(frame, [])
        for j, gb in enumerate(gts):
            if matched.get(frame) is not None and matched[frame][j]:
                continue
            current_iou = iou(pb, gb)
            if current_iou > best_iou:
                best_iou = current_iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0:
            tp[i] = 1
            matched[frame][best_j] = True
        else:
            fp[i] = 1
    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)
    if len(tp_cum) == 0:
        return 0.0
    recalls = tp_cum / (total_gts + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)
    mprec = np.concatenate(([0.0], precisions, [0.0]))
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    for k in range(len(mprec) - 1, 0, -1):
        mprec[k - 1] = max(mprec[k - 1], mprec[k])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for k in idx:
        ap += (mrec[k + 1] - mrec[k]) * mprec[k + 1]
    return float(ap)


def load_predictions_json(json_path: str) -> List[dict]:
    with open(json_path, 'r') as f:
        data = json.load(f)
    preds: List[dict] = []
    if isinstance(data, dict):
        for k, items in data.items():
            frame = int(k)
            for it in items:
                if len(it) == 5:
                    x1, y1, x2, y2, score = it
                elif len(it) == 4:
                    x1, y1, x2, y2 = it
                    score = 1.0
                else:
                    continue
                preds.append({'frame': frame, 'bbox': (float(x1), float(y1), float(x2), float(y2)), 'score': float(score)})
    elif isinstance(data, list):
        for entry in data:
            frame = int(entry.get('frame'))
            bbox = entry.get('bbox')
            score = float(entry.get('score', 1.0))
            preds.append({'frame': frame, 'bbox': tuple(map(float, bbox)), 'score': score})
    return preds


def extract_alpha_from_filename(filename: str) -> float:
    base = os.path.basename(filename)
    m = re.search(r'T1_preds_([0-9]+(?:\.[0-9]+)?)', base)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)', base)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    return float('nan')


def plot_alpha_map(results_list: List[Tuple[float, float, str]],
                   out_path: str,
                   iou_thresh: float = 0.5,
                   figsize=(8, 5)) -> None:

    if not results_list:
        print("No data to plot")
        return

    # sort by alpha
    results_list = sorted(results_list, key=lambda x: x[0])

    alphas = [r[0] for r in results_list]
    maps = [r[1] for r in results_list]

    plt.figure(figsize=figsize)
    plt.plot(alphas, maps, marker='o')
    plt.xlabel("alpha")
    plt.ylabel(f"mAP@{iou_thresh:.2f}")
    plt.title(f"alpha vs mAP@IoU={iou_thresh:.2f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Saved plot to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate AP between predicted boxes and CVAT annotations')
    p.add_argument('--cvat', default='data/annotation.xml', help='Path to CVAT annotations.xml')
    p.add_argument('--pred', help='Single predictions JSON file (optional)')
    p.add_argument('--pred_folder', help='Folder with prediction JSONs (optional)')
    p.add_argument('--out', default='alpha_map_plot.png', help='Output PNG for alpha vs mAP')
    p.add_argument('--iou', type=float, default=0.5, help='IoU threshold (default 0.5)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()


    gt = parse_cvat_xml(args.cvat)

    results = []

    def evaluate(pred_file):
        preds = load_predictions_json(pred_file)
        ap = compute_ap(preds, gt, iou_thresh=args.iou)
        alpha = extract_alpha_from_filename(pred_file)
        results.append((alpha, ap, pred_file))
        print(f"{os.path.basename(pred_file)} -> "
              f"AP@{args.iou:.2f}: {ap:.6f} (alpha={alpha})")

    # Single prediction file
    if args.pred:
        if not os.path.exists(args.pred):
            raise FileNotFoundError(f"Prediction file not found: {args.pred}")
        evaluate(args.pred)

    # Prediction folder
    if args.pred_folder:
        if not os.path.isdir(args.pred_folder):
            raise NotADirectoryError(f"{args.pred_folder} is not a directory")

        files = sorted(
            glob.glob(os.path.join(args.pred_folder, 'T1_preds_*.json')) or
            glob.glob(os.path.join(args.pred_folder, '*.json'))
        )

        for f in files:
            try:
                evaluate(f)
            except Exception as e:
                print(f"Skipping {f}: {e}")

    if not results:
        print("No prediction files evaluated. Use --pred or --pred_folder.")
        exit(1)

    # Sort and plot
    results.sort(key=lambda x: float('inf') if x[0] != x[0] else x[0])
    plot_alpha_map(results, args.out, iou_thresh=args.iou)
