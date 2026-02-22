import argparse
import glob
import json
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import numpy as np

import matplotlib.pyplot as plt

import cv2

import imageio



def parse_cvat_xml(xml_path: str) -> List[dict]:
    """Parse a CVAT 1.1 annotations.xml file and return a list of GT entries.

    Supports both <track> (video) and <image> (image sequence) formats.

    Returns a list of dicts in the same format as predictions JSON entries:
      [{'frame': int, 'bbox': (x1,y1,x2,y2), 'score': 1.0}, ...]
    (GT entries have score=1.0 by convention.)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    gt_list: List[dict] = []

    # handle <track> entries (video tasks)
    for track in root.findall('track'):
        for box in track.findall('box'):
            frame = int(box.attrib.get('frame', '0'))
            xtl = float(box.attrib.get('xtl', '0'))
            ytl = float(box.attrib.get('ytl', '0'))
            xbr = float(box.attrib.get('xbr', '0'))
            ybr = float(box.attrib.get('ybr', '0'))
            # check parked attribute: prefer to skip boxes explicitly marked as parked="true"
            parked_val = None
            for attr in box.findall('attribute'):
                if attr.attrib.get('name') == 'parked':
                    parked_val = (attr.text or '').strip().lower()
                    break
            if parked_val == 'true':
                continue
            gt_list.append({'frame': frame, 'bbox': (xtl, ytl, xbr, ybr), 'score': 1.0})

    # handle <image> entries (CVAT for images)
    for image in root.findall('image'):
        # image id attribute may be present; fallback to parsing name if necessary
        img_id = image.attrib.get('id')
        if img_id is not None:
            frame = int(img_id)
        else:
            # try to parse from name like frame_00001.jpg
            name = image.attrib.get('name', '')
            # extract digits
            nums = ''.join([c for c in name if c.isdigit()])
            frame = int(nums) if nums else 0
        for box in image.findall('box'):
            xtl = float(box.attrib.get('xtl', '0'))
            ytl = float(box.attrib.get('ytl', '0'))
            xbr = float(box.attrib.get('xbr', '0'))
            ybr = float(box.attrib.get('ybr', '0'))
            # check parked attribute and skip if explicitly parked
            parked_val = None
            for attr in box.findall('attribute'):
                if attr.attrib.get('name') == 'parked':
                    parked_val = (attr.text or '').strip().lower()
                    break
            if parked_val == 'true':
                continue
            gt_list.append({'frame': frame, 'bbox': (xtl, ytl, xbr, ybr), 'score': 1.0})

    return gt_list


def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two boxes in (x1,y1,x2,y2) format."""
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
    """Compute Average Precision (AP) at given IoU threshold.

    predictions: list of dicts with keys: 'frame' (int), 'bbox' (x1,y1,x2,y2), 'score' (float)
    ground_truths: dict mapping frame->list of GT boxes

    Returns AP (float between 0 and 1).
    """
    # If predictions start at a later frame, only evaluate GTs from that frame onward.
    # This avoids penalizing for ground-truths before predictions begin.
    if len(predictions) > 0:
        min_pred_frame = min(int(p['frame']) for p in predictions)
        gt_filtered = {f: boxes for f, boxes in ground_truths.items() if f >= min_pred_frame}
    else:
        gt_filtered = ground_truths

    # Flatten GTs and count per frame (using filtered GTs)
    gt_counter_per_frame = {f: len(boxes) for f, boxes in gt_filtered.items()}
    total_gts = sum(gt_counter_per_frame.values())

    # sort predictions by score descending
    preds = sorted(predictions, key=lambda x: x.get('score', 0.0), reverse=True)

    tp = np.zeros(len(preds), dtype=np.int32)
    fp = np.zeros(len(preds), dtype=np.int32)

    # keep track of matched GT boxes per frame
    # keep track of matched GT boxes per frame (for filtered GTs)
    matched = {f: np.zeros(len(boxes), dtype=bool) for f, boxes in gt_filtered.items()}

    for i, p in enumerate(preds):
        frame = int(p['frame'])
        pb = p['bbox']
        best_iou = 0.0
        best_j = -1
        # only consider GTs in filtered set
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

    # accumulate
    tp_cum = np.cumsum(tp).astype(float)
    fp_cum = np.cumsum(fp).astype(float)

    if len(tp_cum) == 0:
        return 0.0

    recalls = tp_cum / (total_gts + 1e-8)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-8)

    # ensure precision is non-increasing (precision envelope)
    mprec = np.concatenate(([0.0], precisions, [0.0]))
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    for i in range(len(mprec) - 1, 0, -1):
        mprec[i - 1] = max(mprec[i - 1], mprec[i])

    # integrate area under PR curve
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = 0.0
    for i in idx:
        ap += (mrec[i + 1] - mrec[i]) * mprec[i + 1]
    return float(ap)


def load_predictions_json(json_path: str) -> List[dict]:
    """Load predictions from a JSON file with structure: {frame_idx: [[x1,y1,x2,y2,score], ...], ...}

    Returns a list of dicts: {'frame': int, 'bbox': (x1,y1,x2,y2), 'score': float}
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    preds: List[dict] = []
    # data might be dict mapping string frame->list or list of entries
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
        # list of entries with explicit dicts
        for entry in data:
            frame = int(entry.get('frame'))
            bbox = entry.get('bbox')
            score = float(entry.get('score', 1.0))
            preds.append({'frame': frame, 'bbox': tuple(map(float, bbox)), 'score': score})
    return preds


def extract_alpha_from_filename(filename: str) -> float:
    """Extract a float alpha value from filename like T1_preds_7.5.json.

    If not found using the T1_preds_ pattern, will try to find the first float in the basename.
    Returns NaN if nothing found.
    """
    base = os.path.basename(filename)
    m = re.search(r'T1_preds_([0-9]+(?:\.[0-9]+)?)', base)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    # fallback: first float-like substring
    m2 = re.search(r'([0-9]+(?:\.[0-9]+)?)', base)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            pass
    return float('nan')


def get_frame_from_video(video_path: str, frame_idx: int):
    """Return BGR image for given 0-based frame index from video, or None if not available."""
    if cv2 is None:
        return None
    if not os.path.exists(video_path):
        return None
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def draw_boxes_on_image(img, gt_boxes, pred_boxes, score_thresh=0.0):
    """Draw GT and predicted boxes on image.

    gt_boxes: list of (x1,y1,x2,y2)
    pred_boxes: list of (x1,y1,x2,y2,score)
    Returns annotated BGR image.
    """
    if cv2 is None:
        return img
    out = img.copy()
    # draw GT in green
    for b in gt_boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # draw preds in red with score
    for p in pred_boxes:
        if len(p) == 5:
            x1, y1, x2, y2, s = p
        else:
            x1, y1, x2, y2 = p
            s = 1.0
        if s < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, f"{s:.2f}", (x1, max(10, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    return out


def visualize_predictions(preds_list, ground_truths, alpha, out_dir, repo_root='.', video_path=None, max_images=10, score_thresh=0.0):
    """Create a short GIF comparing GT and predictions over a small set of frames.

    Draws GT boxes in green and predicted boxes in red (with score). Does not indicate matches.
    Produces a single GIF per alpha at <out_dir>/alpha_<alpha>.gif
    """
    if cv2 is None:
        print("opencv (cv2) not available; cannot create visualizations. Install opencv-python and retry.")
        return
    if imageio is None:
        print("imageio not available; install imageio to create GIFs (pip install imageio)")
        return

    os.makedirs(out_dir, exist_ok=True)

    # map frame->preds
    preds_per_frame = {}
    for p in preds_list:
        f = int(p['frame'])
        preds_per_frame.setdefault(f, []).append((*p['bbox'], p.get('score', 1.0)))

    frames = sorted(set(list(ground_truths.keys()) + list(preds_per_frame.keys())))
    # If preds start at a later frame, only visualize frames from that smallest predicted frame onward
    if len(preds_per_frame) > 0:
        min_pred_frame = min(preds_per_frame.keys())
        frames = [f for f in frames if f >= min_pred_frame]
    if not frames:
        print(f"No frames to visualize for alpha={alpha}")
        return

    # pick video: prefer provided video_path, otherwise try repo T1_fg_<alpha>.mp4, then data/vdo.avi
    chosen_video = None
    if video_path and os.path.exists(video_path):
        chosen_video = video_path
    else:
        try:
            if alpha == alpha:
                fg_name = os.path.join(repo_root, f'T1_fg_{alpha}.mp4')
            else:
                fg_name = None
        except Exception:
            fg_name = None
        if fg_name and os.path.exists(fg_name):
            chosen_video = fg_name
        else:
            alt = os.path.join(repo_root, 'data', 'vdo.avi')
            if os.path.exists(alt):
                chosen_video = alt

    if not chosen_video:
        print(f"No video found to extract frames for visualization (alpha={alpha}). Skipping visuals.")
        return

    images = []
    saved = 0
    for f in frames:
        if saved >= max_images:
            break
        img = get_frame_from_video(chosen_video, f)
        if img is None:
            continue
        gts = ground_truths.get(f, [])
        preds = preds_per_frame.get(f, [])
        annotated = draw_boxes_on_image(img, gts, preds, score_thresh=score_thresh)
        # convert BGR to RGB for GIF
        rgb = annotated[:, :, ::-1]
        images.append(rgb)
        saved += 1

    if not images:
        print(f"No frames were captured for alpha={alpha}; no GIF created.")
        return

    out_gif = os.path.join(out_dir, f'alpha_{alpha}.gif')
    try:
        imageio.mimsave(out_gif, images, duration=0.2)
        print(f"Saved GIF {out_gif} ({len(images)} frames)")
    except Exception as e:
        print(f"Failed to save GIF {out_gif}: {e}")


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate AP@IoU=0.5 between predicted boxes and CVAT 1.1 annotations')
    p.add_argument('--cvat', default='data/annotations.xml', help='Path to CVAT 1.1 annotations.xml')
    p.add_argument('--pred', help='Path to a single predictions JSON file (optional)')
    p.add_argument('--pred_folder', help='Path to a folder containing predictions files named like T1_preds_<alpha>.json (optional)')
    p.add_argument('--out', default='alpha_map_plot.png', help='Output path for the saved plot (PNG)')

    p.add_argument('--visualize', action='store_true', help='If set, create per-frame visualizations with GT and predictions overlaid')
    p.add_argument('--vis_out', default='vis', help='Directory to save visualization images (will create per-alpha subfolders)')
    p.add_argument('--video', help='Path to video to read frames from; if not set, script will try repo T1_fg_<alpha>.mp4 or data/vdo.avi')
    p.add_argument('--max_images', type=int, default=10, help='Maximum number of frames to visualize per alpha')
    p.add_argument('--score_thresh', type=float, default=0.0, help='Minimum prediction score to visualize')
    p.add_argument('--iou', type=float, default=0.5, help='IoU threshold (default 0.5)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # allow user to override default cvat path if the file is actually named differently
    if not os.path.exists(args.cvat):
        # try common alternate name
        alt = os.path.join(os.path.dirname(args.cvat), 'annotation.xml')
        if os.path.exists(alt):
            args.cvat = alt

    gt_list = parse_cvat_xml(args.cvat)
    # convert list-of-dicts GT into dict mapping frame->list of boxes (x1,y1,x2,y2)
    gt: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for e in gt_list:
        f = int(e['frame'])
        bbox = e['bbox']
        gt.setdefault(f, []).append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))

    results = []  # list of (alpha, ap)

    # if single pred provided, evaluate it too (alpha may be NaN)
    if args.pred:
        if not os.path.exists(args.pred):
            raise FileNotFoundError(f"Prediction file not found: {args.pred}")
        preds = load_predictions_json(args.pred)
        ap50 = compute_ap(preds, gt, iou_thresh=args.iou)
        alpha = extract_alpha_from_filename(args.pred)
        results.append((alpha, ap50, args.pred))
        print(f"{os.path.basename(args.pred)} -> AP@{args.iou:.2f}: {ap50:.6f} (alpha={alpha})")
        if getattr(args, 'visualize', False):
            vis_dir = args.vis_out
            visualize_predictions(preds, gt, alpha, vis_dir, repo_root='.', video_path=args.video, max_images=args.max_images, score_thresh=args.score_thresh)

    # if folder provided, find files matching T1_preds_*.json
    if args.pred_folder:
        if not os.path.isdir(args.pred_folder):
            raise NotADirectoryError(f"pred_folder is not a directory: {args.pred_folder}")
        pattern = os.path.join(args.pred_folder, 'T1_preds_*.json')
        files = sorted(glob.glob(pattern))
        if not files:
            # also try any preds in folder
            files = sorted(glob.glob(os.path.join(args.pred_folder, '*.json')))
        for f in files:
            try:
                preds = load_predictions_json(f)
            except Exception as e:
                print(f"Skipping {f}: failed to load predictions ({e})")
                continue
            ap50 = compute_ap(preds, gt, iou_thresh=args.iou)
            alpha = extract_alpha_from_filename(f)
            results.append((alpha, ap50, f))
            print(f"{os.path.basename(f)} -> AP@{args.iou:.2f}: {ap50:.6f} (alpha={alpha})")
            if getattr(args, 'visualize', False):
                vis_dir = args.vis_out
                visualize_predictions(preds, gt, alpha, vis_dir, repo_root='.', video_path=args.video, max_images=args.max_images, score_thresh=args.score_thresh)

    if not results:
        print("No prediction files were evaluated. Provide --pred or --pred_folder with valid files.")
        exit(1)

    # sort by alpha (put NaNs at end)
    results_sorted = sorted(results, key=lambda x: (float('inf') if (x[0] != x[0]) else x[0]))
    alphas = [r[0] for r in results_sorted]
    maps = [r[1] for r in results_sorted]

    # prepare plot (replace NaN alphas with index positions for plotting but label them)
    plot_x = []
    xticks = []
    xtick_labels = []
    for i, a in enumerate(alphas):
        if a == a:  # not NaN
            plot_x.append(a)
            xticks.append(a)
            xtick_labels.append(str(a))
        else:
            # place NaN entries at end with index offset
            pos = max(plot_x) + 1 if plot_x else i
            plot_x.append(pos)
            xticks.append(pos)
            xtick_labels.append(os.path.basename(results_sorted[i][2]))

    plt.figure(figsize=(8, 5))
    plt.plot(plot_x, maps, marker='o', linestyle='-')
    plt.xlabel('alpha')
    plt.ylabel(f'mAP@{args.iou:.2f}')
    plt.title('alpha vs mAP@IoU={:.2f}'.format(args.iou))
    plt.grid(True)
    plt.xticks(xticks, xtick_labels, rotation=45)
    plt.tight_layout()
    out_path = args.out
    plt.savefig(out_path)
    print(f"Saved alpha vs mAP plot to {out_path}")
