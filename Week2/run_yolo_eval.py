import argparse
import json
import os
import tempfile
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

import xml.etree.ElementTree as ET

from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import time

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
        # for attr in box.findall('attribute'):
        #     if attr.attrib.get('name') == 'parked' and (attr.text or '').strip().lower() == 'true':
        #         return


        annotations.setdefault(frame, []).append((xtl, ytl, xbr, ybr))

    # Track-based annotations
    for track in root.findall('track'):
        # only cars
        if track.attrib.get('label', '').lower() != 'car':
            continue
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


def run_detection_and_eval(video_path, ann_path, model_path='yolov8n.pt', imgsz=640, out_video='pred_folder/result.mp4', save_output_video=True, verbose=True):
    # parse GT (CVAT XML -> frame -> boxes)
    print('Parsing CVAT XML...')
    gt = parse_cvat_xml(ann_path)
    # gt is a dict mapping frame (int) -> list of boxes (xtl, ytl, xbr, ybr)
    gt_image_ids = set(gt.keys())

    # temp files for GT and predictions
    tmpdir = tempfile.mkdtemp(prefix='yolo_eval_')
    gt_json = os.path.join(tmpdir, 'gt.json')
    preds_json = os.path.join(tmpdir, 'preds.json')

    with open(gt_json, 'w') as f:
        json.dump(gt, f)

    model = YOLO(model_path)
    if verbose:
        print(f'Loaded model: {model_path}')

    # keep car and bikes (bicycle + motorcycle)
    ALLOWED_CLASS_NAMES = {"car", "bicycle", "motorcycle"}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video {video_path}')

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if save_output_video and out_video is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        os.makedirs(os.path.dirname(out_video), exist_ok=True)
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    else:
        writer = None

    detections = []
    frame_idx = 0
    if verbose:
        print('Running detection on video and writing output video...' if writer is not None else 'Running detection on video (no video will be written)...')
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing frames")
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # run model inference
        res = model(frame, imgsz=imgsz, verbose=False)
        r = res[0]

        boxes = []
        if r.boxes is not None:
            # iterate detections and filter by allowed class names
            for (x1, y1, x2, y2), score, class_id in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
                r.boxes.cls.cpu().numpy()
            ):
                cls_id = int(class_id)
                cls_name = model.names.get(cls_id, str(cls_id)).lower()
                if cls_name not in ALLOWED_CLASS_NAMES:
                    continue
                boxes.append((
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                    float(score),
                    cls_id
                ))
        for (x1, y1, x2, y2, conf, cls) in boxes:
            x1i, y1i, x2i, y2i = map(int, (x1, y1, x2, y2))
            cls_name = model.names.get(cls, str(cls))
            label = f'{cls_name} {conf:.2f}'
            cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1i, max(15, y1i - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # add detections (only for frames that have GT annotations)
        if frame_idx in gt_image_ids:
            for (x1, y1, x2, y2, conf, cls) in boxes:
                cls_name = model.names.get(cls, str(cls))
                detections.append({
                    'image_id': frame_idx,
                    'category_id': cls_name,
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'score': float(conf),
                })

        # draw ground-truth boxes (if any) in red for this frame
        if frame_idx in gt_image_ids:
            for (xtl, ytl, xbr, ybr) in gt.get(frame_idx, []):
                try:
                    x1g, y1g, x2g, y2g = map(int, (xtl, ytl, xbr, ybr))
                except Exception:
                    # fallback to casting individually
                    x1g, y1g, x2g, y2g = int(xtl), int(ytl), int(xbr), int(ybr)
                cv2.rectangle(frame, (x1g, y1g), (x2g, y2g), (0, 0, 255), 2)
                cv2.putText(frame, 'GT', (x1g, max(15, y1g - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if writer is not None:
            writer.write(frame)
        frame_idx += 1
        pbar.update(1)

    pbar.close()

    cap.release()
    if writer is not None:
        writer.release()
        if verbose:
            print(f'Output video written to {out_video}')

    # convert detections to the format expected by compute_ap:
    # predictions: list of { 'frame': int, 'bbox': [x1,y1,x2,y2], 'score': float }

    predictions_for_eval = []
    # avoid flooding output with detections
    for d in detections:
        frame = int(d['image_id'])
        x, y, w, h = d['bbox']
        x1, y1, x2, y2 = float(x), float(y), float(x + w), float(y + h)
        predictions_for_eval.append({
            'frame': frame,
            'bbox': [x1, y1, x2, y2],
            'score': float(d['score']),
        })

    # save predictions (optional)
    with open(preds_json, 'w') as f:
        json.dump(predictions_for_eval, f)

    # run evaluation using compute_ap
    if verbose:
        print('Running evaluation using compute_ap...')
    # mAP@0.50
    ap_50 = compute_ap(predictions_for_eval, gt, iou_thresh=0.5)

    # mAP over IoU=0.50:0.95 (step 0.05) approximate COCO-style
    iou_thresholds = list(np.arange(0.5, 0.96, 0.05))
    aps = [compute_ap(predictions_for_eval, gt, iou_thresh=float(t)) for t in iou_thresholds]
    ap_50_95 = float(np.mean(aps)) if len(aps) > 0 else 0.0

    elapsed = time.time() - start_time
    frames_processed = int(frame_idx)

    if verbose:
        print('\nResults:')
        print(f'  mAP (IoU=0.50:0.95) : {ap_50_95:.4f}')
        print(f'  mAP@0.50           : {ap_50:.4f}')
        print(f'  Elapsed time (s)   : {elapsed:.2f}')

    # return metrics so callers (sweep) can use them
    return float(ap_50), float(ap_50_95), float(elapsed), int(frames_processed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='data/vdo.avi')
    parser.add_argument('--ann', default='data/annotation.xml')
    parser.add_argument('--model', default='yolo26n.pt')
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--out', default='pred_folder/result.mp4')
    parser.add_argument('--heatmap', action='store_true', help='Run model/imgsz sweep and save heatmaps (map50 and map50-95)')
    parser.add_argument('--models', default='YOLO26n,YOLO26s,YOLO26m,YOLO26l', help='Comma-separated model names to sweep (e.g. YOLO26n,YOLO26s)')
    parser.add_argument('--imgszs', default='320,640,960,1280', help='Comma-separated image sizes to sweep (e.g. 640,960,1280)')
    args = parser.parse_args()

    def name_to_model_path(name: str) -> str:
        # map common names to local filenames when possible, case-insensitive
        n = name.strip().lower()
        mapping = {
            'yolo26n': 'yolo26n.pt',
            'yolo26s': 'yolo26s.pt',
            'yolo26m': 'yolo26m.pt',
        }
        return mapping.get(n, name)

    if args.heatmap:
        # build lists
        model_names = [m.strip() for m in args.models.split(',') if m.strip()]
        imgsz_list = [int(s.strip()) for s in args.imgszs.split(',') if s.strip()]

        # prepare grid results
        results_map50 = np.zeros((len(imgsz_list), len(model_names)), dtype=float)
        results_map50_95 = np.zeros((len(imgsz_list), len(model_names)), dtype=float)

        for i, imgsz in enumerate(imgsz_list):
            for j, mname in enumerate(model_names):
                mpath = name_to_model_path(mname)
                # avoid writing many videos during sweep; pass empty string and disable saving
                outvid = ''
                print(f'Running sweep: model={mname} (path={mpath}), imgsz={imgsz}')
                try:
                    ap50, ap5095, elapsed, frames = run_detection_and_eval(
                        args.video,
                        args.ann,
                        model_path=mpath,
                        imgsz=imgsz,
                        out_video=outvid,
                        save_output_video=False,
                        verbose=True,
                    )
                except Exception as e:
                    print(f'Error running {mname}@{imgsz}: {e}')
                    ap50, ap5095, elapsed, frames = 0.0, 0.0, 0.0, 0
                results_map50[i, j] = ap50
                results_map50_95[i, j] = ap5095
                # store FPS (frames / second)
                if 'results_fps' not in locals():
                    results_fps = np.zeros((len(imgsz_list), len(model_names)), dtype=float)
                fps = (frames / elapsed) if elapsed > 0 else 0.0
                results_fps[i, j] = float(fps)

        # Plot heatmaps
        try:
            # mAP@0.50 heatmap with increased contrast
            fig, ax = plt.subplots()
            vmin = np.min(results_map50)
            vmax = np.max(results_map50)
            delta = max(1e-6, (vmax - vmin))
            vmin_plot = max(0.0, vmin - 0.1 * delta)
            vmax_plot = min(1.0, vmax + 0.1 * delta)
            im = ax.imshow(results_map50, vmin=vmin_plot, vmax=vmax_plot, cmap='plasma')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names)
            ax.set_yticks(range(len(imgsz_list)))
            ax.set_yticklabels([str(s) for s in imgsz_list])
            ax.set_xlabel('model')
            ax.set_ylabel('imgsz')
            ax.set_title('mAP@0.50')
            mid = (vmin_plot + vmax_plot) / 2.0
            for ii in range(results_map50.shape[0]):
                for jj in range(results_map50.shape[1]):
                    val = results_map50[ii, jj]
                    txt_color = 'white' if val < mid else 'black'
                    ax.text(jj, ii, f'{val:.3f}', ha='center', va='center', color=txt_color)
            fig.colorbar(im, ax=ax)
            os.makedirs('pred_folder', exist_ok=True)
            fig.savefig('pred_folder/heatmap_map50.png', bbox_inches='tight', dpi=150)
            plt.close(fig)

            # mAP@0.50:0.95 heatmap with increased contrast
            fig, ax = plt.subplots()
            vmin = np.min(results_map50_95)
            vmax = np.max(results_map50_95)
            delta = max(1e-6, (vmax - vmin))
            vmin_plot = max(0.0, vmin - 0.1 * delta)
            vmax_plot = min(1.0, vmax + 0.1 * delta)
            im = ax.imshow(results_map50_95, vmin=vmin_plot, vmax=vmax_plot, cmap='plasma')
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names)
            ax.set_yticks(range(len(imgsz_list)))
            ax.set_yticklabels([str(s) for s in imgsz_list])
            ax.set_xlabel('model')
            ax.set_ylabel('imgsz')
            ax.set_title('mAP@0.50:0.95')
            mid = (vmin_plot + vmax_plot) / 2.0
            for ii in range(results_map50_95.shape[0]):
                for jj in range(results_map50_95.shape[1]):
                    val = results_map50_95[ii, jj]
                    txt_color = 'white' if val < mid else 'black'
                    ax.text(jj, ii, f'{val:.3f}', ha='center', va='center', color=txt_color)
            fig.colorbar(im, ax=ax)
            fig.savefig('pred_folder/heatmap_map50_95.png', bbox_inches='tight', dpi=150)
            plt.close(fig)


            # Plot FPS heatmap (frames per second) with better contrast
            try:
                fig, ax = plt.subplots()
                vmin = np.min(results_fps)
                vmax = np.max(results_fps)
                delta = max(1e-6, (vmax - vmin))
                vmin_plot = max(0.0, vmin - 0.1 * delta)
                vmax_plot = vmax + 0.1 * delta
                im = ax.imshow(results_fps, vmin=vmin_plot, vmax=vmax_plot, cmap='inferno')
                ax.set_xticks(range(len(model_names)))
                ax.set_xticklabels(model_names)
                ax.set_yticks(range(len(imgsz_list)))
                ax.set_yticklabels([str(s) for s in imgsz_list])
                ax.set_xlabel('model')
                ax.set_ylabel('imgsz')
                ax.set_title('FPS (frames/s)')
                mid = (vmin_plot + vmax_plot) / 2.0
                for ii in range(results_fps.shape[0]):
                    for jj in range(results_fps.shape[1]):
                        val = results_fps[ii, jj]
                        txt_color = 'white' if val < mid else 'black'
                        ax.text(jj, ii, f'{val:.2f}', ha='center', va='center', color=txt_color)
                fig.colorbar(im, ax=ax)
                fig.savefig('pred_folder/heatmap_fps.png', bbox_inches='tight', dpi=150)
                plt.close(fig)
            except Exception:
                pass

            print('Heatmaps saved to pred_folder/heatmap_map50.png, pred_folder/heatmap_map50_95.png and pred_folder/heatmap_fps.png')
        except Exception as e:
            print(f'Could not plot heatmaps: {e}')

        return

    # default single run
    try:
        ap50, ap5095, elapsed, frames = run_detection_and_eval(args.video, args.ann, model_path=name_to_model_path(args.model), imgsz=args.imgsz, out_video=args.out)
    except Exception as e:
        print(f'Run failed: {e}')
        return
    if args.heatmap is False:
        fps = (frames / elapsed) if elapsed > 0 else 0.0
        print(f'Run finished: mAP@0.50={ap50:.4f}, mAP@0.50:0.95={ap5095:.4f}, time(s)={elapsed:.2f}, FPS={fps:.2f}')


if __name__ == '__main__':
    main()
