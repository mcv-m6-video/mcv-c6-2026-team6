from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch


def iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    return interArea / denom if denom > 0 else 0.0


class SimpleTracker:
    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.tracks = {}  # id -> {box, last_frame, age}
        self.next_id = 1

    def update(self, detections: List[Tuple[float, float, float, float, float]], frame_id: int):
        # detections: list of (x1,y1,x2,y2,score)
        assigned = set()
        matches = []
        for tid, t in list(self.tracks.items()):
            best_iou = 0.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                b = det[:4]
                cur_iou = iou(t["box"], b)
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= self.iou_thresh:
                # update
                self.tracks[tid]["box"] = detections[best_idx][:4]
                self.tracks[tid]["last_frame"] = frame_id
                self.tracks[tid]["age"] = 0
                assigned.add(best_idx)
                matches.append((tid, detections[best_idx]))
            else:
                self.tracks[tid]["age"] += 1

        # create new tracks for unassigned detections
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = {"box": det[:4], "last_frame": frame_id, "age": 0}
            matches.append((tid, det))

        # remove old tracks
        to_del = [tid for tid, t in self.tracks.items() if t["age"] > self.max_age]
        for tid in to_del:
            del self.tracks[tid]

        return matches


def run_on_camera(video_path: Path, model: torch.nn.Module, device: torch.device, out_mtsc: Path, conf_thresh: float, iou_thresh: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    tracker = SimpleTracker(iou_thresh=iou_thresh, max_age=30)
    frame_id = 0
    out_rows = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        with torch.no_grad():
            preds = model(img_t)

        # Expect preds as tensor (N,6): x1,y1,x2,y2,score,class OR list of dicts like Ultralytics
        dets = []
        if isinstance(preds, (list, tuple)) and len(preds) > 0:
            p = preds[0]
            if isinstance(p, dict) and "boxes" in p:
                boxes = p["boxes"].cpu().numpy()
                scores = p.get("scores", torch.ones(len(boxes))).cpu().numpy()
                for b, s in zip(boxes, scores):
                    if s >= conf_thresh:
                        dets.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(s)))
            elif torch.is_tensor(p):
                arr = p.cpu().numpy()
                for row in arr:
                    if row[4] >= conf_thresh:
                        dets.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))
        elif torch.is_tensor(preds):
            arr = preds.cpu().numpy()
            for row in arr:
                if row[4] >= conf_thresh:
                    dets.append((float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])))
        else:
            # fallback: try calling .predict on model
            if hasattr(model, "predict"):
                out = model.predict(img)
                for row in out:
                    if row[4] >= conf_thresh:
                        dets.append(tuple(map(float, row[:5])))

        matches = tracker.update(dets, frame_id)
        for tid, det in matches:
            x1, y1, x2, y2, s = det
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            # MTSC/MOT format expects: frame, id, left, top, width, height, 1, -1, -1, -1
            out_rows.append((int(frame_id), int(tid), int(round(x1)), int(round(y1)), int(round(w)), int(round(h)), 1, -1, -1, -1))

    cap.release()

    out_mtsc.parent.mkdir(parents=True, exist_ok=True)
    with out_mtsc.open("w", encoding="utf-8") as f:
        for row in out_rows:
            # write as integers separated by spaces
            f.write(" ".join(str(int(r)) for r in row) + "\n")


def load_torch_model(path: Path, device: torch.device) -> torch.nn.Module:
    # Try torch.jit first
    try:
        m = torch.jit.load(str(path), map_location=device)
        m.to(device)
        m.eval()
        return m
    except Exception:
        # Try raw load
        ck = torch.load(str(path), map_location=device)
        if isinstance(ck, dict) and "model" in ck:
            # user saved checkpoint with state_dict
            model = ck["model"]
            if isinstance(model, torch.nn.Module):
                model.to(device)
                model.eval()
                return model
        # As last resort, try load as state_dict into a generic wrapper that expects scripted model
        raise RuntimeError("Unable to load detector model. Provide a TorchScript model or a callable module saved with torch.jit or returning detections tensor.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--model", required=True, help="Path to detector .pt (TorchScript or module)")
    parser.add_argument("--mtsc-name", default="mtsc_yolo26.txt", help="Output mtsc filename placed in each cam/mtsc/")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.3)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    seq = args.sequence
    cams_dir = data_root / "train" / seq
    if not cams_dir.exists():
        cams_dir = data_root / "validation" / seq
    if not cams_dir.exists():
        cams_dir = data_root / "test" / seq
    if not cams_dir.exists():
        raise FileNotFoundError(f"Sequence folder not found for {seq} under {data_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_torch_model(Path(args.model), device)

    for d in sorted(cams_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("c"):
            continue
        video = d / "vdo.avi"
        if not video.exists():
            continue
        out = d / "mtsc" / args.mtsc_name
        print(f"Running detector on {d.name} -> {out}")
        run_on_camera(video, model, device, out, args.conf_thresh, args.iou_thresh)


if __name__ == "__main__":
    main()
