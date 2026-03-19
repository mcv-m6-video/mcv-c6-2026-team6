from __future__ import annotations

import argparse
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

from mtmc.data import cam_numeric_id, camera_dir_from_sequence, load_timestamps, load_tracklets_for_sequence
from mtmc.models import DGVehicleNet
from mtmc.utils import load_yaml


@dataclass
class Tracklet:
    camera: str
    local_id: int
    start_abs: float
    end_abs: float
    feat: np.ndarray
    length: int
    mean_area: float


class DSU:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def time_gap(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < b0:
        return b0 - a1
    if b1 < a0:
        return a0 - b1
    return 0.0


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x) + 1e-12
    return x / n


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2n(a), l2n(b)))


def load_model(checkpoint: Path, device: torch.device) -> tuple[DGVehicleNet, tuple[int, int]]:
    ckpt = torch.load(checkpoint, map_location=device)
    model = DGVehicleNet(
        num_classes=int(ckpt["num_classes"]),
        id_dim=int(ckpt["id_dim"]),
        style_dim=int(ckpt["style_dim"]),
        backbone_name=str(ckpt.get("backbone", "resnet18")),
        pretrained_backbone=bool(ckpt.get("pretrained_backbone", False)),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    image_size = tuple(ckpt.get("image_size", [256, 256]))
    return model, (int(image_size[0]), int(image_size[1]))


def extract_tracklet_feature(
    rows: pd.DataFrame,
    video_path: Path,
    model: DGVehicleNet,
    tfm,
    device: torch.device,
    max_samples: int = 8,
) -> np.ndarray:
    rows = rows.sort_values("frame")
    if len(rows) > max_samples:
        idx = np.linspace(0, len(rows) - 1, max_samples).astype(int)
        rows = rows.iloc[idx]

    cap = cv2.VideoCapture(str(video_path))
    feats = []
    with torch.no_grad():
        for _, r in rows.iterrows():
            frame_id = int(r.frame)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
            ok, frame = cap.read()
            if not ok:
                continue

            x1 = max(0, int(round(r.left)))
            y1 = max(0, int(round(r.top)))
            x2 = min(frame.shape[1] - 1, int(round(r.left + r.width)))
            y2 = min(frame.shape[0] - 1, int(round(r.top + r.height)))
            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            x = tfm(crop).unsqueeze(0).to(device)
            out = model(x)
            feats.append(out["id_feat"].cpu().numpy()[0])

    cap.release()

    if not feats:
        return np.zeros((model.classifier.in_features,), dtype=np.float32)

    return l2n(np.mean(np.stack(feats, axis=0), axis=0)).astype(np.float32)


def associate(tracklets: List[Tracklet], min_sim: float, max_time_gap_sec: float) -> Dict[Tuple[str, int], int]:
    n = len(tracklets)
    dsu = DSU(n)

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            a = tracklets[i]
            b = tracklets[j]
            if a.camera == b.camera:
                continue
            tg = time_gap(a.start_abs, a.end_abs, b.start_abs, b.end_abs)
            if tg > max_time_gap_sec:
                continue
            sim = cosine_sim(a.feat, b.feat)
            if sim >= min_sim:
                pairs.append((sim, i, j))

    pairs.sort(reverse=True, key=lambda x: x[0])

    cluster_cams: Dict[int, set] = {i: {tracklets[i].camera} for i in range(n)}
    for _sim, i, j in pairs:
        ri, rj = dsu.find(i), dsu.find(j)
        if ri == rj:
            continue
        # Keep one tracklet per camera inside each cluster.
        if cluster_cams[ri].intersection(cluster_cams[rj]):
            continue
        dsu.union(ri, rj)
        r = dsu.find(ri)
        o = rj if r == ri else ri
        cluster_cams[r] = cluster_cams[ri].union(cluster_cams[rj])
        if o in cluster_cams:
            del cluster_cams[o]

    root_to_gid: Dict[int, int] = {}
    next_gid = 1
    out: Dict[Tuple[str, int], int] = {}
    for i, t in enumerate(tracklets):
        r = dsu.find(i)
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        out[(t.camera, t.local_id)] = root_to_gid[r]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MTMC association and export track1 format")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--mtsc-file", default=None, help="Override MTSC input filename")
    parser.add_argument("--detector-model", default=None, help="Optional detector .pt to run per-camera when mtsc missing")
    parser.add_argument("--detector-mtsc-name", default=None, help="Name for generated mtsc files (overrides mtsc-file when running detector)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    mtmc_cfg = cfg["mtmc"]
    fps_default = float(mtmc_cfg["fps_default"])
    fps_overrides = mtmc_cfg.get("fps_overrides", {})
    min_track_length = int(mtmc_cfg.get("min_track_length", 1))
    min_mean_area = float(mtmc_cfg.get("min_mean_area", 0.0))
    keep_only_multicam = bool(mtmc_cfg.get("keep_only_multicam", False))
    min_global_cameras = int(mtmc_cfg.get("min_global_cameras", 2))
    min_global_rows = int(mtmc_cfg.get("min_global_rows", 1))
    max_global_ids = int(mtmc_cfg.get("max_global_ids", 0))
    cam_id_offset = int(mtmc_cfg.get("camera_id_offset", 0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_image_size = load_model(Path(args.checkpoint), device)

    tfm = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(ckpt_image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    data_root = Path(args.data_root)
    cams = camera_dir_from_sequence(data_root, args.sequence)
    timestamps = load_timestamps(data_root, args.sequence)
    mtsc_file = args.mtsc_file if args.mtsc_file else mtmc_cfg["mtsc_file"]
    # If some per-camera mtsc files are missing and a detector model is provided,
    # run the detector script to generate them.
    missing = False
    for cam_name, cam_dir in cams.items():
        path = cam_dir / "mtsc" / mtsc_file
        if not path.exists():
            missing = True
            break

    if missing and args.detector_model:
        det_mtsc_name = args.detector_mtsc_name if args.detector_mtsc_name else mtsc_file
        script = ROOT / "tools" / "detect_and_track_yolo.py"
        cmd = [sys.executable, str(script), "--data-root", str(data_root), "--sequence", args.sequence, "--model", args.detector_model, "--mtsc-name", det_mtsc_name]
        print("Running detector to generate missing MTSC files:", " ".join(cmd))
        subprocess.check_call(cmd)
        mtsc_file = det_mtsc_name

    df = load_tracklets_for_sequence(data_root, args.sequence, mtsc_file)

    tracklets: List[Tracklet] = []
    grouped = df.groupby(["camera", "obj_id"])

    for (camera, local_id), rows in tqdm(grouped, desc="Tracklets"):
        length = int(len(rows))
        mean_area = float((rows["width"] * rows["height"]).mean())
        if length < min_track_length or mean_area < min_mean_area:
            continue

        cam_dir = cams[camera]
        video_path = cam_dir / "vdo.avi"
        feat = extract_tracklet_feature(rows, video_path, model, tfm, device)

        fps = float(fps_overrides.get(camera, fps_default))
        start_abs = float(timestamps.get(camera, 0.0)) + float(rows["frame"].min()) / fps
        end_abs = float(timestamps.get(camera, 0.0)) + float(rows["frame"].max()) / fps

        tracklets.append(
            Tracklet(
                camera=camera,
                local_id=int(local_id),
                start_abs=start_abs,
                end_abs=end_abs,
                feat=feat,
                length=length,
                mean_area=mean_area,
            )
        )

    if not tracklets:
        raise RuntimeError(
            "No tracklets left after filtering. "
            "Try lowering mtmc.min_track_length or mtmc.min_mean_area in config."
        )

    gid_map = associate(
        tracklets,
        min_sim=float(mtmc_cfg["min_sim"]),
        max_time_gap_sec=float(mtmc_cfg["max_time_gap_sec"]),
    )

    gid_cams: Dict[int, set] = {}
    gid_rows: Dict[int, int] = {}
    for (cam_name, _local_id), gid in gid_map.items():
        if gid not in gid_cams:
            gid_cams[gid] = set()
        gid_cams[gid].add(cam_name)

    for _, r in df.iterrows():
        key = (str(r.camera), int(r.obj_id))
        if key in gid_map:
            gid = gid_map[key]
            gid_rows[gid] = gid_rows.get(gid, 0) + 1

    candidate_gids = [
        gid
        for gid in gid_cams
        if len(gid_cams.get(gid, set())) >= min_global_cameras and gid_rows.get(gid, 0) >= min_global_rows
    ]

    candidate_gids.sort(key=lambda g: (gid_rows.get(g, 0), len(gid_cams.get(g, set()))), reverse=True)
    if max_global_ids > 0:
        candidate_gids = candidate_gids[:max_global_ids]
    keep_gids = set(candidate_gids)

    out_rows = []
    for _, r in df.iterrows():
        cam_name = str(r.camera)
        local_id = int(r.obj_id)
        key = (cam_name, local_id)
        if key not in gid_map:
            continue
        gid = gid_map[key]
        if keep_only_multicam and len(gid_cams.get(gid, set())) < 2:
            continue
        if gid not in keep_gids:
            continue
        out_rows.append(
                [
                    cam_numeric_id(cam_name) + cam_id_offset,
                    gid,
                    int(r.frame),
                    int(round(r.left)),
                    int(round(r.top)),
                    int(round(r.width)),
                    int(round(r.height)),
                    -1,
                    -1,
                ]
            )

    out_rows.sort(key=lambda x: (x[0], x[2], x[1]))

    out_file = Path(args.output_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(" ".join(str(v) for v in row) + "\n")

    print(f"Wrote {len(out_rows)} lines to {out_file}")


if __name__ == "__main__":
    main()
