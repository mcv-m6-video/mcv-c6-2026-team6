from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import cv2
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mtmc.data.cityflow import camera_dir_from_sequence, parse_mot_txt


def clamp_bbox(x: float, y: float, w: float, h: float, width: int, height: int):
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width - 1, int(round(x + w)))
    y2 = min(height - 1, int(round(y + h)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ReID crops from CityFlow GT")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--sequences", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=900)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    records = []

    for seq in args.sequences:
        cam_dirs = camera_dir_from_sequence(data_root, seq)
        for cam_name, cam_dir in tqdm(cam_dirs.items(), desc=f"Sequence {seq}"):
            gt_file = cam_dir / "gt" / "gt.txt"
            vdo_file = cam_dir / "vdo.avi"
            if not gt_file.exists() or not vdo_file.exists():
                continue

            gt = parse_mot_txt(gt_file)
            gt = gt[(gt["frame"] % args.frame_stride) == 0].copy()
            gt = gt[(gt["width"] * gt["height"]) >= args.min_area].copy()
            if gt.empty:
                continue

            cap = cv2.VideoCapture(str(vdo_file))
            if not cap.isOpened():
                continue

            frames = sorted(gt["frame"].unique().tolist())
            frame_to_rows = {f: g for f, g in gt.groupby("frame")}

            for frame_id in frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id - 1))
                ok, frame = cap.read()
                if not ok:
                    continue

                h, w = frame.shape[:2]
                rows = frame_to_rows[frame_id]
                for _, row in rows.iterrows():
                    bbox = clamp_bbox(row.left, row.top, row.width, row.height, w, h)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    obj_id = int(row.obj_id)
                    rel_path = Path(seq) / cam_name / f"{obj_id:04d}" / f"f{int(frame_id):06d}.jpg"
                    out_path = crops_dir / rel_path
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_path), crop)

                    records.append(
                        {
                            "sequence": seq,
                            "camera": cam_name,
                            "obj_id": obj_id,
                            "frame": int(frame_id),
                            "crop_path": str(out_path.resolve()),
                        }
                    )

            cap.release()

    index_path = output_dir / "train_index.csv"
    pd.DataFrame(records).to_csv(index_path, index=False)
    print(f"Saved {len(records)} crops to {index_path}")


if __name__ == "__main__":
    main()
