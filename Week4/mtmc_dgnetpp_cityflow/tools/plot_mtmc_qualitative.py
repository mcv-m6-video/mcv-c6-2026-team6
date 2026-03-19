from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd


def parse_track1(path: Path) -> pd.DataFrame:
    cols = ["cam", "gid", "frame", "left", "top", "width", "height", "xworld", "yworld"]
    df = pd.read_csv(path, sep=r"\s+", header=None, names=cols)
    for c in ["cam", "gid", "frame"]:
        df[c] = df[c].astype(int)
    for c in ["left", "top", "width", "height"]:
        df[c] = df[c].astype(float)
    df["area"] = df["width"] * df["height"]
    return df


def list_sequence_cameras(data_root: Path, sequence: str) -> Dict[int, str]:
    seq_dirs = [data_root / "train" / sequence, data_root / "validation" / sequence, data_root / "test" / sequence]
    base = next((p for p in seq_dirs if p.exists()), None)
    if base is None:
        raise FileNotFoundError(f"Sequence folder not found for '{sequence}' under {data_root}")

    out: Dict[int, str] = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("c"):
            cam_num = int(d.name[1:])
            out[cam_num] = d.name
    if not out:
        raise RuntimeError(f"No camera folders found for sequence '{sequence}'")
    return out


def infer_camera_id_offset(track_cam_ids: Iterable[int], available_cam_nums: Iterable[int]) -> int:
    track_ids = sorted(set(int(x) for x in track_cam_ids))
    available = set(int(x) for x in available_cam_nums)

    if all(tid in available for tid in track_ids):
        return 0

    # In run_mtmc export: written_cam = real_cam + offset.
    # So real_cam = written_cam - offset.
    best_offset = 0
    best_hits = -1
    for offset in range(-200, 201):
        hits = sum(1 for tid in track_ids if (tid - offset) in available)
        if hits > best_hits:
            best_hits = hits
            best_offset = offset

    return best_offset


def read_video_frame(video_path: Path, frame_id_1based: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_id_1based - 1))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    return frame


def draw_detection(frame_bgr: np.ndarray, row: pd.Series, title: str) -> np.ndarray:
    img = frame_bgr.copy()
    x1 = int(round(row.left))
    y1 = int(round(row.top))
    x2 = int(round(row.left + row.width))
    y2 = int(round(row.top + row.height))

    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))

    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 240, 0), 3)
    cv2.rectangle(img, (x1, max(0, y1 - 38)), (min(w - 1, x1 + 360), y1), (0, 240, 0), -1)
    cv2.putText(img, title, (x1 + 6, max(14, y1 - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def fit_to_tile(img: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(tile_w / max(1, w), tile_h / max(1, h))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((tile_h, tile_w, 3), 245, dtype=np.uint8)
    x0 = (tile_w - nw) // 2
    y0 = (tile_h - nh) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def select_row_for_camera(cam_df: pd.DataFrame) -> pd.Series:
    # Pick the clearest view: largest bbox area.
    pos = int(np.argmax(cam_df["area"].to_numpy()))
    return cam_df.iloc[pos]


def best_global_ids(df: pd.DataFrame, min_cameras: int, max_examples: int) -> List[int]:
    gid_to_cams: Dict[int, set] = {}
    gid_to_rows: Dict[int, int] = {}
    for _, r in df.iterrows():
        gid = int(r["gid"])
        cam = int(r["cam"])
        if gid not in gid_to_cams:
            gid_to_cams[gid] = set()
        gid_to_cams[gid].add(cam)
        gid_to_rows[gid] = gid_to_rows.get(gid, 0) + 1

    candidates = [gid for gid, cams in gid_to_cams.items() if len(cams) >= int(min_cameras)]
    candidates.sort(key=lambda g: (len(gid_to_cams[g]), gid_to_rows[g], -g), reverse=True)
    return candidates[:max_examples]


def build_gid_canvas(
    gid: int,
    rows_gid: pd.DataFrame,
    cam_num_to_name: Dict[int, str],
    seq_root: Path,
    camera_id_offset: int,
    tile_w: int,
    tile_h: int,
    columns: int,
) -> np.ndarray | None:
    panels: List[np.ndarray] = []

    cam_ids = sorted({int(v) for v in rows_gid["cam"].tolist()})
    for written_cam_int in cam_ids:
        cam_rows = rows_gid.loc[rows_gid["cam"].map(int) == written_cam_int]
        if isinstance(cam_rows, pd.Series):
            cam_rows = cam_rows.to_frame().T
        if cam_rows.empty:
            continue
        real_cam_num = written_cam_int - int(camera_id_offset)
        cam_name = cam_num_to_name.get(real_cam_num)
        if cam_name is None:
            continue

        video_path = seq_root / cam_name / "vdo.avi"
        if not video_path.exists():
            continue

        r = select_row_for_camera(cam_rows)
        frame = read_video_frame(video_path, int(r.frame))
        if frame is None:
            continue

        title = f"{cam_name} | GID {gid} | f={int(r.frame)}"
        ann = draw_detection(frame, r, title)
        tile = fit_to_tile(ann, tile_w, tile_h)
        panels.append(tile)

    if not panels:
        return None

    n = len(panels)
    cols = max(1, min(columns, n))
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * tile_h, cols * tile_w, 3), 255, dtype=np.uint8)

    for i, p in enumerate(panels):
        rr = i // cols
        cc = i % cols
        y0 = rr * tile_h
        x0 = cc * tile_w
        canvas[y0 : y0 + tile_h, x0 : x0 + tile_w] = p

    # Top banner.
    banner_h = 52
    banner = np.full((banner_h, canvas.shape[1], 3), 235, dtype=np.uint8)
    txt = f"Global ID {gid} | cameras shown: {len(panels)}"
    cv2.putText(banner, txt, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2, cv2.LINE_AA)
    return np.vstack([banner, canvas])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate qualitative MTMC examples with same global ID across cameras")
    parser.add_argument("--track-file", required=True, help="Path to track1 txt (output of run_mtmc.py)")
    parser.add_argument("--data-root", required=True, help="CityFlow root (e.g., ../AI_CITY_CHALLENGE_2022_TRAIN)")
    parser.add_argument("--sequence", required=True, help="Sequence name (e.g., S03)")
    parser.add_argument("--output-dir", default="./outputs/qualitative", help="Directory to save qualitative images")
    parser.add_argument("--min-cameras", type=int, default=3, help="Only visualize IDs appearing in at least this many cameras")
    parser.add_argument("--max-examples", type=int, default=24, help="Maximum number of global IDs to export")
    parser.add_argument("--camera-id-offset", type=int, default=None, help="Offset used when writing track file; auto-inferred if omitted")
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=360)
    parser.add_argument("--columns", type=int, default=3, help="Montage columns")
    args = parser.parse_args()

    track_path = Path(args.track_file)
    data_root = Path(args.data_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = parse_track1(track_path)
    if df.empty:
        raise RuntimeError(f"Empty track file: {track_path}")

    cam_num_to_name = list_sequence_cameras(data_root, args.sequence)
    seq_root = next(
        (p for p in [data_root / "train" / args.sequence, data_root / "validation" / args.sequence, data_root / "test" / args.sequence] if p.exists()),
        None,
    )
    if seq_root is None:
        raise RuntimeError(f"Could not locate sequence root for {args.sequence}")

    if args.camera_id_offset is None:
        offset = infer_camera_id_offset(df["cam"].tolist(), cam_num_to_name.keys())
        print(f"[INFO] Auto-inferred camera_id_offset={offset}")
    else:
        offset = int(args.camera_id_offset)
        print(f"[INFO] Using provided camera_id_offset={offset}")

    gids = best_global_ids(df, min_cameras=args.min_cameras, max_examples=args.max_examples)
    if not gids:
        raise RuntimeError(
            "No global IDs satisfy min-cameras condition. "
            "Try lowering --min-cameras or check your track file."
        )

    summary_rows = []
    exported = 0
    for gid in gids:
        rows_gid = df.loc[df["gid"] == gid]
        if isinstance(rows_gid, pd.Series):
            rows_gid = rows_gid.to_frame().T
        else:
            rows_gid = rows_gid.copy()
        canvas = build_gid_canvas(
            gid=gid,
            rows_gid=rows_gid,
            cam_num_to_name=cam_num_to_name,
            seq_root=seq_root,
            camera_id_offset=offset,
            tile_w=int(args.tile_width),
            tile_h=int(args.tile_height),
            columns=int(args.columns),
        )
        if canvas is None:
            continue

        save_path = out_dir / f"gid_{gid:05d}.jpg"
        cv2.imwrite(str(save_path), canvas)

        cams = sorted(set(int(c) for c in rows_gid["cam"].tolist()))
        summary_rows.append({"gid": int(gid), "written_cams": ",".join(str(c) for c in cams), "rows": int(len(rows_gid)), "image": save_path.name})
        exported += 1

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[OK] Exported {exported} qualitative examples to: {out_dir}")
    print(f"[OK] Summary table: {summary_csv}")


if __name__ == "__main__":
    main()
