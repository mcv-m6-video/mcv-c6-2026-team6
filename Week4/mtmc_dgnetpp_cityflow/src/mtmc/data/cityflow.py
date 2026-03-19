from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class TrackRow:
    frame: int
    obj_id: int
    left: float
    top: float
    width: float
    height: float


def parse_mot_txt(path: str | Path) -> pd.DataFrame:
    cols = [
        "frame",
        "obj_id",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "x",
        "y",
        "z",
    ]
    df = pd.read_csv(path, header=None, names=cols)
    return df


def camera_dir_from_sequence(data_root: str | Path, sequence: str) -> Dict[str, Path]:
    root = Path(data_root)
    if not root.exists():
        parent = root.parent if root.parent.exists() else Path.cwd()
        siblings = [p.name for p in parent.iterdir() if p.is_dir()]
        maybe = difflib.get_close_matches(root.name, siblings, n=1)
        hint = f" Did you mean '{maybe[0]}'?" if maybe else ""
        raise FileNotFoundError(f"Data root not found: {root}.{hint}")

    train_dir = root / "train" / sequence
    val_dir = root / "validation" / sequence
    test_dir = root / "test" / sequence

    if train_dir.exists():
        base = train_dir
    elif val_dir.exists():
        base = val_dir
    elif test_dir.exists():
        base = test_dir
    else:
        available = []
        for split in ("train", "validation", "test"):
            split_dir = root / split
            if split_dir.exists():
                for d in split_dir.iterdir():
                    if d.is_dir() and d.name.startswith("S"):
                        available.append(d.name)
        available = sorted(set(available))
        maybe = difflib.get_close_matches(sequence, available, n=1)
        hint = f" Did you mean '{maybe[0]}'?" if maybe else ""
        extra = f" Available sequences: {', '.join(available)}." if available else ""
        raise FileNotFoundError(f"Sequence not found: {sequence}.{hint}{extra}")

    cams = {}
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("c"):
            cams[d.name] = d
    return cams


def load_timestamps(data_root: str | Path, sequence: str) -> Dict[str, float]:
    ts_file = Path(data_root) / "cam_timestamp" / f"{sequence}.txt"
    out: Dict[str, float] = {}
    with ts_file.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            cam, ts = parts
            out[cam] = float(ts)
    return out


def cam_numeric_id(cam_name: str) -> int:
    return int(cam_name[1:])


def load_tracklets_for_sequence(data_root: str | Path, sequence: str, mtsc_file: str) -> pd.DataFrame:
    cams = camera_dir_from_sequence(data_root, sequence)
    all_rows: List[pd.DataFrame] = []

    for cam_name, cam_dir in cams.items():
        file_path = cam_dir / "mtsc" / mtsc_file
        if not file_path.exists():
            raise FileNotFoundError(f"Missing MTSC file: {file_path}")

        df = parse_mot_txt(file_path)
        df["camera"] = cam_name
        all_rows.append(df)

    if not all_rows:
        return pd.DataFrame()
    return pd.concat(all_rows, ignore_index=True)
