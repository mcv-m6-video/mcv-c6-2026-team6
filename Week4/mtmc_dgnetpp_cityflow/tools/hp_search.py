#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import shutil
import subprocess
import sys
from pathlib import Path
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def merge_override(cfg: dict, key_path: str, value):
    parts = key_path.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact HP grid search for ReID+MTMC")
    parser.add_argument("--config", required=True, help="Base config yaml")
    parser.add_argument("--data-root", default="../AI_CITY_CHALLENGE_2022_TRAIN", help="CityFlow data root")
    parser.add_argument("--train-index", default="outputs/reid_data/train_index.csv", help="Train index CSV")
    parser.add_argument("--sequence", default="S03", help="Sequence for MTMC association")
    parser.add_argument("--out-dir", default="outputs/hp_search", help="Where to write configs, models and logs")
    parser.add_argument("--max-runs", type=int, default=0, help="Limit number of runs (0 = no limit)")
    parser.add_argument("--dry", action="store_true", help="Only write configs and print commands")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(Path(args.config).read_text())

    # Compact grid of important params (small initial sweep)
    grid = {
        "reid.lr": [3e-4, 1e-4],
        "reid.batch_size": [16, 32],
        "reid.weight_decay": [5e-4],
        "reid.margin": [0.3, 0.5],
        "reid.lambda_triplet": [1.0, 2.0],
        "mtmc.min_sim": [0.75, 0.85],
        "mtmc.min_track_length": [10, 20],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    out_dir = Path(args.out_dir)
    cfg_out_dir = out_dir / "configs"
    models_out_dir = out_dir / "models"
    assoc_out_dir = out_dir / "assoc"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_out_dir.mkdir(parents=True, exist_ok=True)
    models_out_dir.mkdir(parents=True, exist_ok=True)
    assoc_out_dir.mkdir(parents=True, exist_ok=True)

    results_csv = out_dir / "results.csv"
    header = ["exp_id"] + keys + ["train_rc", "assoc_rc", "assoc_lines", "cfg_path", "model_dir", "assoc_out"]

    combos = list(itertools.product(*values))
    if args.max_runs > 0:
        combos = combos[: args.max_runs]

    if args.dry:
        print(f"Dry run: will create {len(combos)} experiments")

    with results_csv.open("w", newline="", encoding="utf-8") as rf:
        writer = csv.writer(rf)
        writer.writerow(header)

        for idx, combo in enumerate(combos, start=1):
            exp_id = f"exp{idx:03d}"
            cfg_copy = dict(base_cfg)
            for k, v in zip(keys, combo):
                merge_override(cfg_copy, k, v)

            cfg_path = cfg_out_dir / f"{exp_id}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg_copy))

            model_dir = models_out_dir / exp_id
            assoc_out = assoc_out_dir / f"{exp_id}_track1.txt"

            train_cmd = [sys.executable, str(TOOLS / "train_reid.py"), "--config", str(cfg_path), "--train-index", args.train_index, "--output-dir", str(model_dir)]
            assoc_cmd = [sys.executable, str(TOOLS / "run_mtmc.py"), "--config", str(cfg_path), "--data-root", args.data_root, "--sequence", args.sequence, "--checkpoint", str(model_dir / "best.pt"), "--output-file", str(assoc_out)]

            print("---")
            print(f"{exp_id}: params={dict(zip(keys, combo))}")
            print("train:", " ".join(train_cmd))
            print("assoc:", " ".join(assoc_cmd))

            train_rc = None
            assoc_rc = None
            assoc_lines = -1

            if not args.dry:
                t0 = time.time()
                train_rc = subprocess.call(train_cmd)
                t1 = time.time()
                print(f"Train finished (rc={train_rc}) in {t1-t0:.1f}s")

                # pick best.pt if available, otherwise last.pt
                ckpt = model_dir / "best.pt"
                if not ckpt.exists():
                    ckpt = model_dir / "last.pt"

                if not ckpt.exists():
                    print(f"Warning: no checkpoint found for {exp_id} at {model_dir}")
                else:
                    assoc_rc = subprocess.call(assoc_cmd)
                    if assoc_out.exists():
                        assoc_lines = sum(1 for _ in assoc_out.open("r", encoding="utf-8"))

            writer.writerow([exp_id] + list(combo) + [train_rc, assoc_rc, assoc_lines, str(cfg_path), str(model_dir), str(assoc_out)])
            rf.flush()

    print(f"Done. Results logged to {results_csv}")


if __name__ == "__main__":
    main()
