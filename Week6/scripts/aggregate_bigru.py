#!/usr/bin/env python3
"""
scripts/aggregate_bigru.py
---------------------------
Reads results.json for all BiGRU experiments and prints
one table per ablation stage, making it easy to pick the
best value at each stage.

Usage:
    python scripts/aggregate_bigru.py
    python scripts/aggregate_bigru.py --stage 1
    python scripts/aggregate_bigru.py --save_root /custom/path
"""

import os
import json
import argparse
from tabulate import tabulate

SAVE_ROOT = "/ghome/group06/Ouss/c6/results/ablations"

# AP10 class indices (0-indexed), excludes FREE KICK (10) and GOAL (11)
CLASS_NAMES = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT",
    "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]
AP10_CLASSES = CLASS_NAMES[:10]

STAGE_LABELS = {
    1: "Hidden Dim",
    2: "Num Layers",
    3: "Dropout",
    4: "Learning Rate",
    5: "Loss Function",
}

ABLATION_VAR_LABEL = {
    "hidden_dim":    "hidden_dim",
    "num_layers":    "num_layers",
    "dropout":       "dropout",
    "learning_rate": "lr",
    "loss":          "loss",
}


def load_all_results(save_root, config_dir="config"):
    """Load all results.json and their configs."""
    records = []
    if not os.path.exists(config_dir):
        print(f"Config dir '{config_dir}' not found. Run from repo root.")
        return records

    for fname in sorted(os.listdir(config_dir)):
        if not fname.startswith("bigru_") or not fname.endswith(".json"):
            continue
        exp = fname[:-5]
        cfg_path = os.path.join(config_dir, fname)
        res_path = os.path.join(save_root, exp, "results.json")

        with open(cfg_path) as f:
            cfg = json.load(f)

        result = None
        if os.path.exists(res_path):
            with open(res_path) as f:
                result = json.load(f)

        records.append({
            "exp":          exp,
            "stage":        cfg.get("stage", "?"),
            "ablation_var": cfg.get("ablation_var", "?"),
            "hidden_dim":   cfg.get("gru_hidden_dim", "?"),
            "num_layers":   cfg.get("gru_num_layers", "?"),
            "dropout":      cfg.get("gru_dropout", "?"),
            "lr":           cfg.get("learning_rate", "?"),
            "loss":         cfg.get("loss", "ce"),
            "result":       result,
        })
    return records


def print_stage(records, stage_num, baseline_ap10=None):
    stage_records = [r for r in records if r["stage"] == stage_num]
    if not stage_records:
        return

    label = STAGE_LABELS.get(stage_num, f"Stage {stage_num}")
    print(f"\n{'='*60}")
    print(f"  STAGE {stage_num}: Ablation over {label}")
    print(f"{'='*60}")

    rows = []
    for r in stage_records:
        res = r["result"]
        if res is None:
            ap10 = "—"
            ap12 = "—"
            per_cls = ["—"] * 10
            status = "PENDING"
        else:
            ap10 = f"{res['AP10']:.2f}"
            ap12 = f"{res['AP12']:.2f}"
            per_cls = [f"{res['per_class'].get(c, 0):.1f}" for c in AP10_CLASSES]
            delta = ""
            if baseline_ap10 is not None:
                d = res["AP10"] - baseline_ap10
                delta = f"  ({'+' if d>=0 else ''}{d:.2f} vs baseline)"
            status = f"DONE{delta}"

        rows.append([
            r["hidden_dim"], r["num_layers"], r["dropout"],
            r["lr"], r["loss"],
            ap10, ap12,
            *per_cls,
            status
        ])

    # Sort by AP10 descending (done runs first)
    rows.sort(key=lambda x: float(x[5]) if x[5] != "—" else -999, reverse=True)

    headers = ["hid_dim", "layers", "dropout", "lr", "loss",
               "AP10↑", "AP12",
               *[c[:8] for c in AP10_CLASSES],
               "status"]
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Best row
    done = [r for r in rows if r[-1].startswith("DONE")]
    if done:
        best = done[0]
        print(f"\n  ★  Best in Stage {stage_num}:")
        print(f"     hidden_dim={best[0]}, layers={best[1]}, "
              f"dropout={best[2]}, lr={best[3]}, loss={best[4]}  →  AP10={best[5]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", default=SAVE_ROOT)
    parser.add_argument("--config_dir", default="config")
    parser.add_argument("--stage", type=int, default=None,
                        help="Show only this stage (default: all)")
    parser.add_argument("--baseline_ap10", type=float, default=None,
                        help="Baseline AP10 to compute deltas against")
    args = parser.parse_args()

    records = load_all_results(args.save_root, args.config_dir)
    if not records:
        print("No BiGRU configs found in config/. Run generate_configs.py first.")
        return

    stages = sorted(set(r["stage"] for r in records if isinstance(r["stage"], int)))
    if args.stage:
        stages = [s for s in stages if s == args.stage]

    for s in stages:
        print_stage(records, s, baseline_ap10=args.baseline_ap10)

    # Summary: best per stage
    print(f"\n{'='*60}")
    print("  SUMMARY — Best AP10 per Stage")
    print(f"{'='*60}")
    summary_rows = []
    for s in sorted(set(r["stage"] for r in records if isinstance(r["stage"], int))):
        done = [r for r in records if r["stage"] == s and r["result"] is not None]
        if not done:
            summary_rows.append([STAGE_LABELS.get(s, s), "—", "—", "PENDING"])
        else:
            best = max(done, key=lambda r: r["result"]["AP10"])
            summary_rows.append([
                STAGE_LABELS.get(s, s),
                best["exp"],
                f"{best['result']['AP10']:.2f}",
                "✓"
            ])
    print(tabulate(summary_rows,
                   headers=["Stage", "Best Experiment", "AP10", "Status"],
                   tablefmt="simple"))
    print()


if __name__ == "__main__":
    main()
