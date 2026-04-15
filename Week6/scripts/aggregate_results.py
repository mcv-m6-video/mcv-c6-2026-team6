import os
import json
import argparse
from tabulate import tabulate

CLASS_NAMES = [
    "PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT",
    "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK",
    "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"
]

SAVE_ROOT = "/ghome/group06/Ouss/c6/results/ablations"

EXPERIMENTS = [
    "baseline",
    "lstm",
    "lstm_2layer",
    "tcn",
    "transformer",
    "focal_loss",
]


def load_result(exp_name):
    results_path = os.path.join(SAVE_ROOT, exp_name, 'results.json')
    if not os.path.exists(results_path):
        return None
    with open(results_path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_root', default=SAVE_ROOT,
                        help='Root directory where experiments are saved')
    parser.add_argument('--sort_by', default='AP10',
                        choices=['AP10', 'AP12'],
                        help='Sort table by this metric (descending)')
    args = parser.parse_args()

    rows = []
    missing = []

    for exp in EXPERIMENTS:
        res_path = os.path.join(args.save_root, exp, 'results.json')
        if not os.path.exists(res_path):
            missing.append(exp)
            continue
        with open(res_path) as f:
            res = json.load(f)

        row = [exp, f"{res['AP10']:.2f}", f"{res['AP12']:.2f}"]
        for cls in CLASS_NAMES:
            val = res['per_class'].get(cls, float('nan'))
            row.append(f"{val:.2f}" if val == val else "—")
        rows.append((res['AP10'], row))

    if not rows:
        print("No results found yet. Run some experiments first.")
        return

    # Sort by chosen metric
    rows.sort(key=lambda x: x[0], reverse=True)
    rows = [r for _, r in rows]

    headers = ["Model", "AP10 ↑", "AP12 ↑"] + CLASS_NAMES
    print("\n" + "=" * 60)
    print("  ABLATION STUDY RESULTS")
    print("=" * 60)
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Baseline delta
    baseline_row = next((r for r in rows if r[0] == 'baseline'), None)
    if baseline_row:
        baseline_ap10 = float(baseline_row[1])
        print("\n  Delta vs baseline (AP10):")
        delta_rows = []
        for row in rows:
            if row[0] == 'baseline':
                continue
            delta = float(row[1]) - baseline_ap10
            sign = "+" if delta >= 0 else ""
            delta_rows.append([row[0], f"{sign}{delta:.2f}"])
        print(tabulate(delta_rows, headers=["Model", "ΔAP10"], tablefmt="simple"))

    if missing:
        print(f"\n  Not yet available: {', '.join(missing)}")

    print()


if __name__ == '__main__':
    main()
