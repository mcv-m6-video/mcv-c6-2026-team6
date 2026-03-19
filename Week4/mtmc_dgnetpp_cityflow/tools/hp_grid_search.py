#!/usr/bin/env python3
"""Grid search for MTMC hyperparameters.

Usage example:
  python tools/hp_grid_search.py \
    --config configs/default.yaml \
    --data-root ../AI_CITY_CHALLENGE_2022_TRAIN \
    --sequence S03 \
    --checkpoint ./outputs/reid_model/best.pt \
    --gt AI_CITY_CHALLENGE_2022_TRAIN/eval/ground_truth_s03.txt \
    --out-dir outputs/hp_search

The script will write `results.csv` under the output directory with one row per hyperparameter combination.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from itertools import product
from pathlib import Path

try:
    import yaml
except Exception:
    print("Please install PyYAML (pip install pyyaml)")
    raise


def parse_list(arg, typ=float):
    return [typ(x) for x in str(arg).split(',') if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gt', required=True, help='Ground truth file for eval')
    p.add_argument('--eval-script', default='AI_CITY_CHALLENGE_2022_TRAIN/eval/eval.py')
    p.add_argument('--out-dir', default='outputs/hp_search')

    p.add_argument('--min-sim', default='0.65', help='Comma-separated values')
    p.add_argument('--min-track-length', default='10', help='Comma-separated values')
    p.add_argument('--min-mean-area', default='2000.0', help='Comma-separated values')

    p.add_argument('--keep-temp', action='store_true', help='Keep temporary configs and outputs')
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg_path = Path(args.config)
    with base_cfg_path.open('r', encoding='utf-8') as fh:
        base_cfg = yaml.safe_load(fh)

    sims = parse_list(args.min_sim, float)
    lengths = parse_list(args.min_track_length, int)
    areas = parse_list(args.min_mean_area, float)

    combos = list(product(sims, lengths, areas))

    results_path = out_dir / 'results.csv'
    with results_path.open('w', newline='', encoding='utf-8') as csvfh:
        writer = csv.writer(csvfh)
        writer.writerow(['min_sim', 'min_track_length', 'min_mean_area', 'idf1', 'idp', 'idr', 'lines', 'run_time_sec', 'config', 'output'])

        for idx, (ms, ml, ma) in enumerate(combos, start=1):
            print(f'Running {idx}/{len(combos)}: min_sim={ms}, min_track_length={ml}, min_mean_area={ma}')
            cfg = dict(base_cfg)
            # shallow copy for mtmc section
            mtmc = dict(cfg.get('mtmc', {}))
            mtmc['min_sim'] = float(ms)
            mtmc['min_track_length'] = int(ml)
            mtmc['min_mean_area'] = float(ma)
            cfg['mtmc'] = mtmc

            temp_cfg_file = out_dir / f'cfg_ms{ms}_ml{ml}_ma{int(ma)}.yaml'
            with temp_cfg_file.open('w', encoding='utf-8') as fh:
                yaml.safe_dump(cfg, fh)

            out_file = out_dir / f'track_ms{ms}_ml{ml}_ma{int(ma)}.txt'

            start = time.time()
            try:
                cmd = [sys.executable, 'tools/run_mtmc.py', '--config', str(temp_cfg_file), '--data-root', args.data_root, '--sequence', args.sequence, '--checkpoint', args.checkpoint, '--output-file', str(out_file)]
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print('Run failed:', e)
                idf1 = idp = idr = None
                lines = 0
                run_time = time.time() - start
                writer.writerow([ms, ml, ma, idf1, idp, idr, lines, run_time, str(temp_cfg_file), str(out_file)])
                continue

            run_time = time.time() - start

            # count lines in output
            lines = 0
            if out_file.exists():
                with out_file.open('r', encoding='utf-8') as fh:
                    for _ in fh:
                        lines += 1

            # evaluate using eval script in machine-readable JSON mode
            try:
                eval_cmd = [sys.executable, args.eval_script, args.gt, str(out_file), '-m', '--dstype', 'train']
                proc = subprocess.run(eval_cmd, capture_output=True, text=True, check=True)
                out = proc.stdout.strip()
                # expected output: {"results":{...}}
                data = json.loads(out)
                results = data.get('results', {})
                idf1 = results.get('idf1')
                idp = results.get('idp')
                idr = results.get('idr')
            except Exception as e:
                print('Eval failed or could not parse JSON:', e)
                idf1 = idp = idr = None

            writer.writerow([ms, ml, ma, idf1, idp, idr, lines, round(run_time, 2), str(temp_cfg_file), str(out_file)])

    print('Grid search finished. Results saved to', results_path)
    if not args.keep_temp:
        print('Temporary files kept in', out_dir)


if __name__ == '__main__':
    main()
