#!/usr/bin/env python3
"""
generate_gt_mtmc.py
===================
Generate gt_mtmc.txt for CityFlow sequences from per-camera gt/gt.txt files.

In the CityFlow AIC22 dataset, the per-camera gt/gt.txt files for training
sequences (S01, S04) use GLOBALLY CONSISTENT vehicle IDs — the same vehicle
ID number means the same physical vehicle across all cameras in the sequence.

So gt_mtmc.txt can be generated directly by aggregating gt/gt.txt entries.

Output format (matches CityFlow official gt_mtmc.txt):
  <cam_id> <global_vehicle_id> <frame_id> <left> <top> <width> <height> -1 -1

Usage:
  python generate_gt_mtmc.py --seq-dir data/aicity/S01
  python generate_gt_mtmc.py --seq-dir data/aicity/S04
  python generate_gt_mtmc.py --seq-dir data/aicity/S03  # test set - generates from gt if available
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def generate_gt_mtmc(seq_dir: str, verify: bool = True):
    seq_path = Path(seq_dir)
    out_path = seq_path / 'gt_mtmc.txt'

    cameras = sorted([d for d in seq_path.iterdir()
                      if d.is_dir() and d.name.startswith('c')])

    if not cameras:
        print(f"No camera dirs found in {seq_path}")
        return

    rows = []
    global_id_seen = defaultdict(set)  # global_id -> set of cam_names

    for cam_dir in cameras:
        cam_name = cam_dir.name
        gt_file  = cam_dir / 'gt' / 'gt.txt'
        if not gt_file.exists():
            print(f"  [skip] {cam_name}: no gt/gt.txt")
            continue

        n = 0
        with open(gt_file, newline='') as f:
            for row in csv.reader(f):
                if not row or row[0].startswith('#') or len(row) < 7:
                    continue
                try:
                    active = int(float(row[6]))
                except (ValueError, IndexError):
                    active = 1
                if active == 0:
                    continue

                frame    = int(row[0])
                vid      = int(row[1])   # This IS the global vehicle ID in CityFlow
                x, y     = float(row[2]), float(row[3])
                w, h     = float(row[4]), float(row[5])

                rows.append(f"{cam_name} {vid} {frame} "
                             f"{x:.2f} {y:.2f} {w:.2f} {h:.2f} -1 -1")
                global_id_seen[vid].add(cam_name)
                n += 1
        print(f"  {cam_name}: {n} GT detections")

    if not rows:
        print(f"ERROR: No GT data found in {seq_path}")
        return

    # Sort by camera, then frame
    rows.sort(key=lambda r: (r.split()[0], int(r.split()[2])))

    with open(out_path, 'w') as f:
        f.write('\n'.join(rows) + '\n')

    # Stats
    multi_cam = {gid: cams for gid, cams in global_id_seen.items() if len(cams) >= 2}
    single_cam = {gid: cams for gid, cams in global_id_seen.items() if len(cams) < 2}
    total_veh = len(global_id_seen)

    print(f"\n  Written: {out_path}")
    print(f"  Total rows    : {len(rows)}")
    print(f"  Unique vehicles: {total_veh}")
    print(f"  Multi-cam vehicles (appear in >=2 cameras): {len(multi_cam)}")
    print(f"  Single-cam vehicles: {len(single_cam)}")

    if verify and multi_cam:
        print(f"\n  Sample multi-cam vehicles:")
        for gid, cams in list(multi_cam.items())[:5]:
            print(f"    vehicle {gid:4d} → cameras: {sorted(cams)}")

    return out_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--seq-dir', required=True,
                   help='Path to sequence dir (e.g. data/aicity/S01)')
    p.add_argument('--all-seqs', nargs='+', default=None,
                   help='Process multiple sequences: --all-seqs data/S01 data/S03 data/S04')
    args = p.parse_args()

    if args.all_seqs:
        for seq in args.all_seqs:
            print(f"\n{'='*50}\n  {seq}\n{'='*50}")
            generate_gt_mtmc(seq)
    else:
        generate_gt_mtmc(args.seq_dir)


if __name__ == '__main__':
    main()