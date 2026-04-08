#!/usr/bin/env python3
"""Verify SN-BAS-2025 paths under data/sn_bas_2025 match starter split JSONs."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels_root",
        default=None,
        help="Root of the unpacked dataset (default: <repo_root>/data/sn_bas_2025)"
    )
    parser.add_argument(
        "--starter_dir",
        default=None,
        help="starter/ directory (default: <repo_root>/starter)"
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    labels_root = Path(args.labels_root) if args.labels_root else repo_root / "data" / "sn_bas_2025"
    starter_dir = Path(args.starter_dir) if args.starter_dir else repo_root / "starter"

    print(f"labels_root: {labels_root}")

    split_files = [
        starter_dir / "data" / "soccernetball" / "train.json",
        starter_dir / "data" / "soccernetball" / "val.json",
        starter_dir / "data" / "soccernetball" / "test.json",
    ]

    ok = []
    missing = []

    for split_file in split_files:
        if not split_file.exists():
            print(f"WARNING: split file not found: {split_file}", file=sys.stderr)
            continue
        games = load_json(str(split_file))
        for game in games:
            video = game["video"]
            label_path = labels_root / video / "Labels-ball.json"
            if label_path.exists():
                ok.append((str(split_file.name), video))
            else:
                missing.append((str(split_file.name), video))

    total = len(ok) + len(missing)
    print(f"checked: {total} games  ok: {len(ok)}   missing: {len(missing)}")

    if missing:
        print("\nMissing label files:", file=sys.stderr)
        for split_name, video in missing[:10]:
            print(f"  [{split_name}]  {labels_root / video / 'Labels-ball.json'}", file=sys.stderr)
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more", file=sys.stderr)

        # Diagnose empty match dirs
        first_video = missing[0][1]
        game_dir = labels_root / first_video
        if game_dir.is_dir() and not any(game_dir.iterdir()):
            print(
                "\n  Match folder exists but is empty — GNU unzip cannot extract WinZip AES",
                file=sys.stderr,
            )
            print(
                "  in these archives. Use: pip install pyzipper; then re-run prepare_sn_bas_data.sh",
                file=sys.stderr,
            )
            print(
                "  with SN_BAS_ZIP_PASSWORD set to your SoccerNet NDA password.",
                file=sys.stderr,
            )
        elif not game_dir.exists():
            # Check if england_efl tree is one level deeper
            subs = [d.name for d in labels_root.iterdir() if d.is_dir()][:5]
            print(f"\n  Match folder does not exist at: {game_dir}", file=sys.stderr)
            print(f"  Top-level dirs under labels_root: {subs}", file=sys.stderr)
            # Check if there's a nested england_efl
            nested = labels_root / "england_efl"
            if not nested.exists():
                for sub in labels_root.iterdir():
                    if (sub / "england_efl").exists():
                        print(f"  Found england_efl under {sub} — update labels_dir in config JSONs to:", file=sys.stderr)
                        print(f"    {sub}", file=sys.stderr)
                        break

        sys.exit(1)
    else:
        print("All label files found.")


if __name__ == "__main__":
    main()
