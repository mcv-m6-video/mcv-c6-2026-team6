from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-filter track1 predictions by global-id statistics")
    parser.add_argument("--input", required=True, help="Input track1 file")
    parser.add_argument("--output", required=True, help="Output track1 file")
    parser.add_argument("--min-global-cameras", type=int, default=2)
    parser.add_argument("--min-global-rows", type=int, default=1)
    parser.add_argument("--max-global-ids", type=int, default=0, help="0 means keep all")
    parser.add_argument(
        "--rank-by",
        choices=["rows", "cams"],
        default="rows",
        help="Ranking key when max-global-ids > 0",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)

    gid_to_lines = defaultdict(list)
    gid_to_cams = defaultdict(set)

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 9:
                continue
            cam = int(float(parts[0]))
            gid = int(float(parts[1]))
            gid_to_lines[gid].append(parts)
            gid_to_cams[gid].add(cam)

    gids = []
    for gid, lines in gid_to_lines.items():
        cams = len(gid_to_cams[gid])
        rows = len(lines)
        if cams < args.min_global_cameras:
            continue
        if rows < args.min_global_rows:
            continue
        gids.append((gid, cams, rows))

    if args.rank_by == "rows":
        gids.sort(key=lambda x: (x[2], x[1], -x[0]), reverse=True)
    else:
        gids.sort(key=lambda x: (x[1], x[2], -x[0]), reverse=True)

    if args.max_global_ids > 0:
        gids = gids[: args.max_global_ids]

    keep = {g for g, _c, _r in gids}

    out_lines = []
    for gid in keep:
        for p in gid_to_lines[gid]:
            out_lines.append(
                [
                    int(float(p[0])),
                    int(float(p[1])),
                    int(float(p[2])),
                    int(float(p[3])),
                    int(float(p[4])),
                    int(float(p[5])),
                    int(float(p[6])),
                    int(float(p[7])),
                    int(float(p[8])),
                ]
            )

    out_lines.sort(key=lambda x: (x[0], x[2], x[1]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_lines:
            f.write(" ".join(str(v) for v in row) + "\n")

    print(f"Input IDs: {len(gid_to_lines)}, kept IDs: {len(keep)}")
    print(f"Wrote {len(out_lines)} rows to {out_path}")


if __name__ == "__main__":
    main()
