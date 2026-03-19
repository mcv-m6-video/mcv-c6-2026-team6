"""
paths.py
========
Central path configuration. Edit the three lines under "USER CONFIG" to
match your setup, then run: python -m src.paths  to verify all paths exist.
"""

from pathlib import Path
import os

# ── USER CONFIG ───────────────────────────────────────────────────────────────
# ROOT: directory containing this project (has src/, scripts/, checkpoints/ etc.)
# Can also be set via environment variable: export MTMC_ROOT=/your/path
ROOT = Path(os.environ.get('MTMC_ROOT', Path(__file__).resolve().parent.parent))

# SEQS_ROOT: directory that directly contains S01/, S03/, S04/ folders.
# Set via env var:  export MTMC_SEQS_ROOT=/data/.../aicity
# Your layout:      /data/113-1/users/smazumder/C6/data/aicity/S01/
#                                                              ^^^^^^ this is SEQS_ROOT
SEQS_ROOT = Path(os.environ.get('MTMC_SEQS_ROOT', ROOT /'Week_4'/ 'data' / 'aicity'))

# ── DERIVED PATHS (no need to edit) ──────────────────────────────────────────
OUTPUTS     = ROOT / 'outputs'
RESULTS     = OUTPUTS / 'results'
IMAGES      = OUTPUTS / 'images'
CHECKPOINTS = ROOT / 'checkpoints'

# Sequence paths — S01/S04 train, S03 test
SEQ_TRAIN = [SEQS_ROOT / 'S01', SEQS_ROOT / 'S04']
SEQ_TEST  =  SEQS_ROOT / 'S03'

# Auto-create output dirs
for _d in [RESULTS, IMAGES, CHECKPOINTS]:
    _d.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    print(f"ROOT        : {ROOT}")
    print(f"SEQS_ROOT   : {SEQS_ROOT}")
    print(f"SEQ_TRAIN   : {[str(s) for s in SEQ_TRAIN]}")
    print(f"SEQ_TEST    : {SEQ_TEST}")
    print(f"CHECKPOINTS : {CHECKPOINTS}")
    print(f"RESULTS     : {RESULTS}")
    print()
    for s in SEQ_TRAIN + [SEQ_TEST]:
        status = '✓ exists' if s.exists() else '✗ NOT FOUND'
        print(f"  {status}  {s}")


# ── Camera synchronisation offsets ───────────────────────────────────────────
def load_sync_offsets(offset_file=None) -> dict:
    """
    Load camera synchronisation offsets in seconds.
    Returns {cam_name: offset_seconds}
    real_time = frame_no / fps + offset_seconds
    """
    if offset_file is None:
        offset_file = Path(__file__).resolve().parent / 'camera_sync_offsets.txt'
    
    offsets = {}
    if not Path(offset_file).exists():
        return offsets
    
    with open(offset_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('['):
                continue
            parts = line.split()
            if len(parts) == 2:
                offsets[parts[0]] = float(parts[1])
    return offsets

# Pre-loaded sync offsets for direct import
CAM_SYNC_OFFSETS = load_sync_offsets()