from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-csv", required=True)
    parser.add_argument("--train-out", default="outputs/reid_data/train_split.csv")
    parser.add_argument("--val-out", default="outputs/reid_data/val_split.csv")
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.index_csv)
    ids = sorted(df["obj_id"].unique().tolist())
    rng = np.random.RandomState(args.seed)
    rng.shuffle(ids)

    n_val = max(1, int(len(ids) * args.val_frac))
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    df_train = df[df["obj_id"].isin(train_ids)].reset_index(drop=True)
    df_val = df[df["obj_id"].isin(val_ids)].reset_index(drop=True)

    out_train = Path(args.train_out)
    out_val = Path(args.val_out)
    out_train.parent.mkdir(parents=True, exist_ok=True)
    df_train.to_csv(out_train, index=False)
    df_val.to_csv(out_val, index=False)

    print(f"Total ids: {len(ids)}, train ids: {len(train_ids)}, val ids: {len(val_ids)}")
    print(f"Train rows: {len(df_train)}, Val rows: {len(df_val)}")


if __name__ == "__main__":
    main()
