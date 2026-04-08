#!/usr/bin/env python3
"""Count parameters and FLOPs for the classification model (thop / pytorch-OpCounter style)."""
from __future__ import annotations
import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Config name, e.g. baseline_c6")
    parser.add_argument("--clip_len", type=int, default=None, help="Override clip length")
    args = parser.parse_args()

    # Locate repo root (this script lives in scripts/)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    starter_dir = os.path.join(repo_root, "starter")
    sys.path.insert(0, starter_dir)

    from thop import clever_format, profile
    import torch

    from util.io import load_json

    config_path = os.path.join(starter_dir, "config", args.model + ".json")
    config = load_json(config_path)

    clip_len = args.clip_len if args.clip_len is not None else config["clip_len"]

    # Build a minimal args namespace
    import types
    model_args = types.SimpleNamespace(
        feature_arch=config["feature_arch"],
        num_classes=config["num_classes"],
        device="cpu",
    )

    from model.model_classification import Model
    model = Model(args=model_args)
    model._model.eval()

    # Input: (B=1, T=clip_len, C=3, H=224, W=398)
    dummy = torch.zeros(1, clip_len, 3, 224, 398)

    macs, params = profile(model._model, inputs=(dummy,), verbose=False)
    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")

    try:
        from tabulate import tabulate
        table = [
            ["Config", args.model],
            ["feature_arch", config["feature_arch"]],
            ["clip_len", clip_len],
            ["Params", params_fmt],
            ["MACs (FLOPs/2)", macs_fmt],
        ]
        print(tabulate(table, tablefmt="grid"))
    except ImportError:
        print(f"Config:  {args.model}")
        print(f"feature_arch: {config['feature_arch']}")
        print(f"clip_len:     {clip_len}")
        print(f"Params:       {params_fmt}")
        print(f"MACs:         {macs_fmt}")


if __name__ == "__main__":
    main()
