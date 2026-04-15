# pip install ptflops

# Usage:
#     python scripts/count_macs.py

import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from util.io import load_json
from model.model_spotting import Model

# ── Config ────────────────────────────────────────────────────────────────────
EXPERIMENTS = [
    "baseline",
    "lstm",
    "lstm_2layer",
    "tcn",
    "transformer",
    "focal_loss",
]

# Input shape: (batch=1, clip_len=50, C=3, H=224, W=398)
CLIP_LEN = 50
H, W = 224, 398


class _ArgsFromDict:
    """Mimics argparse Namespace from a dict."""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __contains__(self, key):
        return hasattr(self, key)


def load_model(exp_name):
    config_path = f'config/{exp_name}.json'
    config = load_json(config_path)
    config.setdefault('temporal_arch', 'none')
    config.setdefault('loss', 'ce')
    config.setdefault('device', 'cpu')

    args = _ArgsFromDict(config)
    args.model = exp_name
    args.seed  = 1

    # Patch save/store dirs so model doesn't fail on init
    args.save_dir  = '/tmp'
    args.store_dir = '/tmp'

    model = Model(args=args)
    return model._model


def count_macs_ptflops(model, input_shape):
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        print("  ptflops not installed. Run: pip install ptflops")
        return None, None

    macs, params = get_model_complexity_info(
        model, input_shape,
        as_strings=True,
        print_per_layer_stat=False,
        verbose=False
    )
    return macs, params


def main():
    from tabulate import tabulate

    rows = []
    for exp in EXPERIMENTS:
        print(f"Processing {exp}...")
        try:
            model = load_model(exp)
            model.eval()

            # Total params
            total_params = sum(p.numel() for p in model.parameters())
            total_params_M = total_params / 1e6

            # MACs via ptflops
            # Input to Impl.forward: (B, T, C, H, W) → strip batch dim for ptflops
            input_shape = (CLIP_LEN, 3, H, W)
            macs_str, _ = count_macs_ptflops(model, input_shape)

            rows.append([exp, f"{total_params_M:.2f}M", macs_str or "—"])
        except Exception as e:
            rows.append([exp, "ERROR", str(e)])

    headers = ["Model", "Params", "MACs (one clip)"]
    print("\n" + tabulate(rows, headers=headers, tablefmt="grid"))


if __name__ == '__main__':
    main()
