from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

from mtmc.models import DGVehicleNet


def build_transform(image_size: tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, index_csv: str | Path, image_size: tuple[int, int]):
        self.index = pd.read_csv(index_csv)
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        x = self.transform(img)
        label = int(row["obj_id"]) if "obj_id" in row else int(row["id"])
        return x, label


def extract_features(model: torch.nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device)
            out = model(x)
            f = out["id_feat"].cpu().numpy()
            feats.append(f)
            labels.extend(y.numpy().tolist())

    feats = np.vstack(feats)
    labels = np.array(labels, dtype=int)
    return feats, labels


def compute_metrics(feats: np.ndarray, labels: np.ndarray, topk=(1, 5, 10)):
    # Compute distances in chunks to avoid building a full NxN array in memory.
    n, d = feats.shape
    norms = np.sum(feats * feats, axis=1)

    aps = []
    cmc_counts = {k: 0 for k in topk}
    valid_q = 0

    # Choose a reasonable default chunk size; caller can pre-split if needed.
    chunk_size = 256
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        q = feats[start:end]  # (qbs, d)
        q_norms = np.sum(q * q, axis=1)

        # dot product between queries and all features -> (qbs, n)
        dots = q @ feats.T
        # squared distances
        d2 = q_norms[:, None] + norms[None, :] - 2.0 * dots
        d2 = np.maximum(d2, 0.0)
        dist_chunk = np.sqrt(d2)

        for i_local in range(end - start):
            i = start + i_local
            dist_row = dist_chunk[i_local]
            order = np.argsort(dist_row)
            order = order[order != i]

            matches = (labels[order] == labels[i]).astype(int)
            num_pos = matches.sum()
            if num_pos == 0:
                continue
            valid_q += 1

            cum = np.cumsum(matches)
            precision_at_k = cum / (np.arange(len(matches)) + 1)
            ap = (precision_at_k * matches).sum() / num_pos
            aps.append(ap)

            for k in topk:
                if matches[:k].sum() > 0:
                    cmc_counts[k] += 1

    mAP = float(np.mean(aps)) if len(aps) > 0 else 0.0
    cmc = {k: (cmc_counts[k] / max(1, valid_q)) for k in topk}
    return mAP, cmc, valid_q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index-csv", required=True, help="CSV with columns: crop_path,obj_id")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    num_classes = int(ckpt.get("num_classes", 1000))
    id_dim = int(ckpt.get("id_dim", 256))
    style_dim = int(ckpt.get("style_dim", 64))
    backbone = ckpt.get("backbone", "resnet18")
    pretrained_backbone = bool(ckpt.get("pretrained_backbone", False))
    image_size = tuple(ckpt.get("image_size", [256, 256]))

    model = DGVehicleNet(
        num_classes=num_classes,
        id_dim=id_dim,
        style_dim=style_dim,
        backbone_name=backbone,
        pretrained_backbone=pretrained_backbone,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)

    ds = EvalDataset(args.index_csv, image_size=tuple(image_size))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    feats, labels = extract_features(model, dl, device)

    mAP, cmc, valid_q = compute_metrics(feats, labels)

    print(f"Evaluated queries with positives: {valid_q}/{len(labels)}")
    print(f"mAP: {mAP:.4f}")
    for k, v in cmc.items():
        print(f"CMC@{k}: {v:.4f}")


if __name__ == "__main__":
    main()
