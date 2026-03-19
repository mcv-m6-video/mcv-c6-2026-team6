from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mtmc.data import ReIDCropDataset
from mtmc.models import BatchHardTripletLoss, DGVehicleNet
from mtmc.utils import load_yaml, set_seed


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    mean = MEAN.to(x.device)
    std = STD.to(x.device)
    return (x * std + mean).clamp(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DG-style ReID model for CityFlow")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backbone",
        default=None,
        choices=["resnet18", "resnet34"],
        help="Override config reid.backbone",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("H", "W"),
        help="Override config reid.image_size",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))

    reid_cfg = cfg["reid"]
    image_size = tuple(args.image_size) if args.image_size is not None else tuple(reid_cfg["image_size"])
    backbone_name = str(args.backbone if args.backbone is not None else reid_cfg.get("backbone", "resnet18"))
    pretrained_backbone = bool(reid_cfg.get("pretrained_backbone", False))

    ds = ReIDCropDataset(args.train_index, image_size=image_size)
    dl = DataLoader(
        ds,
        batch_size=int(reid_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(reid_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
    )

    num_classes = len(ds.id_to_label)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DGVehicleNet(
        num_classes=num_classes,
        id_dim=int(reid_cfg["id_dim"]),
        style_dim=int(reid_cfg["style_dim"]),
        backbone_name=backbone_name,
        pretrained_backbone=pretrained_backbone,
    ).to(device)

    ce_loss = nn.CrossEntropyLoss()
    triplet_loss = BatchHardTripletLoss(margin=float(reid_cfg["margin"]))

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(reid_cfg["lr"]),
        weight_decay=float(reid_cfg["weight_decay"]),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    for epoch in range(1, int(reid_cfg["epochs"]) + 1):
        model.train()
        running = 0.0
        count = 0

        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for imgs, labels, _cam_ids in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss_id = ce_loss(out["logits"], labels)
            loss_tri = triplet_loss(out["id_feat"], labels)

            recon_target = F.interpolate(unnormalize(imgs), size=(64, 64), mode="bilinear", align_corners=False)
            loss_rec = F.mse_loss(out["recon"], recon_target)

            loss = (
                float(reid_cfg["lambda_id"]) * loss_id
                + float(reid_cfg["lambda_triplet"]) * loss_tri
                + float(reid_cfg["lambda_recon"]) * loss_rec
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            count += 1
            pbar.set_postfix(loss=f"{running / max(1, count):.4f}")

        epoch_loss = running / max(1, count)
        ckpt = {
            "model": model.state_dict(),
            "num_classes": num_classes,
            "id_dim": int(reid_cfg["id_dim"]),
            "style_dim": int(reid_cfg["style_dim"]),
            "backbone": backbone_name,
            "pretrained_backbone": pretrained_backbone,
            "image_size": list(image_size),
            "epoch": epoch,
            "train_loss": epoch_loss,
        }
        torch.save(ckpt, out_dir / "last.pt")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(ckpt, out_dir / "best.pt")

        print(f"Epoch {epoch}: loss={epoch_loss:.4f}, best={best_loss:.4f}")


if __name__ == "__main__":
    main()
