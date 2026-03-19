from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ReIDCropDataset(Dataset):
    def __init__(self, index_csv: str | Path, image_size: tuple[int, int] = (256, 256)) -> None:
        self.index = pd.read_csv(index_csv)
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        unique_ids = sorted(self.index["obj_id"].unique().tolist())
        self.id_to_label = {oid: i for i, oid in enumerate(unique_ids)}

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]
        img = Image.open(row["crop_path"]).convert("RGB")
        x = self.transform(img)
        label = self.id_to_label[int(row["obj_id"])]
        cam_id = int(str(row["camera"])[1:])
        return x, label, cam_id
