from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from mtmc.models import DGVehicleNet


def build_transform(image_size: tuple[int, int]):
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ReIDEvalDataset(Dataset):
    def __init__(self, index_csv: str | Path, image_size: tuple[int, int]):
        self.index = pd.read_csv(index_csv)
        self.transform = build_transform(image_size)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index.iloc[idx]
        crop_path = str(row["crop_path"])
        img = Image.open(crop_path).convert("RGB")
        x = self.transform(img)

        obj_id = int(row["obj_id"] if "obj_id" in row else row["id"])
        cam = str(row.get("camera", "unknown"))
        seq = str(row.get("sequence", "unknown"))
        return x, obj_id, cam, seq, crop_path


def collate_batch(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    obj_ids = np.array([b[1] for b in batch], dtype=np.int64)
    cams = [b[2] for b in batch]
    seqs = [b[3] for b in batch]
    paths = [b[4] for b in batch]
    return xs, obj_ids, cams, seqs, paths


def extract_features(model: torch.nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    feats = []
    obj_ids = []
    cams = []
    seqs = []
    paths = []

    with torch.no_grad():
        for x, y_obj, y_cam, y_seq, y_path in dl:
            x = x.to(device)
            out = model(x)
            feat = out["id_feat"].cpu().numpy()

            feats.append(feat)
            obj_ids.append(y_obj)
            cams.extend(y_cam)
            seqs.extend(y_seq)
            paths.extend(y_path)

    if not feats:
        raise RuntimeError("No features extracted. Check your index CSV.")

    all_feats = np.vstack(feats).astype(np.float32)
    all_obj_ids = np.concatenate(obj_ids).astype(np.int64)
    return all_feats, all_obj_ids, np.array(cams), np.array(seqs), np.array(paths)


def filter_and_subsample(
    feats: np.ndarray,
    obj_ids: np.ndarray,
    cams: np.ndarray,
    seqs: np.ndarray,
    paths: np.ndarray,
    min_samples_per_id: int,
    max_samples: int,
    seed: int,
):
    unique_ids, counts = np.unique(obj_ids, return_counts=True)
    keep_ids = set(unique_ids[counts >= min_samples_per_id].tolist())
    keep_mask = np.array([oid in keep_ids for oid in obj_ids], dtype=bool)

    feats = feats[keep_mask]
    obj_ids = obj_ids[keep_mask]
    cams = cams[keep_mask]
    seqs = seqs[keep_mask]
    paths = paths[keep_mask]

    if len(obj_ids) == 0:
        raise RuntimeError(
            "No samples left after filtering by --min-samples-per-id. "
            "Try a smaller threshold."
        )

    if max_samples > 0 and len(obj_ids) > max_samples:
        rng = np.random.default_rng(seed)
        unique_ids = np.unique(obj_ids)
        per_id_budget = max(1, max_samples // len(unique_ids))

        selected = []
        leftovers = []
        for oid in unique_ids:
            idxs = np.where(obj_ids == oid)[0]
            rng.shuffle(idxs)
            take = min(per_id_budget, len(idxs))
            selected.extend(idxs[:take].tolist())
            leftovers.extend(idxs[take:].tolist())

        selected = np.array(selected, dtype=np.int64)
        if len(selected) < max_samples and leftovers:
            leftovers = np.array(leftovers, dtype=np.int64)
            rng.shuffle(leftovers)
            needed = max_samples - len(selected)
            selected = np.concatenate([selected, leftovers[:needed]])

        selected = np.sort(selected)
        feats = feats[selected]
        obj_ids = obj_ids[selected]
        cams = cams[selected]
        seqs = seqs[selected]
        paths = paths[selected]

    return feats, obj_ids, cams, seqs, paths


def _l2_normalize(feats: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12
    return feats / norms


def extract_no_reid_rgb_hist_features(paths: np.ndarray, bins: int) -> np.ndarray:
    """Handcrafted baseline features with no ReID model: RGB histograms per crop."""
    feats = []
    for p in paths:
        img = Image.open(str(p)).convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
        parts = []
        for ch in range(3):
            hist, _ = np.histogram(arr[:, :, ch], bins=bins, range=(0, 256), density=True)
            parts.append(hist.astype(np.float32))
        feats.append(np.concatenate(parts, axis=0))
    return _l2_normalize(np.vstack(feats).astype(np.float32))


def project_tsne(feats: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    if len(feats) < 3:
        raise RuntimeError("Need at least 3 samples for t-SNE.")

    max_valid_perplexity = max(2.0, min(float(perplexity), float((len(feats) - 1) // 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=max_valid_perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
    )
    return tsne.fit_transform(feats)


def project_umap(feats: np.ndarray, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        import umap
    except ImportError as exc:
        raise RuntimeError(
            "UMAP requested but package 'umap-learn' is not installed. "
            "Install it with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    return reducer.fit_transform(feats)


def _camera_color_map(cams: np.ndarray):
    unique = sorted(np.unique(cams).tolist())
    cmap = cm.get_cmap("tab20")
    color_map = {cam: cmap(i % 20) for i, cam in enumerate(unique)}
    return color_map, unique


def save_plot(coords: np.ndarray, obj_ids: np.ndarray, cams: np.ndarray, out_path: Path, title: str, point_size: float, alpha: float):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=160)

    # Left: identity-colored embedding.
    n_ids = len(np.unique(obj_ids))
    axes[0].scatter(coords[:, 0], coords[:, 1], c=obj_ids, cmap="turbo", s=point_size, alpha=alpha, linewidths=0)
    axes[0].set_title(f"{title} | colored by vehicle ID ({n_ids} IDs)")
    axes[0].set_xlabel("dim-1")
    axes[0].set_ylabel("dim-2")

    # Right: camera-colored embedding.
    cam_colors, cams_unique = _camera_color_map(cams)
    for cam in cams_unique:
        mask = cams == cam
        axes[1].scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            color=cam_colors[cam],
            label=cam,
        )
    axes[1].set_title(f"{title} | colored by camera ({len(cams_unique)} cameras)")
    axes[1].set_xlabel("dim-1")
    axes[1].set_ylabel("dim-2")

    if len(cams_unique) <= 16:
        axes[1].legend(loc="best", frameon=True, fontsize=8)

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def save_comparison_plot(
    coords_reid: np.ndarray,
    coords_no_reid: np.ndarray,
    obj_ids: np.ndarray,
    out_path: Path,
    method_name: str,
    point_size: float,
    alpha: float,
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=170)
    n_ids = len(np.unique(obj_ids))

    axes[0].scatter(coords_no_reid[:, 0], coords_no_reid[:, 1], c=obj_ids, cmap="turbo", s=point_size, alpha=alpha, linewidths=0)
    axes[0].set_title(f"No ReID baseline ({method_name}) | {n_ids} IDs")
    axes[0].set_xlabel("dim-1")
    axes[0].set_ylabel("dim-2")

    axes[1].scatter(coords_reid[:, 0], coords_reid[:, 1], c=obj_ids, cmap="turbo", s=point_size, alpha=alpha, linewidths=0)
    axes[1].set_title(f"With ReID model ({method_name}) | {n_ids} IDs")
    axes[1].set_xlabel("dim-1")
    axes[1].set_ylabel("dim-2")

    fig.tight_layout()
    fig.savefig(str(out_path))
    plt.close(fig)


def build_model_from_checkpoint(ckpt_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, tuple[int, int]]:
    ckpt = torch.load(ckpt_path, map_location="cpu")

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
    return model, image_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize ReID embedding quality with t-SNE/UMAP")
    parser.add_argument("--checkpoint", required=True, help="Path to ReID checkpoint (best.pt)")
    parser.add_argument("--index-csv", required=True, help="CSV with columns: crop_path,obj_id,camera,sequence")
    parser.add_argument("--output-dir", default="./outputs/reid_embedding_viz", help="Where to save plots and CSVs")
    parser.add_argument("--method", choices=["tsne", "umap", "both"], default="both")
    parser.add_argument("--compare-no-reid", action="store_true", help="Also compute handcrafted no-ReID features and save side-by-side comparison")
    parser.add_argument("--no-reid-bins", type=int, default=16, help="Histogram bins per RGB channel for no-ReID baseline")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="cuda/cpu; auto if omitted")
    parser.add_argument("--max-samples", type=int, default=5000, help="0 keeps all samples")
    parser.add_argument("--min-samples-per-id", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)

    parser.add_argument("--point-size", type=float, default=8.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    model, image_size = build_model_from_checkpoint(args.checkpoint, device)

    print(f"[INFO] Loading index CSV: {args.index_csv}")
    ds = ReIDEvalDataset(args.index_csv, image_size=image_size)
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )

    feats, obj_ids, cams, seqs, paths = extract_features(model, dl, device)
    feats, obj_ids, cams, seqs, paths = filter_and_subsample(
        feats=feats,
        obj_ids=obj_ids,
        cams=cams,
        seqs=seqs,
        paths=paths,
        min_samples_per_id=args.min_samples_per_id,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    no_reid_feats = None
    if args.compare_no_reid:
        print("[INFO] Computing no-ReID baseline features (RGB histograms)...")
        no_reid_feats = extract_no_reid_rgb_hist_features(paths=paths, bins=args.no_reid_bins)

    print(f"[INFO] Samples used: {len(obj_ids)}")
    print(f"[INFO] Unique IDs: {len(np.unique(obj_ids))}")
    print(f"[INFO] Unique cameras: {len(np.unique(cams))}")

    if args.method in {"tsne", "both"}:
        print("[INFO] Computing t-SNE...")
        coords_tsne = project_tsne(feats, seed=args.seed, perplexity=args.tsne_perplexity)

        tsne_csv = out_dir / "tsne_points.csv"
        pd.DataFrame({
            "x": coords_tsne[:, 0],
            "y": coords_tsne[:, 1],
            "obj_id": obj_ids,
            "camera": cams,
            "sequence": seqs,
        }).to_csv(tsne_csv, index=False)

        tsne_png = out_dir / "tsne_embedding.png"
        save_plot(
            coords=coords_tsne,
            obj_ids=obj_ids,
            cams=cams,
            out_path=tsne_png,
            title="ReID embedding (t-SNE)",
            point_size=args.point_size,
            alpha=args.alpha,
        )
        print(f"[OK] Saved: {tsne_csv}")
        print(f"[OK] Saved: {tsne_png}")

        if no_reid_feats is not None:
            print("[INFO] Computing t-SNE for no-ReID baseline...")
            coords_tsne_no_reid = project_tsne(no_reid_feats, seed=args.seed, perplexity=args.tsne_perplexity)
            tsne_no_reid_csv = out_dir / "tsne_points_no_reid.csv"
            pd.DataFrame({
                "x": coords_tsne_no_reid[:, 0],
                "y": coords_tsne_no_reid[:, 1],
                "obj_id": obj_ids,
                "camera": cams,
                "sequence": seqs,
            }).to_csv(tsne_no_reid_csv, index=False)

            tsne_no_reid_png = out_dir / "tsne_embedding_no_reid.png"
            save_plot(
                coords=coords_tsne_no_reid,
                obj_ids=obj_ids,
                cams=cams,
                out_path=tsne_no_reid_png,
                title="No-ReID baseline embedding (t-SNE)",
                point_size=args.point_size,
                alpha=args.alpha,
            )

            tsne_cmp_png = out_dir / "tsne_reid_vs_no_reid.png"
            save_comparison_plot(
                coords_reid=coords_tsne,
                coords_no_reid=coords_tsne_no_reid,
                obj_ids=obj_ids,
                out_path=tsne_cmp_png,
                method_name="t-SNE",
                point_size=args.point_size,
                alpha=args.alpha,
            )
            print(f"[OK] Saved: {tsne_no_reid_csv}")
            print(f"[OK] Saved: {tsne_no_reid_png}")
            print(f"[OK] Saved: {tsne_cmp_png}")

    if args.method in {"umap", "both"}:
        print("[INFO] Computing UMAP...")
        coords_umap = project_umap(
            feats,
            seed=args.seed,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
        )

        umap_csv = out_dir / "umap_points.csv"
        pd.DataFrame({
            "x": coords_umap[:, 0],
            "y": coords_umap[:, 1],
            "obj_id": obj_ids,
            "camera": cams,
            "sequence": seqs,
        }).to_csv(umap_csv, index=False)

        umap_png = out_dir / "umap_embedding.png"
        save_plot(
            coords=coords_umap,
            obj_ids=obj_ids,
            cams=cams,
            out_path=umap_png,
            title="ReID embedding (UMAP)",
            point_size=args.point_size,
            alpha=args.alpha,
        )
        print(f"[OK] Saved: {umap_csv}")
        print(f"[OK] Saved: {umap_png}")

        if no_reid_feats is not None:
            print("[INFO] Computing UMAP for no-ReID baseline...")
            coords_umap_no_reid = project_umap(
                no_reid_feats,
                seed=args.seed,
                n_neighbors=args.umap_neighbors,
                min_dist=args.umap_min_dist,
            )

            umap_no_reid_csv = out_dir / "umap_points_no_reid.csv"
            pd.DataFrame({
                "x": coords_umap_no_reid[:, 0],
                "y": coords_umap_no_reid[:, 1],
                "obj_id": obj_ids,
                "camera": cams,
                "sequence": seqs,
            }).to_csv(umap_no_reid_csv, index=False)

            umap_no_reid_png = out_dir / "umap_embedding_no_reid.png"
            save_plot(
                coords=coords_umap_no_reid,
                obj_ids=obj_ids,
                cams=cams,
                out_path=umap_no_reid_png,
                title="No-ReID baseline embedding (UMAP)",
                point_size=args.point_size,
                alpha=args.alpha,
            )

            umap_cmp_png = out_dir / "umap_reid_vs_no_reid.png"
            save_comparison_plot(
                coords_reid=coords_umap,
                coords_no_reid=coords_umap_no_reid,
                obj_ids=obj_ids,
                out_path=umap_cmp_png,
                method_name="UMAP",
                point_size=args.point_size,
                alpha=args.alpha,
            )

            print(f"[OK] Saved: {umap_no_reid_csv}")
            print(f"[OK] Saved: {umap_no_reid_png}")
            print(f"[OK] Saved: {umap_cmp_png}")


if __name__ == "__main__":
    main()
