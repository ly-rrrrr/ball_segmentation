#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))
from filter_sam_masks_by_ball_seeds import read_png_gray

FEATURE_NAMES = [
    "log_area",
    "log_bbox_w",
    "log_bbox_h",
    "log_bbox_max",
    "log_bbox_min",
    "aspect_minmax",
    "extent",
    "log_crop_side",
    "area_over_crop_area",
    "bbox_max_over_crop_side",
]


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        return torch, nn, F, DataLoader, Dataset
    except Exception as e:
        raise SystemExit("PyTorch is required. Install torch before running this script.") from e


def parse_box(raw: str) -> List[float]:
    try:
        v = json.loads(raw or "[]")
        if len(v) >= 4:
            return [float(v[0]), float(v[1]), float(v[2]), float(v[3])]
    except Exception:
        pass
    return [0.0, 0.0, 1.0, 1.0]


def raw_features(row: Dict[str, Any]) -> List[float]:
    box = parse_box(str(row.get("upscaled_bbox_xyxy", "")))
    w = max(1.0, box[2] - box[0])
    h = max(1.0, box[3] - box[1])
    bmax = max(w, h)
    bmin = min(w, h)
    try:
        area = max(1.0, float(row.get("area") or 0.0))
    except Exception:
        area = 1.0
    try:
        crop_side = max(1.0, float(row.get("crop_side_upscaled") or 0.0))
    except Exception:
        crop_side = 1.0
    extent = area / max(1.0, w * h)
    aspect = bmin / max(1.0, bmax)
    return [
        math.log1p(area),
        math.log1p(w),
        math.log1p(h),
        math.log1p(bmax),
        math.log1p(bmin),
        aspect,
        extent,
        math.log1p(crop_side),
        area / max(1.0, crop_side * crop_side),
        bmax / max(1.0, crop_side),
    ]


def feature_stats(rows: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
    vals = [raw_features(r) for r in rows]
    n = max(1, len(vals))
    means = [sum(v[i] for v in vals) / n for i in range(len(FEATURE_NAMES))]
    stds = []
    for i, m in enumerate(means):
        var = sum((v[i] - m) ** 2 for v in vals) / n
        stds.append(max(1e-6, math.sqrt(var)))
    return means, stds


def normalize_features(row: Dict[str, Any], means: List[float], stds: List[float]) -> List[float]:
    vals = raw_features(row)
    return [(v - means[i]) / stds[i] for i, v in enumerate(vals)]


def read_rows(csv_path: Path, allowed: set[str], split: str = "train") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("label") not in allowed:
                continue
            if split and row.get("split", "train") != split:
                continue
            rows.append(row)
    return rows


def split_rows(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_label.setdefault(str(r["label"]), []).append(r)
    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for items in by_label.values():
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio))) if len(items) > 10 else 0
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def make_dataset_class(torch, Dataset):
    class ScaleAwareDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]], class_to_idx: Dict[str, int], means: List[float], stds: List[float], *, augment: bool, bg_noise: float, bg_jitter: float) -> None:
            self.rows = rows
            self.class_to_idx = class_to_idx
            self.means = means
            self.stds = stds
            self.augment = augment
            self.bg_noise = float(bg_noise)
            self.bg_jitter = float(bg_jitter)

        def __len__(self) -> int:
            return len(self.rows)

        @staticmethod
        def normalize_channel(x):
            return (x - x.mean()) / x.std().clamp_min(1e-6)

        def make_view(self, crop, mask, masked, *, perturb_bg: bool):
            if perturb_bg:
                outside = mask < 0.5
                inside = mask >= 0.5
                base = crop[inside].mean() if inside.any() else crop.mean()
                bg_level = (base + (torch.rand((), dtype=crop.dtype) - 0.5) * self.bg_jitter).clamp(0.0, 1.0)
                noise = torch.randn_like(crop) * self.bg_noise
                aug_crop = crop.clone()
                aug_crop[outside] = (bg_level + noise[outside]).clamp(0.0, 1.0)
            else:
                aug_crop = crop
            return torch.stack([self.normalize_channel(aug_crop), mask, self.normalize_channel(masked)], dim=0)

        def __getitem__(self, idx: int):
            row = self.rows[idx]
            w, h, crop_raw = read_png_gray(Path(row["image_path"]))
            mw, mh, mask_raw = read_png_gray(Path(row["mask_path"]))
            masked_path = row.get("masked_path") or row.get("image_path")
            kw, kh, masked_raw = read_png_gray(Path(masked_path))
            if w != mw or h != mh or w != kw or h != kh:
                raise ValueError(f"crop/mask/masked size mismatch: {row['sample_id']}")
            crop = torch.tensor([float(v) / 255.0 for v in crop_raw], dtype=torch.float32).reshape(h, w)
            mask = torch.tensor([float(v) / 255.0 for v in mask_raw], dtype=torch.float32).reshape(h, w)
            mask = (mask > 0.5).float()
            masked = torch.tensor([float(v) / 255.0 for v in masked_raw], dtype=torch.float32).reshape(h, w)
            feat = torch.tensor(normalize_features(row, self.means, self.stds), dtype=torch.float32)
            x_clean = self.make_view(crop, mask, masked, perturb_bg=False)
            x_aug = self.make_view(crop, mask, masked, perturb_bg=self.augment)
            y = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
            return x_clean, x_aug, feat, y
    return ScaleAwareDataset


def make_model_class(nn, F):
    class ScaleAwarePrototypeCNN(nn.Module):
        def __init__(self, num_classes: int, feature_dim: int, embedding_dim: int = 64, input_channels: int = 3) -> None:
            super().__init__()
            self.cnn = nn.Sequential(
                nn.Conv2d(input_channels, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.feature_mlp = nn.Sequential(
                nn.Linear(feature_dim, 32), nn.LayerNorm(32), nn.ReLU(inplace=True),
                nn.Linear(32, 32), nn.ReLU(inplace=True),
            )
            self.embedding = nn.Sequential(
                nn.Linear(96 + 32, embedding_dim), nn.ReLU(inplace=True),
                nn.Linear(embedding_dim, embedding_dim),
            )
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, x, feat):
            z_img = self.cnn(x).flatten(1)
            z_feat = self.feature_mlp(feat)
            emb = F.normalize(self.embedding(torch.cat([z_img, z_feat], dim=1)), dim=1)
            logits = self.classifier(emb)
            return logits, emb
    return ScaleAwarePrototypeCNN


def evaluate(torch, model, loader, device) -> Dict[str, float]:
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    total = correct = 0
    loss_sum = 0.0
    per_label: Dict[int, List[int]] = {}
    with torch.no_grad():
        for x, _xa, feat, y in loader:
            x, feat, y = x.to(device), feat.to(device), y.to(device)
            logits, _ = model(x, feat)
            loss = ce(logits, y)
            pred = logits.argmax(1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
            loss_sum += float(loss.item()) * int(y.numel())
            for yy, pp in zip(y.detach().cpu().tolist(), pred.detach().cpu().tolist()):
                per_label.setdefault(int(yy), [0, 0])
                per_label[int(yy)][1] += 1
                per_label[int(yy)][0] += int(yy == pp)
    out = {"loss": loss_sum / max(1, total), "accuracy": correct / max(1, total), "count": float(total)}
    out["balanced_accuracy"] = sum(v[0] / max(1, v[1]) for v in per_label.values()) / max(1, len(per_label))
    return out


def compute_prototypes(torch, model, loader, device, num_classes: int):
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, _xa, feat, y in loader:
            x, feat, y = x.to(device), feat.to(device), y.to(device)
            _logits, emb = model(x, feat)
            if sums is None:
                sums = torch.zeros(num_classes, emb.shape[1], dtype=torch.float32, device=device)
            for c in range(num_classes):
                m = y == c
                if m.any():
                    sums[c] += emb[m].sum(0)
                    counts[c] += float(m.sum().item())
    if sums is None:
        raise RuntimeError("no embeddings for prototypes")
    protos = sums / counts.clamp_min(1.0).unsqueeze(1)
    return torch.nn.functional.normalize(protos, dim=1), counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Train v4.1 scale-aware CNN prototype classifier.")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--classes", default="real_ball,shadow_ball,double_ball_cluster,shadow_double_ball_cluster,single_ball_background_mixed,interference")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--consistency-weight", type=float, default=0.20)
    ap.add_argument("--aug-ce-weight", type=float, default=0.5)
    ap.add_argument("--bg-noise", type=float, default=0.18)
    ap.add_argument("--bg-jitter", type=float, default=0.45)
    args = ap.parse_args()

    torch, nn, F, DataLoader, Dataset = require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    classes = [x.strip() for x in args.classes.split(",") if x.strip()]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    rows = read_rows(Path(args.labels_csv), set(classes), args.train_split)
    if not rows:
        raise SystemExit("no labeled rows found")
    train_rows, val_rows = split_rows(rows, args.val_ratio, args.seed)
    means, stds = feature_stats(train_rows)

    DatasetClass = make_dataset_class(torch, Dataset)
    train_ds = DatasetClass(train_rows, class_to_idx, means, stds, augment=True, bg_noise=args.bg_noise, bg_jitter=args.bg_jitter)
    val_ds = DatasetClass(val_rows, class_to_idx, means, stds, augment=False, bg_noise=args.bg_noise, bg_jitter=args.bg_jitter)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Model = make_model_class(nn, F)
    model = Model(len(classes), len(FEATURE_NAMES), args.embedding_dim, input_channels=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    counts = torch.zeros(len(classes), dtype=torch.float32)
    for r in train_rows:
        counts[class_to_idx[r["label"]]] += 1.0
    class_weights = (counts.sum() / counts.clamp_min(1.0) / float(len(classes))).to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    history: List[Dict[str, Any]] = []
    best_score = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = correct = 0
        loss_sum = ce_sum = cons_sum = 0.0
        for x, xa, feat, y in train_loader:
            x, xa, feat, y = x.to(device), xa.to(device), feat.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, emb = model(x, feat)
            logits_aug, emb_aug = model(xa, feat)
            ce_clean = ce(logits, y)
            ce_aug = ce(logits_aug, y)
            consistency = (1.0 - (emb * emb_aug).sum(dim=1)).mean()
            loss = ce_clean + args.aug_ce_weight * ce_aug + args.consistency_weight * consistency
            loss.backward()
            opt.step()
            n = int(y.numel())
            total += n
            correct += int((logits.argmax(1) == y).sum().item())
            loss_sum += float(loss.item()) * n
            ce_sum += float(ce_clean.item()) * n
            cons_sum += float(consistency.item()) * n
        train_metrics = {"loss": loss_sum / max(1, total), "ce_loss": ce_sum / max(1, total), "consistency_loss": cons_sum / max(1, total), "accuracy": correct / max(1, total), "count": float(total)}
        val_metrics = evaluate(torch, model, val_loader, device) if val_rows else {"loss":0.0,"accuracy":0.0,"balanced_accuracy":0.0,"count":0.0}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        score = float(val_metrics.get("balanced_accuracy", val_metrics.get("accuracy", 0.0)))
        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)

    proto_loader = DataLoader(DatasetClass(train_rows, class_to_idx, means, stds, augment=False, bg_noise=args.bg_noise, bg_jitter=args.bg_jitter), batch_size=args.batch_size, shuffle=False, num_workers=0)
    prototypes, proto_counts = compute_prototypes(torch, model, proto_loader, device, len(classes))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cnn_scale_aware_model.pt"
    ckpt = {
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "embedding_dim": args.embedding_dim,
        "input_channels": 3,
        "feature_names": FEATURE_NAMES,
        "feature_means": means,
        "feature_stds": stds,
        "prototypes": prototypes.detach().cpu(),
        "prototype_counts": proto_counts.detach().cpu(),
        "history": history,
        "params": vars(args),
        "version": "v4.1_scale_aware_bg_consistency",
    }
    torch.save(ckpt, model_path)
    summary = {
        "model_path": str(model_path),
        "version": ckpt["version"],
        "classes": classes,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "feature_names": FEATURE_NAMES,
        "feature_means": means,
        "feature_stds": stds,
        "class_weights": class_weights.detach().cpu().tolist(),
        "best_val_balanced_accuracy": best_score,
        "best_val_accuracy": max((h["val"].get("accuracy", 0.0) for h in history), default=0.0),
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(model_path), "best_val_balanced_accuracy": best_score}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
