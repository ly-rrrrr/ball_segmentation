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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))
from filter_sam_masks_by_ball_seeds import read_png_gray


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, Dataset
        return torch, nn, F, DataLoader, Dataset
    except Exception as e:
        raise SystemExit("PyTorch is required. Install torch before running this script.") from e


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


def normalize_channel(vals: List[float]) -> List[float]:
    m = sum(vals) / float(max(1, len(vals)))
    var = sum((v - m) * (v - m) for v in vals) / float(max(1, len(vals)))
    std = max(1e-6, math.sqrt(var))
    return [(v - m) / std for v in vals]


def make_dataset_class(torch, Dataset):
    class CropDataset(Dataset):
        def __init__(self, rows: List[Dict[str, Any]], class_to_idx: Dict[str, int]) -> None:
            self.rows = rows
            self.class_to_idx = class_to_idx

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, idx: int):
            row = self.rows[idx]
            w, h, crop = read_png_gray(Path(row["image_path"]))
            mw, mh, mask = read_png_gray(Path(row["mask_path"]))
            masked_path = row.get("masked_path") or row.get("image_path")
            kw, kh, masked = read_png_gray(Path(masked_path))
            if w != mw or h != mh or w != kw or h != kh:
                raise ValueError(f"crop/mask/masked size mismatch: {row['sample_id']}")
            channels = []
            crop_vals = [float(v) / 255.0 for v in crop]
            channels.append(normalize_channel(crop_vals))
            channels.append([float(v) / 255.0 for v in mask])
            masked_vals = [float(v) / 255.0 for v in masked]
            channels.append(normalize_channel(masked_vals))
            extra_triplets = [
                ("image_path_upscaled", "mask_path_upscaled", "masked_path_upscaled"),
            ]
            for img_key, mask_key, masked_key in extra_triplets:
                if row.get(img_key) and row.get(mask_key) and row.get(masked_key):
                    ew, eh, ecrop = read_png_gray(Path(row[img_key]))
                    emw, emh, emask = read_png_gray(Path(row[mask_key]))
                    ekw, ekh, emasked = read_png_gray(Path(row[masked_key]))
                    if ew != w or eh != h or emw != w or emh != h or ekw != w or ekh != h:
                        raise ValueError(f"multiscale crop size mismatch: {row['sample_id']}")
                    channels.append(normalize_channel([float(v) / 255.0 for v in ecrop]))
                    channels.append([float(v) / 255.0 for v in emask])
                    channels.append(normalize_channel([float(v) / 255.0 for v in emasked]))
            x = torch.tensor(channels, dtype=torch.float32).reshape(len(channels), h, w)
            y = torch.tensor(self.class_to_idx[row["label"]], dtype=torch.long)
            return x, y
    return CropDataset


def make_model_class(nn, F):
    class PrototypeCNN(nn.Module):
        def __init__(self, num_classes: int, embedding_dim: int = 64, input_channels: int = 3) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.Conv2d(24, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.embedding = nn.Linear(96, embedding_dim)
            self.classifier = nn.Linear(embedding_dim, num_classes)

        def forward(self, x):
            z = self.features(x).flatten(1)
            emb = F.normalize(self.embedding(z), dim=1)
            logits = self.classifier(emb)
            return logits, emb
    return PrototypeCNN


def evaluate(torch, model, loader, device) -> Dict[str, float]:
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    ce = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _emb = model(x)
            loss = ce(logits, y)
            pred = logits.argmax(1)
            total += int(y.numel())
            correct += int((pred == y).sum().item())
            loss_sum += float(loss.item()) * int(y.numel())
    return {"loss": loss_sum / float(max(1, total)), "accuracy": correct / float(max(1, total)), "count": float(total)}


def compute_prototypes(torch, model, loader, device, num_classes: int):
    model.eval()
    sums = None
    counts = torch.zeros(num_classes, dtype=torch.float32, device=device)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _logits, emb = model(x)
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
    protos = torch.nn.functional.normalize(protos, dim=1)
    return protos, counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a CNN embedding model and class prototypes for ball mask crops.")
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding-dim", type=int, default=64)
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--classes", default="real_ball,shadow_ball,double_ball_cluster,interference")
    ap.add_argument("--train-split", default="train")
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
    input_channels = 6 if rows and rows[0].get("image_path_upscaled") else 3
    CropDataset = make_dataset_class(torch, Dataset)
    train_ds = CropDataset(train_rows, class_to_idx)
    val_ds = CropDataset(val_rows, class_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PrototypeCNN = make_model_class(nn, F)
    model = PrototypeCNN(len(classes), args.embedding_dim, input_channels=input_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    history: List[Dict[str, Any]] = []
    best_acc = -1.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = loss_sum = correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits, _emb = model(x)
            loss = ce(logits, y)
            loss.backward()
            opt.step()
            total += int(y.numel())
            loss_sum += float(loss.item()) * int(y.numel())
            correct += int((logits.argmax(1) == y).sum().item())
        train_metrics = {"loss": loss_sum / float(max(1, total)), "accuracy": correct / float(max(1, total)), "count": float(total)}
        val_metrics = evaluate(torch, model, val_loader, device) if val_rows else {"loss": 0.0, "accuracy": 0.0, "count": 0.0}
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["accuracy"] >= best_acc:
            best_acc = val_metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(json.dumps(history[-1], ensure_ascii=False), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    proto_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    prototypes, proto_counts = compute_prototypes(torch, model, proto_loader, device, len(classes))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "cnn_prototype_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "embedding_dim": args.embedding_dim,
        "input_channels": input_channels,
        "prototypes": prototypes.detach().cpu(),
        "prototype_counts": proto_counts.detach().cpu(),
        "history": history,
        "params": vars(args),
    }, model_path)
    (output_dir / "training_summary.json").write_text(json.dumps({
        "model_path": str(model_path),
        "classes": classes,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "input_channels": input_channels,
        "best_val_accuracy": best_acc,
        "history": history,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(model_path), "best_val_accuracy": best_acc}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
