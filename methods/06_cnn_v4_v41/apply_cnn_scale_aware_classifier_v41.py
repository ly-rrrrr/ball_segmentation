#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "05_cnn_v3"))
from apply_cnn_prototype_classifier import render_overlays
from train_cnn_scale_aware_classifier_v41 import make_dataset_class, make_model_class, require_torch


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply v4.1 scale-aware CNN prototype classifier.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image")
    ap.add_argument("--summary")
    ap.add_argument("--keep-threshold", type=float, default=0.75)
    ap.add_argument("--keep-classes", default="real_ball,shadow_ball,double_ball_cluster,shadow_double_ball_cluster")
    ap.add_argument("--prototype-weight", type=float, default=0.35)
    ap.add_argument("--max-keep-area", type=int, default=1000)
    args = ap.parse_args()

    torch, nn, F, DataLoader, Dataset = require_torch()
    ckpt = torch.load(Path(args.model), map_location="cpu")
    classes = list(ckpt["classes"])
    class_to_idx = dict(ckpt["class_to_idx"])
    feature_names = list(ckpt["feature_names"])
    means = [float(x) for x in ckpt["feature_means"]]
    stds = [float(x) for x in ckpt["feature_stds"]]
    Model = make_model_class(nn, F)
    model = Model(len(classes), len(feature_names), int(ckpt.get("embedding_dim", 64)), input_channels=int(ckpt.get("input_channels", 3)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    prototypes = ckpt["prototypes"]

    rows = read_csv_rows(Path(args.labels_csv))
    DatasetClass = make_dataset_class(torch, Dataset)
    ds = DatasetClass(rows, {c: i for i, c in enumerate(classes)}, means, stds, augment=False, bg_noise=0.0, bg_jitter=0.0)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    keep_classes = {x.strip() for x in args.keep_classes.split(",") if x.strip()}
    predictions: Dict[Tuple[str, int], Dict[str, Any]] = {}
    out_rows: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for x_clean, _x_aug, feat, _y in loader:
            logits, emb = model(x_clean, feat)
            clf_prob = torch.softmax(logits, dim=1)
            sim = torch.matmul(emb, prototypes.t())
            proto_prob = torch.softmax(sim * 8.0, dim=1)
            prob = (1.0 - args.prototype_weight) * clf_prob + args.prototype_weight * proto_prob
            vals, inds = prob.max(dim=1)
            for b in range(x_clean.shape[0]):
                row = dict(rows[offset + b])
                pred_label = classes[int(inds[b].item())]
                keep_prob = sum(float(prob[b, class_to_idx[c]].item()) for c in keep_classes if c in class_to_idx)
                row.update({
                    "pred_label": pred_label,
                    "pred_confidence": float(vals[b].item()),
                    "keep_probability": keep_prob,
                })
                for i, c in enumerate(classes):
                    row[f"prob_{c}"] = float(prob[b, i].item())
                out_rows.append(row)
                try:
                    predictions[(str(row.get("tile_id")), int(row.get("mask_index", -1)))] = row
                except Exception:
                    pass
            offset += x_clean.shape[0]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = output_dir / "cnn_scale_aware_predictions.csv"
    fields = list(out_rows[0].keys()) if out_rows else []
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(out_rows)
    outputs: Dict[str, str] = {"predictions_csv": str(pred_csv)}
    if args.image and args.summary:
        outputs.update(render_overlays(Path(args.image).resolve(), Path(args.summary).resolve(), predictions, output_dir, keep_classes, args.keep_threshold, args.max_keep_area))
    summary = {
        "model": str(Path(args.model).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "keep_threshold": args.keep_threshold,
        "keep_classes": sorted(keep_classes),
        "prototype_weight": args.prototype_weight,
        "max_keep_area": args.max_keep_area,
        "num_predictions": len(out_rows),
        "version": "v4.1_scale_aware",
        **outputs,
    }
    (output_dir / "cnn_scale_aware_apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
