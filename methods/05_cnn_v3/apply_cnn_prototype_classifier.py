#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "04_learned_seed"))
from filter_sam_masks_by_ball_seeds import read_bmp_rgb_gray, read_png_gray, write_png
from filter_sam_masks_by_learned_seed_arc_shape import save_class_overlay, class_map_to_png
from train_cnn_prototype_classifier import make_dataset_class, make_model_class, normalize_channel, require_torch


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def softmax(vals: List[float]) -> List[float]:
    m = max(vals) if vals else 0.0
    exps = [math.exp(v - m) for v in vals]
    s = sum(exps)
    return [v / max(1e-12, s) for v in exps]


def render_overlays(image_path: Path, summary_path: Path, predictions: Dict[Tuple[str, int], Dict[str, Any]], output_dir: Path, keep_classes: set[str], keep_threshold: float, max_keep_area: int = 1000) -> Dict[str, str]:
    width, height, rgb, _gray = read_bmp_rgb_gray(image_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    union = bytearray(width * height)
    class_map = bytearray(width * height)
    reject_rgb = bytearray(rgb)
    keep_count = reject_count = area_reject_count = 0
    for rec in summary.get("mask_records", []):
        key = (str(rec.get("tile_id")), int(rec.get("mask_index", -1)))
        pred = predictions.get(key)
        if pred is None:
            continue
        base_keep = pred["pred_label"] in keep_classes and float(pred["keep_probability"]) >= keep_threshold
        try:
            area_value = int(float(pred.get("area") or rec.get("area") or 0))
        except Exception:
            area_value = 0
        area_rejected = max_keep_area > 0 and area_value > max_keep_area
        keep = base_keep and not area_rejected
        if keep:
            keep_count += 1
        else:
            reject_count += 1
            if base_keep and area_rejected:
                area_reject_count += 1
        p = Path(str(rec.get("mask_file")))
        if not p.is_file():
            continue
        mw, mh, mask = read_png_gray(p)
        tx0, ty0, tx1, ty1 = [int(v) for v in rec.get("box_xyxy", [0, 0, width, height])]
        cls = 2 if pred["pred_label"] in {"double_ball_cluster", "shadow_double_ball_cluster", "multi_ball_cluster"} else 1
        for y in range(mh):
            gy = ty0 + int((y + 0.5) * max(1, ty1 - ty0) / float(mh))
            if not (0 <= gy < height):
                continue
            moff = y * mw
            uoff = gy * width
            for x in range(mw):
                if mask[moff + x] <= 0:
                    continue
                gx = tx0 + int((x + 0.5) * max(1, tx1 - tx0) / float(mw))
                if not (0 <= gx < width):
                    continue
                i = uoff + gx
                if keep:
                    union[i] = 255
                    if cls == 2 or class_map[i] == 0:
                        class_map[i] = cls
                else:
                    j = i * 3
                    color = (255, 70, 70)
                    reject_rgb[j] = int(reject_rgb[j] * 0.45 + color[0] * 0.55)
                    reject_rgb[j + 1] = int(reject_rgb[j + 1] * 0.45 + color[1] * 0.55)
                    reject_rgb[j + 2] = int(reject_rgb[j + 2] * 0.45 + color[2] * 0.55)
    output_dir.mkdir(parents=True, exist_ok=True)
    union_png = output_dir / "cnn_filtered_union.png"
    class_map_png = output_dir / "cnn_filtered_class_map.png"
    class_overlay_png = output_dir / "cnn_filtered_class_overlay.png"
    reject_overlay_png = output_dir / "cnn_rejected_overlay.png"
    write_png(union_png, width, height, union, 0)
    write_png(class_map_png, width, height, class_map_to_png(class_map), 0)
    save_class_overlay(rgb, class_map, width, height, class_overlay_png)
    write_png(reject_overlay_png, width, height, reject_rgb, 2)
    return {
        "cnn_filtered_union": str(union_png),
        "cnn_filtered_class_map": str(class_map_png),
        "cnn_filtered_class_overlay": str(class_overlay_png),
        "cnn_rejected_overlay": str(reject_overlay_png),
        "num_kept": str(keep_count),
        "num_rejected": str(reject_count),
        "num_area_rejected": str(area_reject_count),
        "max_keep_area": str(max_keep_area),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Apply CNN prototype classifier to exported mask crops.")
    ap.add_argument("--model", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--image")
    ap.add_argument("--summary")
    ap.add_argument("--keep-threshold", type=float, default=0.55)
    ap.add_argument("--keep-classes", default="real_ball,shadow_ball,double_ball_cluster")
    ap.add_argument("--prototype-weight", type=float, default=0.55)
    ap.add_argument("--max-keep-area", type=int, default=1000, help="Reject kept masks whose summary area is larger than this value; set 0 to disable.")
    args = ap.parse_args()

    torch, nn, F, DataLoader, Dataset = require_torch()
    ckpt = torch.load(Path(args.model), map_location="cpu")
    classes = list(ckpt["classes"])
    class_to_idx = dict(ckpt["class_to_idx"])
    PrototypeCNN = make_model_class(nn, F)
    model = PrototypeCNN(len(classes), int(ckpt.get("embedding_dim", 64)), input_channels=int(ckpt.get("input_channels", 3)))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    prototypes = ckpt["prototypes"]
    rows = read_csv_rows(Path(args.labels_csv))
    CropDataset = make_dataset_class(torch, Dataset)
    ds = CropDataset(rows, class_to_idx={c: i for i, c in enumerate(classes)})
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)
    keep_classes = {x.strip() for x in args.keep_classes.split(",") if x.strip()}
    predictions: Dict[Tuple[str, int], Dict[str, Any]] = {}
    out_rows: List[Dict[str, Any]] = []
    offset = 0
    with torch.no_grad():
        for x, _y in loader:
            logits, emb = model(x)
            clf_prob = torch.softmax(logits, dim=1)
            sim = torch.matmul(emb, prototypes.t())
            proto_prob = torch.softmax(sim * 8.0, dim=1)
            prob = (1.0 - args.prototype_weight) * clf_prob + args.prototype_weight * proto_prob
            vals, inds = prob.max(dim=1)
            for b in range(x.shape[0]):
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
                    key = (str(row.get("tile_id")), int(row.get("mask_index", -1)))
                    predictions[key] = row
                except Exception:
                    pass
            offset += x.shape[0]
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = output_dir / "cnn_prototype_predictions.csv"
    fields = list(out_rows[0].keys()) if out_rows else []
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)
    outputs: Dict[str, str] = {"predictions_csv": str(pred_csv)}
    if args.image and args.summary:
        outputs.update(render_overlays(Path(args.image).resolve(), Path(args.summary).resolve(), predictions, output_dir, keep_classes, args.keep_threshold, args.max_keep_area))
    summary = {
        "model": str(Path(args.model).resolve()),
        "labels_csv": str(Path(args.labels_csv).resolve()),
        "keep_threshold": args.keep_threshold,
        "keep_classes": sorted(keep_classes),
        "max_keep_area": args.max_keep_area,
        "num_predictions": len(out_rows),
        **outputs,
    }
    (output_dir / "cnn_prototype_apply_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
