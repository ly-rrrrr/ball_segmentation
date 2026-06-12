#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

LABELS = [
    "real_ball",
    "shadow_ball",
    "double_ball_cluster",
    "shadow_double_ball_cluster",
    "single_ball_background_mixed",
    "interference",
]


def rel_url(path: Path, viewer_dir: Path) -> str:
    return str(Path("../") / path.resolve().relative_to(viewer_dir.parent.resolve())).replace("\\", "/")


def read_predictions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def load_records(summary_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    recs = data.get("mask_records") or []
    if not recs:
        for tile in data.get("tiles", []):
            recs.extend(tile.get("mask_records", []))
    return {(str(r.get("tile_id")), int(r.get("mask_index", -1))): dict(r) for r in recs}


def mask_bbox(mask_path: Path) -> List[int] | None:
    if not mask_path.is_file():
        return None
    im = Image.open(mask_path).convert("L")
    box = im.getbbox()
    if box is None:
        return None
    return [int(v) for v in box]


def select_group(rows: List[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    if len(rows) <= n:
        rows = rows[:]
        rng.shuffle(rows)
        return rows
    # Keep a mixture of high-confidence examples and decision-boundary examples.
    rows_sorted = sorted(rows, key=lambda r: float(r.get("keep_probability") or 0.0), reverse=True)
    high = rows_sorted[: max(1, n // 3)]
    mid_pool = sorted(rows, key=lambda r: abs(float(r.get("keep_probability") or 0.0) - 0.75))
    mid = []
    seen = {id(r) for r in high}
    for r in mid_pool:
        if id(r) not in seen:
            mid.append(r)
        if len(mid) >= max(1, n // 3):
            break
    rest_pool = [r for r in rows if id(r) not in seen and all(id(r) != id(m) for m in mid)]
    rng.shuffle(rest_pool)
    out = high + mid + rest_pool[: max(0, n - len(high) - len(mid))]
    rng.shuffle(out)
    return out[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build six-class 2K annotation candidates from existing SAM/CNN outputs.")
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--result-root", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-csv", required=True)
    ap.add_argument("--per-pred-class", type=int, default=120)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    image_root = Path(args.image_root).resolve()
    result_root = Path(args.result_root).resolve()
    repo_root = Path(__file__).resolve().parents[1]
    viewer_dir = repo_root / "annotation" / "viewers"
    all_items: List[Dict[str, Any]] = []
    image_paths = sorted(p for p in image_root.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"})
    for image_path in image_paths:
        sample = image_path.stem
        sample_root = result_root / sample
        summary_path = sample_root / "sam_split9_upscale" / f"{sample}_sam_auto_summary.json"
        pred_path = sample_root / "cnn_v3_upscaled_t075_mincrop256" / "cnn_prototype_predictions.csv"
        if not summary_path.is_file() or not pred_path.is_file():
            continue
        records = load_records(summary_path)
        predictions = read_predictions(pred_path)
        overlay = sample_root / "sam_split9_upscale" / f"{sample}_sam_auto_overlay.png"
        cnn_overlay = sample_root / "cnn_v3_upscaled_t075_mincrop256" / "cnn_filtered_class_overlay.png"
        for pred in predictions:
            key = (str(pred.get("tile_id")), int(pred.get("mask_index", -1)))
            rec = records.get(key)
            if rec is None:
                continue
            pred_label = str(pred.get("pred_label", ""))
            if pred_label not in LABELS:
                pred_label = "interference"
            try:
                keep_probability = float(pred.get("keep_probability") or 0.0)
            except Exception:
                keep_probability = 0.0
            try:
                pred_confidence = float(pred.get("pred_confidence") or 0.0)
            except Exception:
                pred_confidence = 0.0
            try:
                area = float(pred.get("area") or rec.get("area") or 0.0)
            except Exception:
                area = 0.0
            mask_path = Path(str(rec.get("mask_file") or pred.get("mask_file") or ""))
            upscaled_mask = Path(str(rec.get("upscaled_mask_file") or pred.get("upscaled_mask_file") or ""))
            mask_bbox_local = mask_bbox(mask_path)
            upscaled_bbox = mask_bbox(upscaled_mask)
            if mask_bbox_local is None:
                continue
            tile_box = [int(v) for v in rec.get("box_xyxy", [0, 0, 0, 0])]
            tile_x0, tile_y0 = tile_box[0], tile_box[1]
            bbox = [
                tile_x0 + mask_bbox_local[0],
                tile_y0 + mask_bbox_local[1],
                tile_x0 + mask_bbox_local[2],
                tile_y0 + mask_bbox_local[3],
            ]
            mask_size = Image.open(mask_path).size if mask_path.is_file() else None
            upscaled_mask_size = Image.open(upscaled_mask).size if upscaled_mask.is_file() else None
            item = {
                "sample": sample,
                "image": str(image_path),
                "image_url": rel_url(image_path, viewer_dir),
                "sam_overlay_url": rel_url(overlay, viewer_dir) if overlay.is_file() else "",
                "cnn_overlay_url": rel_url(cnn_overlay, viewer_dir) if cnn_overlay.is_file() else "",
                "tile_id": key[0],
                "mask_index": key[1],
                "area": area,
                "sam_score": rec.get("sam_score", pred.get("sam_score", "")),
                "area_ratio": rec.get("area_ratio", ""),
                "bbox_xyxy": bbox,
                "bbox_w": bbox[2] - bbox[0],
                "bbox_h": bbox[3] - bbox[1],
                "mask_bbox_xyxy": mask_bbox_local,
                "upscaled_bbox_xyxy": upscaled_bbox or [],
                "tile_box_xyxy": tile_box,
                "tile_mask_size": list(mask_size) if mask_size else [],
                "sam_input_mask_size": list(upscaled_mask_size) if upscaled_mask_size else [],
                "mask_file": rec.get("mask_file", ""),
                "upscaled_mask_file": str(upscaled_mask),
                "mask_file_url": rel_url(Path(str(rec.get("mask_file"))), viewer_dir) if rec.get("mask_file") else "",
                "upscaled_mask_file_url": rel_url(upscaled_mask, viewer_dir),
                "pred_label": pred_label,
                "pred_confidence": pred_confidence,
                "keep_probability": keep_probability,
                "selection_rule": f"pred_{pred_label}",
            }
            for lab in LABELS:
                key_prob = f"prob_{lab}"
                if key_prob in pred:
                    item[key_prob] = pred[key_prob]
            all_items.append(item)

    by_label: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in LABELS}
    for item in all_items:
        by_label[item["pred_label"]].append(item)

    selected: List[Dict[str, Any]] = []
    for i, lab in enumerate(LABELS):
        selected.extend(select_group(by_label.get(lab, []), args.per_pred_class, args.seed + i))

    # Add additional near-boundary hard examples across all labels.
    existing = {(x["sample"], x["tile_id"], x["mask_index"]) for x in selected}
    hard_pool = sorted(all_items, key=lambda r: abs(float(r.get("keep_probability") or 0.0) - 0.75))
    for item in hard_pool:
        key = (item["sample"], item["tile_id"], item["mask_index"])
        if key in existing:
            continue
        selected.append(item)
        existing.add(key)
        if len(selected) >= args.per_pred_class * len(LABELS) + 120:
            break

    rng = random.Random(args.seed)
    rng.shuffle(selected)
    for idx, item in enumerate(selected):
        item["candidate_id"] = idx

    output_json = Path(args.output_json).resolve()
    output_csv = Path(args.output_csv).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "task": "2k_six_class_mask_annotation",
        "image_root": str(image_root),
        "result_root": str(result_root),
        "label_schema": LABELS,
        "candidate_count": len(selected),
        "raw_candidate_count": len(all_items),
        "pred_label_counts_raw": {lab: len(by_label.get(lab, [])) for lab in LABELS},
        "pred_label_counts_selected": {lab: sum(1 for x in selected if x.get("pred_label") == lab) for lab in LABELS},
        "note": "Candidates are sampled across all 2K split9 SAM masks from current v3 predictions. Labels are manual ground truth, not pred_label.",
        "candidates": selected,
    }
    output_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["candidate_id", "sample", "tile_id", "mask_index", "pred_label", "keep_probability", "area", "bbox_xyxy", "mask_bbox_xyxy", "upscaled_bbox_xyxy", "tile_box_xyxy", "selection_rule"]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for item in selected:
            wr.writerow({k: item.get(k, "") for k in fields})
    print(json.dumps({k: data[k] for k in ["candidate_count", "raw_candidate_count", "pred_label_counts_raw", "pred_label_counts_selected"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
