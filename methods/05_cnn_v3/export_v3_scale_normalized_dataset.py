#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))

from filter_sam_masks_by_ball_seeds import load_mask_records

KEEP_LABELS = {"real_ball", "shadow_ball", "double_ball_cluster", "shadow_double_ball_cluster"}


def square_crop(img: Image.Image, cx: float, cy: float, side: int, fill: int = 0) -> Image.Image:
    side = max(1, int(side))
    x0 = int(round(cx - side / 2.0)); y0 = int(round(cy - side / 2.0))
    x1 = x0 + side; y1 = y0 + side
    pad_l = max(0, -x0); pad_t = max(0, -y0); pad_r = max(0, x1 - img.width); pad_b = max(0, y1 - img.height)
    if pad_l or pad_t or pad_r or pad_b:
        img = ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=fill)
        x0 += pad_l; y0 += pad_t; x1 += pad_l; y1 += pad_t
    return img.crop((x0, y0, x1, y1))


def mask_centroid(mask_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    arr = np.asarray(mask_img.crop(bbox), dtype=np.uint8) > 0
    ys, xs = np.nonzero(arr)
    if len(xs) == 0:
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return float(xs.mean() + x0), float(ys.mean() + y0)


def ref_object_ratio(labels_csv: Path, keep_labels: set[str]) -> Dict[str, float]:
    ratios: List[float] = []
    bbox_maxes: List[float] = []
    crop_sides: List[float] = []
    with labels_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("split") != "train" or row.get("label") not in keep_labels:
                continue
            try:
                box = json.loads(row["upscaled_bbox_xyxy"])
                bbox_max = max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
                crop_side = float(row["crop_side_upscaled"])
            except Exception:
                continue
            if bbox_max > 0 and crop_side > 0:
                ratios.append(bbox_max / crop_side)
                bbox_maxes.append(bbox_max)
                crop_sides.append(crop_side)
    if not ratios:
        raise SystemExit("no reference keep-label rows found")
    return {
        "object_ratio_median": float(statistics.median(ratios)),
        "bbox_max_median": float(statistics.median(bbox_maxes)),
        "crop_side_median": float(statistics.median(crop_sides)),
        "num_reference": len(ratios),
    }


def export_one(out_root: Path, split: str, sample_id: str, rec: Dict[str, Any], tile_image_by_id: Dict[str, str], image_cache: Dict[str, Image.Image], crop_size: int, object_ratio: float, min_crop_side: int, max_crop_side: int, label: str = "interference") -> Dict[str, Any] | None:
    tile_id = str(rec.get("tile_id"))
    tile_path = tile_image_by_id.get(tile_id)
    if not tile_path or not Path(tile_path).is_file():
        return None
    if tile_id not in image_cache:
        image_cache[tile_id] = Image.open(tile_path).convert("L")
    tile_img = image_cache[tile_id]
    mask_path = Path(str(rec.get("upscaled_mask_file") or ""))
    if not mask_path.is_file():
        return None
    mask_img = Image.open(mask_path).convert("L")
    if mask_img.size != tile_img.size:
        return None
    mask_img = mask_img.point(lambda v: 255 if v > 0 else 0)
    bbox = mask_img.getbbox()
    if bbox is None:
        return None
    cx, cy = mask_centroid(mask_img, bbox)
    x0, y0, x1, y1 = bbox
    bbox_max = max(x1 - x0, y1 - y0)
    side = int(round(bbox_max / max(1e-6, object_ratio)))
    side = max(int(min_crop_side), min(int(max_crop_side), side))
    crop = square_crop(tile_img, cx, cy, side, fill=0).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    msk = square_crop(mask_img, cx, cy, side, fill=0).resize((crop_size, crop_size), Image.Resampling.NEAREST)
    msk = msk.point(lambda v: 255 if v > 0 else 0)
    crop_arr = np.asarray(crop, dtype=np.uint8)
    mask_arr = (np.asarray(msk, dtype=np.uint8) > 0).astype(np.uint8)
    masked = Image.fromarray((crop_arr * mask_arr).astype(np.uint8), mode="L")
    img_dir = out_root / "images" / split; mask_dir = out_root / "masks" / split; masked_dir = out_root / "masked" / split
    for d in (img_dir, mask_dir, masked_dir): d.mkdir(parents=True, exist_ok=True)
    image_path = img_dir / f"{sample_id}.png"; mask_out = mask_dir / f"{sample_id}.png"; masked_out = masked_dir / f"{sample_id}.png"
    crop.save(image_path); msk.save(mask_out); masked.save(masked_out)
    return {
        "sample_id": sample_id, "split": split, "label": label, "source": "scale_normalized_inference",
        "image_path": str(image_path), "mask_path": str(mask_out), "masked_path": str(masked_out),
        "tile_id": tile_id, "mask_index": rec.get("mask_index", ""), "area": rec.get("area", ""), "sam_score": rec.get("sam_score", rec.get("predicted_iou", "")),
        "box_xyxy": json.dumps(rec.get("box_xyxy", []), separators=(",", ":")), "original_mask_file": rec.get("mask_file", ""),
        "upscaled_mask_file": str(mask_path), "sam_input_image": tile_path, "upscaled_bbox_xyxy": json.dumps([x0, y0, x1, y1], separators=(",", ":")),
        "crop_side_upscaled": side, "scale_normalized_bbox_max": bbox_max, "scale_normalized_object_ratio": object_ratio, "candidate_id": "",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export v3-compatible CNN crops with object-size normalization.")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--reference-labels-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--object-ratio", type=float, default=0.0, help="Target bbox_max/crop_side. 0 means infer median from reference labels.")
    ap.add_argument("--min-crop-side", type=int, default=96)
    ap.add_argument("--max-crop-side", type=int, default=512)
    ap.add_argument("--keep-labels", default=",".join(sorted(KEEP_LABELS)))
    args = ap.parse_args()

    keep_labels = {x.strip() for x in args.keep_labels.split(",") if x.strip()}
    ref = ref_object_ratio(Path(args.reference_labels_csv), keep_labels)
    object_ratio = float(args.object_ratio) if args.object_ratio > 0 else ref["object_ratio_median"]
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    records = load_mask_records(summary)
    tile_image_by_id = {str(t.get("tile_id")): str(t.get("saved_sam_input_image")) for t in summary.get("tiles", []) if t.get("saved_sam_input_image")}
    out_root = Path(args.output_dir).resolve(); out_root.mkdir(parents=True, exist_ok=True)
    image_cache: Dict[str, Image.Image] = {}
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        key = (str(rec.get("tile_id")), int(rec.get("mask_index", -1)))
        sample_id = f"infer_{i:05d}_{key[0]}_{key[1]:04d}"
        row = export_one(out_root, "infer", sample_id, rec, tile_image_by_id, image_cache, args.crop_size, object_ratio, args.min_crop_side, args.max_crop_side)
        if row:
            rows.append(row)
    fields = ["sample_id", "split", "label", "source", "image_path", "mask_path", "masked_path", "tile_id", "mask_index", "area", "sam_score", "box_xyxy", "original_mask_file", "upscaled_mask_file", "sam_input_image", "upscaled_bbox_xyxy", "crop_side_upscaled", "scale_normalized_bbox_max", "scale_normalized_object_ratio", "candidate_id"]
    csv_path = out_root / "labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows([{k: r.get(k, "") for k in fields} for r in rows])
    sides = sorted(float(r["crop_side_upscaled"]) for r in rows) if rows else []
    meta = {"labels_csv": str(csv_path), "num_infer": len(rows), "reference": ref, "object_ratio_used": object_ratio, "min_crop_side": args.min_crop_side, "max_crop_side": args.max_crop_side, "crop_side_median": statistics.median(sides) if sides else None, "crop_side_q25": sides[len(sides)//4] if sides else None, "crop_side_q75": sides[len(sides)*3//4] if sides else None}
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
