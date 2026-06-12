#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

VALID_LABELS = {
    "real_ball",
    "shadow_ball",
    "double_ball_cluster",
    "shadow_double_ball_cluster",
    "single_ball_background_mixed",
    "interference",
}


def parse_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = str(row.get("candidate_id", "")).strip()
            label = str(row.get("label", "")).strip()
            if cid and label in VALID_LABELS:
                out[cid] = row
    return out


def square_crop(img: Image.Image, cx: float, cy: float, side: int, fill: int = 0) -> Image.Image:
    side = max(1, int(side))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side
    pad_l = max(0, -x0)
    pad_t = max(0, -y0)
    pad_r = max(0, x1 - img.width)
    pad_b = max(0, y1 - img.height)
    if pad_l or pad_t or pad_r or pad_b:
        img = ImageOps.expand(img, border=(pad_l, pad_t, pad_r, pad_b), fill=fill)
        x0 += pad_l
        y0 += pad_t
        x1 += pad_l
        y1 += pad_t
    return img.crop((x0, y0, x1, y1))


def mask_centroid(mask_img: Image.Image, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    arr = np.asarray(mask_img, dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    sub = arr[y0:y1, x0:x1] > 0
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return float(xs.mean() + x0), float(ys.mean() + y0)


def derive_tile_image(candidate: Dict[str, Any]) -> Path:
    upscaled_mask = Path(str(candidate.get("upscaled_mask_file") or ""))
    sample = str(candidate.get("sample"))
    tile_id = str(candidate.get("tile_id"))
    sam_dir = upscaled_mask.parent.parent
    return sam_dir / f"{sample}_sam_auto_split_tiles" / f"{tile_id}_upscaled.png"


def export_one(out_root: Path, candidate: Dict[str, Any], label: str, crop_size: int, context_scale: float, min_crop_side: int) -> Dict[str, Any] | None:
    cid = str(candidate.get("candidate_id"))
    sample = str(candidate.get("sample"))
    tile_id = str(candidate.get("tile_id"))
    mask_index = int(candidate.get("mask_index", -1))
    tile_path = derive_tile_image(candidate)
    mask_path = Path(str(candidate.get("upscaled_mask_file") or ""))
    if not tile_path.is_file() or not mask_path.is_file():
        return None
    tile_img = Image.open(tile_path).convert("L")
    mask_img = Image.open(mask_path).convert("L")
    if tile_img.size != mask_img.size:
        return None
    bbox = mask_img.getbbox()
    if bbox is None:
        return None
    cx, cy = mask_centroid(mask_img, bbox)
    x0, y0, x1, y1 = bbox
    side = max(min_crop_side, int(round(max(x1 - x0, y1 - y0) * context_scale)))
    crop = square_crop(tile_img, cx, cy, side, fill=0).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    msk = square_crop(mask_img, cx, cy, side, fill=0).resize((crop_size, crop_size), Image.Resampling.NEAREST)
    msk = msk.point(lambda v: 255 if v > 0 else 0)
    crop_arr = np.asarray(crop, dtype=np.uint8)
    mask_arr = (np.asarray(msk, dtype=np.uint8) > 0).astype(np.uint8)
    masked = Image.fromarray((crop_arr * mask_arr).astype(np.uint8), mode="L")

    sample_id = f"ball2k_{int(cid):05d}_s{sample}_{tile_id}_{mask_index:04d}"
    img_dir = out_root / "images" / "train"
    mask_dir = out_root / "masks" / "train"
    masked_dir = out_root / "masked" / "train"
    for d in (img_dir, mask_dir, masked_dir):
        d.mkdir(parents=True, exist_ok=True)
    image_path = img_dir / f"{sample_id}.png"
    mask_out_path = mask_dir / f"{sample_id}.png"
    masked_path = masked_dir / f"{sample_id}.png"
    crop.save(image_path)
    msk.save(mask_out_path)
    masked.save(masked_path)
    return {
        "sample_id": sample_id,
        "split": "train",
        "label": label,
        "source": "ball2k_manual",
        "image_path": str(image_path),
        "mask_path": str(mask_out_path),
        "masked_path": str(masked_path),
        "tile_id": tile_id,
        "mask_index": mask_index,
        "area": candidate.get("area", ""),
        "sam_score": candidate.get("sam_score", ""),
        "box_xyxy": json.dumps(candidate.get("tile_box_xyxy", []), separators=(",", ":")),
        "original_mask_file": candidate.get("mask_file", ""),
        "upscaled_mask_file": str(mask_path),
        "sam_input_image": str(tile_path),
        "upscaled_bbox_xyxy": json.dumps([x0, y0, x1, y1], separators=(",", ":")),
        "global_bbox_xyxy": json.dumps(candidate.get("bbox_xyxy", []), separators=(",", ":")),
        "local_mask_bbox_xyxy": json.dumps(candidate.get("mask_bbox_xyxy", []), separators=(",", ":")),
        "crop_side_upscaled": side,
        "candidate_id": cid,
        "sample": sample,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Export 2K six-class manual CNN dataset from candidate JSON and manual labels CSV.")
    ap.add_argument("--candidates-json", required=True)
    ap.add_argument("--labels-csv", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--crop-size", type=int, default=64)
    ap.add_argument("--context-scale", type=float, default=3.5)
    ap.add_argument("--min-crop-side", type=int, default=128)
    args = ap.parse_args()

    data = json.loads(Path(args.candidates_json).read_text(encoding="utf-8"))
    all_candidates = list(data.get("candidates", [])) + list(data.get("legacy_candidates", []))
    candidates = {str(c.get("candidate_id")): dict(c) for c in all_candidates}
    labels = parse_labels(Path(args.labels_csv))
    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for cid, lab_row in sorted(labels.items(), key=lambda kv: int(kv[0])):
        cand = candidates.get(cid)
        if not cand:
            skipped.append({"candidate_id": cid, "reason": "candidate_not_found"})
            continue
        row = export_one(out_root, cand, str(lab_row["label"]).strip(), args.crop_size, args.context_scale, args.min_crop_side)
        if row is None:
            skipped.append({"candidate_id": cid, "reason": "export_failed"})
            continue
        rows.append(row)

    fields = [
        "sample_id", "split", "label", "source", "image_path", "mask_path", "masked_path", "tile_id", "mask_index",
        "area", "sam_score", "box_xyxy", "original_mask_file", "upscaled_mask_file", "sam_input_image",
        "upscaled_bbox_xyxy", "global_bbox_xyxy", "local_mask_bbox_xyxy", "crop_side_upscaled", "candidate_id", "sample",
    ]
    csv_path = out_root / "labels.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fields})
    meta = {
        "labels_csv": str(csv_path),
        "num_train": len(rows),
        "label_counts": {lab: sum(1 for r in rows if r["label"] == lab) for lab in sorted(VALID_LABELS)},
        "sample_counts": {s: sum(1 for r in rows if r["sample"] == s) for s in sorted({r["sample"] for r in rows}, key=lambda x: int(x))},
        "crop_size": args.crop_size,
        "context_scale": args.context_scale,
        "min_crop_side": args.min_crop_side,
        "image_source": "saved_sam_input_image_upscaled_tile",
        "mask_source": "upscaled_mask_file_sam_input_coordinates",
        "skipped": skipped,
    }
    (out_root / "dataset_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
