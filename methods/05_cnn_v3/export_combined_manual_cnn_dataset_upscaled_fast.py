#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))

from filter_sam_masks_by_ball_seeds import load_mask_records


def parse_label_rows(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            cid = str(row.get('candidate_id', '')).strip()
            if cid:
                out[cid] = row
    return out


def load_candidates(path: Path) -> Dict[str, Dict[str, Any]]:
    data = json.loads(path.read_text(encoding='utf-8'))
    return {str(c.get('candidate_id')): dict(c) for c in data.get('candidates', [])}


def source_pairs(raw: str) -> List[Tuple[Path, Path, str]]:
    out: List[Tuple[Path, Path, str]] = []
    for item in raw.split(';'):
        item = item.strip()
        if not item:
            continue
        parts = item.split(':')
        if len(parts) != 3:
            raise SystemExit(f'bad --source-pair item: {item}; expected name:candidates_json:labels_csv')
        name, candidates, labels = parts
        out.append((Path(candidates).resolve(), Path(labels).resolve(), name))
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


def mask_centroid_from_arr(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    sub = arr[y0:y1, x0:x1] > 0
    ys, xs = np.nonzero(sub)
    if len(xs) == 0:
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    return float(xs.mean() + x0), float(ys.mean() + y0)


def export_sample(
    out_root: Path,
    split: str,
    sample_id: str,
    tile_cache: Dict[str, Image.Image],
    tile_image_by_id: Dict[str, str],
    rec: Dict[str, Any],
    label: str,
    source: str,
    crop_size: int,
    context_scale: float,
    min_crop_side: int,
) -> Dict[str, Any] | None:
    tile_id = str(rec.get('tile_id'))
    tile_path = tile_image_by_id.get(tile_id)
    if not tile_path or not Path(tile_path).is_file():
        return None
    if tile_id not in tile_cache:
        tile_cache[tile_id] = Image.open(tile_path).convert('L')
    tile_img = tile_cache[tile_id]
    mask_path = Path(str(rec.get('upscaled_mask_file') or ''))
    if not mask_path.is_file():
        return None
    mask_img = Image.open(mask_path).convert('L')
    if mask_img.size != tile_img.size:
        return None
    bbox = mask_img.getbbox()
    if bbox is None:
        return None
    arr = np.asarray(mask_img)
    cx, cy = mask_centroid_from_arr(arr, bbox)
    x0, y0, x1, y1 = bbox
    side = max(min_crop_side, int(round(max(x1 - x0, y1 - y0) * context_scale)))
    crop = square_crop(tile_img, cx, cy, side, fill=0).resize((crop_size, crop_size), Image.Resampling.BILINEAR)
    msk = square_crop(mask_img, cx, cy, side, fill=0).resize((crop_size, crop_size), Image.Resampling.NEAREST)
    msk = msk.point(lambda v: 255 if v > 0 else 0)
    crop_arr = np.asarray(crop, dtype=np.uint8)
    mask_arr = (np.asarray(msk, dtype=np.uint8) > 0).astype(np.uint8)
    masked_arr = (crop_arr * mask_arr).astype(np.uint8)
    masked = Image.fromarray(masked_arr, mode='L')

    img_dir = out_root / 'images' / split
    mask_dir = out_root / 'masks' / split
    masked_dir = out_root / 'masked' / split
    for d in (img_dir, mask_dir, masked_dir):
        d.mkdir(parents=True, exist_ok=True)
    image_path = img_dir / f'{sample_id}.png'
    mask_out_path = mask_dir / f'{sample_id}.png'
    masked_path = masked_dir / f'{sample_id}.png'
    crop.save(image_path)
    msk.save(mask_out_path)
    masked.save(masked_path)
    return {
        'sample_id': sample_id,
        'split': split,
        'label': label,
        'source': source,
        'image_path': str(image_path),
        'mask_path': str(mask_out_path),
        'masked_path': str(masked_path),
        'tile_id': tile_id,
        'mask_index': rec.get('mask_index', ''),
        'area': rec.get('area', ''),
        'sam_score': rec.get('sam_score', ''),
        'box_xyxy': json.dumps(rec.get('box_xyxy', []), separators=(',', ':')),
        'original_mask_file': rec.get('mask_file', ''),
        'upscaled_mask_file': str(mask_path),
        'sam_input_image': tile_path,
        'upscaled_bbox_xyxy': json.dumps([x0, y0, x1, y1], separators=(',', ':')),
        'crop_side_upscaled': side,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Fast export CNN dataset from SAM upscaled tile images and upscaled masks.')
    ap.add_argument('--summary', required=True)
    ap.add_argument('--source-pair', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--crop-size', type=int, default=64)
    ap.add_argument('--context-scale', type=float, default=3.5)
    ap.add_argument('--min-crop-side', type=int, default=128)
    ap.add_argument('--export-inference-set', action='store_true')
    ap.add_argument('--inference-placeholder-label', default='interference')
    args = ap.parse_args()

    out_root = Path(args.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary = json.loads(Path(args.summary).read_text(encoding='utf-8'))
    records = load_mask_records(summary)
    by_key = {(str(r.get('tile_id')), int(r.get('mask_index', -1))): r for r in records}
    tile_image_by_id = {str(t.get('tile_id')): str(t.get('saved_sam_input_image')) for t in summary.get('tiles', []) if t.get('saved_sam_input_image')}
    tile_cache: Dict[str, Image.Image] = {}

    valid_labels = {'real_ball', 'shadow_ball', 'double_ball_cluster', 'shadow_double_ball_cluster', 'single_ball_background_mixed', 'interference'}
    rows: List[Dict[str, Any]] = []
    seen: Dict[Tuple[str, int], str] = {}
    skipped_conflicts: List[Dict[str, Any]] = []
    for candidates_path, labels_path, source_name in source_pairs(args.source_pair):
        candidates = load_candidates(candidates_path)
        labels = parse_label_rows(labels_path)
        for cid, lab_row in sorted(labels.items(), key=lambda kv: int(kv[0])):
            label = str(lab_row.get('label', '')).strip()
            if label not in valid_labels:
                continue
            cand = candidates.get(cid)
            if not cand:
                continue
            key = (str(cand.get('tile_id')), int(cand.get('mask_index', -1)))
            if key in seen and seen[key] != label:
                skipped_conflicts.append({'tile_id': key[0], 'mask_index': key[1], 'old_label': seen[key], 'new_label': label, 'source': source_name})
                continue
            seen[key] = label
            rec = by_key.get(key)
            if rec is None:
                continue
            sample_id = f'{source_name}_{int(cid):05d}_{key[0]}_{key[1]:04d}'
            row = export_sample(out_root, 'train', sample_id, tile_cache, tile_image_by_id, rec, label, source_name, args.crop_size, args.context_scale, args.min_crop_side)
            if row:
                row['candidate_id'] = cid
                rows.append(row)

    if args.export_inference_set:
        for i, rec in enumerate(records):
            key = (str(rec.get('tile_id')), int(rec.get('mask_index', -1)))
            sample_id = f'infer_{i:05d}_{key[0]}_{key[1]:04d}'
            row = export_sample(out_root, 'infer', sample_id, tile_cache, tile_image_by_id, rec, args.inference_placeholder_label, 'inference_unlabeled', args.crop_size, args.context_scale, args.min_crop_side)
            if row:
                rows.append(row)

    fields = ['sample_id', 'split', 'label', 'source', 'image_path', 'mask_path', 'masked_path', 'tile_id', 'mask_index', 'area', 'sam_score', 'box_xyxy', 'original_mask_file', 'upscaled_mask_file', 'sam_input_image', 'upscaled_bbox_xyxy', 'crop_side_upscaled', 'candidate_id']
    csv_path = out_root / 'labels.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, '') for k in fields})
    train_rows = [r for r in rows if r['split'] == 'train']
    meta = {
        'labels_csv': str(csv_path),
        'num_samples': len(rows),
        'num_train': len(train_rows),
        'num_infer': sum(1 for r in rows if r['split'] == 'infer'),
        'label_counts': {lab: sum(1 for r in train_rows if r['label'] == lab) for lab in sorted({r['label'] for r in train_rows})},
        'source_counts': {src: sum(1 for r in train_rows if r['source'] == src) for src in sorted({r['source'] for r in train_rows})},
        'crop_size': args.crop_size,
        'context_scale': args.context_scale,
        'min_crop_side': args.min_crop_side,
        'image_source': 'saved_sam_input_image_upscaled_tile',
        'mask_source': 'upscaled_mask_file',
        'skipped_conflicts': skipped_conflicts,
    }
    (out_root / 'dataset_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
