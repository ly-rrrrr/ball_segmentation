#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image, ImageOps


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


def load_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    recs = summary.get('mask_records') or []
    if recs:
        return [dict(r) for r in recs]
    out = []
    for tile in summary.get('tiles', []):
        for r in tile.get('mask_records', []):
            rr = dict(r)
            rr.setdefault('tile_id', tile.get('tile_id'))
            rr.setdefault('box_xyxy', tile.get('box_xyxy'))
            out.append(rr)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description='Export inference CNN dataset from split-upscale SAM summary.')
    ap.add_argument('--summary', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--crop-size', type=int, default=64)
    ap.add_argument('--context-scale', type=float, default=3.5)
    ap.add_argument('--min-crop-side', type=int, default=128)
    ap.add_argument('--placeholder-label', default='interference')
    args = ap.parse_args()

    summary_path = Path(args.summary).resolve()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    out_root = Path(args.output_dir).resolve()
    rows: List[Dict[str, Any]] = []
    tile_image_by_id = {str(t.get('tile_id')): str(t.get('saved_sam_input_image')) for t in summary.get('tiles', []) if t.get('saved_sam_input_image')}
    tile_cache: Dict[str, Image.Image] = {}
    for i, rec in enumerate(load_records(summary)):
        tile_id = str(rec.get('tile_id'))
        tile_path = Path(tile_image_by_id.get(tile_id, ''))
        mask_path = Path(str(rec.get('upscaled_mask_file') or ''))
        if not tile_path.is_file() or not mask_path.is_file():
            continue
        if tile_id not in tile_cache:
            tile_cache[tile_id] = Image.open(tile_path).convert('L')
        tile_img = tile_cache[tile_id]
        mask_img = Image.open(mask_path).convert('L')
        if tile_img.size != mask_img.size:
            continue
        bbox = mask_img.getbbox()
        if bbox is None:
            continue
        cx, cy = mask_centroid(mask_img, bbox)
        x0, y0, x1, y1 = bbox
        side = max(args.min_crop_side, int(round(max(x1-x0, y1-y0) * args.context_scale)))
        crop = square_crop(tile_img, cx, cy, side, fill=0).resize((args.crop_size, args.crop_size), Image.Resampling.BILINEAR)
        msk = square_crop(mask_img, cx, cy, side, fill=0).resize((args.crop_size, args.crop_size), Image.Resampling.NEAREST)
        msk = msk.point(lambda v: 255 if v > 0 else 0)
        crop_arr = np.asarray(crop, dtype=np.uint8)
        mask_arr = (np.asarray(msk, dtype=np.uint8) > 0).astype(np.uint8)
        masked = Image.fromarray((crop_arr * mask_arr).astype(np.uint8), mode='L')
        split = 'infer'
        sample_id = f'infer_{i:05d}_{tile_id}_{int(rec.get("mask_index", -1)):04d}'
        img_dir = out_root / 'images' / split
        mask_dir = out_root / 'masks' / split
        masked_dir = out_root / 'masked' / split
        for d in (img_dir, mask_dir, masked_dir):
            d.mkdir(parents=True, exist_ok=True)
        image_path = img_dir / f'{sample_id}.png'
        mask_out = mask_dir / f'{sample_id}.png'
        masked_out = masked_dir / f'{sample_id}.png'
        crop.save(image_path)
        msk.save(mask_out)
        masked.save(masked_out)
        rows.append({
            'sample_id': sample_id, 'split': split, 'label': args.placeholder_label,
            'source': 'inference', 'image_path': str(image_path), 'mask_path': str(mask_out), 'masked_path': str(masked_out),
            'tile_id': tile_id, 'mask_index': rec.get('mask_index', ''), 'area': rec.get('area', ''), 'sam_score': rec.get('sam_score', ''),
            'box_xyxy': json.dumps(rec.get('box_xyxy', []), separators=(',', ':')),
            'original_mask_file': rec.get('mask_file', ''), 'upscaled_mask_file': str(mask_path),
            'sam_input_image': str(tile_path), 'upscaled_bbox_xyxy': json.dumps([x0,y0,x1,y1], separators=(',', ':')),
            'crop_side_upscaled': side, 'candidate_id': '',
        })
    fields = ['sample_id','split','label','source','image_path','mask_path','masked_path','tile_id','mask_index','area','sam_score','box_xyxy','original_mask_file','upscaled_mask_file','sam_input_image','upscaled_bbox_xyxy','crop_side_upscaled','candidate_id']
    csv_path = out_root / 'labels.csv'
    out_root.mkdir(parents=True, exist_ok=True)
    with csv_path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    meta = {'labels_csv': str(csv_path), 'num_infer': len(rows), 'summary': str(summary_path), 'crop_size': args.crop_size, 'context_scale': args.context_scale, 'min_crop_side': args.min_crop_side}
    (out_root / 'dataset_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
