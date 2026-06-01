#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import pipeline


def resolve_device(device_arg: Optional[str]) -> str | int:
    if device_arg is not None:
        text = device_arg.strip().lower()
        if text in {'cpu', '-1'}:
            return 'cpu'
        if text.startswith('cuda:'):
            return text
        if text.isdigit():
            return int(text)
        return device_arg
    if torch.cuda.is_available():
        return 0
    return 'cpu'


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    binary = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    Image.fromarray(binary, mode='L').save(path)


def save_union_overlay(image: Image.Image, mask: np.ndarray, path: Path, alpha: float = 0.45) -> None:
    base = np.asarray(image.convert('RGB'), dtype=np.float32)
    binary = (np.asarray(mask, dtype=np.uint8) > 0)
    color = np.array([30.0, 144.0, 255.0], dtype=np.float32)
    if binary.any():
        base[binary] = base[binary] * (1.0 - alpha) + color * alpha
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode='RGB').save(path)


def summarize_outputs(outputs: Dict[str, Any]) -> Dict[str, Any]:
    masks = outputs.get('masks', []) or []
    iou_scores = outputs.get('iou_scores')
    raw_scores = outputs.get('scores')
    bbox_count = len(outputs.get('bounding_boxes', []) or [])

    areas: List[int] = []
    for mask in masks:
        arr = np.asarray(mask, dtype=np.uint8)
        areas.append(int(arr.sum()))

    summary: Dict[str, Any] = {
        'num_masks': len(masks),
        'num_bounding_boxes': bbox_count,
        'mask_areas': areas,
    }
    if iou_scores is not None:
        summary['iou_scores'] = [float(x) for x in iou_scores]
    if raw_scores is not None:
        try:
            summary['scores'] = [float(x) for x in raw_scores]
        except Exception:
            summary['scores'] = raw_scores
    return summary


def make_equal_tiles(width: int, height: int, split_count: int) -> List[Dict[str, int | str]]:
    grid_size = int(round(split_count ** 0.5))
    if grid_size * grid_size != split_count:
        raise ValueError(f'split count must be a square number, got: {split_count}')

    x_edges = [round(i * width / grid_size) for i in range(grid_size + 1)]
    y_edges = [round(i * height / grid_size) for i in range(grid_size + 1)]
    legacy_2x2_names = {
        (0, 0): 'top_left',
        (0, 1): 'top_right',
        (1, 0): 'bottom_left',
        (1, 1): 'bottom_right',
    }

    tiles: List[Dict[str, int | str]] = []
    for row in range(grid_size):
        for col in range(grid_size):
            tile_id = legacy_2x2_names[(row, col)] if grid_size == 2 else f'r{row}_c{col}'
            tiles.append({
                'tile_id': tile_id,
                'row': row,
                'col': col,
                'x0': x_edges[col],
                'y0': y_edges[row],
                'x1': x_edges[col + 1],
                'y1': y_edges[row + 1],
            })
    return tiles


def update_union_from_masks(
    masks: List[Any],
    union_mask: np.ndarray,
    *,
    area_reference: int,
    union_max_area_ratio: float,
    paste_box: Optional[Dict[str, int | str]] = None,
    mask_output_size: Optional[tuple[int, int]] = None,
    save_individual_masks: bool = False,
    mask_dir: Optional[Path] = None,
    save_upscaled_masks: bool = False,
    upscaled_mask_dir: Optional[Path] = None,
    mask_prefix: str = 'mask',
) -> Dict[str, Any]:
    mask_files: List[str] = []
    upscaled_mask_files: List[str] = []
    mask_records: List[Dict[str, Any]] = []
    kept_indices: List[int] = []
    filtered_large_indices: List[int] = []

    for idx, mask in enumerate(masks):
        input_mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)
        input_mask_area = int(input_mask_arr.sum())

        upscaled_mask_path: Optional[Path] = None
        if save_upscaled_masks and upscaled_mask_dir is not None:
            upscaled_mask_path = upscaled_mask_dir / f'{mask_prefix}_{idx:04d}_upscaled.png'
            save_binary_mask(input_mask_arr, upscaled_mask_path)
            upscaled_mask_files.append(str(upscaled_mask_path))

        mask_arr = input_mask_arr
        if mask_output_size is not None:
            out_w, out_h = int(mask_output_size[0]), int(mask_output_size[1])
            if mask_arr.shape != (out_h, out_w):
                mask_img = Image.fromarray(mask_arr * 255, mode='L')
                mask_img = mask_img.resize((out_w, out_h), Image.NEAREST)
                mask_arr = (np.asarray(mask_img, dtype=np.uint8) > 0).astype(np.uint8)

        mask_area = int(mask_arr.sum())
        mask_area_ratio = (mask_area / area_reference) if area_reference else 0.0

        union_kept = mask_area_ratio <= union_max_area_ratio
        if union_kept:
            kept_indices.append(idx)
            if paste_box is None:
                union_mask[:] = np.logical_or(union_mask > 0, mask_arr > 0).astype(np.uint8)
            else:
                x0, y0 = int(paste_box['x0']), int(paste_box['y0'])
                x1, y1 = int(paste_box['x1']), int(paste_box['y1'])
                region = union_mask[y0:y1, x0:x1]
                union_mask[y0:y1, x0:x1] = np.logical_or(region > 0, mask_arr > 0).astype(np.uint8)
        else:
            filtered_large_indices.append(idx)

        mask_path: Optional[Path] = None
        if save_individual_masks and mask_dir is not None:
            mask_path = mask_dir / f'{mask_prefix}_{idx:04d}.png'
            save_binary_mask(mask_arr, mask_path)
            mask_files.append(str(mask_path))

        record: Dict[str, Any] = {
            'mask_index': idx,
            'area': mask_area,
            'area_ratio': mask_area_ratio,
            'union_kept': bool(union_kept),
            'filtered_large': bool(not union_kept),
            'mask_file': str(mask_path) if mask_path is not None else None,
            'upscaled_mask_file': str(upscaled_mask_path) if upscaled_mask_path is not None else None,
            'upscaled_mask_area': input_mask_area,
            'sam_input_mask_shape': [int(input_mask_arr.shape[0]), int(input_mask_arr.shape[1])],
        }
        if paste_box is not None:
            record['box_xyxy'] = [
                int(paste_box['x0']),
                int(paste_box['y0']),
                int(paste_box['x1']),
                int(paste_box['y1']),
            ]
        mask_records.append(record)

    return {
        'kept_indices': kept_indices,
        'filtered_large_indices': filtered_large_indices,
        'kept_count': len(kept_indices),
        'filtered_large_count': len(filtered_large_indices),
        'mask_files': mask_files,
        'upscaled_mask_files': upscaled_mask_files,
        'mask_records': mask_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Run Segment Anything automatic mask generation on a local image.')
    parser.add_argument('--model-path', required=True, help='Local Hugging Face SAM model directory.')
    parser.add_argument('--image', required=True, help='Input image path.')
    parser.add_argument('--output-dir', default=None, help='Directory for default outputs. Explicit output paths still take priority.')
    parser.add_argument('--output-overlay', default=None, help='Where to save the overlay PNG. Defaults next to the image or --output-dir.')
    parser.add_argument('--output-summary', default=None, help='Where to save the JSON summary. Defaults next to the image or --output-dir.')
    parser.add_argument('--output-mask-dir', default=None, help='Where to save per-mask binary PNG files if --save-individual-masks is enabled. Defaults next to the image or --output-dir.')
    parser.add_argument('--output-upscaled-mask-dir', default=None, help='Where to save SAM-input-size masks if --save-upscaled-masks is enabled. Defaults next to the image or --output-dir.')
    parser.add_argument('--output-union-mask', default=None, help='Where to save the merged binary union mask PNG. Defaults next to the image or --output-dir.')
    parser.add_argument('--save-individual-masks', action='store_true', help='Also export each resized/stitchable mask as a separate binary PNG.')
    parser.add_argument('--save-upscaled-masks', action='store_true', help='Also export each original SAM-input-size mask. Useful for high-resolution shape filtering in split-upscale mode.')
    parser.add_argument('--split', nargs='?', const=4, default=0, type=int, help='Split the image into N equal square-grid tiles, run SAM on each tile, and stitch back. If no value is given, defaults to 4. Examples: 4, 9, 16.')
    parser.add_argument('--split-upscale', action='store_true', help='In --split mode, resize each tile to the original full image size before SAM, then resize masks back before stitching.')
    parser.add_argument('--save-split-tiles', action='store_true', help='Save the tile images used as SAM inputs. In --split-upscale mode these are the upscaled tiles.')
    parser.add_argument('--split-tile-dir', default=None, help='Directory for saved split tile images. Defaults under --output-dir or next to the input image.')
    parser.add_argument('--union-max-area-ratio', type=float, default=0.9, help='Exclude masks larger than this area ratio from the union mask. In --split mode, the ratio is computed per tile.')
    parser.add_argument('--points-per-batch', type=int, default=256, help='Batch size for prompt points.')
    parser.add_argument('--device', default=None, help='Pipeline device. Examples: cpu, 0, cuda:0.')
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    image_path = Path(args.image).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f'model path not found: {model_path}')
    if not image_path.is_file():
        raise SystemExit(f'image not found: {image_path}')
    if not (0.0 < args.union_max_area_ratio <= 1.0):
        raise SystemExit(f'union max area ratio must be in (0, 1], got: {args.union_max_area_ratio}')
    if args.split < 0:
        raise SystemExit(f'--split must be non-negative, got: {args.split}')
    if args.split:
        split_grid_size = int(round(args.split ** 0.5))
        if split_grid_size * split_grid_size != args.split:
            raise SystemExit(f'--split must be a square number such as 4, 9, or 16, got: {args.split}')
    else:
        split_grid_size = 0
    if args.split_upscale and not args.split:
        raise SystemExit('--split-upscale requires --split')
    if args.save_split_tiles and not args.split:
        raise SystemExit('--save-split-tiles requires --split')
    if args.save_upscaled_masks and not args.save_individual_masks:
        raise SystemExit('--save-upscaled-masks requires --save-individual-masks so stitchable masks are still available')

    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None
    default_dir = output_dir if output_dir is not None else image_path.parent
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    overlay_path = Path(args.output_overlay).expanduser().resolve() if args.output_overlay else default_dir / f'{image_path.stem}_sam_auto_overlay.png'
    summary_path = Path(args.output_summary).expanduser().resolve() if args.output_summary else default_dir / f'{image_path.stem}_sam_auto_summary.json'
    mask_dir = Path(args.output_mask_dir).expanduser().resolve() if args.output_mask_dir else default_dir / f'{image_path.stem}_sam_auto_masks'
    upscaled_mask_dir = Path(args.output_upscaled_mask_dir).expanduser().resolve() if args.output_upscaled_mask_dir else default_dir / f'{image_path.stem}_sam_auto_upscaled_masks'
    union_mask_path = Path(args.output_union_mask).expanduser().resolve() if args.output_union_mask else default_dir / f'{image_path.stem}_sam_auto_union.png'
    split_tile_dir = Path(args.split_tile_dir).expanduser().resolve() if args.split_tile_dir else default_dir / f'{image_path.stem}_sam_auto_split_tiles'

    device = resolve_device(args.device)
    print(json.dumps({
        'model_path': str(model_path),
        'image': str(image_path),
        'device': str(device),
        'points_per_batch': args.points_per_batch,
        'output_dir': str(output_dir) if output_dir is not None else None,
        'output_overlay': str(overlay_path),
        'output_summary': str(summary_path),
        'save_individual_masks': args.save_individual_masks,
        'save_upscaled_masks': args.save_upscaled_masks,
        'split': bool(args.split),
        'split_count': args.split,
        'split_grid_size': split_grid_size,
        'split_upscale': args.split_upscale,
        'save_split_tiles': args.save_split_tiles,
        'split_tile_dir': str(split_tile_dir) if args.save_split_tiles else None,
        'output_mask_dir': str(mask_dir) if args.save_individual_masks else None,
        'output_upscaled_mask_dir': str(upscaled_mask_dir) if args.save_upscaled_masks else None,
        'output_union_mask': str(union_mask_path),
        'union_max_area_ratio': args.union_max_area_ratio,
    }, ensure_ascii=False), flush=True)

    raw_image = Image.open(image_path).convert('RGB')
    print('loading_sam_pipeline', flush=True)
    generator = pipeline(
        'mask-generation',
        model=str(model_path),
        device=device,
        points_per_batch=args.points_per_batch,
    )
    union_mask = np.zeros((raw_image.height, raw_image.width), dtype=np.uint8)
    if args.save_individual_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_upscaled_masks:
        upscaled_mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_split_tiles:
        split_tile_dir.mkdir(parents=True, exist_ok=True)

    if args.split:
        print(f'running_automatic_mask_generation_split_tiles split_count={args.split} grid={split_grid_size}x{split_grid_size}', flush=True)
        tile_summaries: List[Dict[str, Any]] = []
        saved_tile_files: List[str] = []
        total_masks = 0
        total_bboxes = 0
        all_mask_files: List[str] = []
        all_upscaled_mask_files: List[str] = []
        all_mask_records: List[Dict[str, Any]] = []
        all_kept = 0
        all_filtered = 0

        for tile in make_equal_tiles(raw_image.width, raw_image.height, args.split):
            x0, y0 = int(tile['x0']), int(tile['y0'])
            x1, y1 = int(tile['x1']), int(tile['y1'])
            tile_id = str(tile['tile_id'])
            tile_image = raw_image.crop((x0, y0, x1, y1))
            sam_input_image = tile_image
            if args.split_upscale:
                sam_input_image = tile_image.resize((raw_image.width, raw_image.height), Image.BICUBIC)
            saved_tile_path = None
            if args.save_split_tiles:
                suffix = 'upscaled' if args.split_upscale else 'tile'
                saved_tile_path = split_tile_dir / f'{tile_id}_{suffix}.png'
                sam_input_image.save(saved_tile_path)
                saved_tile_files.append(str(saved_tile_path))
            print(
                f'running_tile={tile_id} box=({x0},{y0},{x1},{y1}) '
                f'tile_size={tile_image.width}x{tile_image.height} '
                f'sam_input_size={sam_input_image.width}x{sam_input_image.height}',
                flush=True,
            )
            outputs = generator(sam_input_image, points_per_batch=args.points_per_batch)
            masks = outputs.get('masks', []) or []
            print(f'tile={tile_id} generated_masks={len(masks)}', flush=True)

            tile_summary = summarize_outputs(outputs)
            tile_area = tile_image.width * tile_image.height
            union_info = update_union_from_masks(
                masks,
                union_mask,
                area_reference=tile_area,
                union_max_area_ratio=args.union_max_area_ratio,
                paste_box=tile,
                mask_output_size=(tile_image.width, tile_image.height) if args.split_upscale else None,
                save_individual_masks=args.save_individual_masks,
                mask_dir=mask_dir if args.save_individual_masks else None,
                save_upscaled_masks=args.save_upscaled_masks,
                upscaled_mask_dir=upscaled_mask_dir if args.save_upscaled_masks else None,
                mask_prefix=f'{tile_id}_mask',
            )
            tile_summary.update({
                'tile_id': tile_id,
                'box_xyxy': [x0, y0, x1, y1],
                'image_size': {'width': tile_image.width, 'height': tile_image.height},
                'sam_input_size': {'width': sam_input_image.width, 'height': sam_input_image.height},
                'upscaled_before_sam': bool(args.split_upscale),
                'saved_sam_input_image': str(saved_tile_path) if saved_tile_path is not None else None,
                'area_reference': tile_area,
                'union_kept_mask_indices': union_info['kept_indices'],
                'union_filtered_large_mask_indices': union_info['filtered_large_indices'],
                'union_kept_mask_count': union_info['kept_count'],
                'union_filtered_large_mask_count': union_info['filtered_large_count'],
            })
            if args.save_individual_masks:
                tile_records = []
                for rec in union_info['mask_records']:
                    rec = dict(rec)
                    rec['tile_id'] = tile_id
                    tile_records.append(rec)
                tile_summary['mask_files'] = union_info['mask_files']
                tile_summary['upscaled_mask_files'] = union_info['upscaled_mask_files']
                tile_summary['mask_records'] = tile_records
                all_mask_files.extend(union_info['mask_files'])
                all_upscaled_mask_files.extend(union_info['upscaled_mask_files'])
                all_mask_records.extend(tile_records)

            total_masks += int(tile_summary['num_masks'])
            total_bboxes += int(tile_summary['num_bounding_boxes'])
            all_kept += int(union_info['kept_count'])
            all_filtered += int(union_info['filtered_large_count'])
            tile_summaries.append(tile_summary)

        summary: Dict[str, Any] = {
            'mode': 'split_tiles_upscale' if args.split_upscale else 'split_tiles',
            'split_count': args.split,
            'split_grid_size': split_grid_size,
            'num_masks': total_masks,
            'num_bounding_boxes': total_bboxes,
            'tiles': tile_summaries,
            'image_size': {'width': raw_image.width, 'height': raw_image.height},
            'image_path': str(image_path),
            'model_path': str(model_path),
            'points_per_batch': args.points_per_batch,
            'device': str(device),
            'union_max_area_ratio': args.union_max_area_ratio,
            'union_area_ratio_reference': 'per_tile_after_mask_resize',
            'split_upscale': bool(args.split_upscale),
            'save_split_tiles': bool(args.save_split_tiles),
            'save_upscaled_masks': bool(args.save_upscaled_masks),
            'split_tile_dir': str(split_tile_dir) if args.save_split_tiles else None,
            'saved_split_tile_files': saved_tile_files,
            'union_kept_mask_count': all_kept,
            'union_filtered_large_mask_count': all_filtered,
            'overlay_mode': 'filtered_split_tiles_union_binary_mask',
        }
        if args.save_individual_masks:
            summary['mask_dir'] = str(mask_dir)
            summary['mask_files'] = all_mask_files
            if args.save_upscaled_masks:
                summary['upscaled_mask_dir'] = str(upscaled_mask_dir)
                summary['upscaled_mask_files'] = all_upscaled_mask_files
            summary['mask_records'] = all_mask_records
    else:
        print('running_automatic_mask_generation', flush=True)
        outputs = generator(raw_image, points_per_batch=args.points_per_batch)
        masks = outputs.get('masks', []) or []
        print(f'generated_masks={len(masks)}', flush=True)
        summary = summarize_outputs(outputs)
        summary['image_size'] = {'width': raw_image.width, 'height': raw_image.height}
        summary['image_path'] = str(image_path)
        summary['model_path'] = str(model_path)
        summary['points_per_batch'] = args.points_per_batch
        summary['device'] = str(device)
        summary['union_max_area_ratio'] = args.union_max_area_ratio

        image_area = raw_image.width * raw_image.height
        union_info = update_union_from_masks(
            masks,
            union_mask,
            area_reference=image_area,
            union_max_area_ratio=args.union_max_area_ratio,
            paste_box=None,
            save_individual_masks=args.save_individual_masks,
            mask_dir=mask_dir if args.save_individual_masks else None,
            save_upscaled_masks=args.save_upscaled_masks,
            upscaled_mask_dir=upscaled_mask_dir if args.save_upscaled_masks else None,
            mask_prefix='mask',
        )
        summary['union_kept_mask_indices'] = union_info['kept_indices']
        summary['union_filtered_large_mask_indices'] = union_info['filtered_large_indices']
        summary['union_kept_mask_count'] = union_info['kept_count']
        summary['union_filtered_large_mask_count'] = union_info['filtered_large_count']
        summary['overlay_mode'] = 'filtered_union_binary_mask'
        if args.save_individual_masks:
            summary['mask_dir'] = str(mask_dir)
            summary['mask_files'] = union_info['mask_files']
            if args.save_upscaled_masks:
                summary['upscaled_mask_dir'] = str(upscaled_mask_dir)
                summary['upscaled_mask_files'] = union_info['upscaled_mask_files']
            summary['mask_records'] = union_info['mask_records']

    save_binary_mask(union_mask, union_mask_path)
    summary['union_mask_path'] = str(union_mask_path)

    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    save_union_overlay(raw_image, union_mask, overlay_path)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps({
        'num_masks': summary['num_masks'],
        'overlay_path': str(overlay_path),
        'summary_path': str(summary_path),
        'union_mask_path': str(union_mask_path),
        'mask_dir': str(mask_dir) if args.save_individual_masks else None,
        'upscaled_mask_dir': str(upscaled_mask_dir) if args.save_upscaled_masks else None,
    }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
