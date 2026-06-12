#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw


def load_summary(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_binary_mask(mask: np.ndarray, path: Path) -> None:
    arr = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    Image.fromarray(arr, mode="L").save(path)


def save_union_overlay(image: Image.Image, mask: np.ndarray, path: Path, alpha: float = 0.45) -> None:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    binary = np.asarray(mask, dtype=np.uint8) > 0
    color = np.array([30.0, 144.0, 255.0], dtype=np.float32)
    if binary.any():
        base[binary] = base[binary] * (1.0 - alpha) + color * alpha
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB").save(path)


def load_binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def compute_boundary(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    center = padded[1:-1, 1:-1]
    eroded4 = (
        center
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return binary & ~eroded4


def _cross(o: Tuple[int, int], a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts
    lower: List[Tuple[int, int]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[int, int]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def rasterize_polygon(points: Sequence[Tuple[int, int]], shape: Tuple[int, int]) -> np.ndarray:
    if len(points) < 3:
        return np.zeros(shape, dtype=bool)
    h, w = shape
    canvas = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(canvas)
    draw.polygon([(int(x), int(y)) for x, y in points], outline=1, fill=1)
    return np.asarray(canvas, dtype=np.uint8) > 0


def compute_circle_iou(mask: np.ndarray, area: int, xs: np.ndarray, ys: np.ndarray) -> float:
    if area <= 0:
        return 0.0
    cx = float(xs.mean())
    cy = float(ys.mean())
    radius = math.sqrt(area / math.pi)
    x0 = max(0, int(math.floor(min(float(xs.min()), cx - radius))) - 1)
    y0 = max(0, int(math.floor(min(float(ys.min()), cy - radius))) - 1)
    x1 = min(mask.shape[1], int(math.ceil(max(float(xs.max()), cx + radius))) + 2)
    y1 = min(mask.shape[0], int(math.ceil(max(float(ys.max()), cy + radius))) + 2)
    local_mask = mask[y0:y1, x0:x1]
    yy, xx = np.indices(local_mask.shape)
    circle = ((xx + x0 - cx) ** 2 + (yy + y0 - cy) ** 2) <= radius ** 2
    inter = np.logical_and(local_mask, circle).sum()
    union = np.logical_or(local_mask, circle).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


def compute_shape_features(mask: np.ndarray) -> Dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    area = int(binary.sum())
    if area <= 0:
        return {
            "area_pixels": 0,
            "bbox_width": 0,
            "bbox_height": 0,
            "bbox_area": 0,
            "aspect_ratio": None,
            "extent": None,
            "perimeter_pixels": 0,
            "circularity": None,
            "touches_border": False,
            "solidity": None,
            "convexity": None,
            "circle_iou": None,
        }

    ys, xs = np.nonzero(binary)
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    bbox_area = bbox_w * bbox_h
    aspect_ratio = float(max(bbox_w, bbox_h) / max(1, min(bbox_w, bbox_h)))
    extent = float(area / bbox_area) if bbox_area > 0 else None

    boundary = compute_boundary(binary)
    perimeter = int(boundary.sum())
    circularity = float(4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else None
    touches_border = bool(
        binary[0, :].any() or binary[-1, :].any() or binary[:, 0].any() or binary[:, -1].any()
    )

    bys, bxs = np.nonzero(boundary)
    hull_points = convex_hull(zip(bxs.tolist(), bys.tolist()))
    hull_mask = rasterize_polygon(hull_points, binary.shape)
    hull_area = int(hull_mask.sum())
    hull_perimeter = int(compute_boundary(hull_mask).sum()) if hull_area > 0 else 0
    solidity = float(area / hull_area) if hull_area > 0 else None
    convexity = float(hull_perimeter / perimeter) if perimeter > 0 and hull_perimeter > 0 else None
    circle_iou = compute_circle_iou(binary, area, xs, ys)

    return {
        "area_pixels": area,
        "bbox_width": bbox_w,
        "bbox_height": bbox_h,
        "bbox_area": bbox_area,
        "aspect_ratio": aspect_ratio,
        "extent": extent,
        "perimeter_pixels": perimeter,
        "circularity": circularity,
        "touches_border": touches_border,
        "solidity": solidity,
        "convexity": convexity,
        "circle_iou": circle_iou,
    }


def collect_reference_areas(summary: Dict[str, Any]) -> List[int]:
    areas: List[int] = []
    for tile in summary.get("tiles", []) or []:
        for area in tile.get("mask_areas", []) or []:
            try:
                areas.append(int(area))
            except Exception:
                continue
    for area in summary.get("mask_areas", []) or []:
        try:
            areas.append(int(area))
        except Exception:
            continue
    return areas


def collect_target_records(summary: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], str, bool]:
    if isinstance(summary.get("kept_records"), list):
        return list(summary["kept_records"]), "kept_records", True
    if isinstance(summary.get("mask_records"), list):
        return list(summary["mask_records"]), "mask_records", False
    records: List[Dict[str, Any]] = []
    for tile in summary.get("tiles", []) or []:
        for rec in tile.get("mask_records", []) or []:
            item = dict(rec)
            if "tile_id" not in item and tile.get("tile_id") is not None:
                item["tile_id"] = tile["tile_id"]
            records.append(item)
    return records, "tiles.mask_records", False


def get_record_filter_area(rec: Dict[str, Any]) -> Optional[int]:
    for key in ("filter_area", "upscaled_mask_area", "area", "saved_mask_area"):
        value = rec.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return None


def fit_area_distribution(reference_areas: List[int], args: argparse.Namespace) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Dict[str, Any]]:
    areas = np.asarray(reference_areas, dtype=np.float64)
    stats: Dict[str, Any] = {"reference_area_count_raw": int(areas.size)}
    if areas.size == 0:
        return None, None, None, None, stats

    if args.reference_min_area is not None:
        areas = areas[areas >= float(args.reference_min_area)]
    if args.reference_max_area is not None:
        areas = areas[areas <= float(args.reference_max_area)]
    stats["reference_area_count_after_hard_clip"] = int(areas.size)
    if areas.size == 0:
        return None, None, None, None, stats

    if args.reference_max_quantile is not None and 0.0 < args.reference_max_quantile < 1.0:
        q = float(np.quantile(areas, args.reference_max_quantile))
        areas = areas[areas <= q]
        stats["reference_max_quantile_value"] = q
        stats["reference_area_count_after_quantile"] = int(areas.size)
        if areas.size == 0:
            return None, None, None, None, stats

    bin_width = float(args.bin_width)
    if bin_width > 0.0 and areas.size > 0:
        lo = float(areas.min())
        hi = float(areas.max()) + bin_width
        bins = np.arange(lo, hi + bin_width, bin_width)
        if bins.size >= 2:
            hist, edges = np.histogram(areas, bins=bins)
            mode_idx = int(hist.argmax())
            mode_center = float((edges[mode_idx] + edges[mode_idx + 1]) * 0.5)
            stats["mode_center"] = mode_center
            if args.mode_window_ratio is not None and args.mode_window_ratio > 0:
                ratio = float(args.mode_window_ratio)
                lo_keep = mode_center * max(0.0, 1.0 - ratio)
                hi_keep = mode_center * (1.0 + ratio)
                areas = areas[(areas >= lo_keep) & (areas <= hi_keep)]
                stats["mode_window_keep_range"] = [lo_keep, hi_keep]
                stats["reference_area_count_after_mode_window"] = int(areas.size)
                if areas.size == 0:
                    return None, None, None, None, stats

    trim_iters = max(0, int(args.trim_iters))
    trim_std = float(args.trim_std)
    for _ in range(trim_iters):
        if areas.size <= 1:
            break
        mu = float(areas.mean())
        sigma = float(areas.std(ddof=0))
        if sigma <= 0.0:
            break
        lo_keep = mu - trim_std * sigma
        hi_keep = mu + trim_std * sigma
        next_areas = areas[(areas >= lo_keep) & (areas <= hi_keep)]
        if next_areas.size == 0 or next_areas.size == areas.size:
            break
        areas = next_areas

    if areas.size == 0:
        return None, None, None, None, stats

    mu = float(areas.mean())
    sigma = float(areas.std(ddof=0))
    lo = mu - float(args.num_std) * sigma
    hi = mu + float(args.num_std) * sigma
    stats["reference_area_count_final"] = int(areas.size)
    return mu, sigma, lo, hi, stats


def resolve_shape_mask_file(rec: Dict[str, Any], source_mode: str) -> Tuple[Optional[str], str]:
    if source_mode == "mask_file":
        return rec.get("mask_file"), "mask_file"
    if source_mode == "upscaled_mask_file":
        return rec.get("upscaled_mask_file"), "upscaled_mask_file"
    if rec.get("upscaled_mask_file"):
        return rec.get("upscaled_mask_file"), "upscaled_mask_file"
    return rec.get("mask_file"), "mask_file"


def is_enabled(value: Optional[float]) -> bool:
    return value is not None


def single_ball_decision(features: Dict[str, Any], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    circularity = features.get("circularity")
    aspect_ratio = features.get("aspect_ratio")
    extent = features.get("extent")
    solidity = features.get("solidity")
    convexity = features.get("convexity")
    circle_iou = features.get("circle_iou")

    if args.reject_border_touching and features.get("touches_border"):
        reasons.append("touches_border")
    if is_enabled(args.min_circularity) and (circularity is None or circularity < args.min_circularity):
        reasons.append("low_circularity")
    if is_enabled(args.max_aspect_ratio) and (aspect_ratio is None or aspect_ratio > args.max_aspect_ratio):
        reasons.append("high_aspect_ratio")
    if is_enabled(args.min_extent) and (extent is None or extent < args.min_extent):
        reasons.append("low_extent")
    if is_enabled(args.max_extent) and (extent is None or extent > args.max_extent):
        reasons.append("high_extent")
    if is_enabled(args.min_solidity) and (solidity is None or solidity < args.min_solidity):
        reasons.append("low_solidity")
    if is_enabled(args.min_convexity) and (convexity is None or convexity < args.min_convexity):
        reasons.append("low_convexity")
    if is_enabled(args.min_circle_iou) and (circle_iou is None or circle_iou < args.min_circle_iou):
        reasons.append("low_circle_iou")
    return len(reasons) == 0, reasons


def cluster_decision(features: Dict[str, Any], filter_area: Optional[int], mu: Optional[float], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    aspect_ratio = features.get("aspect_ratio")
    extent = features.get("extent")
    if filter_area is None or mu is None:
        return False, ["missing_area_for_cluster"]
    min_area = float(args.cluster_min_area_multiple) * mu
    max_area = float(args.cluster_max_area_multiple) * mu
    if filter_area < min_area:
        reasons.append("cluster_area_too_small")
    if filter_area > max_area:
        reasons.append("cluster_area_too_large")
    if aspect_ratio is None or aspect_ratio > float(args.cluster_max_aspect_ratio):
        reasons.append("cluster_high_aspect_ratio")
    if extent is None or extent < float(args.cluster_min_extent):
        reasons.append("cluster_low_extent")
    if extent is None or extent > float(args.cluster_max_extent):
        reasons.append("cluster_high_extent")
    return len(reasons) == 0, reasons


def compute_shape_decision(rec: Dict[str, Any], mu: Optional[float], args: argparse.Namespace) -> Tuple[bool, List[str], Optional[Dict[str, Any]], Optional[str], str]:
    if args.shape_filter == "off":
        return True, [], None, None, "off"

    shape_mask_file, shape_mask_source = resolve_shape_mask_file(rec, args.shape_mask_source)
    if not shape_mask_file:
        return False, ["missing_shape_mask_file"], None, None, shape_mask_source

    features = compute_shape_features(load_binary_mask(Path(shape_mask_file)))
    single_ok, single_reasons = single_ball_decision(features, args)
    if args.shape_filter == "ball":
        return single_ok, single_reasons, features, shape_mask_file, shape_mask_source

    filter_area = get_record_filter_area(rec)
    cluster_ok, cluster_reasons = cluster_decision(features, filter_area, mu, args)
    if single_ok or cluster_ok:
        return True, [] if single_ok else cluster_reasons, features, shape_mask_file, shape_mask_source
    return False, single_reasons + cluster_reasons, features, shape_mask_file, shape_mask_source


def combine_decisions(area_pass: bool, shape_pass: bool, args: argparse.Namespace) -> bool:
    if args.combine_mode == "area":
        return area_pass
    if args.combine_mode == "shape":
        return shape_pass
    if args.combine_mode == "or":
        return area_pass or shape_pass
    return area_pass and shape_pass


def paste_mask(full_mask: np.ndarray, tile_mask: np.ndarray, box_xyxy: Sequence[int]) -> None:
    x0, y0, x1, y1 = [int(v) for v in box_xyxy]
    target_h = y1 - y0
    target_w = x1 - x0
    arr = np.asarray(tile_mask, dtype=bool)
    if arr.shape != (target_h, target_w):
        arr_img = Image.fromarray((arr.astype(np.uint8) * 255), mode="L")
        arr = np.asarray(arr_img.resize((target_w, target_h), Image.NEAREST), dtype=np.uint8) > 0
    full_mask[y0:y1, x0:x1] = np.logical_or(full_mask[y0:y1, x0:x1], arr)


def resolve_image_path(summary: Dict[str, Any], summary_path: Path) -> Path:
    image_path = summary.get("image_path")
    if image_path:
        return Path(image_path)
    raise SystemExit(f"image_path missing in summary: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Area + shape filtering for SAM masks.")
    parser.add_argument("--reference-summary", default=None, help="Reference summary for fitting area distribution.")
    parser.add_argument("--target-summary", required=True, help="Raw SAM summary or area-filtered summary.")
    parser.add_argument("--output-dir", required=True, help="Directory for filtered outputs.")

    parser.add_argument("--combine-mode", default="and", choices=["and", "or", "area", "shape"])
    parser.add_argument("--shape-filter", default="off", choices=["off", "ball", "ball_or_cluster"])
    parser.add_argument("--shape-mask-source", default="auto", choices=["auto", "mask_file", "upscaled_mask_file"])

    parser.add_argument("--num-std", type=float, default=2.0)
    parser.add_argument("--reference-min-area", type=float, default=None)
    parser.add_argument("--reference-max-area", type=float, default=None)
    parser.add_argument("--reference-max-quantile", type=float, default=None)
    parser.add_argument("--target-min-area", type=float, default=None)
    parser.add_argument("--target-max-area", type=float, default=None)
    parser.add_argument("--bin-width", type=float, default=64.0)
    parser.add_argument("--mode-window-ratio", type=float, default=0.5)
    parser.add_argument("--trim-std", type=float, default=2.5)
    parser.add_argument("--trim-iters", type=int, default=2)

    parser.add_argument("--min-circularity", type=float, default=None)
    parser.add_argument("--max-aspect-ratio", type=float, default=None)
    parser.add_argument("--min-extent", type=float, default=None)
    parser.add_argument("--max-extent", type=float, default=None)
    parser.add_argument("--min-solidity", type=float, default=None)
    parser.add_argument("--min-convexity", type=float, default=None)
    parser.add_argument("--min-circle-iou", type=float, default=None)
    parser.add_argument("--reject-border-touching", action="store_true")

    parser.add_argument("--cluster-min-area-multiple", type=float, default=1.8)
    parser.add_argument("--cluster-max-area-multiple", type=float, default=50.0)
    parser.add_argument("--cluster-max-aspect-ratio", type=float, default=4.0)
    parser.add_argument("--cluster-min-extent", type=float, default=0.12)
    parser.add_argument("--cluster-max-extent", type=float, default=0.98)

    args = parser.parse_args()

    target_summary_path = Path(args.target_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_summary = load_summary(target_summary_path)
    target_records, input_record_source, input_area_preselected = collect_target_records(target_summary)
    image_path = resolve_image_path(target_summary, target_summary_path)
    raw_image = Image.open(image_path).convert("RGB")
    union_mask = np.zeros((raw_image.height, raw_image.width), dtype=bool)

    mu: Optional[float] = None
    sigma: Optional[float] = None
    area_lo: Optional[float] = None
    area_hi: Optional[float] = None
    fit_stats: Dict[str, Any] = {}
    hard_target_area_enabled = args.target_min_area is not None or args.target_max_area is not None
    if args.reference_summary is not None:
        ref_summary_path = Path(args.reference_summary).expanduser().resolve()
        reference_summary = load_summary(ref_summary_path)
        reference_areas = collect_reference_areas(reference_summary)
        mu, sigma, area_lo, area_hi, fit_stats = fit_area_distribution(reference_areas, args)
    elif args.combine_mode in {"and", "or", "area"} and not input_area_preselected and not hard_target_area_enabled:
        raise SystemExit(
            "reference-summary is required for distribution-based area filtering on raw target summaries, "
            "unless target-min-area and/or target-max-area is provided."
        )

    area_pass_count = 0
    shape_pass_count = 0
    area_shape_pass_count = 0
    area_only_pass_count = 0
    shape_only_pass_count = 0
    keep_reason_counts: Counter[str] = Counter()
    reject_reason_counts: Counter[str] = Counter()

    kept_records: List[Dict[str, Any]] = []
    rejected_records: List[Dict[str, Any]] = []

    for rec in target_records:
        filter_area = get_record_filter_area(rec)
        area_checks: List[bool] = []
        if area_lo is not None and area_hi is not None:
            area_checks.append(
                filter_area is not None and float(area_lo) <= float(filter_area) <= float(area_hi)
            )
        if args.target_min_area is not None:
            area_checks.append(filter_area is not None and float(filter_area) >= float(args.target_min_area))
        if args.target_max_area is not None:
            area_checks.append(filter_area is not None and float(filter_area) <= float(args.target_max_area))

        if area_checks:
            area_pass = all(area_checks)
        else:
            area_pass = bool(input_area_preselected)
        if area_pass:
            area_pass_count += 1

        shape_pass, shape_reasons, shape_features, shape_mask_file, shape_mask_source = compute_shape_decision(rec, mu, args)
        if shape_pass:
            shape_pass_count += 1

        if area_pass and shape_pass:
            area_shape_pass_count += 1
        elif area_pass and not shape_pass:
            area_only_pass_count += 1
        elif shape_pass and not area_pass:
            shape_only_pass_count += 1

        keep = combine_decisions(area_pass, shape_pass, args)
        out_rec = dict(rec)
        out_rec["filter_area"] = filter_area
        out_rec["area_pass"] = area_pass
        out_rec["shape_pass"] = shape_pass
        out_rec["shape_reasons"] = shape_reasons
        out_rec["shape_features"] = shape_features
        out_rec["shape_mask_file"] = shape_mask_file
        out_rec["shape_mask_source"] = shape_mask_source

        if keep:
            reason = (
                "area_and_shape"
                if area_pass and shape_pass
                else "area_only"
                if area_pass
                else "shape_only"
                if shape_pass
                else "kept_unexpected"
            )
            out_rec["keep_reason"] = reason
            keep_reason_counts[reason] += 1
            mask_file = rec.get("mask_file")
            box_xyxy = rec.get("box_xyxy")
            if mask_file and box_xyxy is not None:
                paste_mask(union_mask, load_binary_mask(Path(mask_file)), box_xyxy)
            kept_records.append(out_rec)
        else:
            reason = (
                "area_and_shape_fail"
                if (not area_pass and not shape_pass)
                else "area_fail"
                if not area_pass
                else "shape_fail"
            )
            out_rec["reject_reason"] = reason
            reject_reason_counts[reason] += 1
            rejected_records.append(out_rec)

    stem = image_path.stem
    union_path = output_dir / f"{stem}_area_filtered_union.png"
    overlay_path = output_dir / f"{stem}_area_filtered_overlay.png"
    summary_out_path = output_dir / f"{stem}_area_filtered_summary.json"

    save_binary_mask(union_mask, union_path)
    save_union_overlay(raw_image, union_mask, overlay_path)

    out_summary: Dict[str, Any] = {
        "reference_summary": str(Path(args.reference_summary).expanduser().resolve()) if args.reference_summary else None,
        "target_summary": str(target_summary_path),
        "image_path": str(image_path),
        "input_record_source": input_record_source,
        "input_area_preselected": input_area_preselected,
        "combine_mode": args.combine_mode,
        "shape_filter": args.shape_filter,
        "shape_mask_preference": args.shape_mask_source,
        "shape_filter_params": {
            "min_circularity": args.min_circularity,
            "max_aspect_ratio": args.max_aspect_ratio,
            "min_extent": args.min_extent,
            "max_extent": args.max_extent,
            "min_solidity": args.min_solidity,
            "min_convexity": args.min_convexity,
            "min_circle_iou": args.min_circle_iou,
            "reject_border_touching": args.reject_border_touching,
        },
        "cluster_params": {
            "cluster_min_area_multiple": args.cluster_min_area_multiple,
            "cluster_max_area_multiple": args.cluster_max_area_multiple,
            "cluster_max_aspect_ratio": args.cluster_max_aspect_ratio,
            "cluster_min_extent": args.cluster_min_extent,
            "cluster_max_extent": args.cluster_max_extent,
        },
        "area_filter_params": {
            "num_std": args.num_std,
            "reference_min_area": args.reference_min_area,
            "reference_max_area": args.reference_max_area,
            "reference_max_quantile": args.reference_max_quantile,
            "target_min_area": args.target_min_area,
            "target_max_area": args.target_max_area,
            "bin_width": args.bin_width,
            "mode_window_ratio": args.mode_window_ratio,
            "trim_std": args.trim_std,
            "trim_iters": args.trim_iters,
        },
        "mu": mu,
        "sigma": sigma,
        "area_keep_range": [area_lo, area_hi] if area_lo is not None and area_hi is not None else None,
        "target_hard_area_range": [args.target_min_area, args.target_max_area],
        "fit_stats": fit_stats,
        "target_record_count": len(target_records),
        "area_pass_count": area_pass_count,
        "shape_pass_count": shape_pass_count,
        "area_shape_pass_count": area_shape_pass_count,
        "area_only_pass_count": area_only_pass_count,
        "shape_only_pass_count": shape_only_pass_count,
        "kept_count": len(kept_records),
        "rejected_count": len(rejected_records),
        "keep_reason_counts": dict(keep_reason_counts),
        "reject_reason_counts": dict(reject_reason_counts),
        "output_union_mask": str(union_path),
        "output_overlay": str(overlay_path),
        "kept_records": kept_records,
        "rejected_records": rejected_records,
    }

    with open(summary_out_path, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    print(
        json.dumps(
            {
                "mu": mu,
                "sigma": sigma,
                "area_keep_range": [area_lo, area_hi] if area_lo is not None and area_hi is not None else None,
                "target_hard_area_range": [args.target_min_area, args.target_max_area],
                "target_record_count": len(target_records),
                "area_pass_count": area_pass_count,
                "shape_pass_count": shape_pass_count,
                "area_shape_pass_count": area_shape_pass_count,
                "kept_count": len(kept_records),
                "rejected_count": len(rejected_records),
                "output_union_mask": str(union_path),
                "output_overlay": str(overlay_path),
                "output_summary": str(summary_out_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
