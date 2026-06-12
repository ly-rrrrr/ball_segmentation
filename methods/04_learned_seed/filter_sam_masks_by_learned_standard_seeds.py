#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "methods" / "03_seed_heuristic"))

from filter_sam_masks_by_ball_seeds import (
    build_integral,
    load_mask_records,
    mask_bbox,
    percentile,
    radial_balance,
    read_bmp_rgb_gray,
    read_png_gray,
    rect_mean,
    save_overlay,
    write_png,
)


DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))


def median(vals: List[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def mad(vals: List[float], center: float) -> float:
    return max(1e-6, median([abs(v - center) for v in vals]))


def sample_gray(gray: bytearray, width: int, height: int, x: int, y: int) -> float:
    x = max(0, min(width - 1, x))
    y = max(0, min(height - 1, y))
    return float(gray[y * width + x])


def feature_at(gray: bytearray, integ: List[int], width: int, height: int, x: int, y: int, r: int) -> Dict[str, Any]:
    cr = max(1, r // 2)
    center = rect_mean(integ, width, height, x, y, cr)
    outer = rect_mean(integ, width, height, x, y, r * 2)
    signed = center - outer
    profile: List[float] = []
    for mul in (1, 2):
        for dx, dy in DIRS:
            val = sample_gray(gray, width, height, x + dx * r * mul, y + dy * r * mul)
            profile.append(center - val)
    norm = max(1.0, abs(signed))
    profile_norm = [v / norm for v in profile]
    balance = radial_balance(gray, width, height, x, y, r, center)
    return {
        "x": x,
        "y": y,
        "radius": r,
        "center_mean": center,
        "outer_mean": outer,
        "signed_contrast": signed,
        "abs_contrast": abs(signed),
        "radial_balance": balance,
        "profile": profile_norm,
    }


def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return dot / (na * nb)


def nms_candidates(cands: List[Dict[str, Any]], radius: int, max_seeds: int) -> List[Dict[str, Any]]:
    cands = sorted(cands, key=lambda d: d["score"], reverse=True)
    kept: List[Dict[str, Any]] = []
    r2 = radius * radius
    for c in cands:
        x, y = int(c["x"]), int(c["y"])
        if any((int(k["x"]) - x) ** 2 + (int(k["y"]) - y) ** 2 <= r2 for k in kept):
            continue
        c = dict(c)
        c["seed_id"] = len(kept)
        kept.append(c)
        if len(kept) >= max_seeds:
            break
    return kept


def load_seed_supported_positives(
    image_path: Path,
    summary_path: Path,
    area_low_q: float,
    area_high_q: float,
    max_positive_masks: int,
    seed_stride: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    width, height, _rgb, gray = read_bmp_rgb_gray(image_path)
    integ = build_integral(gray, width, height)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_mask_records(summary)
    kept_areas = [float(r["area"]) for r in records if r.get("union_kept") and r.get("area", 0) > 0]
    area_low = percentile(kept_areas, area_low_q)
    area_high = percentile(kept_areas, area_high_q)
    standard_area = median([a for a in kept_areas if area_low <= a <= area_high])
    r0 = max(2, int(round(math.sqrt(max(1.0, standard_area) / math.pi))))

    pos: List[Dict[str, Any]] = []
    selected = [r for r in records if r.get("union_kept") and area_low <= float(r.get("area", 0)) <= area_high]
    selected.sort(key=lambda r: (r.get("sam_score") is None, -(r.get("sam_score") or 0.0)))
    selected = selected[:max_positive_masks]

    for rec in selected:
        mask_path = Path(str(rec.get("mask_file")))
        if not mask_path.is_file():
            continue
        mw, mh, mask = read_png_gray(mask_path)
        lx0, ly0, lx1, ly1, area = mask_bbox(mask, mw, mh)
        if area <= 0:
            continue
        tx0, ty0, tx1, ty1 = [int(v) for v in rec.get("box_xyxy", [0, 0, width, height])]
        tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
        # Use mask centroid as a robust positive center. This avoids relying on the old seed detector.
        sx_sum = sy_sum = count = 0
        for yy in range(ly0, ly1, max(1, seed_stride)):
            off = yy * mw
            for xx in range(lx0, lx1, max(1, seed_stride)):
                if mask[off + xx] > 0:
                    sx_sum += xx
                    sy_sum += yy
                    count += 1
        if count == 0:
            continue
        mx = int(round(sx_sum / count))
        my = int(round(sy_sum / count))
        gx = tx0 + int((mx + 0.5) * tw / float(mw))
        gy = ty0 + int((my + 0.5) * th / float(mh))
        if not (0 <= gx < width and 0 <= gy < height):
            continue
        pos.append(feature_at(gray, integ, width, height, gx, gy, r0))

    meta = {
        "clean_width": width,
        "clean_height": height,
        "area_low": area_low,
        "area_high": area_high,
        "standard_area": standard_area,
        "standard_radius": r0,
        "positive_count": len(pos),
        "selected_mask_count": len(selected),
    }
    return pos, meta


def learn_model(pos: List[Dict[str, Any]], meta: Dict[str, Any]) -> Dict[str, Any]:
    if not pos:
        raise SystemExit("no positive samples selected from clean sample")
    profiles = [p["profile"] for p in pos]
    plen = len(profiles[0])
    template = [median([prof[i] for prof in profiles]) for i in range(plen)]
    signed_vals = [float(p["signed_contrast"]) for p in pos]
    abs_vals = [float(p["abs_contrast"]) for p in pos]
    bal_vals = [float(p["radial_balance"]) for p in pos if float(p["radial_balance"]) < 999]
    signed_med = median(signed_vals)
    abs_med = median(abs_vals)
    bal_med = median(bal_vals) if bal_vals else 1.0
    model = {
        **meta,
        "template_profile": template,
        "signed_contrast_median": signed_med,
        "signed_contrast_mad": mad(signed_vals, signed_med),
        "abs_contrast_median": abs_med,
        "abs_contrast_mad": mad(abs_vals, abs_med),
        "radial_balance_median": bal_med,
        "radial_balance_mad": mad(bal_vals, bal_med) if bal_vals else 1.0,
    }
    return model


def score_feature(f: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, float]:
    profile_sim = max(0.0, cosine(f["profile"], model["template_profile"]))
    abs_med = float(model["abs_contrast_median"])
    abs_mad = float(model["abs_contrast_mad"])
    signed_med = float(model["signed_contrast_median"])
    signed_mad = float(model["signed_contrast_mad"])
    bal_med = float(model["radial_balance_median"])
    bal_mad = float(model["radial_balance_mad"])

    # Contrast magnitude is important, but dirty backgrounds can change it, so score it softly.
    abs_z = abs(float(f["abs_contrast"]) - abs_med) / max(1.0, 2.5 * abs_mad)
    abs_score = max(0.0, 1.0 - abs_z)

    # Bright/dark polarity is an auxiliary learned cue, not a hard gate.
    signed_z = abs(float(f["signed_contrast"]) - signed_med) / max(1.0, 3.5 * signed_mad)
    signed_score = max(0.0, 1.0 - signed_z)

    # Radial balance: lower is better. Permit some degradation in dirty samples.
    bal_limit = bal_med + 4.0 * bal_mad
    bal_score = max(0.0, 1.0 - max(0.0, float(f["radial_balance"]) - bal_med) / max(1.0, bal_limit - bal_med))

    score = 0.45 * profile_sim + 0.25 * abs_score + 0.15 * bal_score + 0.15 * signed_score
    return {
        "score": score,
        "profile_similarity": profile_sim,
        "abs_contrast_score": abs_score,
        "signed_contrast_score": signed_score,
        "radial_balance_score": bal_score,
    }


def apply_detector(image_path: Path, model: Dict[str, Any], args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], bytearray, Dict[str, Any]]:
    width, height, rgb, gray = read_bmp_rgb_gray(image_path)
    integ = build_integral(gray, width, height)
    r0 = int(model["standard_radius"])
    candidate_radii = sorted(set(max(2, r0 + d) for d in (-1, 0, 1)))
    margin = max(candidate_radii) * 2 + 2
    cands: List[Dict[str, Any]] = []
    raw_count = 0
    for y in range(margin, height - margin, args.scan_stride):
        for x in range(margin, width - margin, args.scan_stride):
            best: Dict[str, Any] | None = None
            for r in candidate_radii:
                f = feature_at(gray, integ, width, height, x, y, r)
                sc = score_feature(f, model)
                raw_count += 1
                item = {**f, **sc}
                if best is None or item["score"] > best["score"]:
                    best = item
            if best is not None and best["score"] >= args.seed_score_thresh and best["abs_contrast"] >= args.min_abs_contrast:
                cands.append(best)
    seeds = nms_candidates(cands, args.seed_nms_radius, args.max_seeds)
    seed_map = bytearray(width * height)
    for s in seeds:
        x, y = int(s["x"]), int(s["y"])
        seed_map[y * width + x] = 255
    stats = {
        "raw_scored_candidates": raw_count,
        "pre_nms_candidates": len(cands),
        "seed_count": len(seeds),
        "candidate_radii": candidate_radii,
    }
    return seeds, seed_map, stats


def draw_seed_overlay(rgb: bytearray, width: int, height: int, seeds: List[Dict[str, Any]]) -> bytearray:
    out = bytearray(rgb)
    for seed in seeds:
        x, y = int(seed["x"]), int(seed["y"])
        score = float(seed.get("score", 0.0))
        color = (255, 40, 40) if score >= 0.75 else ((255, 180, 40) if score >= 0.6 else (80, 220, 255))
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            xx, yy = x + dx, y + dy
            if 0 <= xx < width and 0 <= yy < height:
                j = (yy * width + xx) * 3
                out[j:j + 3] = bytes(color)
    return out


def filter_masks_with_seeds(image_path: Path, summary_path: Path, seeds: List[Dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    width, height, rgb, _gray = read_bmp_rgb_gray(image_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_mask_records(summary)
    seed_positions = [(int(s["x"]), int(s["y"])) for s in seeds]
    union = bytearray(width * height)
    filtered_records: List[Dict[str, Any]] = []
    kept = removed = 0

    for n, rec in enumerate(records):
        mask_path = Path(str(rec.get("mask_file")))
        if not mask_path.is_file():
            continue
        mw, mh, mask = read_png_gray(mask_path)
        lx0, ly0, lx1, ly1, area = mask_bbox(mask, mw, mh)
        tx0, ty0, tx1, ty1 = [int(v) for v in rec.get("box_xyxy", [0, 0, width, height])]
        tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
        rec_area_ratio = rec.get("area_ratio")
        if rec_area_ratio is None:
            rec_area_ratio = area / float(max(1, mw * mh))
        rec_area_ratio = float(rec_area_ratio)
        large_area_filtered = rec_area_ratio > args.union_max_area_ratio
        seed_count = 0
        for sx, sy in seed_positions:
            if tx0 <= sx < tx1 and ty0 <= sy < ty1:
                mx = int((sx - tx0) * mw / float(tw))
                my = int((sy - ty0) * mh / float(th))
                if 0 <= mx < mw and 0 <= my < mh and mask[my * mw + mx] > 0:
                    seed_count += 1
        touch_tile_boundary = lx0 <= 1 or ly0 <= 1 or lx1 >= mw - 1 or ly1 >= mh - 1
        sam_score = rec.get("sam_score")
        high_score_keep = bool(args.keep_high_score_no_seed and sam_score is not None and float(sam_score) >= args.keep_high_score_no_seed)
        keep = (not large_area_filtered) and (seed_count >= args.min_seeds_in_mask or high_score_keep)
        if keep:
            kept += 1
            for y in range(mh):
                gy = ty0 + int((y + 0.5) * th / float(mh))
                if not (0 <= gy < height):
                    continue
                moff = y * mw
                uoff = gy * width
                for x in range(mw):
                    if mask[moff + x] > 0:
                        gx = tx0 + int((x + 0.5) * tw / float(mw))
                        if 0 <= gx < width:
                            union[uoff + gx] = 255
        else:
            removed += 1
        out = dict(rec)
        out.update({
            "kept_by_learned_standard_seed_filter": bool(keep),
            "learned_seed_count": seed_count,
            "touch_tile_boundary": bool(touch_tile_boundary),
            "area_ratio": rec_area_ratio,
            "large_area_filtered": bool(large_area_filtered),
            "learned_seed_filter_reason": (
                "large_area_ratio" if large_area_filtered else
                "learned_seed_inside_mask" if seed_count >= args.min_seeds_in_mask else
                "high_sam_score_protection" if high_score_keep else
                "no_learned_seed_support"
            ),
        })
        filtered_records.append(out)
        if (n + 1) % 500 == 0:
            print(f"processed_masks={n + 1} kept={kept} removed={removed}", flush=True)

    union_png = output_dir / "filtered_union.png"
    overlay_png = output_dir / "filtered_overlay.png"
    write_png(union_png, width, height, union, 0)
    save_overlay(rgb, union, width, height, overlay_png)
    return {
        "num_input_masks": len(records),
        "num_kept_masks": kept,
        "num_removed_masks": removed,
        "filtered_union": str(union_png),
        "filtered_overlay": str(overlay_png),
        "mask_records": filtered_records,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Learn standard-ball seeds from a clean sample and apply them to a target SAM result.")
    ap.add_argument("--clean-image", required=True)
    ap.add_argument("--clean-summary", required=True)
    ap.add_argument("--target-image", required=True)
    ap.add_argument("--target-summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--clean-area-low-q", type=float, default=25.0)
    ap.add_argument("--clean-area-high-q", type=float, default=75.0)
    ap.add_argument("--max-positive-masks", type=int, default=2500)
    ap.add_argument("--positive-mask-scan-stride", type=int, default=1)
    ap.add_argument("--scan-stride", type=int, default=2)
    ap.add_argument("--seed-score-thresh", type=float, default=0.58)
    ap.add_argument("--min-abs-contrast", type=float, default=8.0)
    ap.add_argument("--seed-nms-radius", type=int, default=3)
    ap.add_argument("--max-seeds", type=int, default=30000)
    ap.add_argument("--min-seeds-in-mask", type=int, default=1)
    ap.add_argument("--keep-high-score-no-seed", type=float, default=0.985)
    ap.add_argument("--union-max-area-ratio", type=float, default=0.9)
    args = ap.parse_args()

    clean_image = Path(args.clean_image).expanduser().resolve()
    clean_summary = Path(args.clean_summary).expanduser().resolve()
    target_image = Path(args.target_image).expanduser().resolve()
    target_summary = Path(args.target_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("learning_standard_ball_model_from_clean_sample", flush=True)
    positives, meta = load_seed_supported_positives(
        clean_image, clean_summary, args.clean_area_low_q, args.clean_area_high_q,
        args.max_positive_masks, args.positive_mask_scan_stride,
    )
    model = learn_model(positives, meta)
    model_path = output_dir / "standard_ball_seed_model.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"model": str(model_path), "meta": meta}, ensure_ascii=False), flush=True)

    print("applying_learned_standard_ball_detector", flush=True)
    seeds, seed_map, seed_stats = apply_detector(target_image, model, args)
    width, height, rgb, _gray = read_bmp_rgb_gray(target_image)
    seed_map_png = output_dir / "learned_seed_map.png"
    seed_overlay_png = output_dir / "learned_seed_overlay.png"
    write_png(seed_map_png, width, height, seed_map, 0)
    write_png(seed_overlay_png, width, height, draw_seed_overlay(rgb, width, height, seeds), 2)
    seed_json = output_dir / "learned_seeds.json"
    seed_json.write_text(json.dumps({"seeds": seeds, "seed_stats": seed_stats}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("filtering_target_masks_by_learned_seeds", flush=True)
    filter_result = filter_masks_with_seeds(target_image, target_summary, seeds, output_dir, args)

    out_summary = {
        "method": "learned_standard_ball_seed_filter",
        "clean_image": str(clean_image),
        "clean_summary": str(clean_summary),
        "target_image": str(target_image),
        "target_summary": str(target_summary),
        "params": vars(args),
        "model_path": str(model_path),
        "model": {k: v for k, v in model.items() if k != "template_profile"},
        "seed_stats": seed_stats,
        "learned_seed_map": str(seed_map_png),
        "learned_seed_overlay": str(seed_overlay_png),
        "learned_seeds_json": str(seed_json),
        **filter_result,
    }
    out_path = output_dir / "filtered_summary.json"
    out_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "filtered_summary": str(out_path),
        "filtered_overlay": filter_result["filtered_overlay"],
        "learned_seed_overlay": str(seed_overlay_png),
        "seed_stats": seed_stats,
        "kept": filter_result["num_kept_masks"],
        "removed": filter_result["num_removed_masks"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
