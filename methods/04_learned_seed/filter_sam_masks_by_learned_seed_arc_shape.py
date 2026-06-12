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
    read_bmp_rgb_gray,
    read_png_gray,
    save_overlay,
    write_png,
)
from filter_sam_masks_by_learned_standard_seeds import (
    apply_detector,
    draw_seed_overlay,
    feature_at,
    learn_model,
    load_seed_supported_positives,
    score_feature,
)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * q / 100.0))
    return vals[max(0, min(len(vals) - 1, idx))]


def fit_reference_area_model(clean_summary_path: Path, low_q: float, high_q: float) -> Dict[str, float]:
    summary = json.loads(clean_summary_path.read_text(encoding="utf-8"))
    records = load_mask_records(summary)
    areas = [float(r.get("area", 0)) for r in records if r.get("union_kept") and float(r.get("area", 0)) > 0]
    low = percentile(areas, low_q)
    high = percentile(areas, high_q)
    mid = [a for a in areas if low <= a <= high]
    single = percentile(mid, 50.0) if mid else percentile(areas, 50.0)
    return {"single_area_low": low, "single_area_high": high, "single_area": single, "reference_area_count": float(len(areas))}


def boundary_points(mask: bytearray, width: int, height: int, bbox: Tuple[int, int, int, int, int]) -> List[Tuple[int, int]]:
    x0, y0, x1, y1, _area = bbox
    pts: List[Tuple[int, int]] = []
    for y in range(y0, y1):
        off = y * width
        for x in range(x0, x1):
            if mask[off + x] <= 0:
                continue
            if (
                x == 0 or mask[off + x - 1] <= 0 or
                x == width - 1 or mask[off + x + 1] <= 0 or
                y == 0 or mask[off - width + x] <= 0 or
                y == height - 1 or mask[off + width + x] <= 0
            ):
                pts.append((x, y))
    return pts


def shape_basics(mask: bytearray, width: int, height: int, bbox: Tuple[int, int, int, int, int]) -> Dict[str, float]:
    x0, y0, x1, y1, area = bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    aspect = max(bw, bh) / float(max(1, min(bw, bh)))
    extent = area / float(max(1, bw * bh))
    perim = len(boundary_points(mask, width, height, bbox))
    circularity = 4.0 * math.pi * area / float(perim * perim) if perim else 0.0
    return {"area": float(area), "aspect_ratio": aspect, "extent": extent, "perimeter": float(perim), "circularity": circularity}


def seeds_inside_mask(mask: bytearray, mw: int, mh: int, tx0: int, ty0: int, tx1: int, ty1: int, seeds: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
    inside: List[Dict[str, Any]] = []
    for seed in seeds:
        sx, sy = int(seed["x"]), int(seed["y"])
        if not (tx0 <= sx < tx1 and ty0 <= sy < ty1):
            continue
        mx = int((sx - tx0) * mw / float(tw))
        my = int((sy - ty0) * mh / float(th))
        if 0 <= mx < mw and 0 <= my < mh and mask[my * mw + mx] > 0:
            s = dict(seed)
            s["mask_x"] = mx
            s["mask_y"] = my
            inside.append(s)
    return inside


def arc_evidence_for_seed(boundary: List[Tuple[int, int]], cx: int, cy: int, r0: float, tol: float, angle_bins: int) -> Dict[str, float]:
    if not boundary:
        return {"arc_points": 0.0, "arc_coverage": 0.0, "mean_radius_error": 999.0}
    bins = [0] * angle_bins
    arc_points = 0
    err_sum = 0.0
    for x, y in boundary:
        dx, dy = x - cx, y - cy
        d = math.hypot(dx, dy)
        err = abs(d - r0)
        if err <= tol:
            arc_points += 1
            err_sum += err
            ang = math.atan2(dy, dx)
            if ang < 0:
                ang += 2.0 * math.pi
            bi = min(angle_bins - 1, int(ang / (2.0 * math.pi) * angle_bins))
            bins[bi] = 1
    return {
        "arc_points": float(arc_points),
        "arc_coverage": sum(bins) / float(angle_bins),
        "mean_radius_error": err_sum / float(max(1, arc_points)),
    }


def mask_arc_components(boundary: List[Tuple[int, int]], seed_list: List[Dict[str, Any]], r0: float, args: argparse.Namespace) -> Dict[str, Any]:
    if not boundary:
        return {
            "component_count": 0,
            "seed_hit_count": 0,
            "boundary_arc_coverage": 0.0,
            "components": [],
        }
    tol = float(args.arc_radius_tolerance)
    cell = max(1, int(math.ceil(r0 + tol)))
    grid: Dict[Tuple[int, int], List[int]] = {}
    for idx, (x, y) in enumerate(boundary):
        grid.setdefault((x // cell, y // cell), []).append(idx)

    arc_points = set()
    point_seed_hits: Dict[int, set[int]] = {}
    for seed_idx, seed in enumerate(seed_list):
        cx, cy = int(seed["mask_x"]), int(seed["mask_y"])
        gcx, gcy = cx // cell, cy // cell
        for gy in range(gcy - 1, gcy + 2):
            for gx in range(gcx - 1, gcx + 2):
                for pi in grid.get((gx, gy), []):
                    x, y = boundary[pi]
                    d = math.hypot(x - cx, y - cy)
                    if abs(d - r0) <= tol:
                        arc_points.add(pi)
                        point_seed_hits.setdefault(pi, set()).add(seed_idx)

    if not arc_points:
        return {
            "component_count": 0,
            "seed_hit_count": 0,
            "boundary_arc_coverage": 0.0,
            "components": [],
        }

    arc_lookup = {boundary[i]: i for i in arc_points}
    visited = set()
    components: List[Dict[str, Any]] = []
    for pi in list(arc_points):
        if pi in visited:
            continue
        stack = [pi]
        visited.add(pi)
        pts: List[int] = []
        seeds_hit = set()
        while stack:
            cur = stack.pop()
            pts.append(cur)
            seeds_hit.update(point_seed_hits.get(cur, set()))
            x, y = boundary[cur]
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ni = arc_lookup.get((x + dx, y + dy))
                    if ni is not None and ni not in visited:
                        visited.add(ni)
                        stack.append(ni)
        if len(pts) >= args.min_arc_component_points:
            components.append({
                "point_count": len(pts),
                "seed_hit_count": len(seeds_hit),
                "seed_hits": sorted(int(x) for x in seeds_hit),
            })

    components.sort(key=lambda c: c["point_count"], reverse=True)
    all_seed_hits = set()
    for comp in components:
        all_seed_hits.update(comp["seed_hits"])
    return {
        "component_count": len(components),
        "seed_hit_count": len(all_seed_hits),
        "boundary_arc_coverage": len(arc_points) / float(max(1, len(boundary))),
        "components": components[:12],
    }


def plausible_cluster_geometry(seed_list: List[Dict[str, Any]], comp_info: Dict[str, Any], r0: float, args: argparse.Namespace) -> Dict[str, Any]:
    hit_seed_ids = set()
    for comp in comp_info.get("components", []):
        hit_seed_ids.update(int(i) for i in comp.get("seed_hits", []))
    hit_seed_ids = {i for i in hit_seed_ids if 0 <= i < len(seed_list)}
    hit_seeds = [seed_list[i] for i in sorted(hit_seed_ids)]

    low = float(args.cluster_min_center_dist_multiple) * r0
    high = float(args.cluster_max_center_dist_multiple) * r0
    plausible_pairs = 0
    distances: List[float] = []
    for i, a in enumerate(hit_seeds):
        ax, ay = float(a["mask_x"]), float(a["mask_y"])
        for b in hit_seeds[i + 1:]:
            bx, by = float(b["mask_x"]), float(b["mask_y"])
            d = math.hypot(ax - bx, ay - by)
            if low <= d <= high:
                plausible_pairs += 1
                distances.append(d)

    ok = len(hit_seeds) >= args.cluster_min_seed_hits and plausible_pairs >= args.cluster_min_plausible_pairs
    return {
        "ok": bool(ok),
        "hit_seed_count": len(hit_seeds),
        "plausible_pair_count": plausible_pairs,
        "mean_plausible_pair_distance": sum(distances) / float(len(distances)) if distances else 0.0,
        "distance_low": low,
        "distance_high": high,
    }


def cluster_photometric_evidence(seed_list: List[Dict[str, Any]], comp_info: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    hit_seed_ids = set()
    for comp in comp_info.get("components", []):
        hit_seed_ids.update(int(i) for i in comp.get("seed_hits", []))
    hit_seed_ids = {i for i in hit_seed_ids if 0 <= i < len(seed_list)}
    hit_seeds = [seed_list[i] for i in sorted(hit_seed_ids)]

    scores = [float(s.get("score", 0.0)) for s in hit_seeds]
    contrasts = [float(s.get("abs_contrast", 0.0)) for s in hit_seeds]
    profile_sims = [float(s.get("profile_similarity", 0.0)) for s in hit_seeds]
    strong = [s for s in hit_seeds if (
        float(s.get("score", 0.0)) >= args.cluster_min_hit_seed_score
        and float(s.get("abs_contrast", 0.0)) >= args.cluster_min_hit_seed_abs_contrast
    )]

    mean_score = sum(scores) / float(len(scores)) if scores else 0.0
    mean_contrast = sum(contrasts) / float(len(contrasts)) if contrasts else 0.0
    mean_profile = sum(profile_sims) / float(len(profile_sims)) if profile_sims else 0.0
    ok = (
        len(strong) >= args.cluster_min_strong_hit_seeds
        and mean_score >= args.cluster_min_mean_hit_seed_score
        and mean_profile >= args.cluster_min_mean_profile_similarity
    )
    return {
        "ok": bool(ok),
        "hit_seed_count": len(hit_seeds),
        "strong_hit_seed_count": len(strong),
        "mean_hit_seed_score": mean_score,
        "mean_hit_seed_abs_contrast": mean_contrast,
        "mean_hit_seed_profile_similarity": mean_profile,
        "min_hit_seed_score": min(scores) if scores else 0.0,
        "max_hit_seed_score": max(scores) if scores else 0.0,
    }


def qualified_cluster_arc_evidence(comp_info: Dict[str, Any], boundary_count: int, args: argparse.Namespace) -> Dict[str, Any]:
    comps = []
    for comp in comp_info.get("components", []):
        if (
            int(comp.get("point_count", 0)) >= args.cluster_min_qualified_arc_points
            and int(comp.get("seed_hit_count", 0)) >= args.cluster_min_qualified_arc_seed_hits
        ):
            comps.append(comp)
    seed_hits = set()
    point_sum = 0
    for comp in comps:
        point_sum += int(comp.get("point_count", 0))
        seed_hits.update(int(i) for i in comp.get("seed_hits", []))
    coverage = point_sum / float(max(1, boundary_count))
    ok = (
        len(comps) >= args.cluster_min_qualified_arc_components
        and len(seed_hits) >= args.cluster_min_qualified_arc_seed_hits_total
        and coverage >= args.cluster_min_qualified_arc_coverage
    )
    return {
        "ok": bool(ok),
        "qualified_component_count": len(comps),
        "qualified_seed_hit_count": len(seed_hits),
        "qualified_point_count": point_sum,
        "qualified_arc_coverage": coverage,
        "qualified_components": comps[:8],
    }


def arc_shape_decision(mask: bytearray, mw: int, mh: int, bbox: Tuple[int, int, int, int, int], seed_list: List[Dict[str, Any]], r0: float, args: argparse.Namespace) -> Tuple[bool, str, Dict[str, Any]]:
    boundary = boundary_points(mask, mw, mh, bbox)
    seed_count = len(seed_list)
    comp_info = mask_arc_components(boundary, seed_list, r0, args)
    component_count = int(comp_info["component_count"])
    seed_hit_count = int(comp_info["seed_hit_count"])
    boundary_arc_coverage = float(comp_info["boundary_arc_coverage"])
    cluster_geometry = plausible_cluster_geometry(seed_list, comp_info, r0, args)
    cluster_photo = cluster_photometric_evidence(seed_list, comp_info, args)
    qualified_arc = qualified_cluster_arc_evidence(comp_info, len(boundary), args)

    # Cluster: the whole mask boundary has multiple standard-radius arc components,
    # and these arc components are supported by multiple learned seeds. Seeds do not
    # need to each explain their own arc; they only confirm that arcs correspond to
    # ball-like centers inside the same mask.
    if (
        component_count >= args.cluster_min_arc_components
        and seed_hit_count >= args.cluster_min_seed_hits
        and boundary_arc_coverage >= args.cluster_min_boundary_arc_coverage
        and qualified_arc["ok"]
        and cluster_geometry["ok"]
        and cluster_photo["ok"]
    ):
        return True, "ball_cluster_multi_arc_mask", {
            "class": "ball_cluster",
            "seed_count": seed_count,
            "arc_component_count": component_count,
            "arc_seed_hit_count": seed_hit_count,
            "boundary_arc_coverage": boundary_arc_coverage,
            "cluster_geometry": cluster_geometry,
            "cluster_photometric_evidence": cluster_photo,
            "qualified_cluster_arc_evidence": qualified_arc,
            "components": comp_info["components"],
        }

    # Single ball: one dominant standard-radius arc component with seed support.
    if (
        component_count >= 1
        and seed_hit_count >= 1
        and boundary_arc_coverage >= args.single_min_boundary_arc_coverage
    ):
        return True, "single_ball_arc_mask", {
            "class": "single_ball",
            "seed_count": seed_count,
            "arc_component_count": component_count,
            "arc_seed_hit_count": seed_hit_count,
            "boundary_arc_coverage": boundary_arc_coverage,
            "cluster_geometry": cluster_geometry,
            "cluster_photometric_evidence": cluster_photo,
            "qualified_cluster_arc_evidence": qualified_arc,
            "components": comp_info["components"],
        }

    if seed_count >= args.seed_count_protection and component_count >= 1:
        return True, "multi_seed_weak_arc_mask_protection", {
            "class": "probable_cluster",
            "seed_count": seed_count,
            "arc_component_count": component_count,
            "arc_seed_hit_count": seed_hit_count,
            "boundary_arc_coverage": boundary_arc_coverage,
            "cluster_geometry": cluster_geometry,
            "cluster_photometric_evidence": cluster_photo,
            "qualified_cluster_arc_evidence": qualified_arc,
            "components": comp_info["components"],
        }

    return False, "no_standard_ball_arc", {
        "class": "artifact",
        "seed_count": seed_count,
        "arc_component_count": component_count,
        "arc_seed_hit_count": seed_hit_count,
        "boundary_arc_coverage": boundary_arc_coverage,
        "cluster_geometry": cluster_geometry,
        "cluster_photometric_evidence": cluster_photo,
        "qualified_cluster_arc_evidence": qualified_arc,
        "components": comp_info["components"],
    }


def save_class_overlay(rgb: bytearray, class_map: bytearray, width: int, height: int, path: Path) -> None:
    out = bytearray(rgb)
    for i, cls in enumerate(class_map):
        if not cls:
            continue
        j = i * 3
        if cls == 2:
            color = (255, 40, 40)
        else:
            color = (30, 144, 255)
        out[j] = int(out[j] * 0.50 + color[0] * 0.50)
        out[j + 1] = int(out[j + 1] * 0.50 + color[1] * 0.50)
        out[j + 2] = int(out[j + 2] * 0.50 + color[2] * 0.50)
    write_png(path, width, height, out, 2)


def class_map_to_png(class_map: bytearray) -> bytearray:
    out = bytearray(len(class_map))
    for i, cls in enumerate(class_map):
        if cls == 1:
            out[i] = 128
        elif cls == 2:
            out[i] = 255
    return out


def mask_centroid_global(
    mask: bytearray,
    mw: int,
    mh: int,
    bbox: Tuple[int, int, int, int, int],
    tx0: int,
    ty0: int,
    tx1: int,
    ty1: int,
) -> Tuple[int, int]:
    x0, y0, x1, y1, area = bbox
    sx = sy = count = 0
    for y in range(y0, y1):
        off = y * mw
        for x in range(x0, x1):
            if mask[off + x] > 0:
                sx += x
                sy += y
                count += 1
    if count <= 0:
        mx = (x0 + x1) * 0.5
        my = (y0 + y1) * 0.5
    else:
        mx = sx / float(count)
        my = sy / float(count)
    tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
    gx = tx0 + int((mx + 0.5) * tw / float(max(1, mw)))
    gy = ty0 + int((my + 0.5) * th / float(max(1, mh)))
    return gx, gy


def best_local_seedless_feature(
    gray: bytearray,
    integ: List[int],
    width: int,
    height: int,
    model: Dict[str, Any],
    cx: int,
    cy: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    r0 = int(model["standard_radius"])
    best: Dict[str, Any] | None = None
    search = int(args.rescue_local_search_radius)
    for y in range(cy - search, cy + search + 1):
        for x in range(cx - search, cx + search + 1):
            if not (2 * r0 + 2 <= x < width - 2 * r0 - 2 and 2 * r0 + 2 <= y < height - 2 * r0 - 2):
                continue
            f = feature_at(gray, integ, width, height, x, y, r0)
            sc = score_feature(f, model)
            item = {**f, **sc}
            if best is None or float(item["score"]) > float(best["score"]):
                best = item
    return best or {"score": 0.0, "abs_contrast": 0.0, "profile_similarity": 0.0, "radial_balance_score": 0.0}


def seedless_single_rescue_decision(
    mask: bytearray,
    mw: int,
    mh: int,
    bbox: Tuple[int, int, int, int, int],
    tx0: int,
    ty0: int,
    tx1: int,
    ty1: int,
    gray: bytearray,
    integ: List[int],
    width: int,
    height: int,
    area_model: Dict[str, float],
    model: Dict[str, Any],
    basics: Dict[str, float],
    large_union: bool,
    seed_count: int,
    args: argparse.Namespace,
) -> Tuple[bool, str, Dict[str, Any]]:
    area = float(bbox[4])
    single_lo = area_model["single_area_low"] * args.rescue_single_area_low_scale
    single_hi = area_model["single_area_high"] * args.rescue_single_area_high_scale
    meta: Dict[str, Any] = {
        "enabled": bool(args.enable_seedless_single_rescue),
        "single_area_low": single_lo,
        "single_area_high": single_hi,
    }
    if not args.enable_seedless_single_rescue:
        return False, "rescue_disabled", meta
    if large_union or seed_count > 0:
        return False, "rescue_not_applicable", meta
    if not (single_lo <= area <= single_hi):
        return False, "rescue_area_fail", meta
    if basics["aspect_ratio"] > args.rescue_shape_max_aspect_ratio:
        return False, "rescue_shape_fail", meta
    if basics["extent"] < args.rescue_shape_min_extent or basics["extent"] > args.rescue_shape_max_extent:
        return False, "rescue_shape_fail", meta
    if basics["circularity"] < args.rescue_shape_min_circularity:
        return False, "rescue_shape_fail", meta

    cx, cy = mask_centroid_global(mask, mw, mh, bbox, tx0, ty0, tx1, ty1)
    best = best_local_seedless_feature(gray, integ, width, height, model, cx, cy, args)
    meta.update({
        "centroid_x": cx,
        "centroid_y": cy,
        "best_score": float(best.get("score", 0.0)),
        "best_abs_contrast": float(best.get("abs_contrast", 0.0)),
        "best_profile_similarity": float(best.get("profile_similarity", 0.0)),
        "best_radial_balance_score": float(best.get("radial_balance_score", 0.0)),
    })
    ok = (
        float(best.get("score", 0.0)) >= args.rescue_min_seed_score
        and float(best.get("abs_contrast", 0.0)) >= args.rescue_min_abs_contrast
        and float(best.get("profile_similarity", 0.0)) >= args.rescue_min_profile_similarity
        and float(best.get("radial_balance_score", 0.0)) >= args.rescue_min_radial_balance_score
    )
    return bool(ok), ("seedless_single_rescue" if ok else "rescue_photometric_fail"), meta


def two_lobe_centers_global(
    mask: bytearray,
    mw: int,
    mh: int,
    bbox: Tuple[int, int, int, int, int],
    tx0: int,
    ty0: int,
    tx1: int,
    ty1: int,
) -> Dict[str, Any]:
    x0, y0, x1, y1, area = bbox
    pts: List[Tuple[int, int]] = []
    sx = sy = 0.0
    for y in range(y0, y1):
        off = y * mw
        for x in range(x0, x1):
            if mask[off + x] > 0:
                pts.append((x, y))
                sx += x
                sy += y
    if len(pts) < 2:
        return {"ok": False, "reason": "too_few_pixels"}
    mx = sx / float(len(pts))
    my = sy / float(len(pts))
    cxx = cyy = cxy = 0.0
    for x, y in pts:
        dx, dy = x - mx, y - my
        cxx += dx * dx
        cyy += dy * dy
        cxy += dx * dy
    angle = 0.5 * math.atan2(2.0 * cxy, cxx - cyy) if abs(cxy) > 1e-9 or abs(cxx - cyy) > 1e-9 else 0.0
    ux, uy = math.cos(angle), math.sin(angle)
    proj = [(x - mx) * ux + (y - my) * uy for x, y in pts]
    split = percentile([float(p) for p in proj], 50.0)
    left = [(x, y) for (x, y), p in zip(pts, proj) if p <= split]
    right = [(x, y) for (x, y), p in zip(pts, proj) if p > split]
    min_lobe_area = max(3, int(round(area * 0.22)))
    if len(left) < min_lobe_area or len(right) < min_lobe_area:
        return {"ok": False, "reason": "imbalanced_lobes", "left_area": len(left), "right_area": len(right)}

    def center(group: List[Tuple[int, int]]) -> Tuple[float, float]:
        return (sum(x for x, _ in group) / float(len(group)), sum(y for _, y in group) / float(len(group)))

    lcx, lcy = center(left)
    rcx, rcy = center(right)
    tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
    g1x = tx0 + int((lcx + 0.5) * tw / float(max(1, mw)))
    g1y = ty0 + int((lcy + 0.5) * th / float(max(1, mh)))
    g2x = tx0 + int((rcx + 0.5) * tw / float(max(1, mw)))
    g2y = ty0 + int((rcy + 0.5) * th / float(max(1, mh)))
    local_dist = math.hypot(lcx - rcx, lcy - rcy)
    global_dist = math.hypot(g1x - g2x, g1y - g2y)
    return {
        "ok": True,
        "left_area": len(left),
        "right_area": len(right),
        "local_centers": [[lcx, lcy], [rcx, rcy]],
        "global_centers": [[g1x, g1y], [g2x, g2y]],
        "local_center_distance": local_dist,
        "global_center_distance": global_dist,
        "principal_angle": angle,
    }


def two_ball_connected_rescue_decision(
    mask: bytearray,
    mw: int,
    mh: int,
    bbox: Tuple[int, int, int, int, int],
    tx0: int,
    ty0: int,
    tx1: int,
    ty1: int,
    gray: bytearray,
    integ: List[int],
    width: int,
    height: int,
    area_model: Dict[str, float],
    model: Dict[str, Any],
    basics: Dict[str, float],
    large_union: bool,
    seed_count: int,
    args: argparse.Namespace,
) -> Tuple[bool, str, Dict[str, Any]]:
    area = float(bbox[4])
    single_area = float(area_model["single_area"])
    area_lo = single_area * args.two_ball_area_low_multiple
    area_hi = single_area * args.two_ball_area_high_multiple
    meta: Dict[str, Any] = {
        "enabled": bool(args.enable_two_ball_connected_rescue),
        "area_low": area_lo,
        "area_high": area_hi,
        "seed_count": int(seed_count),
    }
    if not args.enable_two_ball_connected_rescue:
        return False, "two_ball_rescue_disabled", meta
    if large_union:
        return False, "two_ball_large_union", meta
    if not (area_lo <= area <= area_hi):
        return False, "two_ball_area_fail", meta
    if seed_count < args.two_ball_min_seeds_in_mask:
        return False, "two_ball_seed_guard_fail", meta
    if basics["aspect_ratio"] > args.two_ball_max_aspect_ratio:
        return False, "two_ball_shape_fail", meta
    if basics["extent"] < args.two_ball_min_extent or basics["extent"] > args.two_ball_max_extent:
        return False, "two_ball_shape_fail", meta
    if basics["circularity"] < args.two_ball_min_circularity:
        return False, "two_ball_shape_fail", meta

    lobes = two_lobe_centers_global(mask, mw, mh, bbox, tx0, ty0, tx1, ty1)
    meta["lobe_geometry"] = lobes
    if not lobes.get("ok"):
        return False, "two_ball_lobe_fail", meta
    r0 = float(model["standard_radius"])
    dist = float(lobes.get("global_center_distance", 0.0))
    dist_lo = r0 * args.two_ball_min_center_dist_multiple
    dist_hi = r0 * args.two_ball_max_center_dist_multiple
    meta.update({"center_distance_low": dist_lo, "center_distance_high": dist_hi})
    if not (dist_lo <= dist <= dist_hi):
        return False, "two_ball_lobe_distance_fail", meta

    scores: List[Dict[str, Any]] = []
    for gx, gy in lobes.get("global_centers", []):
        best = best_local_seedless_feature(gray, integ, width, height, model, int(gx), int(gy), args)
        scores.append(best)
    strong = [s for s in scores if (
        float(s.get("score", 0.0)) >= args.two_ball_min_lobe_score
        and float(s.get("profile_similarity", 0.0)) >= args.two_ball_min_lobe_profile_similarity
        and float(s.get("abs_contrast", 0.0)) >= args.two_ball_min_lobe_abs_contrast
    )]
    mean_score = sum(float(s.get("score", 0.0)) for s in scores) / float(max(1, len(scores)))
    mean_profile = sum(float(s.get("profile_similarity", 0.0)) for s in scores) / float(max(1, len(scores)))
    mean_contrast = sum(float(s.get("abs_contrast", 0.0)) for s in scores) / float(max(1, len(scores)))
    meta.update({
        "lobe_scores": scores,
        "strong_lobe_count": len(strong),
        "mean_lobe_score": mean_score,
        "mean_lobe_profile_similarity": mean_profile,
        "mean_lobe_abs_contrast": mean_contrast,
    })
    ok = (
        len(strong) >= args.two_ball_min_strong_lobes
        and mean_score >= args.two_ball_min_mean_lobe_score
        and mean_profile >= args.two_ball_min_mean_lobe_profile_similarity
        and mean_contrast >= args.two_ball_min_mean_lobe_abs_contrast
    )
    return bool(ok), ("two_ball_connected_rescue" if ok else "two_ball_photometric_fail"), meta


def single_area_decision(area: float, area_model: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, str, Dict[str, float]]:
    single_lo = area_model["single_area_low"] * args.single_area_low_scale
    single_hi = area_model["single_area_high"] * args.single_area_high_scale
    ok = single_lo <= area <= single_hi
    reason = "single_area_ok" if ok else ("area_too_small" if area < single_lo else "area_too_large")
    return ok, reason, {
        "area_mode": "single",
        "single_area_low": single_lo,
        "single_area_high": single_hi,
    }


def class_aware_area_decision(area: float, arc_class: str, area_model: Dict[str, float], args: argparse.Namespace, arc_features: Dict[str, Any] | None = None) -> Tuple[bool, str, Dict[str, float]]:
    if arc_class in {"ball_cluster", "probable_cluster"}:
        single_lo = area_model["single_area_low"] * args.single_area_low_scale
        single_hi = area_model["single_area_high"] * args.single_area_high_scale
        photo = (arc_features or {}).get("cluster_photometric_evidence", {})
        verified_instances = int(photo.get("strong_hit_seed_count", 0))
        if verified_instances <= 0:
            verified_instances = int((arc_features or {}).get("arc_seed_hit_count", 0))
        max_supported_area = max(1, verified_instances) * single_hi * args.cluster_max_area_per_verified_seed
        ok = area <= max_supported_area
        return ok, ("cluster_instance_area_supported" if ok else "cluster_area_under_supported"), {
            "area_mode": "cluster_instance_supported",
            "single_area_low": single_lo,
            "single_area_high": single_hi,
            "verified_instances": float(verified_instances),
            "max_supported_area": max_supported_area,
            "area_per_verified_instance": area / float(max(1, verified_instances)),
        }
    if arc_class == "single_ball":
        return single_area_decision(area, area_model, args)
    return False, "artifact_area_not_evaluated", {"area_mode": "artifact"}


def loose_shape_ok(features: Dict[str, float], args: argparse.Namespace) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if features["aspect_ratio"] > args.shape_max_aspect_ratio:
        reasons.append("high_aspect_ratio")
    if features["extent"] < args.shape_min_extent:
        reasons.append("low_extent")
    if features["extent"] > args.shape_max_extent:
        reasons.append("high_extent")
    return len(reasons) == 0, reasons


def main() -> None:
    ap = argparse.ArgumentParser(description="Learned seed + standard-radius arc-shape filtering for ball masks.")
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
    ap.add_argument("--seed-score-thresh", type=float, default=0.66)
    ap.add_argument("--min-abs-contrast", type=float, default=12.0)
    ap.add_argument("--seed-nms-radius", type=int, default=3)
    ap.add_argument("--max-seeds", type=int, default=20000)
    ap.add_argument("--min-seeds-in-mask", type=int, default=1)
    ap.add_argument("--enable-seedless-single-rescue", action="store_true")
    ap.add_argument("--rescue-single-area-low-scale", type=float, default=0.90)
    ap.add_argument("--rescue-single-area-high-scale", type=float, default=1.10)
    ap.add_argument("--rescue-shape-max-aspect-ratio", type=float, default=1.45)
    ap.add_argument("--rescue-shape-min-extent", type=float, default=0.55)
    ap.add_argument("--rescue-shape-max-extent", type=float, default=0.98)
    ap.add_argument("--rescue-shape-min-circularity", type=float, default=0.75)
    ap.add_argument("--rescue-local-search-radius", type=int, default=2)
    ap.add_argument("--rescue-min-seed-score", type=float, default=0.72)
    ap.add_argument("--rescue-min-abs-contrast", type=float, default=18.0)
    ap.add_argument("--rescue-min-profile-similarity", type=float, default=0.58)
    ap.add_argument("--rescue-min-radial-balance-score", type=float, default=0.35)
    ap.add_argument("--union-max-area-ratio", type=float, default=0.9)

    ap.add_argument("--area-reference-low-q", type=float, default=25.0)
    ap.add_argument("--area-reference-high-q", type=float, default=75.0)
    ap.add_argument("--single-area-low-scale", type=float, default=0.85)
    ap.add_argument("--single-area-high-scale", type=float, default=1.25)
    ap.add_argument("--cluster-min-seeds", type=int, default=2)
    ap.add_argument("--cluster-area-min-multiple", type=float, default=1.35)
    ap.add_argument("--cluster-area-max-multiple", type=float, default=8.0)

    ap.add_argument("--arc-radius-tolerance", type=float, default=1.6)
    ap.add_argument("--min-arc-component-points", type=int, default=3)
    ap.add_argument("--single-min-boundary-arc-coverage", type=float, default=0.12)
    ap.add_argument("--cluster-min-arc-components", type=int, default=2)
    ap.add_argument("--cluster-min-seed-hits", type=int, default=2)
    ap.add_argument("--cluster-min-boundary-arc-coverage", type=float, default=0.12)
    ap.add_argument("--cluster-min-qualified-arc-points", type=int, default=7)
    ap.add_argument("--cluster-min-qualified-arc-seed-hits", type=int, default=1)
    ap.add_argument("--cluster-min-qualified-arc-components", type=int, default=2)
    ap.add_argument("--cluster-min-qualified-arc-seed-hits-total", type=int, default=2)
    ap.add_argument("--cluster-min-qualified-arc-coverage", type=float, default=0.10)
    ap.add_argument("--cluster-min-center-dist-multiple", type=float, default=1.2)
    ap.add_argument("--cluster-max-center-dist-multiple", type=float, default=4.2)
    ap.add_argument("--cluster-min-plausible-pairs", type=int, default=1)
    ap.add_argument("--cluster-min-strong-hit-seeds", type=int, default=2)
    ap.add_argument("--cluster-min-hit-seed-score", type=float, default=0.72)
    ap.add_argument("--cluster-min-hit-seed-abs-contrast", type=float, default=18.0)
    ap.add_argument("--cluster-min-mean-hit-seed-score", type=float, default=0.70)
    ap.add_argument("--cluster-min-mean-profile-similarity", type=float, default=0.52)
    ap.add_argument("--cluster-max-area-per-verified-seed", type=float, default=2.0)
    ap.add_argument("--seed-count-protection", type=int, default=4)
    ap.add_argument("--enable-two-ball-connected-rescue", action="store_true")
    ap.add_argument("--two-ball-area-low-multiple", type=float, default=1.35)
    ap.add_argument("--two-ball-area-high-multiple", type=float, default=2.85)
    ap.add_argument("--two-ball-min-seeds-in-mask", type=int, default=0)
    ap.add_argument("--two-ball-max-aspect-ratio", type=float, default=2.35)
    ap.add_argument("--two-ball-min-extent", type=float, default=0.34)
    ap.add_argument("--two-ball-max-extent", type=float, default=0.94)
    ap.add_argument("--two-ball-min-circularity", type=float, default=0.34)
    ap.add_argument("--two-ball-min-center-dist-multiple", type=float, default=1.15)
    ap.add_argument("--two-ball-max-center-dist-multiple", type=float, default=3.8)
    ap.add_argument("--two-ball-min-lobe-score", type=float, default=0.34)
    ap.add_argument("--two-ball-min-lobe-profile-similarity", type=float, default=0.22)
    ap.add_argument("--two-ball-min-lobe-abs-contrast", type=float, default=4.0)
    ap.add_argument("--two-ball-min-strong-lobes", type=int, default=1)
    ap.add_argument("--two-ball-min-mean-lobe-score", type=float, default=0.30)
    ap.add_argument("--two-ball-min-mean-lobe-profile-similarity", type=float, default=0.18)
    ap.add_argument("--two-ball-min-mean-lobe-abs-contrast", type=float, default=3.0)

    ap.add_argument("--shape-max-aspect-ratio", type=float, default=2.8)
    ap.add_argument("--shape-min-extent", type=float, default=0.14)
    ap.add_argument("--shape-max-extent", type=float, default=0.99)
    args = ap.parse_args()

    clean_image = Path(args.clean_image).expanduser().resolve()
    clean_summary = Path(args.clean_summary).expanduser().resolve()
    target_image = Path(args.target_image).expanduser().resolve()
    target_summary = Path(args.target_summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("learning_standard_ball_model_from_clean_sample", flush=True)
    positives, meta = load_seed_supported_positives(clean_image, clean_summary, args.clean_area_low_q, args.clean_area_high_q, args.max_positive_masks, args.positive_mask_scan_stride)
    model = learn_model(positives, meta)
    model_path = output_dir / "standard_ball_seed_model.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    area_model = fit_reference_area_model(clean_summary, args.area_reference_low_q, args.area_reference_high_q)
    print(json.dumps({"model": str(model_path), "meta": meta, "area_model": area_model}, ensure_ascii=False), flush=True)

    print("applying_strict_learned_standard_ball_detector", flush=True)
    seeds, seed_map, seed_stats = apply_detector(target_image, model, args)
    width, height, rgb, gray = read_bmp_rgb_gray(target_image)
    integ = build_integral(gray, width, height)
    seed_map_png = output_dir / "strict_learned_seed_map.png"
    seed_overlay_png = output_dir / "strict_learned_seed_overlay.png"
    write_png(seed_map_png, width, height, seed_map, 0)
    write_png(seed_overlay_png, width, height, draw_seed_overlay(rgb, width, height, seeds), 2)

    summary = json.loads(target_summary.read_text(encoding="utf-8"))
    records = load_mask_records(summary)
    union = bytearray(width * height)
    class_map = bytearray(width * height)
    out_records: List[Dict[str, Any]] = []
    kept = removed = 0
    reason_counts: Dict[str, int] = {}
    r0 = float(model["standard_radius"])

    print("filtering_by_arc_shape", flush=True)
    for n, rec in enumerate(records):
        mask_path = Path(str(rec.get("mask_file")))
        if not mask_path.is_file():
            continue
        mw, mh, mask = read_png_gray(mask_path)
        bbox = mask_bbox(mask, mw, mh)
        _lx0, _ly0, _lx1, _ly1, area = bbox
        tx0, ty0, tx1, ty1 = [int(v) for v in rec.get("box_xyxy", [0, 0, width, height])]
        seed_list = seeds_inside_mask(mask, mw, mh, tx0, ty0, tx1, ty1, seeds)
        seed_count = len(seed_list)
        rec_area_ratio = float(rec.get("area_ratio") if rec.get("area_ratio") is not None else area / float(max(1, mw * mh)))
        large_union = rec_area_ratio > args.union_max_area_ratio
        basics = shape_basics(mask, mw, mh, bbox)
        shape_ok, shape_reasons = loose_shape_ok(basics, args)
        arc_ok, arc_reason, arc_features = arc_shape_decision(mask, mw, mh, bbox, seed_list, r0, args)
        arc_class = str(arc_features.get("class", "artifact"))
        area_ok, area_reason, area_meta = class_aware_area_decision(float(area), arc_class, area_model, args, arc_features)
        seed_ok = seed_count >= args.min_seeds_in_mask
        rescue_ok = False
        rescue_reason = "not_attempted"
        rescue_features: Dict[str, Any] = {}
        two_ball_ok = False
        two_ball_reason = "not_attempted"
        two_ball_features: Dict[str, Any] = {}

        keep = (not large_union) and seed_ok and area_ok and shape_ok and arc_ok
        if not keep and not seed_ok:
            rescue_ok, rescue_reason, rescue_features = seedless_single_rescue_decision(
                mask, mw, mh, bbox, tx0, ty0, tx1, ty1, gray, integ, width, height,
                area_model, model, basics, large_union, seed_count, args
            )
            if rescue_ok:
                keep = True
                arc_class = "single_ball"
                arc_ok = True
                arc_reason = rescue_reason
                area_ok = True
                area_reason = "single_area_rescued"
                area_meta = {
                    "area_mode": "seedless_single_rescue",
                    "single_area_low": rescue_features.get("single_area_low"),
                    "single_area_high": rescue_features.get("single_area_high"),
                }
                shape_ok = True
                shape_reasons = []
                arc_features = {
                    "class": "single_ball",
                    "seed_count": 0,
                    "rescue_features": rescue_features,
                }
        if not keep:
            two_ball_ok, two_ball_reason, two_ball_features = two_ball_connected_rescue_decision(
                mask, mw, mh, bbox, tx0, ty0, tx1, ty1, gray, integ, width, height,
                area_model, model, basics, large_union, seed_count, args
            )
            if two_ball_ok:
                keep = True
                arc_class = "ball_cluster"
                arc_ok = True
                arc_reason = two_ball_reason
                area_ok = True
                area_reason = "two_ball_connected_area_rescued"
                area_meta = {
                    "area_mode": "two_ball_connected_rescue",
                    "area_low": two_ball_features.get("area_low"),
                    "area_high": two_ball_features.get("area_high"),
                }
                shape_ok = True
                shape_reasons = []
                arc_features = {
                    "class": "ball_cluster",
                    "subclass": "two_ball_connected",
                    "seed_count": seed_count,
                    "two_ball_connected_rescue_features": two_ball_features,
                }
        if keep:
            kept += 1
            class_value = 2 if arc_class in {"ball_cluster", "probable_cluster"} else 1
            for y in range(mh):
                gy = ty0 + int((y + 0.5) * max(1, ty1 - ty0) / float(mh))
                if not (0 <= gy < height):
                    continue
                moff = y * mw
                uoff = gy * width
                for x in range(mw):
                    if mask[moff + x] > 0:
                        gx = tx0 + int((x + 0.5) * max(1, tx1 - tx0) / float(mw))
                        if 0 <= gx < width:
                            ui = uoff + gx
                            union[ui] = 255
                            if class_value == 2 or class_map[ui] == 0:
                                class_map[ui] = class_value
            reason = arc_reason
        else:
            removed += 1
            if large_union:
                reason = "large_union_area"
            elif not seed_ok:
                reason = rescue_reason if rescue_reason not in {"not_attempted", "rescue_disabled", "rescue_not_applicable"} else "no_strict_seed"
            elif not area_ok:
                reason = area_reason
            elif not shape_ok:
                reason = "shape_fail"
            elif not arc_ok:
                reason = arc_reason
            else:
                reason = "removed"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

        out = dict(rec)
        out.update({
            "kept_by_arc_shape_filter": bool(keep),
            "learned_seed_count": seed_count,
            "seedless_rescue_attempted": bool(rescue_reason not in {"not_attempted", "rescue_disabled", "rescue_not_applicable"}),
            "seedless_rescue_pass": bool(rescue_ok),
            "seedless_rescue_reason": rescue_reason,
            "seedless_rescue_features": rescue_features,
            "two_ball_connected_rescue_attempted": bool(two_ball_reason != "not_attempted"),
            "two_ball_connected_rescue_pass": bool(two_ball_ok),
            "two_ball_connected_rescue_reason": two_ball_reason,
            "two_ball_connected_rescue_features": two_ball_features,
            "area_pass": bool(area_ok),
            "area_reason": area_reason,
            "area_model": area_meta,
            "shape_pass": bool(shape_ok),
            "shape_reasons": shape_reasons,
            "shape_basics": basics,
            "arc_pass": bool(arc_ok),
            "arc_reason": arc_reason,
            "arc_features": arc_features,
            "large_union_filtered": bool(large_union),
            "final_reason": reason,
        })
        out_records.append(out)
        if (n + 1) % 500 == 0:
            print(f"processed_masks={n + 1} kept={kept} removed={removed}", flush=True)

    union_png = output_dir / "filtered_union.png"
    overlay_png = output_dir / "filtered_overlay.png"
    class_map_png = output_dir / "filtered_class_map.png"
    class_overlay_png = output_dir / "filtered_class_overlay.png"
    write_png(union_png, width, height, union, 0)
    save_overlay(rgb, union, width, height, overlay_png)
    write_png(class_map_png, width, height, class_map_to_png(class_map), 0)
    save_class_overlay(rgb, class_map, width, height, class_overlay_png)
    out_summary = {
        "method": "learned_seed_arc_shape_single_area_cluster_area_skipped_filter",
        "clean_image": str(clean_image),
        "clean_summary": str(clean_summary),
        "target_image": str(target_image),
        "target_summary": str(target_summary),
        "params": vars(args),
        "model_path": str(model_path),
        "model": {k: v for k, v in model.items() if k != "template_profile"},
        "area_reference_model": area_model,
        "active_filter_modules": [
            "learned_standard_seed",
            "single_reference_area",
            "mask_arc_geometry",
            "cluster_photometric_evidence",
            "cluster_instance_area_support",
            "optional_seedless_single_rescue",
            "optional_two_ball_connected_rescue"
        ],
        "pruned_filter_modules": [
            "cluster_spatial_explanation",
            "cluster_boundary_image_evidence"
        ],
        "seed_stats": seed_stats,
        "strict_learned_seed_map": str(seed_map_png),
        "strict_learned_seed_overlay": str(seed_overlay_png),
        "num_input_masks": len(records),
        "num_kept_masks": kept,
        "num_removed_masks": removed,
        "reason_counts": reason_counts,
        "filtered_union": str(union_png),
        "filtered_overlay": str(overlay_png),
        "filtered_class_map": str(class_map_png),
        "filtered_class_overlay": str(class_overlay_png),
        "class_overlay_legend": {"single_ball": "blue", "ball_cluster_or_probable_cluster": "red"},
        "mask_records": out_records,
    }
    out_path = output_dir / "filtered_summary.json"
    out_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "filtered_summary": str(out_path),
        "filtered_overlay": str(overlay_png),
        "filtered_class_overlay": str(class_overlay_png),
        "strict_learned_seed_overlay": str(seed_overlay_png),
        "seed_stats": seed_stats,
        "kept": kept,
        "removed": removed,
        "reason_counts": reason_counts,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
