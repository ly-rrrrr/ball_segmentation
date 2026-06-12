#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_bmp_rgb_gray(path: Path) -> Tuple[int, int, bytearray, bytearray]:
    data = path.read_bytes()
    if data[:2] != b"BM":
        raise ValueError(f"not a BMP file: {path}")
    off = struct.unpack_from("<I", data, 10)[0]
    header_size = struct.unpack_from("<I", data, 14)[0]
    if header_size < 40:
        raise ValueError("unsupported BMP header")
    width = struct.unpack_from("<i", data, 18)[0]
    raw_height = struct.unpack_from("<i", data, 22)[0]
    planes = struct.unpack_from("<H", data, 26)[0]
    bits = struct.unpack_from("<H", data, 28)[0]
    comp = struct.unpack_from("<I", data, 30)[0]
    if planes != 1 or bits not in (24, 32) or comp != 0:
        raise ValueError(f"unsupported BMP format: bits={bits}, compression={comp}")
    height = abs(raw_height)
    top_down = raw_height < 0
    row_stride = ((width * bits + 31) // 32) * 4
    rgb = bytearray(width * height * 3)
    gray = bytearray(width * height)
    bpp = bits // 8
    for y in range(height):
        src_y = y if top_down else (height - 1 - y)
        row_off = off + src_y * row_stride
        for x in range(width):
            b = data[row_off + x * bpp]
            g = data[row_off + x * bpp + 1]
            r = data[row_off + x * bpp + 2]
            i = y * width + x
            j = i * 3
            rgb[j:j + 3] = bytes((r, g, b))
            gray[i] = (77 * r + 150 * g + 29 * b) >> 8
    return width, height, rgb, gray


def png_unfilter(ft: int, row: bytearray, prev: bytearray, bpp: int) -> bytearray:
    out = bytearray(row)
    if ft == 0:
        return out
    if ft == 1:
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            out[i] = (out[i] + left) & 255
    elif ft == 2:
        for i in range(len(out)):
            out[i] = (out[i] + prev[i]) & 255
    elif ft == 3:
        for i in range(len(out)):
            left = out[i - bpp] if i >= bpp else 0
            up = prev[i]
            out[i] = (out[i] + ((left + up) >> 1)) & 255
    elif ft == 4:
        for i in range(len(out)):
            a = out[i - bpp] if i >= bpp else 0
            b = prev[i]
            c = prev[i - bpp] if i >= bpp else 0
            p = a + b - c
            pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
            pr = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
            out[i] = (out[i] + pr) & 255
    else:
        raise ValueError(f"unsupported PNG filter: {ft}")
    return out


def read_png_gray(path: Path) -> Tuple[int, int, bytearray]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG file: {path}")
    pos = 8
    width = height = color_type = bit_depth = None
    idat = bytearray()
    while pos < len(data):
        n = struct.unpack_from(">I", data, pos)[0]
        typ = data[pos + 4:pos + 8]
        chunk = data[pos + 8:pos + 8 + n]
        pos += 12 + n
        if typ == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack_from(">IIBB", chunk, 0)
        elif typ == b"IDAT":
            idat.extend(chunk)
        elif typ == b"IEND":
            break
    if bit_depth != 8 or color_type not in (0, 2, 6):
        raise ValueError(f"unsupported PNG format: bit_depth={bit_depth}, color_type={color_type}, path={path}")
    channels = 1 if color_type == 0 else (3 if color_type == 2 else 4)
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    prev = bytearray(stride)
    out = bytearray(width * height)
    p = 0
    for y in range(height):
        ft = raw[p]
        p += 1
        row = png_unfilter(ft, bytearray(raw[p:p + stride]), prev, channels)
        p += stride
        if channels == 1:
            out[y * width:(y + 1) * width] = row
        else:
            dst = y * width
            for x in range(width):
                r, g, b = row[x * channels], row[x * channels + 1], row[x * channels + 2]
                out[dst + x] = (77 * r + 150 * g + 29 * b) >> 8
        prev = row
    return width, height, out


def write_png(path: Path, width: int, height: int, data: bytearray, color_type: int) -> None:
    channels = 1 if color_type == 0 else 3
    raw = bytearray()
    stride = width * channels
    for y in range(height):
        raw.append(0)
        raw.extend(data[y * stride:(y + 1) * stride])

    def chunk(typ: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + typ
            + payload
            + struct.pack(">I", zlib.crc32(typ + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))


def build_integral(gray: bytearray, width: int, height: int) -> List[int]:
    integ = [0] * ((width + 1) * (height + 1))
    sw = width + 1
    for y in range(height):
        row_sum = 0
        base = y * width
        out = (y + 1) * sw
        prev = y * sw
        for x in range(width):
            row_sum += gray[base + x]
            integ[out + x + 1] = integ[prev + x + 1] + row_sum
    return integ


def rect_sum(integ: List[int], width: int, x0: int, y0: int, x1: int, y1: int) -> int:
    sw = width + 1
    return integ[y1 * sw + x1] - integ[y0 * sw + x1] - integ[y1 * sw + x0] + integ[y0 * sw + x0]


def rect_mean(integ: List[int], width: int, height: int, x: int, y: int, r: int) -> float:
    x0, y0 = max(0, x - r), max(0, y - r)
    x1, y1 = min(width, x + r + 1), min(height, y + r + 1)
    return rect_sum(integ, width, x0, y0, x1, y1) / float((x1 - x0) * (y1 - y0))


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * q / 100.0))
    return vals[max(0, min(len(vals) - 1, idx))]


def radial_balance(gray: bytearray, width: int, height: int, x: int, y: int, r: int, center: float) -> float:
    diffs: List[float] = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        sx = max(0, min(width - 1, x + dx * r))
        sy = max(0, min(height - 1, y + dy * r))
        diffs.append(abs(float(gray[sy * width + sx]) - center))
    m = sum(diffs) / len(diffs)
    if m <= 1e-6:
        return 999.0
    return (max(diffs) - min(diffs)) / m


def detect_ball_seeds(
    gray: bytearray,
    width: int,
    height: int,
    radii: List[int],
    stride: int,
    q: float,
    min_response: float,
    max_seeds: int,
    nms_radius: int,
    radial_balance_max: float,
) -> Tuple[List[Dict[str, Any]], bytearray, Dict[str, float]]:
    integ = build_integral(gray, width, height)
    margin = max(radii) * 2 + 1
    candidates: List[Tuple[float, int, int, int, float, float]] = []
    responses: List[float] = []
    for y in range(margin, height - margin, stride):
        for x in range(margin, width - margin, stride):
            best = 0.0
            best_r = radii[0]
            best_center = 0.0
            best_ring = 0.0
            for r in radii:
                cr = max(1, r // 2)
                outer = rect_mean(integ, width, height, x, y, r * 2)
                center = rect_mean(integ, width, height, x, y, cr)
                response = abs(center - outer)
                if response > best:
                    best = response
                    best_r = r
                    best_center = center
                    best_ring = outer
            responses.append(best)
            candidates.append((best, x, y, best_r, best_center, best_ring))

    threshold = max(min_response, percentile(responses, q))
    candidates = [c for c in candidates if c[0] >= threshold]
    candidates.sort(reverse=True, key=lambda c: c[0])

    occupied = bytearray(width * height)
    seed_map = bytearray(width * height)
    seeds: List[Dict[str, Any]] = []
    rr = max(1, nms_radius)
    for response, x, y, r, center, ring in candidates:
        if occupied[y * width + x]:
            continue
        balance = radial_balance(gray, width, height, x, y, r, center)
        if balance > radial_balance_max:
            continue
        seed_id = len(seeds)
        seeds.append({
            "seed_id": seed_id,
            "x": x,
            "y": y,
            "radius": r,
            "response": round(response, 4),
            "center_mean": round(center, 4),
            "ring_mean": round(ring, 4),
            "radial_balance": round(balance, 4),
        })
        seed_map[y * width + x] = 255
        for yy in range(max(0, y - rr), min(height, y + rr + 1)):
            off = yy * width
            for xx in range(max(0, x - rr), min(width, x + rr + 1)):
                occupied[off + xx] = 1
        if len(seeds) >= max_seeds:
            break

    stats = {
        "response_threshold": threshold,
        "response_percentile": q,
        "raw_candidate_count": len(candidates),
        "seed_count": len(seeds),
    }
    return seeds, seed_map, stats


def load_mask_records(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    score_by_tile_index: Dict[Tuple[str, int], float] = {}
    for tile in summary.get("tiles", []) or []:
        tile_id = tile.get("tile_id")
        for i, score in enumerate(tile.get("scores", []) or []):
            score_by_tile_index[(str(tile_id), i)] = float(score)

    records = summary.get("mask_records", []) or []
    if not records:
        for tile in summary.get("tiles", []) or []:
            tile_id = str(tile.get("tile_id"))
            for rec in tile.get("mask_records", []) or []:
                item = dict(rec)
                item.setdefault("tile_id", tile_id)
                records.append(item)

    out: List[Dict[str, Any]] = []
    for rec in records:
        item = dict(rec)
        key = (str(item.get("tile_id")), int(item.get("mask_index", -1)))
        item["sam_score"] = score_by_tile_index.get(key)
        out.append(item)
    return out


def mask_bbox(mask: bytearray, width: int, height: int) -> Tuple[int, int, int, int, int]:
    min_x, min_y, max_x, max_y = width, height, -1, -1
    area = 0
    for y in range(height):
        off = y * width
        for x in range(width):
            if mask[off + x] > 0:
                area += 1
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
    if area == 0:
        return 0, 0, 0, 0, 0
    return min_x, min_y, max_x + 1, max_y + 1, area


def draw_seed_map(rgb: bytearray, width: int, height: int, seeds: List[Dict[str, Any]]) -> bytearray:
    out = bytearray(rgb)
    for seed in seeds:
        x, y = int(seed["x"]), int(seed["y"])
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            xx, yy = x + dx, y + dy
            if 0 <= xx < width and 0 <= yy < height:
                j = (yy * width + xx) * 3
                out[j:j + 3] = bytes((255, 30, 30))
    return out


def save_overlay(rgb: bytearray, union: bytearray, width: int, height: int, path: Path) -> None:
    out = bytearray(rgb)
    for i, v in enumerate(union):
        if v:
            j = i * 3
            out[j] = int(out[j] * 0.55 + 30 * 0.45)
            out[j + 1] = int(out[j + 1] * 0.55 + 144 * 0.45)
            out[j + 2] = int(out[j + 2] * 0.55 + 255 * 0.45)
    write_png(path, width, height, out, 2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Filter SAM masks by verified ball seed coverage.")
    ap.add_argument("--image", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed-radii", default="2,3,4,5,6")
    ap.add_argument("--seed-stride", type=int, default=2)
    ap.add_argument("--seed-percentile", type=float, default=97.5)
    ap.add_argument("--min-seed-response", type=float, default=4.0)
    ap.add_argument("--max-seeds", type=int, default=15000)
    ap.add_argument("--seed-nms-radius", type=int, default=3)
    ap.add_argument("--radial-balance-max", type=float, default=2.5)
    ap.add_argument("--min-seeds-in-mask", type=int, default=1)
    ap.add_argument("--boundary-nearby-padding", type=int, default=6)
    ap.add_argument("--keep-high-score-no-seed", type=float, default=0.0)
    args = ap.parse_args()

    image_path = Path(args.image).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    width, height, rgb, gray = read_bmp_rgb_gray(image_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    records = load_mask_records(summary)
    radii = [int(x) for x in args.seed_radii.split(",") if x.strip()]

    print("detecting_ball_seeds", flush=True)
    seeds, seed_map, seed_stats = detect_ball_seeds(
        gray, width, height, radii, args.seed_stride, args.seed_percentile,
        args.min_seed_response, args.max_seeds, args.seed_nms_radius, args.radial_balance_max,
    )
    seed_positions = [(int(s["x"]), int(s["y"])) for s in seeds]

    seed_map_png = output_dir / "ball_seed_map.png"
    seed_overlay_png = output_dir / "ball_seed_overlay.png"
    write_png(seed_map_png, width, height, seed_map, 0)
    write_png(seed_overlay_png, width, height, draw_seed_map(rgb, width, height, seeds), 2)

    union = bytearray(width * height)
    filtered_records: List[Dict[str, Any]] = []
    kept = removed = 0

    print(f"filtering_masks records={len(records)} seeds={len(seeds)}", flush=True)
    for n, rec in enumerate(records):
        mask_path = Path(str(rec.get("mask_file")))
        if not mask_path.is_file():
            continue
        mw, mh, mask = read_png_gray(mask_path)
        lx0, ly0, lx1, ly1, area = mask_bbox(mask, mw, mh)
        tx0, ty0, tx1, ty1 = [int(v) for v in rec.get("box_xyxy", [0, 0, width, height])]
        tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)

        seed_count = 0
        nearby_seed_count = 0
        gx0 = tx0 + int(math.floor(lx0 * tw / float(mw)))
        gy0 = ty0 + int(math.floor(ly0 * th / float(mh)))
        gx1 = tx0 + int(math.ceil(lx1 * tw / float(mw)))
        gy1 = ty0 + int(math.ceil(ly1 * th / float(mh)))
        pad = args.boundary_nearby_padding

        for sx, sy in seed_positions:
            if tx0 <= sx < tx1 and ty0 <= sy < ty1:
                mx = int((sx - tx0) * mw / float(tw))
                my = int((sy - ty0) * mh / float(th))
                if 0 <= mx < mw and 0 <= my < mh and mask[my * mw + mx] > 0:
                    seed_count += 1
                if gx0 - pad <= sx < gx1 + pad and gy0 - pad <= sy < gy1 + pad:
                    nearby_seed_count += 1

        touch_tile_boundary = lx0 <= 1 or ly0 <= 1 or lx1 >= mw - 1 or ly1 >= mh - 1
        sam_score = rec.get("sam_score")
        high_score_keep = bool(args.keep_high_score_no_seed and sam_score is not None and float(sam_score) >= args.keep_high_score_no_seed)
        keep = seed_count >= args.min_seeds_in_mask or (touch_tile_boundary and nearby_seed_count > 0) or high_score_keep

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
            "kept_by_ball_seed_filter": bool(keep),
            "seed_count": seed_count,
            "nearby_seed_count": nearby_seed_count,
            "touch_tile_boundary": bool(touch_tile_boundary),
            "mask_bbox_global_xyxy": [gx0, gy0, gx1, gy1],
            "ball_seed_filter_reason": (
                "seed_inside_mask" if seed_count >= args.min_seeds_in_mask else
                "boundary_nearby_seed" if touch_tile_boundary and nearby_seed_count > 0 else
                "high_sam_score_protection" if high_score_keep else
                "no_ball_seed_support"
            ),
        })
        filtered_records.append(out)
        if (n + 1) % 500 == 0:
            print(f"processed={n + 1} kept={kept} removed={removed}", flush=True)

    union_png = output_dir / "filtered_union.png"
    overlay_png = output_dir / "filtered_overlay.png"
    write_png(union_png, width, height, union, 0)
    save_overlay(rgb, union, width, height, overlay_png)

    out_summary = {
        "method": "ball_seed_support_filter",
        "image_path": str(image_path),
        "summary_path": str(summary_path),
        "image_size": {"width": width, "height": height},
        "params": vars(args),
        "seed_stats": seed_stats,
        "num_input_masks": len(records),
        "num_kept_masks": kept,
        "num_removed_masks": removed,
        "seed_map": str(seed_map_png),
        "seed_overlay": str(seed_overlay_png),
        "filtered_union": str(union_png),
        "filtered_overlay": str(overlay_png),
        "mask_records": filtered_records,
    }
    out_summary_path = output_dir / "filtered_summary.json"
    out_summary_path.write_text(json.dumps(out_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "filtered_overlay": str(overlay_png),
        "filtered_union": str(union_png),
        "filtered_summary": str(out_summary_path),
        "seed_count": len(seeds),
        "kept": kept,
        "removed": removed,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
