#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(json.dumps({"running": cmd}, ensure_ascii=False), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run split36-upscale SAM + CNN v3 upscaled-only filtering for new membrane BMP samples.")
    ap.add_argument("--input-root", default="data/new_membrane")
    ap.add_argument("--output-root", default="outputs/new_membrane_cnn_v3")
    ap.add_argument("--sam-model", default="weights/sam")
    ap.add_argument("--cnn-model", default="checkpoints/cnn_v3/cnn_prototype_model.pt")
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--gpu", default="2")
    ap.add_argument("--points-per-batch", type=int, default=256)
    ap.add_argument("--max-keep-area", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    input_root = (repo / args.input_root).resolve()
    output_root = (repo / args.output_root).resolve()
    sam_model = (repo / args.sam_model).resolve()
    cnn_model = (repo / args.cnn_model).resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input root not found: {input_root}")
    if not sam_model.is_dir():
        raise SystemExit(f"SAM model not found: {sam_model}")
    if not cnn_model.is_file():
        raise SystemExit(f"CNN model not found: {cnn_model}")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    dummy_dir = output_root / "_empty_source_pair"
    dummy_dir.mkdir(parents=True, exist_ok=True)
    candidates_json = dummy_dir / "empty_candidates.json"
    labels_csv = dummy_dir / "empty_labels.csv"
    candidates_json.write_text(json.dumps({"candidates": []}, ensure_ascii=False), encoding="utf-8")
    labels_csv.write_text("candidate_id,label\n", encoding="utf-8")
    source_pair = f"empty:{candidates_json}:{labels_csv}"

    images = sorted(p for p in input_root.rglob("*.bmp") if p.is_file())
    if not images:
        raise SystemExit(f"no bmp files found under {input_root}")

    results = []
    for image in images:
        rel_parent = image.parent.relative_to(input_root)
        sample_root = output_root / rel_parent / image.stem
        sam_dir = sample_root / "sam_split36_upscale"
        dataset_dir = sample_root / "cnn_v3_upscaled_dataset"
        apply_dir = sample_root / f"cnn_v3_upscaled_t{int(round(args.threshold * 100)):03d}"
        summary_path = sam_dir / f"{image.stem}_sam_auto_summary.json"
        overlay_path = sam_dir / f"{image.stem}_sam_auto_overlay.png"
        labels_path = dataset_dir / "labels.csv"
        apply_summary = apply_dir / "cnn_prototype_apply_summary.json"

        if args.overwrite or not summary_path.is_file():
            run([
                sys.executable,
                str(repo / "methods/01_sam_generation/run_sam_automatic_mask.py"),
                "--model-path", str(sam_model),
                "--image", str(image),
                "--output-dir", str(sam_dir),
                "--split", "36",
                "--split-upscale",
                "--save-split-tiles",
                "--save-individual-masks",
                "--save-upscaled-masks",
                "--union-max-area-ratio", "0.9",
                "--points-per-batch", str(args.points_per_batch),
                "--device", "0",
            ], env)
        else:
            print(json.dumps({"skip_sam_existing": str(summary_path)}, ensure_ascii=False), flush=True)

        if args.overwrite or not labels_path.is_file():
            run([
                sys.executable,
                str(repo / "methods/05_cnn_v3/export_combined_manual_cnn_dataset_upscaled_fast.py"),
                "--summary", str(summary_path),
                "--source-pair", source_pair,
                "--output-dir", str(dataset_dir),
                "--crop-size", "64",
                "--context-scale", "3.5",
                "--min-crop-side", "128",
                "--export-inference-set",
            ], env)
        else:
            print(json.dumps({"skip_dataset_existing": str(labels_path)}, ensure_ascii=False), flush=True)

        if args.overwrite or not apply_summary.is_file():
            run([
                sys.executable,
                str(repo / "methods/05_cnn_v3/apply_cnn_prototype_classifier.py"),
                "--model", str(cnn_model),
                "--labels-csv", str(labels_path),
                "--output-dir", str(apply_dir),
                "--image", str(image),
                "--summary", str(summary_path),
                "--keep-threshold", str(args.threshold),
                "--keep-classes", "real_ball,shadow_ball,double_ball_cluster,shadow_double_ball_cluster",
                "--max-keep-area", str(args.max_keep_area),
            ], env)
        else:
            print(json.dumps({"skip_apply_existing": str(apply_summary)}, ensure_ascii=False), flush=True)

        result = {
            "image": str(image),
            "sam_summary": str(summary_path),
            "sam_overlay": str(overlay_path),
            "dataset_labels": str(labels_path),
            "cnn_apply_summary": str(apply_summary),
            "cnn_overlay": str(apply_dir / "cnn_filtered_class_overlay.png"),
            "cnn_rejected_overlay": str(apply_dir / "cnn_rejected_overlay.png"),
        }
        if apply_summary.is_file():
            try:
                result.update(json.loads(apply_summary.read_text(encoding="utf-8")))
            except Exception:
                pass
        results.append(result)

    output_root.mkdir(parents=True, exist_ok=True)
    batch_summary = output_root / "batch_summary.json"
    batch_summary.write_text(json.dumps({
        "input_root": str(input_root),
        "output_root": str(output_root),
        "num_images": len(images),
        "threshold": args.threshold,
        "gpu": args.gpu,
        "max_keep_area": args.max_keep_area,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"batch_summary": str(batch_summary), "num_images": len(images)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
