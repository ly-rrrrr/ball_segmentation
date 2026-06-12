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
    ap = argparse.ArgumentParser(description="Dynamic SAM split-upscale + CNN v3 upscaled-only batch runner.")
    ap.add_argument("--input-root", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--sam-model", default="weights/sam")
    ap.add_argument("--cnn-model", default="checkpoints/cnn_v3/cnn_prototype_model.pt")
    ap.add_argument("--split", type=int, default=9)
    ap.add_argument("--threshold", type=float, default=0.75)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--worker-index", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--points-per-batch", type=int, default=None, help="Optional SAM points_per_batch override. Omit to keep SAM script default.")
    ap.add_argument("--min-crop-side", type=int, default=128)
    ap.add_argument("--context-scale", type=float, default=3.5)
    ap.add_argument("--crop-size", type=int, default=64)
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
    if args.num_workers < 1 or not (0 <= args.worker_index < args.num_workers):
        raise SystemExit("bad worker-index/num-workers")

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
    images = [p for i, p in enumerate(images) if i % args.num_workers == args.worker_index]
    if not images:
        print(json.dumps({"worker": args.worker_index, "num_images": 0}, ensure_ascii=False), flush=True)
        return

    results = []
    split_tag = f"split{args.split}_upscale"
    threshold_tag = f"t{int(round(args.threshold * 100)):03d}"
    for image in images:
        try:
            rel_parent = image.parent.relative_to(input_root)
        except ValueError:
            rel_parent = Path()
        sample_root = output_root / rel_parent / image.stem
        sam_dir = sample_root / f"sam_{split_tag}"
        dataset_dir = sample_root / f"cnn_v3_upscaled_dataset_mincrop{args.min_crop_side}"
        apply_dir = sample_root / f"cnn_v3_upscaled_{threshold_tag}_mincrop{args.min_crop_side}"
        summary_path = sam_dir / f"{image.stem}_sam_auto_summary.json"
        labels_path = dataset_dir / "labels.csv"
        apply_summary = apply_dir / "cnn_prototype_apply_summary.json"

        if args.overwrite or not summary_path.is_file():
            sam_cmd = [
                sys.executable,
                str(repo / "methods/01_sam_generation/run_sam_automatic_mask.py"),
                "--model-path", str(sam_model),
                "--image", str(image),
                "--output-dir", str(sam_dir),
                "--split", str(args.split),
                "--split-upscale",
                "--save-split-tiles",
                "--save-individual-masks",
                "--save-upscaled-masks",
                "--union-max-area-ratio", "0.9",
                "--device", "0",
            ]
            if args.points_per_batch is not None:
                sam_cmd.extend(["--points-per-batch", str(args.points_per_batch)])
            run(sam_cmd, env)
        else:
            print(json.dumps({"skip_sam_existing": str(summary_path)}, ensure_ascii=False), flush=True)

        if args.overwrite or not labels_path.is_file():
            run([
                sys.executable,
                str(repo / "methods/05_cnn_v3/export_combined_manual_cnn_dataset_upscaled_fast.py"),
                "--summary", str(summary_path),
                "--source-pair", source_pair,
                "--output-dir", str(dataset_dir),
                "--crop-size", str(args.crop_size),
                "--context-scale", str(args.context_scale),
                "--min-crop-side", str(args.min_crop_side),
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
            "sam_overlay": str(sam_dir / f"{image.stem}_sam_auto_overlay.png"),
            "dataset_labels": str(labels_path),
            "cnn_apply_summary": str(apply_summary),
            "cnn_overlay": str(apply_dir / "cnn_filtered_class_overlay.png"),
            "cnn_rejected_overlay": str(apply_dir / "cnn_rejected_overlay.png"),
        }
        if apply_summary.is_file():
            result.update(json.loads(apply_summary.read_text(encoding="utf-8")))
        results.append(result)

    output_root.mkdir(parents=True, exist_ok=True)
    worker_summary = output_root / f"batch_summary_worker{args.worker_index}.json"
    worker_summary.write_text(json.dumps({
        "input_root": str(input_root),
        "output_root": str(output_root),
        "split": args.split,
        "threshold": args.threshold,
        "gpu": args.gpu,
        "worker_index": args.worker_index,
        "num_workers": args.num_workers,
        "min_crop_side": args.min_crop_side,
        "max_keep_area": args.max_keep_area,
        "num_images": len(images),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"worker_summary": str(worker_summary), "num_images": len(images)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
