#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def find_summary(sample_dir: Path, sample_stem: str) -> Path | None:
    candidates = [
        sample_dir / "sam_split9_upscale" / f"{sample_stem}_sam_auto_summary.json",
        sample_dir / "sam_split36_upscale" / f"{sample_stem}_sam_auto_summary.json",
    ]
    for p in candidates:
        if p.is_file():
            return p
    found = sorted(sample_dir.glob("**/*_sam_auto_summary.json"))
    return found[0] if found else None


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch export/apply v3 scale-normalized CNN crops from existing SAM summaries.")
    ap.add_argument("--input-root", required=True, help="Original image root containing bmp files.")
    ap.add_argument("--sam-output-root", required=True, help="Existing batch output root containing per-sample SAM summary dirs.")
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--model", default="checkpoints/cnn_v3/cnn_prototype_model.pt")
    ap.add_argument("--reference-labels-csv", default="work/cnn_v3_reference/labels.csv")
    ap.add_argument("--python", default=None, help="Python executable; defaults to the current interpreter.")
    ap.add_argument("--keep-threshold", type=float, default=0.75)
    ap.add_argument("--keep-classes", default="real_ball,shadow_ball,double_ball_cluster,shadow_double_ball_cluster")
    ap.add_argument("--max-keep-area", type=int, default=1000)
    ap.add_argument("--min-crop-side", type=int, default=96)
    ap.add_argument("--max-crop-side", type=int, default=512)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    python = args.python or sys.executable
    repo = Path(__file__).resolve().parents[1]

    input_root = Path(args.input_root)
    sam_root = Path(args.sam_output_root)
    out_root = Path(args.output_root)
    images = sorted([p for p in input_root.iterdir() if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}])
    done = []
    skipped = []
    for img in images:
        stem = img.stem
        sample_dir = sam_root / stem
        summary = find_summary(sample_dir, stem)
        if summary is None:
            skipped.append({"sample": stem, "reason": "missing_summary"})
            continue
        sample_out = out_root / stem
        dataset_dir = sample_out / "cnn_v3_scale_normalized_dataset"
        apply_dir = sample_out / "cnn_v3_scale_normalized_t075"
        if apply_dir.joinpath("cnn_prototype_apply_summary.json").is_file() and not args.overwrite:
            done.append({"sample": stem, "status": "exists"})
            continue
        run([
            python,
            str(repo / "methods/05_cnn_v3/export_v3_scale_normalized_dataset.py"),
            "--summary", str(summary),
            "--reference-labels-csv", args.reference_labels_csv,
            "--output-dir", str(dataset_dir),
            "--min-crop-side", str(args.min_crop_side),
            "--max-crop-side", str(args.max_crop_side),
        ])
        run([
            python,
            str(repo / "methods/05_cnn_v3/apply_cnn_prototype_classifier.py"),
            "--model", args.model,
            "--labels-csv", str(dataset_dir / "labels.csv"),
            "--output-dir", str(apply_dir),
            "--image", str(img),
            "--summary", str(summary),
            "--keep-threshold", str(args.keep_threshold),
            "--keep-classes", args.keep_classes,
            "--max-keep-area", str(args.max_keep_area),
        ])
        done.append({"sample": stem, "status": "processed", "summary": str(summary), "output": str(apply_dir)})
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"input_root": str(input_root), "sam_output_root": str(sam_root), "output_root": str(out_root), "done": done, "skipped": skipped, "params": vars(args)}
    (out_root / "batch_v3_scale_normalized_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
