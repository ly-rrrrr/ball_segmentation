#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def normalize(value: Any, marker: str) -> Any:
    if isinstance(value, dict):
        return {key: normalize(item, marker) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize(item, marker) for item in value]
    if not isinstance(value, str):
        return value

    normalized = value.replace("\\", "/")
    token = f"/{marker.strip('/')}/"
    if token in normalized:
        return marker.strip("/") + "/" + normalized.split(token, 1)[1]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert project-local absolute paths in annotation JSON to portable relative paths."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-marker", default="ball_segmentation")
    args = parser.parse_args()

    source = Path(args.input)
    output = Path(args.output)
    data = json.loads(source.read_text(encoding="utf-8"))
    portable = normalize(data, args.project_marker)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(portable, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"input": str(source), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
