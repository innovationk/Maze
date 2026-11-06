#!/usr/bin/env python3
"""Count puzzle pieces in images and write the result as JSON.

Examples
--------
    python count_pieces.py ../TESTS -o results/tests.json
    python count_pieces.py ../TESTS -o results/tests.json --viz-dir results/viz --details
    python count_pieces.py ../dataset --labels ../dataset/dataset.csv -o results/dataset.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from puzzle_counter import CounterConfig, PieceCounter
from puzzle_counter.pipeline import render

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def collect_images(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS))
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    return paths


def load_labels(csv_path: str) -> dict[str, int]:
    with open(csv_path, newline="") as f:
        return {row["label"]: int(row["count"]) for row in csv.DictReader(f)}


def parse_args() -> argparse.Namespace:
    d = CounterConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="Image files and/or directories")
    ap.add_argument("-o", "--output", help="JSON output file (default: print to stdout)")
    ap.add_argument("--details", action="store_true", help="Include per-piece boxes and scores in the JSON")
    ap.add_argument("--viz-dir", help="Save annotated images (masks + numbered boxes) to this directory")
    ap.add_argument("--labels", help="CSV with columns label,count to evaluate against")
    ap.add_argument("--prompt", default=d.prompt)
    ap.add_argument("--box-threshold", type=float, default=d.box_threshold)
    ap.add_argument("--text-threshold", type=float, default=d.text_threshold)
    ap.add_argument("--points-per-side", type=int, default=d.points_per_side, help="SAM 2 point-grid density")
    ap.add_argument("--detector", default=d.detector_id)
    ap.add_argument("--segmenter", default=d.segmenter_id)
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: auto)")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger("count_pieces")

    config = CounterConfig(
        detector_id=args.detector,
        segmenter_id=args.segmenter,
        prompt=args.prompt,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        points_per_side=args.points_per_side,
        device=args.device,
    )
    images = collect_images(args.inputs)
    if not images:
        raise SystemExit("No images found.")
    labels = load_labels(args.labels) if args.labels else {}
    viz_dir = Path(args.viz_dir) if args.viz_dir else None
    if viz_dir:
        viz_dir.mkdir(parents=True, exist_ok=True)

    counter = PieceCounter(config)
    results = []
    for path in images:
        start = time.perf_counter()
        image = Image.open(path)
        result = counter.count(image)
        elapsed = time.perf_counter() - start

        entry: dict = {"image": path.name, "count": result.count, "typical_piece_area": result.typical_piece_area}
        if path.name in labels:
            entry["expected"] = labels[path.name]
            entry["error"] = result.count - labels[path.name]
        entry["seconds"] = round(elapsed, 2)
        if args.details:
            entry["pieces"] = [asdict(p) for p in result.pieces]
        results.append(entry)

        suffix = f" (expected {entry['expected']})" if "expected" in entry else ""
        log.info("%s -> %d%s in %.1fs", path.name, result.count, suffix, elapsed)
        if viz_dir:
            render(image, result).save(viz_dir / f"{path.stem}.jpg", quality=90)

    report: dict = {"config": asdict(config), "results": results}
    evaluated = [r for r in results if "expected" in r]
    if evaluated:
        abs_err = [abs(r["error"]) for r in evaluated]
        report["metrics"] = {
            "images": len(evaluated),
            "mae": round(sum(abs_err) / len(abs_err), 2),
            "mape": round(100 * sum(abs(r["error"]) / r["expected"] for r in evaluated) / len(evaluated), 1),
            "exact": sum(e == 0 for e in abs_err),
        }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        log.info("Wrote %s", args.output)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
