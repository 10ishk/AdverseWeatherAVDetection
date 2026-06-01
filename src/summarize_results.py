"""Summarize the final row of a YOLOv5 results.csv file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLUMN_MAP = {
    "precision": "metrics/precision",
    "recall": "metrics/recall",
    "mAP@0.5": "metrics/mAP_0.5",
    "mAP@0.5:0.95": "metrics/mAP_0.5:0.95",
    "train_box_loss": "train/box_loss",
    "train_obj_loss": "train/obj_loss",
    "val_box_loss": "val/box_loss",
    "val_obj_loss": "val/obj_loss",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a compact metrics summary from YOLOv5 results.csv.")
    parser.add_argument("--results-csv", required=True, type=Path)
    parser.add_argument("--output", default=Path("results/metrics_summary.json"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.results_csv.exists():
        raise FileNotFoundError(f"Could not find results CSV: {args.results_csv}")

    df = pd.read_csv(args.results_csv, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    final_row = df.iloc[-1]

    summary = {"epoch": int(final_row.get("epoch", len(df) - 1))}
    for output_name, column_name in METRIC_COLUMN_MAP.items():
        if column_name in final_row:
            summary[output_name] = float(final_row[column_name])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Metrics summary written to {args.output}")


if __name__ == "__main__":
    main()
