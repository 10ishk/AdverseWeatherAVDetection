"""Evaluate a trained YOLOv5 model on the validation split."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv5 weights on the adverse-weather validation split.")
    parser.add_argument("--yolov5-dir", default=Path("yolov5"), type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", default=Path("data.yaml"), type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=Path("runs/val"), type=Path)
    parser.add_argument("--name", default="adverse_weather_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    val_py = args.yolov5_dir / "val.py"
    if not val_py.exists():
        raise FileNotFoundError(f"YOLOv5 val.py not found at {val_py}. Clone YOLOv5 first.")

    command = [
        sys.executable,
        str(val_py),
        "--weights", str(args.weights),
        "--data", str(args.data),
        "--imgsz", str(args.imgsz),
        "--device", str(args.device),
        "--project", str(args.project),
        "--name", args.name,
        "--task", "val",
    ]

    print("Running evaluation command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
