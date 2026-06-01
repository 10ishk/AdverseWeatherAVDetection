"""Launch YOLOv5 training with reproducible command-line arguments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv5 on the adverse-weather dataset.")
    parser.add_argument("--yolov5-dir", default=Path("yolov5"), type=Path)
    parser.add_argument("--data", default=Path("data.yaml"), type=Path)
    parser.add_argument("--weights", default="yolov5s.pt")
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=Path("runs"), type=Path)
    parser.add_argument("--name", default="adverse_weather_exp")
    parser.add_argument("--workers", default=4, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_py = args.yolov5_dir / "train.py"
    if not train_py.exists():
        raise FileNotFoundError(f"YOLOv5 train.py not found at {train_py}. Clone YOLOv5 first.")

    command = [
        sys.executable,
        str(train_py),
        "--data", str(args.data),
        "--imgsz", str(args.imgsz),
        "--batch-size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--weights", str(args.weights),
        "--device", str(args.device),
        "--project", str(args.project),
        "--name", args.name,
        "--workers", str(args.workers),
    ]

    print("Running training command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
