"""Run YOLOv5 inference on images, folders, or videos."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with trained YOLOv5 weights.")
    parser.add_argument("--yolov5-dir", default=Path("yolov5"), type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--conf-thres", default=0.4, type=float)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=Path("runs"), type=Path)
    parser.add_argument("--name", default="adverse_weather_infer")
    parser.add_argument("--save-txt", action="store_true")
    parser.add_argument("--save-conf", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detect_py = args.yolov5_dir / "detect.py"
    if not detect_py.exists():
        raise FileNotFoundError(f"YOLOv5 detect.py not found at {detect_py}. Clone YOLOv5 first.")

    command = [
        sys.executable,
        str(detect_py),
        "--weights", str(args.weights),
        "--source", str(args.source),
        "--imgsz", str(args.imgsz),
        "--conf-thres", str(args.conf_thres),
        "--device", str(args.device),
        "--project", str(args.project),
        "--name", args.name,
    ]
    if args.save_txt:
        command.append("--save-txt")
    if args.save_conf:
        command.append("--save-conf")

    print("Running inference command:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
