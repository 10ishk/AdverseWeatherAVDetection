"""Filter BDD100K images by adverse weather and convert labels to YOLO format.

This script intentionally keeps dataset paths configurable because BDD100K is not
included in the repository. It supports a few common BDD100K folder layouts and
writes a dataset_summary.json file so experiments can be audited later.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm


DEFAULT_WEATHER = ["rainy", "foggy", "snowy"]
DEFAULT_CLASSES = ["car"]


@dataclass
class SplitPaths:
    images_dir: Path
    labels_file: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a YOLO dataset from BDD100K adverse-weather images.")
    parser.add_argument("--bdd-root", required=True, type=Path, help="Path to local BDD100K root folder.")
    parser.add_argument("--output-dir", default=Path("dataset"), type=Path, help="Output YOLO dataset folder.")
    parser.add_argument("--weather", nargs="+", default=DEFAULT_WEATHER, help="Weather labels to include.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, help="BDD100K object classes to include.")
    parser.add_argument("--max-train", type=int, default=1000, help="Maximum training images to sample.")
    parser.add_argument("--max-val", type=int, default=200, help="Maximum validation images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    return parser.parse_args()


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def resolve_split_paths(bdd_root: Path, split: str) -> SplitPaths:
    images_dir = first_existing(
        [
            bdd_root / split / "images",
            bdd_root / "images" / split,
            bdd_root / "images" / "100k" / split,
        ]
    )

    labels_file = first_existing(
        [
            bdd_root / split / "annotations" / f"bdd100k_labels_images_{split}.json",
            bdd_root / "labels" / f"bdd100k_labels_images_{split}.json",
            bdd_root / "labels" / "det_20" / f"det_{split}.json",
        ]
    )

    if images_dir is None:
        raise FileNotFoundError(f"Could not locate images directory for split '{split}' under {bdd_root}")
    if labels_file is None:
        raise FileNotFoundError(f"Could not locate labels JSON for split '{split}' under {bdd_root}")

    return SplitPaths(images_dir=images_dir, labels_file=labels_file)


def convert_bbox_to_yolo(box: dict[str, float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height


def load_annotations(labels_file: Path) -> list[dict]:
    with labels_file.open("r", encoding="utf-8") as file:
        annotations = json.load(file)
    if not isinstance(annotations, list):
        raise ValueError(f"Expected a list of BDD100K annotations in {labels_file}")
    return annotations


def filter_split(
    bdd_root: Path,
    output_dir: Path,
    split: str,
    weather_labels: set[str],
    class_to_id: dict[str, int],
    max_images: int,
) -> dict[str, int]:
    paths = resolve_split_paths(bdd_root, split)
    annotations = load_annotations(paths.labels_file)

    adverse_items = [
        item for item in annotations
        if item.get("attributes", {}).get("weather") in weather_labels
    ]
    selected_items = random.sample(adverse_items, min(max_images, len(adverse_items)))

    image_output_dir = output_dir / "images" / split
    label_output_dir = output_dir / "labels" / split
    image_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)

    copied_images = 0
    written_boxes = 0
    skipped_missing_images = 0
    skipped_unreadable_images = 0

    for item in tqdm(selected_items, desc=f"Processing {split}"):
        image_name = item.get("name")
        if not image_name:
            continue

        source_image = paths.images_dir / image_name
        if not source_image.exists():
            skipped_missing_images += 1
            continue

        image = cv2.imread(str(source_image))
        if image is None:
            skipped_unreadable_images += 1
            continue

        image_height, image_width = image.shape[:2]
        shutil.copy2(source_image, image_output_dir / image_name)
        copied_images += 1

        label_file = label_output_dir / f"{Path(image_name).stem}.txt"
        with label_file.open("w", encoding="utf-8") as output:
            for label in item.get("labels", []):
                category = label.get("category")
                box = label.get("box2d")
                if category not in class_to_id or not box:
                    continue

                x_center, y_center, width, height = convert_bbox_to_yolo(box, image_width, image_height)
                output.write(
                    f"{class_to_id[category]} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                )
                written_boxes += 1

    return {
        "candidate_adverse_images": len(adverse_items),
        "selected_images": len(selected_items),
        "copied_images": copied_images,
        "written_boxes": written_boxes,
        "skipped_missing_images": skipped_missing_images,
        "skipped_unreadable_images": skipped_unreadable_images,
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    class_to_id = {class_name: idx for idx, class_name in enumerate(args.classes)}
    summary = {
        "weather_filter": args.weather,
        "classes": class_to_id,
        "seed": args.seed,
        "splits": {},
    }

    summary["splits"]["train"] = filter_split(
        args.bdd_root, args.output_dir, "train", set(args.weather), class_to_id, args.max_train
    )
    summary["splits"]["val"] = filter_split(
        args.bdd_root, args.output_dir, "val", set(args.weather), class_to_id, args.max_val
    )

    summary_path = args.output_dir / "dataset_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Dataset summary written to {summary_path}")


if __name__ == "__main__":
    main()
