# Case Study: Adverse Weather Object Detection for Autonomous Vehicles

## Context

Autonomous-driving perception systems need to detect vehicles reliably across changing road and weather conditions. Rain, fog, snow, glare, reduced contrast, and road reflections can reduce object-detection quality, especially for smaller or partially occluded vehicles.

## Problem

The objective was to build a focused object-detection workflow for vehicle detection in adverse-weather driving scenes. Instead of training on a general driving dataset only, the workflow filters BDD100K by weather attributes and trains a YOLOv5s detector on the weather-specific subset.

## Dataset Strategy

BDD100K was used as the source dataset. Images were filtered using weather metadata such as rainy, foggy, and snowy scenes. BDD100K bounding-box annotations were converted into YOLO format with normalized center coordinates, width, and height.

The public repository excludes dataset files because BDD100K is large and should be downloaded from the official source.

## Technical Approach

1. Load BDD100K annotation JSON files.
2. Filter image records by adverse-weather labels.
3. Copy selected images into YOLO train/validation folders.
4. Convert BDD100K `box2d` annotations into YOLO labels.
5. Train YOLOv5s using transfer learning.
6. Monitor precision, recall, mAP@0.5, and mAP@0.5:0.95.
7. Run inference on adverse-weather driving scenes.

## Model Choice

YOLOv5s was selected because it is lightweight, fast to train on limited GPU memory, and suitable for creating a practical baseline. The recorded experiment used an RTX 3050 Laptop GPU, so batch size and image size were selected to balance performance and hardware constraints.

## Results

The recorded local run produced the following final metrics:

| Metric | Value |
|---|---:|
| Precision | 0.7516 |
| Recall | 0.5036 |
| mAP@0.5 | 0.5800 |
| mAP@0.5:0.95 | 0.3292 |

These results should be interpreted as a project baseline, not as production-level autonomous-driving validation.

## What Makes This More Than a Basic Notebook

The refined project includes a reusable dataset-filtering pipeline, portable training/evaluation/inference scripts, config files, metric summary output, model-card documentation, and future-scope planning. This makes it easier to reproduce, review, and extend compared with a single notebook-only project.

## Limitations

- Current public configuration focuses on the `car` class.
- No trained weights or BDD100K images are included in the repository.
- No production deployment or real-time vehicle integration is claimed.
- The model needs deeper evaluation by weather type, time of day, object size, and occlusion.

## Next Improvements

The most valuable next steps are multi-class detection, weather-segmented evaluation, a small inference demo, and model comparison against newer detectors such as YOLOv8 or RT-DETR.
