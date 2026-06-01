# Experiment Notes

## Recorded Local Setup

- Model: YOLOv5s
- Input size: 416 x 416
- Epochs: 10
- Confidence threshold used for inference examples: 0.4
- Hardware in recorded run: NVIDIA GeForce RTX 3050 Laptop GPU
- CUDA version in recorded run: 11.8

## Important Reproducibility Notes

- BDD100K is not included in this repository.
- YOLOv5 is cloned separately and excluded from version control.
- Training outputs, weights, and inference images are excluded to keep the repository lightweight and public-safe.
- Metrics can vary depending on dataset selection, seed, hardware, PyTorch version, and YOLOv5 version.

## Suggested Experiment Tracking

For a stronger version of this project, track:

- Dataset split seed
- Number of images per weather type
- Number of boxes per class
- Training loss curves
- Per-weather mAP
- Inference latency
- Failure examples
- Confidence threshold sensitivity
