# Model Card: YOLOv5s Adverse-Weather Vehicle Detector

## Model Summary

- **Model family:** YOLOv5
- **Variant:** YOLOv5s
- **Task:** Object detection
- **Primary class:** Car
- **Target context:** Road scenes under adverse weather such as rain, fog, and snow

## Intended Use

This model is intended as a portfolio and experimentation baseline for adverse-weather object detection. It can be used to study how a lightweight detector performs on degraded driving scenes.

## Not Intended For

- Real autonomous-driving deployment
- Safety-critical decision-making
- Direct use in production vehicles
- Performance claims without dataset-specific reproduction

## Training Data

The workflow filters BDD100K records by weather metadata and converts selected bounding-box annotations into YOLO format. The public repository does not include BDD100K data.

## Recorded Metrics

| Metric | Value |
|---|---:|
| Precision | 0.7516 |
| Recall | 0.5036 |
| mAP@0.5 | 0.5800 |
| mAP@0.5:0.95 | 0.3292 |

## Known Limitations

- Single-class car detection in the current configuration
- Weather imbalance may affect generalization
- Limited training epochs in the recorded run
- Further testing needed on unseen images and videos

## Ethical and Safety Notes

This project is for experimentation and portfolio demonstration. Autonomous-driving perception systems require extensive validation, robustness testing, safety engineering, and regulatory review before any real-world use.
