# Adverse Weather Object Detection for Autonomous Vehicles

A computer vision project focused on vehicle detection in challenging autonomous-driving conditions such as rain, fog, and snow. The project uses a filtered BDD100K adverse-weather subset, YOLOv5 transfer learning, configurable dataset preparation, and reusable training/evaluation/inference scripts.

The goal is not just to train a generic object detector. The project explores how object detection performance changes when the input data is limited to visually degraded road scenes where contrast, visibility, glare, and occlusion can make vehicle detection harder.

## Project Highlights

- Filtered BDD100K images by adverse weather attributes such as rainy, foggy, and snowy scenes
- Converted BDD100K bounding-box annotations into YOLO format
- Trained a YOLOv5s detector for the `car` class using transfer learning
- Used weather-oriented augmentations such as saturation/value shifts, scaling, mosaic, and horizontal flips
- Recorded training-run metrics for precision, recall, mAP@0.5, and mAP@0.5:0.95
- Added reusable scripts for dataset filtering, training, evaluation, inference, and metric summarization
- Structured the repo for public GitHub presentation without uploading BDD100K data or model weights

## Why This Project Matters

Autonomous vehicle perception systems must work under degraded visual conditions. Rain, fog, snow, low visibility, and road reflections can reduce object-detection reliability. This project demonstrates a practical computer-vision workflow for adapting a lightweight YOLOv5 model to adverse-weather driving scenes.

## Current Implementation

The current implementation includes:

- BDD100K adverse-weather filtering logic
- YOLO annotation conversion for vehicle bounding boxes
- YOLOv5s training workflow
- Configurable training parameters
- Evaluation and inference scripts
- Metrics summary from a recorded local training run
- Clean project documentation for reproducibility

## Results From Recorded Local Training Run

The notebook output recorded the following final training-run metrics:

| Metric | Value |
|---|---:|
| Precision | 0.7516 |
| Recall | 0.5036 |
| mAP@0.5 | 0.5800 |
| mAP@0.5:0.95 | 0.3292 |

These metrics are provided as a local experiment summary. They may vary depending on dataset split, YOLOv5 version, CUDA/PyTorch versions, training epochs, and confidence thresholds.

## Tech Stack

- Python
- PyTorch
- YOLOv5
- OpenCV
- Pandas
- NumPy
- PyYAML
- BDD100K dataset
- Jupyter Notebook

## Repository Structure

```text
adverse-weather-av-detection/
  README.md
  requirements.txt
  data.yaml
  .gitignore

  src/
    dataset_filter.py
    train_yolov5.py
    evaluate_yolov5.py
    infer_yolov5.py
    summarize_results.py

  configs/
    data_bdd100k_car_adverse.yaml
    training_config.yaml

  notebooks/
    adverse_weather_detection_original.ipynb

  docs/
    case_study.md
    model_card.md
    experiment_notes.md
    future_scope.md

  results/
    metrics_summary.json

  assets/
    screenshots/
    diagrams/
    sample_predictions/

  outputs/
    sample_results/
```

## Dataset

This project uses the BDD100K dataset. The dataset is not included in this repository because of size and licensing constraints.

The dataset filtering step selects images with adverse-weather attributes, such as:

- Rainy
- Foggy
- Snowy

The current implementation focuses on the `car` class. The pipeline can be extended to other BDD100K categories such as bus, truck, person, rider, bike, and traffic sign.

## Setup

### 1. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Clone YOLOv5

```bash
git clone https://github.com/ultralytics/yolov5.git
pip install -r yolov5/requirements.txt
```

### 4. Prepare BDD100K locally

Place BDD100K outside this repository or inside a local ignored folder such as:

```text
bdd100k/
  train/
    images/
    annotations/
  val/
    images/
    annotations/
```

The script also supports common BDD100K-style image/label paths where possible. Update the paths in `configs/training_config.yaml` if your local dataset layout is different.

## How to Run

### 1. Filter BDD100K adverse-weather images

```bash
python src/dataset_filter.py \
  --bdd-root /path/to/bdd100k \
  --output-dir dataset \
  --weather rainy foggy snowy \
  --classes car \
  --max-train 1000 \
  --max-val 200 \
  --seed 42
```

### 2. Train YOLOv5s

```bash
python src/train_yolov5.py \
  --yolov5-dir yolov5 \
  --data data.yaml \
  --weights yolov5s.pt \
  --imgsz 416 \
  --batch-size 2 \
  --epochs 10 \
  --device 0 \
  --project runs \
  --name adverse_weather_exp
```

### 3. Evaluate the trained model

```bash
python src/evaluate_yolov5.py \
  --yolov5-dir yolov5 \
  --weights runs/adverse_weather_exp/weights/best.pt \
  --data data.yaml \
  --imgsz 416 \
  --device 0
```

### 4. Run inference

```bash
python src/infer_yolov5.py \
  --yolov5-dir yolov5 \
  --weights runs/adverse_weather_exp/weights/best.pt \
  --source path/to/test_images \
  --imgsz 416 \
  --conf-thres 0.4 \
  --device 0 \
  --project runs \
  --name adverse_weather_infer
```

### 5. Summarize training metrics

```bash
python src/summarize_results.py \
  --results-csv runs/adverse_weather_exp/results.csv \
  --output results/metrics_summary.json
```

## Example Use Cases

- Testing vehicle detection robustness under rain, fog, and snow
- Building a baseline object detector for adverse-weather autonomous-driving datasets
- Comparing clear-weather vs adverse-weather detection performance
- Creating a foundation for weather-aware perception research
- Demonstrating applied computer vision and ML engineering workflow

## Limitations

- The current public repo does not include BDD100K images, annotations, trained weights, or inference images.
- The current training configuration focuses on the `car` class only.
- Metrics are from a local experiment and should be reproduced before making production claims.
- The model is not deployed as a real-time autonomous-driving system.
- Further evaluation is needed across weather type, lighting, object scale, and occlusion severity.

## Future Scope

Planned improvements include:

- Multi-class detection for cars, buses, trucks, pedestrians, riders, traffic lights, and traffic signs
- Separate metrics by weather condition: rainy vs foggy vs snowy
- Baseline comparison against clear-weather subsets
- Confusion matrix and per-class error analysis
- Streamlit or FastAPI demo for image upload inference
- Export to ONNX for lightweight deployment testing
- Real-time video inference demo
- Experiment tracking with MLflow or Weights & Biases
- Model comparison between YOLOv5s, YOLOv8n, and RT-DETR

## Public-Safe Notes

Large datasets, trained weights, and generated inference outputs are intentionally excluded from GitHub. Add sample screenshots or prediction images only if they are license-safe and do not contain private data.

