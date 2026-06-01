# Future Scope

The following improvements would make the project stronger for computer-vision and applied-AI roles.

## 1. Multi-Class Detection

Extend from `car` detection to additional BDD100K categories:

- bus
- truck
- person
- rider
- bike
- traffic light
- traffic sign

## 2. Weather-Segmented Evaluation

Report separate metrics for:

- rainy scenes
- foggy scenes
- snowy scenes
- night + adverse weather
- daytime + adverse weather

## 3. Baseline Comparison

Compare performance across:

- clear-weather subset
- adverse-weather subset
- combined dataset
- YOLOv5s vs YOLOv8n vs RT-DETR

## 4. Error Analysis

Add qualitative analysis for:

- missed distant vehicles
- false positives from reflections
- occluded vehicles
- low-contrast fog scenes
- small objects in rain or snow

## 5. Deployment Demo

Add a lightweight demo using Streamlit or FastAPI where users can upload an image and view predicted bounding boxes.

## 6. Export and Optimization

Export the model to ONNX and test lightweight inference speed for edge-style deployment experiments.
