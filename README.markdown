# Adverse Weather Detection for Autonomous Vehicles

This project trains a YOLOv5 model to detect objects in adverse weather conditions (e.g., rain, fog, snow) for autonomous vehicles. It uses a filtered subset of the BDD100K dataset and provides a Jupyter notebook to build the model.

## Project Structure
- `data.yaml`: Configuration file for the dataset paths and classes used by YOLOv5.
- `adverse_weather_detection.ipynb`: Jupyter notebook to train the YOLOv5 model on the filtered dataset.
- `.gitignore`: Excludes datasets, dependencies, and output directories.

## Prerequisites
- **Python 3.8–3.11** (YOLOv5 may not fully support Python 3.12 due to compatibility issues).
- **PyTorch 2.0.1+cu118** (or compatible version for your GPU; adjust based on CUDA version).
- **YOLOv5 Repository**: Clone from GitHub.
- **BDD100K Dataset**: Download and filter for adverse weather (not included in this repo).
- **Jupyter Notebook**: Required to run the training notebook.

## Setup and Running Locally
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/10ishk/adverse-weather-detection.git
   cd adverse-weather-detection
   ```

2. **Install Dependencies**:
   Install Python and required libraries:
   ```bash
   pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==0.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
   pip install pandas numpy opencv-python jupyter
   ```

3. **Clone YOLOv5**:
   Clone the YOLOv5 repository into the project directory:
   ```bash
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   pip install -r requirements.txt
   cd ..
   ```

4. **Prepare the Dataset**:
   - Download the BDD100K dataset from [its official website](https://bdd-data.berkeley.edu/).
   - Filter the dataset for adverse weather conditions (e.g., rainy, foggy, snowy) and convert annotations to YOLO format. This step is not included in the repo; you’ll need to create a script to process the dataset.
   - Place the filtered dataset in `dataset/` with `images/` and `labels/` subdirectories:
     ```
     adverse-weather-detection/
     ├── dataset/
     │   ├── images/
     │   └── labels/
     ├── data.yaml
     ├── adverse_weather_detection.ipynb
     └── ...
     ```
   - Update `data.yaml` with the correct paths to `dataset/images/` and `dataset/labels/`.

5. **Train the Model**:
   Launch Jupyter Notebook and run the training notebook:
   ```bash
   jupyter notebook adverse_weather_detection.ipynb
   ```
   - Follow the steps in the notebook to train the YOLOv5 model. It will save weights to `runs/adverse_weather_expX/weights/best.pt`.
   - Note: The notebook assumes `yolov5s.pt` (pretrained weights) is available; download it from the YOLOv5 repository if missing.

6. **Run Inference**:
   Test the model on a new image using the YOLOv5 `detect.py` script:
   ```bash
   python yolov5/detect.py --weights runs/adverse_weather_expX/weights/best.pt --source path/to/your/image.jpg --img 416 --conf-thres 0.4 --device 0
   ```
   Results will be saved in `runs/detect/expX/`.

## Dataset and Training Process
### Dataset
- **BDD100K Dataset**: A large-scale dataset for autonomous driving with 100,000 images, including diverse weather conditions.
- **Filtering**: Selected images labeled with adverse weather (rainy, foggy, snowy) to focus on challenging conditions.

### Training Steps
1. **Preprocessing**:
   - Filtered BDD100K images for adverse weather conditions.
   - Converted BDD100K annotations (COCO format) to YOLO format (normalized bounding boxes: class, x_center, y_center, width, height).
   - Split the filtered dataset into training and validation sets (e.g., 80-20 split).
   - Organized the dataset into `dataset/images/` and `dataset/labels/`, with paths defined in `data.yaml`.

2. **Model Selection**:
   - Used YOLOv5s (small variant) as the base model for efficiency, starting with pretrained weights (`yolov5s.pt`).

3. **Training**:
   - Trained the model using the `adverse_weather_detection.ipynb` notebook for 10 epochs (based on observed metrics).
   - Image size: 416x416 to balance accuracy and speed.
   - Confidence threshold: 0.4 for detections.
   - Device: GPU (NVIDIA GeForce RTX 3050) for faster training.
   - Techniques:
     - **Data Augmentation**: Applied YOLOv5’s default augmentations (e.g., mosaic, flip, scale) to improve robustness.
     - **Transfer Learning**: Leveraged pretrained weights to accelerate convergence.
     - **Hyperparameter Tuning**: Used default YOLOv5 hyperparameters but adjusted learning rate schedules (`x/lr0`, `x/lr1`, `x/lr2`) for stability.

4. **Evaluation**:
   - Monitored metrics: precision, recall, mAP@0.5, and mAP@0.5:0.95.
   - Final metrics (epoch 9): precision 0.74, recall 0.475, mAP@0.5 0.525, mAP@0.5:0.95 0.30.
   - Saved the best model weights to `runs/adverse_weather_expX/weights/best.pt`.

5. **Inference**:
   - Tested the model on new images with adverse weather, using a confidence threshold of 0.4.
   - Saved detection results (bounding boxes, labels) for analysis.

## Uniqueness of This Project
- **Focus on Adverse Weather**: Specifically targets challenging weather conditions (rain, fog, snow) critical for autonomous vehicle safety, unlike general object detection models.
- **Custom Dataset Filtering**: Processes the BDD100K dataset to create a specialized adverse weather subset, improving model performance in niche scenarios.
- **Practical Application**: Designed for real-world autonomous driving, where reliable detection in poor weather is a significant challenge.
- **Notebook-Driven Workflow**: Provides an interactive Jupyter notebook (`adverse_weather_detection.ipynb`) for training, making it accessible for experimentation and learning.

## Functionality
This project builds an object detection system for autonomous vehicles in adverse weather conditions using YOLOv5. Key components:
- **Training**: `adverse_weather_detection.ipynb` provides a Jupyter notebook to train a YOLOv5 model on a filtered dataset, improving detection accuracy in challenging conditions.
- **Inference**: The trained model can detect objects (e.g., cars, pedestrians) in adverse weather images.

## Notes
- The model was trained with a confidence threshold of 0.4 and image size of 416x416.
- If inference yields no detections, try lowering the confidence threshold (e.g., `--conf-thres 0.25`).
- Ensure your GPU drivers and CUDA toolkit are compatible with PyTorch for faster training/inference.