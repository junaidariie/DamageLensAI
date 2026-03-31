# Car Damage AI

[Link 1](#) | [Link 2](#)

> 🎥 **Video Demo:** _Add your project video link or embed here._

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Run the API](#run-the-api)
  - [Run the Streamlit App](#run-the-streamlit-app)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Notebooks](#notebooks)
- [Files and Scripts](#files-and-scripts)
- [Next Steps](#next-steps)

---

## Overview

`Car Damage AI` is a vehicle damage analysis project built with a fusion of deep learning models and object detection:

- `ResNet18` for car damage classification
- `DeiT` for transformer-based image classification
- `YOLO` for damage localization and bounding-box predictions
- Grad-CAM for visual model explainability

The project includes a FastAPI backend, a Streamlit frontend, and training notebooks for both ResNet and DeiT fine-tuning.

---

## Features

- Image upload and inference through a visually polished Streamlit app
- Prediction modes: `Fusion`, `ResNet Only`, `DeiT Only`
- Grad-CAM explainability for both ResNet and DeiT
- YOLO-based damage localization with bounding-box output
- Endpoints for ResNet, DeiT, fusion prediction, Grad-CAM generation, and YOLO detection
- Training notebooks with dataset splitting, augmentation, metrics, and visualization

---

## Repository Structure

- `app.py` - FastAPI backend that serves model inference and explainability endpoints
- `main.py` - Streamlit frontend connecting to the API
- `scripts/`
  - `prediction_helper.py` - model wrappers for ResNet, DeiT, and fusion prediction
  - `gradcam.py` - Grad-CAM generation for ResNet and DeiT models
  - `yolo.py` - YOLO damage localization and image export
- `checkpoints/` - saved model weights used by the API
- `static/` - uploaded images and generated result visualizations
- `Notebooks/` - training and evaluation notebooks
- `requirements.txt` - Python dependencies

---

## Setup

### 1. Create and activate your virtual environment

```powershell
cd "d:\Car Damage Project"
python -m venv myvenv
myvenv\Scripts\activate
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Confirm required files

Ensure the following checkpoint files exist:

- `checkpoints/best_resnet_model.pt`
- `checkpoints/best_deit_model.pt`
- `checkpoints/damage_detector.pt`

If these files are missing, run the notebooks in `Notebooks/` to train or generate them.

---

## Usage

### Run the API

```powershell
uvicorn app:app --reload
```

The API runs by default at `http://127.0.0.1:8000`.

### Run the Streamlit App

```powershell
streamlit run main.py
```

> The Streamlit app uses `API_URL = "http://127.0.0.1:8000"` in `main.py`.

---

## Model Architecture

### ResNet18

- Uses a fine-tuned `resnet18` backbone
- Freezes early layers
- Fine-tunes `layer3`, `layer4`, and a custom classification head
- Predicts 6 classes:
  - `Front Breakage`
  - `Front Crushed`
  - `Front Normal`
  - `Rear Breakage`
  - `Rear Crushed`
  - `Rear Normal`

### DeiT

- Uses `facebook/deit-base-distilled-patch16-224` from Hugging Face
- Fine-tunes the transformer head for the same 6-class damage classification task

### Fusion

- Combines ResNet and DeiT probability outputs
- Weighted average fusion is used to create a final ensemble prediction
- Default weights are 0.5 / 0.5

### YOLO

- Uses a YOLO-based detector from `checkpoints/damage_detector.pt`
- Detects damage regions and returns boxes, labels, and confidence scores

---

## Training & Evaluation

### Summary from Notebooks

#### ResNet18 Fine-tuning

- Initial epoch: `Train Acc 18.52%`, `Val Acc 29.25%`
- Final reported epoch: `Train Acc 84.67%`, `Val Acc 75.28%`
- Classification evaluation shows overall validation accuracy near `0.75`

#### DeiT Fine-tuning

- Initial epoch: `Train Acc 31.40%`, `Val Acc 40.59%`
- Later epoch metrics: `Train Acc 80.85%`, `Val Acc 71.88%`
- Best logged validation performance approaching `73.70%`
- Classification evaluation shows overall validation accuracy near `0.71`

### Evaluation Outputs

The notebooks generate the following evaluation artifacts:

- Training and validation loss curves
- Training and validation accuracy curves
- Confusion matrix heatmap
- `classification_report` with precision, recall, and F1-score

If you want to record exact values, open the notebook and inspect the last `classification_report` output.

---

## Notebooks

### `Notebooks/Resnet18_fine_tuning.ipynb`

- Prepares dataset split into `train/` and `val/`
- Uses custom `Car_Dataset` and `ImageFolder`
- Fine-tunes a ResNet18 model with data augmentation
- Saves best checkpoint to `checkpoints/best_resnet_model.pt`
- Generates loss and accuracy plots
- Builds confusion matrix and classification report

### `Notebooks/Deit_fine_tuning.ipynb`

- Uses `DeiTImageProcessor` and `DeiTForImageClassification`
- Prepares dataset and tokenizes images for transformer input
- Fine-tunes the DeiT model
- Saves best checkpoint to `checkpoints/best_deit_model.pt`
- Generates evaluation plots and classification metrics

---

## Files and Scripts

### `app.py`

- Implements FastAPI endpoints:
  - `GET /` status check
  - `POST /predict` Grad-CAM generation
  - `POST /predict/resnet` ResNet-only prediction
  - `POST /predict/deit` DeiT-only prediction
  - `POST /predict/fusion` fusion output
  - `POST /predict/yolo` YOLO damage localization

### `main.py`

- Streamlit frontend with a polished dark UI
- Image uploader and inference controls
- Selectable prediction engine
- Displays predictions, Grad-CAM visualizations, and YOLO detections

### `scripts/prediction_helper.py`

- `ResnetCarDamagePredictor`
- `DeitCarDamagePredictor`
- `FusionCarDamagePredictor`

### `scripts/gradcam.py`

- Grad-CAM generation for ResNet and DeiT models
- Saves overlay visualizations to `static/results`

### `scripts/yolo.py`

- Loads the YOLO detector
- Predicts damage boxes and saves annotated images

---

## Next Steps

- Add the demo video link in the placeholder above
- Replace `Link 1` and `Link 2` with repository, demo, or model documentation URLs
- Add a README screenshot or GIF for the Streamlit UI
- Include exact per-class metrics once the notebook evaluation outputs are executed
- Consider adding a `docker-compose` setup for simpler deployment

---

## Notes

- The Streamlit app assumes the FastAPI service is available at `http://127.0.0.1:8000`
- If you change the API host or port, update `API_URL` in `main.py`
- Use `python -m pip install --upgrade pip` if dependency installation fails
