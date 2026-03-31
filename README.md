***

# Car Damage AI

> 🎥 **Video Demo:** _Add your project video link or embed here._
> App Link : https://damagelensai-2ykgnklvvepm5ddzwzugza.streamlit.app/

***

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training & Evaluation](#training--evaluation)
- [Notebooks](#notebooks)
- [Files and Scripts](#files-and-scripts)
- [Next Steps](#next-steps)

***

## Overview

**Car Damage AI** is a vehicle damage analysis system combining deep learning classification and object detection:

- **ResNet18** for car damage classification
- **DeiT** (transformer-based) for image classification  
- **YOLOv8** for damage localization and bounding boxes
- **Grad-CAM** for model explainability

Includes FastAPI backend, Streamlit frontend, and training notebooks.

**⚠️ Important Note:** Models trained on **extremely limited data**:
- Transformer (DeiT) & ResNet18: **2,000 images**
- YOLOv8: **~150 images**

**Attention Map Performance:**
- ✅ **ResNet Grad-CAM: Performing VERY WELL**
- ❌ **DeiT Attention Maps: Performing VERY POORLY**

***

## Features

- Image upload via polished Streamlit app
- Prediction modes: `Fusion`, `ResNet Only`, `DeiT Only`
- Grad-CAM explainability (excellent for ResNet, poor for DeiT)
- YOLO damage localization with bounding boxes
- FastAPI endpoints for all predictions
- Training notebooks with augmentation & metrics

***

## Repository Structure

```
├── app.py                 # FastAPI backend
├── main.py               # Streamlit frontend
├── scripts/
│   ├── prediction_helper.py
│   ├── gradcam.py
│   └── yolo.py
├── checkpoints/          # Model weights
├── static/              # Results & uploads
├── Notebooks/           # Training notebooks
└── requirements.txt
```

***

## Setup

### 1. Environment Setup

```powershell
cd "Car Damage Project"
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Verify Checkpoints

Ensure these exist in `checkpoints/`:
- `best_resnet_model.pt`
- `best_deit_model.pt` 
- `damage_detector.pt`

***

## Usage

### Start API Server
```powershell
uvicorn app:app --reload
```
**Runs at:** `http://127.0.0.1:8000`

### Start Streamlit App
```powershell
streamlit run main.py
```

***

## Model Architecture

### ResNet18
- Fine-tuned backbone with frozen early layers
- Custom head for **6 damage classes**:
  | Class          | Description      |
  |----------------|------------------|
  | Front Breakage | Front minor damage |
  | Front Crushed  | Front severe damage |
  | Front Normal   | No front damage |
  | Rear Breakage  | Rear minor damage |
  | Rear Crushed   | Rear severe damage |
  | Rear Normal    | No rear damage |

### DeiT (Transformer)
- `facebook/deit-base-distilled-patch16-224`
- Fine-tuned transformer head

### Fusion Model
- Weighted average (0.5 ResNet + 0.5 DeiT)

### YOLOv8
- Damage region detection
- Bounding boxes + confidence scores

***

## Training & Evaluation

**Trained on LIMITED data:**
- **ResNet18 & DeiT:** 2,000 images total
- **YOLOv8:** ~150 images

### Key Metrics

| Model    | Train Acc | Val Acc | Best Val Acc |
|----------|-----------|---------|--------------|
| **ResNet18** | 84.67%    | 75.28%  | **75.28%**   |
| **DeiT**     | 80.85%    | 71.88%  | **73.70%**   |

**Grad-CAM Performance:**
- **ResNet:** Excellent attention visualization
- **DeiT:** Very poor attention maps

***

## Notebooks

### 1. `Resnet18_fine_tuning.ipynb`
- Dataset split + augmentation
- Training curves & confusion matrix
- Saves `best_resnet_model.pt`

### 2. `Deit_fine_tuning.ipynb`  
- Transformer preprocessing
- Evaluation plots & metrics
- Saves `best_deit_model.pt`

***

## Files and Scripts

| File | Purpose |
|------|---------|
| `app.py` | FastAPI endpoints (/predict, /yolo, etc.) |
| `main.py` | Streamlit UI with dark theme |
| `prediction_helper.py` | Model wrappers |
| `gradcam.py` | ResNet/DeiT visualization |
| `yolo.py` | Damage detection |

***

## Next Steps

- [ ] Add demo video
- [ ] Include per-class F1-scores
- [ ] UI screenshots/GIFs
- [ ] Docker deployment
- [ ] Improve DeiT attention maps
- [ ] Dataset expansion

***

**Notes:**
- Update `API_URL` in `main.py` if changing host/port
- `pip install --upgrade pip` if dependency issues

***
