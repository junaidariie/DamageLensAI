

---

# 🚗 DamageLensAI

# ATTENTION : Due to some frontend error when users sends first request the frontend throws an error related to api and the backend does't start autometicly, so to use the app u first have to manually restart the backend or huggingface space from huggingface hub here's the repo link : https://huggingface.co/spaces/junaid17/DamageLens

🔗 **Live App:** [LINK](https://junaidariie.github.io/DamageLensAI-frontend/)

---

## 🎥 Demo

[![Watch Demo](https://img.youtube.com/vi/dt2TFUvRHvI/0.jpg)](https://www.youtube.com/watch?v=dt2TFUvRHvI)

## 🔍 Overview

**DamageLensAI** is a multi-model vehicle damage analysis system that combines deep learning classification, transformer-based learning, and object detection to identify and localize car damage.

The system integrates:

- **ResNet18** → Reliable CNN-based classification  
- **DeiT (Vision Transformer)** → Transformer-based classification  
- **YOLOv8** → Damage localization with bounding boxes  
- **Grad-CAM** → Model interpretability via attention maps  

Built with a **FastAPI backend** and an **interactive Streamlit frontend**, the application provides both predictions and visual explanations.

---

## ⚠️ Dataset Constraint (Important)

Models were trained on **extremely limited data**, which directly impacts performance:

- **ResNet18 & DeiT:** ~2,000 images (6 classes)  
- **YOLOv8:** ~150 images  

Despite this, strong generalization behavior was achieved using transfer learning and evaluation-driven training.

---

## ⚙️ Features

- Upload image via polished UI  
- Multiple prediction modes:
  - `Fusion (ResNet + DeiT)`
  - `ResNet Only`
  - `DeiT Only`
- **Grad-CAM visualization**
  - ✅ ResNet → Highly accurate attention maps  
  - ❌ DeiT → Weak / noisy attention maps  
- YOLO-based damage localization  
- FastAPI endpoints for scalable inference  

---

## 🧠 Model Architecture

### 🔹 ResNet18
- Fine-tuned pretrained backbone  
- Early layers frozen  
- Custom classifier head  
- Strong spatial feature learning  

### 🔹 DeiT (Transformer)
- **Model:** `facebook/deit-base-distilled-patch16-224`  
- **Fine-tuning Strategy:** - Initial layers frozen to preserve pretrained transformer knowledge.
    - **Unfrozen the last 4 encoder layers** to adapt to specific vehicle damage features.
    - **LayerNorm parameters unfrozen** for better internal normalization during adaptation.
- Captures global dependencies but struggles with localization.

### 🔹 Fusion Model
- Weighted average of predictions:
  - `0.5 * ResNet + 0.5 * DeiT`

### 🔹 YOLOv8
- Object detection for damage regions  
- Outputs bounding boxes + confidence  

---

## 🏷️ Classes

| Class          | Description              |
|----------------|--------------------------|
| F_Breakage     | Front minor damage       |
| F_Crushed      | Front severe damage      |
| F_Normal       | No front damage          |
| R_Breakage     | Rear minor damage        |
| R_Crushed      | Rear severe damage       |
| R_Normal       | No rear damage           |

---

## 📊 Training & Evaluation

### 🔹 ResNet18 Performance

- **Train Accuracy:** 84.67%  
- **Validation Accuracy:** 75.28%  
- **Generalization Gap:** ~9% (controlled)

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| F_Breakage | 0.84 | 0.78 | 0.81 |
| F_Crushed | 0.70 | 0.75 | 0.72 |
| F_Normal | 0.87 | 0.89 | 0.88 |
| R_Breakage | 0.69 | 0.54 | 0.60 |
| R_Crushed | 0.55 | 0.67 | 0.60 |
| R_Normal | 0.77 | 0.77 | 0.77 |

---

### 🔹 DeiT Performance

- **Train Accuracy:** ~98% (Final Epoch)  
- **Validation Accuracy:** **81%** - **Generalization Gap:** ~17% (High learning capacity)

**Classification Report:**

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| **F_Breakage** | 0.85 | 0.86 | 0.86 | 95 |
| **F_Crushed** | 0.79 | 0.76 | 0.78 | 75 |
| **F_Normal** | 0.90 | 0.92 | 0.91 | 100 |
| **R_Breakage** | 0.83 | 0.54 | 0.65 | 54 |
| **R_Crushed** | 0.62 | 0.74 | 0.67 | 57 |
| **R_Normal** | 0.79 | 0.90 | 0.84 | 60 |
| | | | | |
| **Accuracy** | | | **0.81** | 441 |
| **Macro Avg** | 0.80 | 0.79 | 0.79 | 441 |
| **Weighted Avg** | 0.81 | 0.81 | 0.80 | 441 |

---

## 📈 Key Observations

- **Transformer Scalability:** By unfreezing the last 4 encoder layers, DeiT surpassed ResNet18 in validation accuracy (81% vs 75%).
- **Class Imbalance Sensitivity:** Both models struggle more with "Rear" damage categories compared to "Front" damage, likely due to visual similarities or lower image counts in the dataset.
- **Deep Convergence:** The DeiT model converges to near-perfect training accuracy, suggesting it has successfully captured the complex patterns of the limited dataset.

---

## 🔥 Explainability

- ResNet Grad-CAM: strong localization  
- DeiT attention: weak and diffused  

---

## 📁 Project Structure

├── app.py  
├── main.py  
├── scripts/  
├── checkpoints/  
├── static/  
├── Notebooks/  
└── requirements.txt  

---

## 🚀 Setup

```bash
python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt
```

---

## ▶️ Run

```bash
uvicorn app:app --reload  
streamlit run main.py
```

---

## 🧠 Final Takeaway

**81% accuracy** achieved with the fine-tuned DeiT model. While ResNet is better for spatial interpretability (Grad-CAM), the Vision Transformer (DeiT) provides higher raw classification accuracy when the final encoder blocks are unfrozen.
