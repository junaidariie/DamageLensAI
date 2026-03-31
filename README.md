# 🚗 DamageLensAI

🔗 **Live App:** https://junaidariie.github.io/DamageLensAI/

---

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
- `facebook/deit-base-distilled-patch16-224`  
- Fine-tuned classification head  
- Captures global dependencies but struggles with localization  

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

precision    recall  f1-score   support

F_Breakage     0.84    0.78      0.81       95
F_Crushed      0.70    0.75      0.72       75
F_Normal       0.87    0.89      0.88      100
R_Breakage     0.69    0.54      0.60       54
R_Crushed      0.55    0.67      0.60       57
R_Normal       0.77    0.77      0.77       60

accuracy                           0.75      441
macro avg       0.74    0.73      0.73      441
weighted avg    0.76    0.75      0.75      441

---

### 🔹 DeiT Performance

- **Train Accuracy:** ~81%  
- **Validation Accuracy:** ~71–73%  
- **Generalization Gap:** ~7–9% (stable)

**Classification Report:**

precision    recall  f1-score   support

F_Breakage     0.71    0.82      0.76       95
F_Crushed      0.61    0.51      0.55       75
F_Normal       0.75    0.80      0.78      100
R_Breakage     0.82    0.57      0.67       54
R_Crushed      0.69    0.72      0.71       57
R_Normal       0.67    0.73      0.70       60

accuracy                           0.71      441
macro avg       0.71    0.69      0.70      441
weighted avg    0.71    0.71      0.70      441

---

## 📈 Key Observations

- Controlled overfitting (small train–val gap)  
- ResNet outperforms DeiT  
- DeiT struggles with limited data  

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

python -m venv venv  
venv\Scripts\activate  
pip install -r requirements.txt  

---

## ▶️ Run

uvicorn app:app --reload  
streamlit run main.py  

---

## 🧠 Final Takeaway

~75% accuracy with stable generalization on limited data.  
ResNet proved more reliable and interpretable than DeiT.
