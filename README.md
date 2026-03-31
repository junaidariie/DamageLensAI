```markdown
# 🚗 DamageLens AI

[🌐 Live Demo (Streamlit)](https://damagelensai-2ykgnklvvepm5ddzwzugza.streamlit.app/)  
[🤗 Hugging Face Space](https://huggingface.co/spaces/junaid17/DamageLensAI/tree/main)

> ⚠️ **Important Note:** This project is built on *very limited data* and is intended as a **system design + ML integration project**, not a production-ready model.

---

## 📌 Overview

**DamageLens AI** is an end-to-end car damage analysis system that combines:

- 🧠 Deep Learning Classification (ResNet + DeiT)
- 🔍 Explainability (Grad-CAM)
- 📦 Object Detection (YOLO)

The goal is not just prediction, but:
> **Understanding *what* the model predicts, *where* it looks, and *where damage actually exists*.**

---

## 🚀 Live Applications

- 🔗 **Frontend App:**  
  https://damagelensai-2ykgnklvvepm5ddzwzugza.streamlit.app/

- 🔗 **API / Backend (HF Space):**  
  https://junaid17-damagelensai.hf.space

---

## ✨ Key Features

- Multi-model prediction:
  - Fusion (ResNet + DeiT)
  - ResNet only
  - DeiT only

- 📊 Probability visualization with interactive charts  
- 🔥 Grad-CAM explainability (ResNet + DeiT)  
- 🎯 YOLO-based damage localization  
- ⚡ FastAPI backend with multiple endpoints  
- 🎨 Modern Streamlit UI (dark theme)

---

## 🧠 System Architecture

```

Input Image
↓
[ ResNet ]      [ DeiT ]
↓             ↓
Probabilities → Fusion
↓
Final Prediction

* Grad-CAM (Explainability)
* YOLO (Damage Localization)

```

---

## 🏗️ Models Used

### 🔹 ResNet18 (CNN)
- Fine-tuned on car damage dataset
- Strong spatial feature extraction
- **Grad-CAM performs well**

---

### 🔹 DeiT (Transformer)
- `facebook/deit-base-distilled-patch16-224`
- Fine-tuned for classification
- ⚠️ **Grad-CAM performance is poor (expected behavior for ViTs)**

---

### 🔹 Fusion Model
- Combines ResNet + DeiT predictions
- Weighted average (0.5 / 0.5)

---

### 🔹 YOLO (Detection)
- Detects damage regions
- Outputs bounding boxes + confidence

---

## ⚠️ Dataset Limitations (READ THIS)

This project is intentionally transparent about its limitations:

| Model       | Dataset Size |
|------------|-------------|
| ResNet     | ~2000 images |
| DeiT       | ~2000 images |
| YOLO       | ~150 images  |

### Implications:

- YOLO detections may be inconsistent  
- Transformer explainability is weak  
- Model generalization is limited  

> ⚠️ This is a **low-data experimental system**, not production-grade.

---

## 📊 Performance Summary

### ✅ Works Well
- ResNet classification
- Grad-CAM (ResNet)
- End-to-end system integration

### ⚠️ Weak Areas
- DeiT Grad-CAM (poor attention maps)
- YOLO (limited training data)
- Overall robustness

---

## 🖼️ Outputs Provided

For each image, the system generates:

1. ✅ Final prediction (damage type)
2. 📊 Class probabilities
3. 🔥 Grad-CAM (ResNet + DeiT)
4. 🎯 YOLO bounding boxes

---

## 📂 Project Structure

```

.
├── app.py                  # FastAPI backend
├── main.py                 # Streamlit frontend
├── scripts/
│   ├── prediction_helper.py
│   ├── gradcam.py
│   └── yolo.py
├── checkpoints/            # Model weights
├── static/                 # Uploaded + output images
├── Notebooks/              # Training notebooks
├── requirements.txt

````

---

## ⚙️ Setup

```bash
git clone <repo-url>
cd damage-lens-ai

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
````

---

## ▶️ Run Locally

### Start API

```bash
uvicorn app:app --reload
```

### Start Frontend

```bash
streamlit run main.py
```

---

## 🔌 API Endpoints

| Endpoint          | Description      |
| ----------------- | ---------------- |
| `/predict/fusion` | Final prediction |
| `/predict/resnet` | ResNet output    |
| `/predict/deit`   | DeiT output      |
| `/predict`        | Grad-CAM         |
| `/predict/yolo`   | Damage detection |

---

## 🧪 Notebooks

* `Resnet18_fine_tuning.ipynb`
* `Deit_fine_tuning.ipynb`

Includes:

* Training pipeline
* Evaluation metrics
* Visualization

---

## 🎯 What This Project Demonstrates

This is NOT just a model project.

It demonstrates:

* Multi-model system design
* Explainability integration
* Detection + classification pipeline
* API + frontend deployment

---

## 🚀 Future Improvements

* Train YOLO on larger dataset
* Improve DeiT interpretability
* Combine YOLO + Grad-CAM into single visualization
* Add damage severity scoring
* Deploy frontend separately (Vercel)

---

## ⚠️ Final Note

This project focuses on:

> **Building a complete AI system — not just maximizing accuracy.**

---

## 📬 Contact

If you want to collaborate or discuss improvements, feel free to reach out.

```
