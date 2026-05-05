#  Driver Drowsiness Detection System Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red?style=flat-square&logo=keras)

> **Author:** Venkata Naga Sai Chaitanya TUTIKE  
> **Student ID:** 2876709  
> **Institution:** University of East London  
> **Supervisor:**  Rasha Hafida

---

## 📌 Overview

Driver drowsiness is a leading cause of road accidents, particularly during long drives and night travel. This project builds an intelligent detection system using deep learning to analyze facial cues — eye closure, yawning, and head movement — and classify the driver's alertness state in real time.

Three models were developed and compared:

| Model | Accuracy | F1-Score |
|---|---|---|
| Custom CNN | 82.86% | 82.58% |
| DenseNet201 (Transfer Learning) | 83.00% | 82.55% |
| **Fine-Tuned DenseNet201** ⭐ | **87.34%** | **86.82%** |

---

## 🗂️ Dataset

- **Source:** NTHU Driver Drowsiness Detection (DDD) Dataset — Kaggle
- **Classes:** `notdrowsy`, `yawning`, `slowBlinkWithNodding`, `sleepyCombination`
- **Split:** 70% Train / 15% Validation / 15% Test
- Images resized to `128×128`, normalized, and augmented (rotation, flip, zoom, shift)

---

## 🧠 Models

- **Custom CNN** — Built from scratch with 3 convolutional blocks, dropout, and softmax output. Serves as the baseline.
- **DenseNet201** — Pre-trained on ImageNet with frozen weights and a custom classification head added on top.
- **Fine-Tuned DenseNet201** — Best performing model. Last 30 layers unfrozen and retrained at `lr=1e-5` for domain-specific adaptation, yielding a ~4.3% improvement in F1-score over the baselines.

---

## ⚙️ Setup

```bash
git clone https://github.com/Chaitanya-AI-ML/Driver-drowsiness-detection-system-using-deep-learning-methods.git
cd Driver-drowsiness-detection-system-using-deep-learning-methods
pip install -r requirements.txt
```

**Key dependencies:** `tensorflow`, `keras`, `numpy`, `scikit-learn`, `opencv-python`, `matplotlib`

---

## 🚀 Usage

Open and run the main notebook in Google Colab or Jupyter:

```
notebooks/driver_drowsiness_detection.ipynb
```

The notebook covers dataset loading, preprocessing, model training, evaluation, and Grad-CAM visualization end-to-end.

---

## 🔍 Explainability — Grad-CAM

Grad-CAM heatmaps are generated to visualize which facial regions (eyes, mouth) influenced the model's prediction. This helps verify the model is learning meaningful drowsiness indicators rather than irrelevant background features.

---

## 📈 Key Results

- Fine-tuning DenseNet201 provided the most significant gains — confirming that domain adaptation of pretrained features outperforms both training from scratch and frozen transfer learning.
- The `sleepyCombination` class was the hardest to classify due to its visual overlap with other drowsy states.
- ROC-AUC scores were highest for the Fine-Tuned DenseNet201 across all four classes.

---

## 🛣️ Future Work

- Integrate temporal models (LSTM / 3D-CNN) for sequential drowsiness analysis
- Add multimodal inputs (steering data, physiological signals)
- Optimize for real-time edge deployment using model pruning or MobileNet

---

## 🙏 Acknowledgements

Thanks to ** Rasha Hafida** (University of East London) for her guidance and support throughout this project.