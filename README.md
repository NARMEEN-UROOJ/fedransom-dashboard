# FedRansom — Federated Learning Based Privacy-Preserving Ransomware Detection

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow)](https://tensorflow.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-10B981?style=flat-square)](LICENSE)

> A heterogeneous federated learning framework enabling multiple organizations to collaboratively detect ransomware threats **without sharing a single byte of sensitive data.**

---

## Overview

FedRansom implements a privacy-preserving federated learning system across two independent data modalities:

- **Image Branch** — MobileNetV2 CNN trained on Malimg + MaleBin grayscale malware images (60-class family classification)
- **EMBER Branch** — MLP Dense network trained on EMBER 2018 and EMBER 2024 static PE file features (binary malware/benign detection)

Both branches use **manual FedAvg aggregation** (McMahan et al., 2017) with **FedMD cross-modal knowledge distillation** (Li & Wang, 2019) for heterogeneous model collaboration.

---

## Key Results

| Branch | Model | Accuracy | F1 Score | ROC-AUC |
|--------|-------|----------|----------|---------|
| Image | Centralized CNN | 89.28% | 86.52% | 99.50% |
| Image | Client 3 — Local | 83.89% | 80.51% | 99.30% |
| Image | Client 4 — Local | 88.26% | 86.03% | 99.46% |
| **Image** | **FedAvg Global** | **92.32%** | **90.30%** | **99.77%** |
| EMBER 2018 | Centralized MLP | 95.20% | 95.19% | 98.82% |
| EMBER 2018 | Client 1 IID | 93.41% | 93.45% | 98.57% |
| EMBER 2018 | Client 2 non-IID | 85.90% | 87.53% | 97.59% |
| **EMBER 2018** | **FedAvg Global** | **94.71%** | **94.63%** | **98.51%** |
| EMBER 2024 | Centralized MLP | 94.17% | 94.00% | 98.35% |
| **EMBER 2024** | **FedAvg Global** | **94.53%** | **94.63%** | **98.51%** |

**Key finding:** The Image FedAvg global model (92.32%) **beats the centralized baseline (89.28%) by 3.04%** — without sharing any raw data.

### Non-IID Robustness
| Setup | Accuracy | vs Centralized |
|-------|----------|----------------|
| IID FedAvg | 92.32% | +3.04% |
| Non-IID FedAvg | 92.00% | +2.72% |

Gap between IID and Non-IID: only **0.32%** — proving strong robustness to heterogeneous data distributions.

### Privacy Attack Resistance
| Attack | Result |
|--------|--------|
| Model Inversion (Image) | **FAILED** — outputs are unrecognizable noise |
| Membership Inference (Image) | **50.20%** accuracy (≈ random, gap = 0.20%) |
| Membership Inference (EMBER) | **~50.20%** accuracy (≈ random) |

---

## Dashboard — 8 Pages

| Page | Description |
|------|-------------|
| Home | Hero, animated federated topology map, kill chain visualizer |
| Overview | All 11 models comparison with full metrics |
| Image Branch | FedAvg results, Non-IID experiment, ROC-AUC, Confusion Matrix |
| EMBER Branch | EMBER 2018 + 2024, 3 confusion matrices, 7 SHAP tabs |
| Live Detection — Image | Real-time 60-class detection + LIME explanations |
| Live Detection — EMBER | Binary PE file detection + gauge + feature deviation |
| FedMD Results | Cross-modal distillation results and pipeline |
| Privacy Demo | Model inversion + membership inference attack resistance |

---

## Installation

### Requirements
- Python **3.11** (required — TensorFlow does not support 3.12+)

```bash
# Clone
git clone https://github.com/NARMEEN-UROOJ/fedransom-dashboard.git
cd fedransom-dashboard

# Create virtual environment
py -3.11 -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`

---

## Project Structure

```
fedransom-dashboard/
├── app.py                          # Streamlit dashboard (1300+ lines)
├── requirements.txt
├── README.md
│
├── data/                           # Metrics, configs, test samples
│   ├── X_test_sample.npy           # 4700 image test samples
│   ├── y_test_sample.npy           # One-hot labels (60 classes)
│   ├── ember_test_sample_scaled.npy # 500 EMBER samples (scaled)
│   ├── ember_test_sample_raw.npy   # 500 EMBER samples (raw values)
│   ├── ember_feature_names.json    # 890 selected PE feature names
│   ├── label_encoder.pkl           # Sklearn LabelEncoder (60 classes)
│   ├── image_federated_metrics.json # Per-round FL accuracy history
│   ├── image_full_metrics.json     # Acc/Prec/Recall/F1/ROC-AUC
│   ├── fedavg_round_history.csv    # EMBER round metrics
│   ├── fedavg_comparison.csv       # EMBER model comparison
│   ├── fedavg_global_feature_cols.json # 890 selected EMBER features
│   ├── fedmd_results.json          # FedMD distillation results
│   ├── niid_results.json           # Non-IID experiment results
│   ├── privacy_results.json        # Privacy attack results
│   └── ember_local_vs_global_results.csv
│
├── models/                         # Trained federated models
│   ├── image_federated_global.h5   # Image FedAvg global (92.32%)
│   ├── ember_fedavg_best.keras     # EMBER 2018 FedAvg (94.71%)
│   └── ember24_fedavg_global_model.keras # EMBER 2024 FedAvg (94.53%)
│
├── assets/                         # Pre-computed visualization PNGs
│   ├── image_confusion_matrix_top20.png
│   ├── image_roc_metrics.png
│   ├── ember_confusion_matrix.png
│   ├── ember_central_confusion_matrix.png
│   ├── ember_client1_confusion_matrix.png
│   ├── ember_local_vs_global.png
│   ├── ember_convergence.png
│   ├── niid_comparison.png
│   ├── niid_distribution.png
│   ├── privacy_attack_demo.png
│   ├── shap_feature_importance_bar.png
│   ├── shap_beeswarm.png
│   ├── shap_waterfall_malware.png
│   ├── shap_waterfall_benign.png
│   ├── shap_heatmap.png
│   ├── shap_class_specific.png
│   └── shap_dependence_top1.png
│
└── Notebooks/                      # All training notebooks
    ├── 01_Image_Centralized_CNN.ipynb
    ├── 02_Image_Local_Clients.ipynb
    ├── 03_Image_FedAvg.ipynb
    ├── 04_FedMD_CrossModal.ipynb
    ├── 05_Improvement1_ROC_AUC.ipynb
    ├── 06_Improvement2&3_Privacy_Attack_NonIID.ipynb
    ├── 07_Confusion_Matrix.ipynb
    ├── 08_EMBER_Centralized.ipynb
    ├── 09_EMBER_Client1_Local.ipynb
    ├── 10_EMBER_Client2_Local.ipynb
    ├── 11_EMBER_FedAvg.ipynb
    ├── 12_EMBER24_Centralized.ipynb
    ├── 13_EMBER24_Client1.ipynb
    ├── 14_EMBER24_Client2.ipynb
    └── 15_EMBER24_FedAvg_SHAP.ipynb
```

---

## Datasets

| Dataset | Samples | Features | Task |
|---------|---------|----------|------|
| Malimg + MaleBin | 31,332 images | 128×128 grayscale | 60-class family classification |
| EMBER 2018 | 599,920 PE files | 2,341 static features | Binary malware/benign |
| EMBER 2024 | 720,000 PE files | 890 selected features | Binary malware/benign |

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning | TensorFlow 2.x / Keras |
| CNN Architecture | MobileNetV2 (ImageNet pre-trained, fine-tuned) |
| MLP Architecture | Dense (512→256→128→1) with BatchNorm + Dropout |
| FL Algorithm | Manual FedAvg — McMahan et al. (2017) |
| Cross-modal FL | FedMD — Li & Wang (2019) |
| Image Explainability | LIME (300 perturbations, batch_size=8) |
| Feature Explainability | SHAP (SHapley Additive exPlanations) |
| Dashboard | Streamlit 1.35 |
| Visualization | Plotly + Matplotlib + Seaborn |
| Data Processing | NumPy · Pandas · Scikit-learn |

---

## FL Implementation Details

```
Image Branch:
  Clients:       2 (Client 3 + Client 4)
  Rounds:        10
  Local epochs:  5 per round
  Batch size:    16
  Optimizer:     Adam (lr=1e-4)
  Aggregation:   Sample-proportional weighted average
  BN fix:        8-sample calibration pass after each aggregation

EMBER Branch:
  Clients:       2 (Client 1 IID + Client 2 non-IID)
  Rounds:        10
  Aggregation:   FedAvg with warm-start initialization
  Features:      890 selected from 945 raw EMBER 2024 columns

FedMD (Cross-Modal):
  Rounds:        5
  Models:        Image CNN global + EMBER 2024 MLP global
  Temperature:   T = 3.0 (knowledge distillation)
  Consensus:     0.6 × EMBER + 0.4 × Image (weighted by accuracy)
```

