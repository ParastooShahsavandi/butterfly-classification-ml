# Butterfly Classification ML (EfficientNetB0)

This repository contains code and resources for classifying **40 butterfly species** using **EfficientNetB0** with transfer learning and data augmentation.  
The full project description, methodology, and results are provided in the attached PDF report.

## 📂 Dataset  
The dataset is not stored in this repository due to size limits. You can download it from Kaggle:  
👉 [Butterfly Images (40 Species) – Kaggle](https://www.kaggle.com/datasets/gpiosenka/butterfly-images40-species)

## 📄 Contents  
- `train_model.py` – Script for training the model.  
- `evaluate_model.py` – Script for evaluating the trained model and generating reports.  
- `evaluation_report_2025-05-25_10-24-07.txt` – Accuracy, F1-score, and per-class metrics.  
- `confusion_matrix.png` – Confusion matrix of classification results.  
- *(Optional)* `efficientnet_b0_trained.keras` – Saved trained model (if uploaded).  

## 🚀 Quick Start  
1. Download and extract the dataset from Kaggle.  
2. Organize it into `train` and `valid` folders as described in the PDF.  
3. Run the training script:  
   ```bash
   python train_model.py
4. Evaluate the trained model:
  ```bash
   python evaluate_model.py

