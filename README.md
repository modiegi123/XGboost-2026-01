# Enhancing Transparency and Early Detection of At-Risk Students
# in Higher Education Using Academic and Socioeconomic Indicators

## Author: Josephine Morudu (218075774)
## University of Johannesburg, Academy of Computer Science Software Engineering

---

## Overview

This system implements a from-scratch XGBoost (gradient boosted trees)
classifier for predicting student dropout risk using academic, socioeconomic,
and macroeconomic features. No third-party ML library (scikit-learn, the
`xgboost` package, etc.) is used — the boosting algorithm, decision trees,
SMOTE balancing, and evaluation metrics are all implemented in plain Python.

The pipeline:

- Loads and validates the student dataset
- Encodes features (one-hot + Min-Max normalisation)
- Balances the training set with SMOTE
- Tunes and trains the XGBoost classifier
- Predicts dropout risk for held-out students
- Reports accuracy, precision/recall/F1, AUC-ROC, and feature importance

Earlier exploratory components (PCA visualisation, SHAP explanations,
fairness auditing, the SQLite database layer, and the advisor dashboard)
have been removed, so the project stays focused on the core XGBoost
prediction task.

## Dependencies

This project uses the Python standard library (Python 3.10+)..

## How to Run

Train the model on the full dataset (this also saves the trained model to
`output/model.pkl` for reuse):

```bash
cd student_dropout_prediction
python main.py
```

Predict dropout risk for new student(s) using the saved model, without
retraining. The input CSV needs one row per student with the same feature
columns as the training dataset (no `Target` column):

```bash
python predict.py --input new_students.csv
```

## Dataset

Place the `dataset.csv` file in the `data/` folder.

The dataset is the **"Predict Students' Dropout and Academic Success"** dataset
by Realinho et al. (2022), available from:
- UCI Machine Learning Repository
- Kaggle

It contains 4,424 student records across 35 features with a 3-class target
(Dropout, Enrolled, Graduate).

## Project Structure

```
student_dropout_prediction/
├── main.py                    # Entry point
├── pipeline.py                # XGBoost training/prediction pipeline
├── data_loader.py             # CSV data ingestion and validation
├── config.py                  # Global configuration and constants
├── README.md                  # This file
├── data/
│   └── dataset.csv            # Place dataset here
├── output/                    # Generated artefacts (created on run)
│   └── predictions.csv        # Per-student predictions
├── models/
│   ├── __init__.py
│   ├── student.py             # Student entity
│   ├── encoded_feature_matrix.py  # Feature encoding & normalisation
│   ├── smote_sampler.py       # SMOTE oversampling
│   ├── xgboost_classifier.py  # From-scratch gradient boosted trees classifier
│   └── performance_metrics.py # Model evaluation metrics
```

## Pipeline Stages

1. **Data Ingestion** (FR-01): Load, validate, deduplicate
2. **Feature Encoding** (FR-02, FR-03): Imputation, one-hot + Min-Max normalisation
3. **SMOTE Balancing** (FR-06): Class balancing on the training split only
4. **Training** (FR-07): Grid search + final XGBoost model training
5. **Prediction** (FR-08): Dropout / Enrolled / Graduate classification on the test set
6. **Evaluation** (FR-14): Accuracy, precision/recall/F1, AUC-ROC, confusion matrix
7. **Feature Importance**: Ranked features driving the model's predictions

## Requirements Traceability

| FR | Description | Module |
|----|-------------|--------|
| FR-01 | Data ingestion, zero duplicates | data_loader.py |
| FR-02 | Imputation on training only | encoded_feature_matrix.py |
| FR-03 | One-hot + Min-Max encoding | encoded_feature_matrix.py |
| FR-04 | Reproducibility (fixed seed) | config.py |
| FR-06 | SMOTE within training folds | smote_sampler.py |
| FR-07 | Grid search on training only | xgboost_classifier.py |
| FR-08 | Dropout risk classification | pipeline.py |
| FR-14 | Full performance metrics | performance_metrics.py |
