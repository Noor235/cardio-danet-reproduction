# Reproduction and Extension of Cardiovascular Disease Prediction Models

## Overview

This repository presents a reproducible benchmarking study for cardiovascular disease prediction using structured clinical data.

The study reproduces the experimental framework of Kırboğa (2023) and extends it through:

- Multi-seed evaluation (5 random seeds)
- Deep tabular modeling (DANet)
- SHAP-based interpretability
- Permutation importance validation
- Calibration and Brier score analysis
- Threshold sensitivity evaluation
- Clinical error (false-negative) assessment
- Computational cost comparison

The goal is to evaluate reproducibility, stability, and interpretability of machine learning models in a healthcare risk prediction setting.

---

## Models Evaluated

Classical Machine Learning:
- Logistic Regression
- Random Forest
- Support Vector Machine
- K-Nearest Neighbors
- Decision Tree
- Gaussian Naïve Bayes
- XGBoost

Deep Model Extension:
- Deep Abstract Network (DANet)

---

## Dataset

Dataset: Cardiovascular Disease Dataset  
Source: Kaggle (Sulianova)

Download from:
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

After downloading, place the file as:

data/cardio_train.csv

Note: The dataset is not included due to licensing.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Noor235/cardio-danet-reproduction.git
cd cardio-danet-reproduction 
cd YOUR_REPO_NAME

