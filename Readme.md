# Real-Time Fraud Detection System with Explainable AI & Live Dashboard

## Project Description
This project is a Machine Learning based fraud detection system that helps identify suspicious financial transactions in real time.

The project also explains the model predictions using Explainable AI (SHAP) and provides a live interactive dashboard using Streamlit.

---

# Dataset Used
IEEE-CIS Fraud Detection Dataset

Download Link:
https://www.kaggle.com/c/ieee-fraud-detection/data

Files Used:
- train_transaction.csv
- train_identity.csv

---

# Tools & Technologies Used

## Programming Language
- Python

## Libraries
- Pandas
- NumPy
- Scikit-learn
- LightGBM
- XGBoost
- SHAP
- Streamlit
- Plotly
- Matplotlib
- Seaborn
- SMOTE

---

# Project Steps

## 1. Data Loading
- Loaded transaction and identity datasets
- Merged datasets using TransactionID

## 2. Data Cleaning
- Removed unnecessary columns
- Filled missing values
- Handled data imbalance using SMOTE

## 3. Feature Engineering
Created new features such as:
- HourOfDay
- AmtToMeanRatio
- DeviceRisk

## 4. Model Training
Trained three models:
- LightGBM
- XGBoost
- Isolation Forest

## 5. Model Evaluation
Used evaluation metrics like:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- PR-AUC

## 6. Explainable AI
Used SHAP to:
- Explain predictions
- Identify important features
- Generate SHAP plots

## 7. Visualizations
- SHAP Summary Plot
- Fraud Rate by Hour
- Transaction Amount Distribution
- Risk Tier Donut Chart
- Precision-Recall Curve
- Interactive Scatter Plot

---

