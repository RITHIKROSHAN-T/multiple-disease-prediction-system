# 🚀 Multiple Disease Prediction System  
### *A Machine Learning Based Healthcare Diagnostic Web Application (Kidney • Liver • Parkinson’s)*

---

## 📘 Project Overview  

The **Multiple Disease Prediction System** is an end-to-end Machine Learning project designed to predict the presence of **Chronic Kidney Disease (CKD)**, **Liver Disease**, and **Parkinson’s Disease** using medical input parameters.  

The system integrates:  
- **Data Preprocessing & EDA**  
- **Model Training & Evaluation**  
- **Model Deployment using Streamlit**  
- **Modular, production-ready code architecture**  

This project demonstrates complete ML pipeline knowledge—right from dataset cleaning to building an interactive prediction web app.

---

## 🎯 Problem Statement  

Early detection of diseases like CKD, Liver Disease, and Parkinson’s is critical for effective treatment. Manual diagnosis is time-consuming and often requires specialized medical expertise.  

This system aims to assist healthcare professionals and the general public by providing:  

- **Fast, reliable, ML-based predictions**  
- **Easy-to-use interactive UI**  
- **Disease-specific prediction models**  

---

## 🧠 Key Features  

### ✔ Multi-disease Prediction  
Three separate ML models predict:  
- 🩺 **Chronic Kidney Disease**  
- 🟠 **Liver Disease**  
- 🟣 **Parkinson’s Disease**

### ✔ End-to-End ML Pipeline  
Includes:  
- Data cleaning  
- Exploratory Data Analysis (EDA)  
- Feature engineering  
- Model training  
- Hyperparameter optimization  
- Saving models as `.pkl`  

### ✔ Production-Ready Streamlit App  
- Clean user interface  
- Real-time predictions  
- Disease-specific input forms  
- Model loading & inference pipeline  

### ✔ Modular, Scalable Architecture  
Easily extendable to more diseases and datasets.

---

## 🏛 Project Architecture  



## 📂 Folder Structure Explained

### **/data/**
Raw datasets:
- kidney_disease.csv  
- indian_liver_patient.csv  
- parkinsons.csv  

### **/notebooks/**
EDA notebooks:
- 01_eda_kidney.ipynb  
- 02_eda_liver.ipynb  
- 03_eda_parkinsons.ipynb  

### **/src/training/**
Training scripts:
- train_kidney.py  
- train_liver.py  
- train_parkinsons.py  

### **/models/**
Trained model files:
- kidney_model.pkl  
- liver_model.pkl  
- parkinsons_model.pkl  

### **/app/**
Streamlit frontend:
- app.py  
- kidney_ui.py  
- liver_ui.py  
- parkinsons_ui.py  

### **/assets/**
Screenshots for README

### **/docs/**
Architecture diagram, project report, presentation (optional)

---

## 🧪 Model Performance Summary

| Disease | Best Model | Accuracy | F1 Score |
|---------|------------|----------|----------|
| Kidney | RandomForest | **100%** | **1.00** |
| Liver | Logistic Regression | 73% | 0.83 |
| Parkinson's | Logistic Regression | 92% | 0.94 |

---

