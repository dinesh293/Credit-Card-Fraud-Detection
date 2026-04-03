# 💳 Credit Card Fraud Detection using Machine Learning

## 📌 Overview
This project is a real-time fraud detection system built using Machine Learning to identify fraudulent financial transactions.  
It uses a **Random Forest Classifier** combined with **SMOTE** to handle class imbalance and improve fraud detection performance.

---

## 🎯 Key Highlights
- 🔍 Detects fraudulent transactions in real-time
- ⚖️ Handles imbalanced data using SMOTE
- 📊 Provides detailed model performance metrics
- 📈 Interactive data visualization dashboard
- 🌐 Built as a deployable Streamlit web application

---

## 🧠 Machine Learning Approach
- Model: **Random Forest Classifier**
- Data Balancing: **SMOTE (Synthetic Minority Oversampling)**
- Train-Test Split: 80:20

### 📊 Model Performance
- Accuracy: ~0.90+
- Precision: High (reliable fraud detection)
- Recall: Significantly improved (~30%)
- F1 Score: Balanced performance
- ROC-AUC: ~0.92

---

## ⚙️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn (SMOTE)  
- Matplotlib, Seaborn  
- Streamlit  

---

## 💻 Features
- 📥 Download dataset
- 📊 Fraud vs Non-Fraud visualization
- 📉 Confusion matrix heatmap
- 🔎 Real-time transaction testing
- 📈 Performance metrics display

---

## ▶️ How to Run Locally

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
streamlit run app.py
