import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix, roc_auc_score 
from sklearn.metrics import precision_score,recall_score, f1_score
from imblearn.over_sampling import SMOTE


# Function to generate synthetic data
def generate_synthetic_data(n_samples=1000):
    np.random.seed(42)
    data = {
        'Transaction_Amount': np.random.uniform(10, 1000, n_samples),
        'Transaction_Time': np.random.uniform(0, 24, n_samples),
        'Account_Age': np.random.uniform(1, 10, n_samples),
        'Num_Transactions': np.random.randint(1, 50, n_samples),
        'Fraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
    }
    return pd.DataFrame(data)

# Load dataset
data = generate_synthetic_data(2000)

# Splitting dataset
X = data.drop(columns=['Fraud'])
y = data['Fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handling Imbalanced data

smote = SMOTE()
X_train_smote,y_train_smote = smote.fit_resample(X_train,y_train)

# Model Selection

model = RandomForestClassifier(n_estimators=100,random_state = 42)
model.fit(X_train_smote,y_train_smote)

y_pred =model.predict(X_test)
y_prob =model.predict_proba(X_test)[:,-1]

# Model Evaluation
report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test,y_pred)
confusion_matrix_score = confusion_matrix(y_test,y_pred)
roc_auc = roc_auc_score(y_test,y_prob)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1_score = f1_score(y_test,y_pred)

# Streamlit UI
st.title("Creditsafe AI: Fraud Detection")
st.write("Using Random Forest to detect fraudulent transactions in real-time.")

# Project Description
st.subheader("Project Description")
st.markdown(
    """
    This project aims to detect fraudulent financial transactions using a machine learning approach. 
    We use a **Random Forest Classifier** to identify potentially fraudulent transactions based on various features such as transaction amount, transaction time, account age, and number of transactions.
    """
)

# What We Built
st.subheader("What We Built")
st.markdown(
    """
       1. **Machine Learning Model** - A Random Forest classifier is trained to predict fraudulent transactions.
    2. **Streamlit Web App** - A user-friendly interface to:
       - Visualize the dataset and fraud distribution.
       - Download the dataset for further analysis.
       - Enter new transaction details for real-time fraud detection.
    """
)

# Technologies Used
st.subheader("Technologies Used")
st.markdown(
    """
    - **Python** (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
    - **Machine Learning** (Random Forest Classifier)
    - **Streamlit** (Web App for real-time fraud detection)
    """
)


# Dataset Download
st.subheader("Download Dataset")
csv = data.to_csv(index=False).encode('utf-8')
st.download_button(label="Download CSV", data=csv, file_name="fraud_dataset.csv", mime="text/csv")

# Model Evaluation
st.subheader("📊Model Performance")
col1,col2,col3,col4,col5 = st.columns(5)
col1.metric("Accuracy", f"{accuracy:.3f}")
col2.metric("Precision", f"{precision:.3f}")
col3.metric("Recall", f"{recall:.3f}")
col4.metric("F1 Score", f"{f1_score:.3f}")
col5.metric("ROC AUC", f"{roc_auc:.3f}")
st.text(report)

# Confusion Matrix
st.subheader("📉 Confusion Matrix")

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix_score,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Fraud"],
            yticklabels=["Legitimate", "Fraud"],
            ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# Graphs and Data Visualization
st.subheader("Data Distribution")
col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Fraud", data=data, ax=ax1)
    ax1.set_title("Fraud vs Non-Fraud")
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(data["Transaction_Amount"], bins=30, kde=True, ax=ax2)
    ax2.set_title("Transaction Amount Distribution")
    st.pyplot(fig2)

# Real-time Transaction Testing
st.subheader("🔍 Test a New Transaction")
col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("Transaction Amount", 10.0, 1000.0, 100.0)
    time = st.number_input("Transaction Time (0-24 hrs)", 0.0, 24.0, 12.0)

with col2:
    age = st.number_input("Account Age (Years)", 1.0, 10.0, 5.0)
    num_tx = st.number_input("Number of Transactions", 1, 50, 10)

if st.button("🚨 Predict Fraud"):
    sample = pd.DataFrame(
        [[amount, time, age, num_tx]],
        columns=X.columns
    )

    prediction = model.predict(sample)[0]
    probability = model.predict_proba(sample)[0][1]

    if prediction == 1:
        st.error(f"⚠ Fraudulent Transaction (Risk: {probability:.2%})")
    else:
        st.success(f"✅ Legitimate Transaction (Risk: {probability:.2%})")


