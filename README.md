# Credit Card Fraud Detection using Machine Learning

##  Overview

This project focuses on detecting fraudulent credit card transactions using machine learning techniques. Fraud detection is a critical problem in financial systems, where the goal is to identify suspicious transactions while minimizing false alarms.

The model analyzes transaction patterns and identifies anomalies that may indicate fraudulent activity.

---

##  Objectives

* Detect fraudulent credit card transactions
* Handle highly imbalanced datasets
* Compare different machine learning algorithms
* Build a reliable fraud detection system

---

##  Dataset

The dataset used in this project contains anonymized credit card transactions made by European cardholders.

Features include:

* Time of transaction
* Transaction amount
* PCA transformed features (V1–V28)
* Transaction class label

Class values:

* **0 → Normal Transaction**
* **1 → Fraudulent Transaction**

The dataset is highly imbalanced, where fraudulent transactions represent a very small percentage of the total data.

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

---

##  Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├__Final_Model.py
│── eda.py
│── preprocessing.py
│── models.py
│── evaluation.py
|__ Visualize.py
├── main.py
├── Tuning.py
└── README.md
```

---

##  Workflow

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing
4. Handling Imbalanced Data
5. Model Training
6. Model Evaluation

---

##  Data Preprocessing

The following preprocessing steps are applied:

* Handling missing values
* Feature scaling using StandardScaler
* Train-test split
* Balancing dataset using sampling techniques

---

##  Machine Learning Models

The following models are used for fraud detection:

* Isolation Forest (Anomaly Detection)
* LOF
* OneClass-SVM
---

## 📈 Evaluation Metrics

Because fraud datasets are highly imbalanced, accuracy alone is not reliable. The following metrics are used:

* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC Score

---


---

## 🚀 How to Run the Project

Clone the repository:

```
git clone https://github.com/senthilpandian-21/Credit-Card-Fraud-Detection
```

Navigate to the project folder:

```
cd credit-card-fraud-detection
```


```

Run the project:

```
python main.py
```

---

## 📌 Future Improvements

* Deep learning models for fraud detection
* Real-time fraud detection system
* Hybrid anomaly detection methods
* Model explainability using SHAP

---

## 👨‍💻 Author

Developed by **Senthil Pandian**

Machine Learning Enthusiast | AI Developer
