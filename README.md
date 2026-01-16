# 🛡️ Credit Card Fraud Detection System

This project is a **Streamlit-based Machine Learning application** for detecting fraudulent credit card transactions from **highly imbalanced data**. It provides an interactive interface for analyzing transactions and predicting fraud in real time.

---

## 🔹 Features

* **Random Forest Classifier** for high accuracy fraud detection
* **Imbalanced Data Handling:** SMOTE, Oversampling, Undersampling
* **Train/Test Split with Stratification** to preserve class distribution
* **Feature Scaling** without causing data leakage
* **Comprehensive Evaluation Metrics:**

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Specificity
  * Confusion Matrix & Feature Importance
* **CSV Upload Support** to analyze your own dataset
* **Synthetic Data Generation** if dataset is missing

---

## 🔹 Tech Stack

* **Programming Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning & Data Handling:** Scikit-learn, Imbalanced-learn, Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

---

## 🔹 Dataset

The project uses the **Credit Card Fraud Detection Dataset** from Kaggle, which contains credit card transactions labeled as fraudulent or legitimate.

**Dataset Link:** [Credit Card Fraud Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

---

## 🔹 How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Upload your dataset or use synthetic data generation to test the model.

---

## 🔹 System Workflow

1. **Data Loading:** Upload CSV or generate synthetic data
2. **Data Preprocessing:** Handle missing values, scale features
3. **Data Balancing:** Apply SMOTE or other resampling methods
4. **Model Training:** Train Random Forest classifier
5. **Evaluation:** Use multiple metrics and confusion matrix
6. **Prediction:** Detect fraud in real-time transactions

---

## 🔹 Future Enhancements

* Integration of **Deep Learning models** for better accuracy
* **Real-time API deployment**
* **Interactive dashboard** for monitoring transactions
* Advanced **anomaly detection techniques**

---

## 👩‍💻 Author

**Farzeen Fatima**
Undergraduate Computer Science Student
IBA Sukkur


