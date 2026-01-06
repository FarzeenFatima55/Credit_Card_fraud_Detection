🛡️ Credit Card Fraud Detection System

This project is a Streamlit-based Machine Learning application for detecting
fraudulent credit card transactions from highly imbalanced data.

🔹 Features
- Random Forest Classifier
- Handles imbalanced data (SMOTE, Oversampling, Undersampling)
- Train/Test split with stratification
- Feature scaling without data leakage
- Evaluation Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Specificity
- Confusion Matrix & Feature Importance
- CSV upload support
- Synthetic data generation if dataset is missing

 🔹 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Imbalanced-learn
- Pandas, NumPy
- Matplotlib, Seaborn

 🔹 Dataset
Credit Card Fraud Detection Dataset (Kaggle)

Link:  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset is stored in the `data/` folder.

 🔹 How to Run
```bash
pip install -r requirements.txt
streamlit run app.py
