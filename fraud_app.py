import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Try importing imblearn samplers; show helpful message if missing
try:
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    IMBLEARN_AVAILABLE = True
except Exception as e:
    IMBLEARN_AVAILABLE = False
    IMBLEARN_IMPORT_ERROR = str(e)

# ==========================================
# 1. PAGE CONFIGURATION & UI SETUP
# ==========================================
st.set_page_config(page_title="🛡️ Fraud Detection System", layout="wide", initial_sidebar_state="expanded")

st.title("🛡️ Credit Card Fraud Detection Dashboard")
st.markdown("""
<style>
    .metric-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f0f2f6;
        text-align: center;
        margin: 5px;
    }
</style>
This application analyzes transaction data to identify fraudulent activities using Machine Learning.
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & HANDLING
# ==========================================
@st.cache_data
def load_data():
    """
    Loads creditcard.csv, creditcard.csv.zip, or generates synthetic data 
    if files are missing.
    """
    df = None
    data_source = "Synthetic (Demo)"

    # 1. Try loading raw CSV
    if os.path.exists('creditcard.csv'):
        df = pd.read_csv('creditcard.csv')
        data_source = "Local CSV"
    
    # 2. Try loading Zipped CSV (Useful for GitHub/Large files)
    elif os.path.exists('creditcard.csv.zip'):
        df = pd.read_csv('creditcard.csv.zip', compression='zip')
        data_source = "Local ZIP"
        
    elif os.path.exists('creditcard.zip'):
        df = pd.read_csv('creditcard.zip', compression='zip')
        data_source = "Local ZIP"

    # 3. Generate Synthetic Data if no file found
    if df is None:
        np.random.seed(42)
        n_rows = 5000
        # Generate 28 anonymized PCA features (V1-V28)
        data = np.random.randn(n_rows, 28) 
        df = pd.DataFrame(data, columns=[f'V{i}' for i in range(1, 29)])
        # Time and Amount
        df['Time'] = np.arange(n_rows)
        df['Amount'] = np.random.uniform(0, 500, n_rows)
        # Class (0 = Normal, 1 = Fraud) - Create imbalance (2% fraud)
        df['Class'] = np.random.choice([0, 1], size=n_rows, p=[0.98, 0.02])
    else:
        # Sampling for speed if dataset is huge (optional)
        if len(df) > 50000:
            df = df.sample(50000, random_state=42)
    
    return df, data_source

# Load Data (local default)
with st.spinner('Loading and analyzing data...'):
    df_local, source_local = load_data()

# Sidebar Controls
st.sidebar.header("⚙️ Model Parameters & Data")
# File uploader (this overrides local file if provided)
uploaded = st.sidebar.file_uploader("Upload CSV (optional) — it will override local file", type=['csv'])
if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        source = "Uploaded CSV"
    except Exception as e:
        st.sidebar.error(f"Unable to read uploaded file: {e}")
        df = df_local.copy()
        source = source_local
else:
    df = df_local.copy()
    source = source_local

split_size = st.sidebar.slider('Test Set Size', 0.1, 0.5, 0.2)
random_seed = int(st.sidebar.number_input('Random Seed', value=42, step=1))
st.sidebar.info(f"📁 Data Source: {source}")
st.sidebar.info(f"📊 Total Transactions: {len(df)}")

# Sampling options (applied ONLY on training data - Option B)
st.sidebar.subheader("Sampling (applied on TRAINING data only)")
sampling_method = st.sidebar.radio("Choose sampling method", 
                                   options=["None", "SMOTE", "Random Oversample", "Random Undersample"])
if not IMBLEARN_AVAILABLE and sampling_method != "None":
    st.sidebar.error("`imbalanced-learn` not installed. Run:\npython -m pip install imbalanced-learn")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (Quick View)
# ==========================================
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("1. Class Distribution")
    if 'Class' not in df.columns:
        st.warning("Dataset has no 'Class' column. Make sure your data includes a 'Class' column with 0 (normal) and 1 (fraud).")
        # Show first rows and return early
        st.dataframe(df.head(8), use_container_width=True)
        st.stop()
    counts = df['Class'].value_counts()
    fraud_count = counts.get(1, 0)
    normal_count = counts.get(0, 0)
    
    # Simple pie chart
    fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
    ax_pie.pie([normal_count, fraud_count], labels=['Normal', 'Fraud'], autopct='%1.1f%%', 
               colors=['#66b3ff', '#ff9999'])
    st.pyplot(fig_pie)
    st.caption(f"Normal: {normal_count} | Fraud: {fraud_count}")

with col2:
    st.subheader("2. Dataset Preview")
    st.dataframe(df.head(8), use_container_width=True)
    st.markdown(f"**Insight:** The dataset is likely unbalanced. Standard accuracy is not enough; we need Precision and Recall.")

# ==========================================
# 4. PREPROCESSING & TRAINING  (Option B : sampling on train only)
# ==========================================
st.divider()
st.subheader("3. Model Training & Evaluation")

# Keep a copy for processing
df_proc = df.copy()

# If Amount exists, we'll scale it — but to avoid leakage we'll fit scaler on X_train only (done later)
# Drop Time if present (optional)
if 'Time' in df_proc.columns:
    df_proc = df_proc.drop(['Time'], axis=1)

# Ensure 'Amount' and other numeric columns are retained. We'll create Amount_Scaled after train/test split.

X = df_proc.drop('Class', axis=1)
y = df_proc['Class']

# Train/Test Split (stratify to preserve imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size, random_state=random_seed, stratify=y)

# FEATURE SCALING (fit scaler on training set only)
sc = StandardScaler()
if 'Amount' in X_train.columns:
    X_train_amount = X_train['Amount'].values.reshape(-1, 1)
    X_test_amount = X_test['Amount'].values.reshape(-1, 1)
    X_train['Amount_Scaled'] = sc.fit_transform(X_train_amount)
    X_test['Amount_Scaled'] = sc.transform(X_test_amount)
    # Drop original Amount columns to match your prior behavior
    X_train = X_train.drop(['Amount'], axis=1)
    X_test = X_test.drop(['Amount'], axis=1)
else:
    # If no Amount column, create Amount_Scaled as not present — do nothing
    pass

# Now apply sampling ONLY on training data (Option B)
if sampling_method != "None":
    if not IMBLEARN_AVAILABLE:
        st.error("imbalanced-learn required for sampling. Install with:\npython -m pip install imbalanced-learn")
    else:
        try:
            if sampling_method == "SMOTE":
                sampler = SMOTE(random_state=random_seed)
            elif sampling_method == "Random Oversample":
                sampler = RandomOverSampler(random_state=random_seed)
            elif sampling_method == "Random Undersample":
                sampler = RandomUnderSampler(random_state=random_seed)
            else:
                sampler = None

            if sampler is not None:
                X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
                st.sidebar.success(f"Applied {sampling_method} on training data. New training size: {len(y_train_res)}")
                X_train, y_train = X_train_res, y_train_res
        except Exception as e:
            st.error(f"Sampling failed: {e}")

# Model Training
model = RandomForestClassifier(n_estimators=50, random_state=random_seed)
with st.spinner("Training model..."):
    model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================================
# 5. METRICS CALCULATION
# ==========================================
# Confusion Matrix components for Specificity
cm = confusion_matrix(y_test, y_pred)

# Handle case where confusion_matrix might be shape (1,1) if only one class present
if cm.size == 1:
    # All predictions/labels are same class
    if y_test.unique()[0] == 0:
        tn = cm[0, 0]
        fp = 0
        fn = 0
        tp = 0
    else:
        tn = 0
        fp = 0
        fn = 0
        tp = cm[0, 0]
else:
    # Ensure ravel works
    try:
        tn, fp, fn, tp = cm.ravel()
    except Exception:
        # fallback: pad to 2x2
        cm2 = np.zeros((2,2), dtype=int)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                cm2[i,j] = cm[i,j]
        tn, fp, fn, tp = cm2.ravel()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0) # Sensitivity
f1 = f1_score(y_test, y_pred, zero_division=0)
specificity = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

# Display Metrics using columns
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Accuracy", f"{accuracy:.2%}", help="Overall correctness")
m2.metric("Precision", f"{precision:.2%}", help="How many detected frauds were actually fraud?")
m3.metric("Recall", f"{recall:.2%}", help="How many actual frauds did we catch?")
m4.metric("F1-Score", f"{f1:.2%}", help="Balance between Precision and Recall")
m5.metric("Specificity", f"{specificity:.2%}", help="Ability to identify normal transactions")

# ==========================================
# 6. VISUALIZATION
# ==========================================
c_chart1, c_chart2 = st.columns(2)

with c_chart1:
    st.markdown("### 📉 Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cbar=False, 
                xticklabels=['Pred Normal', 'Pred Fraud'],
                yticklabels=['Actual Normal', 'Actual Fraud'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig_cm)
    st.info("Top-Left: Correctly cleared. Bottom-Right: Fraud caught.")

with c_chart2:
    st.markdown("### 🧠 Feature Importance")
    importances = model.feature_importances_
    # Get top 10 features
    n_features = min(10, len(importances))
    indices = np.argsort(importances)[::-1][:n_features]
    
    fig_feat, ax_feat = plt.subplots()
    plt.title("Top Features Driving Fraud Detection")
    plt.bar(range(n_features), importances[indices], align="center")
    plt.xticks(range(n_features), [X.columns[i] for i in indices], rotation=45)
    plt.xlim([-1, n_features])
    st.pyplot(fig_feat)

# ==========================================
# 7. SUMMARY
# ==========================================
st.success(f"""
**Analysis Complete:** The model successfully processed **{len(X_test)}** test transactions. 
It identified **{tp}** fraudulent transactions correctly out of **{(tp+fn)}** actual frauds.
""")

# Helpful final notes
st.markdown("#### Notes & Tips")
st.markdown("""
- Sampling **was applied only on the training set** (to avoid data leakage).  
""")
if not IMBLEARN_AVAILABLE:
    st.warning(f"imbalanced-learn import error: {IMBLEARN_IMPORT_ERROR}")
