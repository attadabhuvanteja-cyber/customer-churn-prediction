import nbformat as nbf

def get_phase1_cells():
    cells = []
    
    # Intro
    md_intro = """# Telco Customer Churn Prediction
### An End-to-End Machine Learning Portfolio Project

---

## 1. Project Overview & Business Problem

### What is Customer Churn?
**Customer churn** (attrition) is the percentage of subscribers who discontinue their service agreements within a given timeframe. In the telecommunications and subscription economy, managing churn is the single most vital operational determinant of long-term profitability.

### Why Predicting Churn Matters
1. **Asymmetric Customer Economics**: Acquiring a new customer in telecommunications is **5 to 7 times more expensive** than retaining an existing customer. High churn erodes customer lifetime value (LTV) and places an immense burden on customer acquisition marketing.
2. **Proactive Intervention**: By accurately identifying churn indicators before a customer calls to cancel, retention teams can execute surgical, low-cost interventions (e.g., proactive tech support outreach, annual contract incentives, equipment upgrades).
3. **Targeted Retention Spend**: Blanket discounting diminishes profit margins. Machine learning models allow companies to reserve high-value retention budgets only for customers with high churn risk and high potential lifetime value.

### The Asymmetric Cost of Errors:
* **False Negative (Type II Error - Severe)**: The model predicts a customer will **stay**, but they actually **churn**. Result: The customer is lost forever, along with all future recurring revenue and customer lifetime value.
* **False Positive (Type I Error - Moderate)**: The model predicts a customer will **churn**, but they intended to **stay**. Result: The company expends a modest retention offer or outreach campaign on a retained user.

Because **False Negatives are far more damaging than False Positives**, optimizing for **Recall** (capturing as many churners as possible) and **F1-Score / ROC-AUC** is paramount, making simple accuracy misleading on this imbalanced dataset.
"""
    cells.append(nbf.v4.new_markdown_cell(md_intro))
    
    # Imports
    code_imports = """# 1. Environment configuration and library imports
import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn preprocessing & model selection
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    cross_val_predict,
    GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    precision_recall_curve,
    classification_report
)

# Visual styling
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# Global random state for exact reproducibility
RANDOM_STATE = 42

# Ensure reports directory exists
os.makedirs("reports/figures", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Path resolution: works locally from project root, notebooks directory, and in Google Colab
if os.path.exists("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
elif os.path.exists("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    DATA_PATH = "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
else:
    DATA_PATH = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

print(f"Data source configured: {DATA_PATH}")
"""
    cells.append(nbf.v4.new_code_cell(code_imports))
    
    # Phase 1 header
    md_p1 = """## 2. Phase 1: Setup & Data Understanding

In this phase, we load the raw CSV dataset, inspect its shape, examine column types, verify duplicate records, and analyze the distribution of the target variable (`Churn`).
"""
    cells.append(nbf.v4.new_markdown_cell(md_p1))
    
    # Load data
    code_load = """# Load dataset and preview structural dimensions
df_raw = pd.read_csv(DATA_PATH)

print(f"Dataset Dimensions: {df_raw.shape[0]:,} rows and {df_raw.shape[1]} columns")
print("=" * 70)
df_raw.head()
"""
    cells.append(nbf.v4.new_code_cell(code_load))
    
    # Data inspection
    code_info = """# Inspect column types and verify duplicates
print("--- Column Summary & Data Types ---")
df_raw.info()

print("\\n--- Duplicate Rows ---")
num_duplicates = df_raw.duplicated().sum()
print(f"Total duplicate rows: {num_duplicates}")

print("\\n--- Explicit Missing Values (NaN) ---")
print(df_raw.isnull().sum())
"""
    cells.append(nbf.v4.new_code_cell(code_info))
    
    # Target analysis
    md_target = """### Target Variable Analysis (`Churn`)

Examining class prevalence confirms whether the dataset is balanced or imbalanced.
"""
    cells.append(nbf.v4.new_markdown_cell(md_target))
    
    code_target = """# Class distribution of Churn
churn_counts = df_raw['Churn'].value_counts()
churn_rates = df_raw['Churn'].value_counts(normalize=True) * 100

churn_summary = pd.DataFrame({
    'Customer Count': churn_counts,
    'Percentage (%)': churn_rates.round(2)
})

print("Churn Class Balance:")
display(churn_summary)
"""
    cells.append(nbf.v4.new_code_cell(code_target))
    
    return cells
