import nbformat as nbf

def get_phase4_cells():
    cells = []
    
    md_p4 = """## 5. Phase 4: Preprocessing & Data Leakage Prevention

### Preventing Data Leakage
Data leakage occurs when information from outside the training dataset is inadvertently used to create or tune the model.
* **Golden Rule**: We perform a **Stratified 80/20 train/test split FIRST**.
* **Encapsulated Preprocessing**: Preprocessing transformations (`StandardScaler` and `OneHotEncoder`) are encapsulated inside Scikit-learn `Pipeline` and `ColumnTransformer` objects. They are fitted **exclusively on the training folds** and only applied (transformed) on validation and test data.
* **Full Feature Representation**: Categorical features are encoded with `OneHotEncoder(handle_unknown='ignore')` without dropping the first category, ensuring that feature importance charts display all categorical levels explicitly.
* **Class Imbalance Strategy**: Rather than synthetic resampling (which can distort natural class probabilities), models utilize `class_weight='balanced'` where supported, penalizing errors on the minority churn class inversely proportional to class frequencies.

### Why Accuracy is Misleading on Imbalanced Data
In this dataset, 73.46% of customers do not churn. A naive "dummy" model that predicts "No Churn" for every customer would achieve **73.46% accuracy**, yet fail to detect **100% of actual churners**. Therefore, we evaluate models primarily on **Recall** (churn class capture rate), **F1-Score** (harmonic mean of precision and recall), and **ROC-AUC** (discrimination ability).
"""
    cells.append(nbf.v4.new_markdown_cell(md_p4))
    
    code_p4 = """# Separate feature matrix X and target vector y
X = df_clean.drop(columns=['Churn'])
y = df_clean['Churn']

# 1. Stratified Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Training feature set: {X_train.shape[0]:,} samples")
print(f"Test feature set:     {X_test.shape[0]:,} samples")
print(f"Training target churn rate: {y_train.mean():.4f} ({y_train.sum():,} churners)")
print(f"Test target churn rate:     {y_test.mean():.4f} ({y_test.sum():,} churners)")

# 2. Identify feature groups
num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
cat_features = [col for col in X_train.columns if col not in num_features]

print(f"\\nNumerical features ({len(num_features)}): {num_features}")
print(f"Categorical features ({len(cat_features)}): {cat_features}")

# 3. Build ColumnTransformer (StandardScaler for numerics, OneHotEncoder for categoricals)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
    ],
    remainder='drop'
)

print("\\nPreprocessor defined successfully.")
"""
    cells.append(nbf.v4.new_code_cell(code_p4))
    
    return cells
