import nbformat as nbf

def get_phase2_cells():
    cells = []
    
    md_p2 = """## 3. Phase 2: Data Cleaning

In this phase, we address structural data hygiene:
1. **Drop `customerID`**: An arbitrary alphanumeric identifier (e.g., `7590-VHVEG`) containing 7,043 unique values. It holds no generalizable predictive signal and would cause spurious memorization if retained.
2. **Handle `TotalCharges` Blank Strings & Imputation Decision**:
   - `TotalCharges` is stored as an `object` string type because 11 rows contain blank whitespace strings (`" "`).
   - Converting these using `pd.to_numeric(..., errors='coerce')` produces 11 `NaN` values.
   - Crucially, all 11 rows have `tenure == 0`. These are brand-new customers who enrolled in the current billing cycle and have not yet been billed.
   - **Business Justification**: Filling these 11 rows with `0.0` is logically, mathematically, and domain-accurate—they have accumulated exactly $0 in historical charges.
3. **Format `SeniorCitizen`**: Converted from `0/1` integers to descriptive string labels (`"No"` / `"Yes"`) to ensure clean categorical handling and readable EDA plots.
4. **Binary Target Encoding**: Map `Churn` to integer indicators (`"Yes"` = 1, `"No"` = 0).
"""
    cells.append(nbf.v4.new_markdown_cell(md_p2))
    
    code_p2 = """# Create a clean working copy
df_clean = df_raw.copy()

# 1. Drop customerID
df_clean = df_clean.drop(columns=['customerID'])

# 2. Investigate TotalCharges blank spaces
blank_tc_mask = df_clean['TotalCharges'].str.strip() == ''
print(f"Number of rows with blank TotalCharges: {blank_tc_mask.sum()}")
print("Tenure values for blank TotalCharges rows:")
print(df_clean.loc[blank_tc_mask, 'tenure'].value_counts())

# Convert TotalCharges to numeric
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Fill the 11 tenure=0 rows with 0.0 total charges
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
print(f"Missing values in TotalCharges after imputation: {df_clean['TotalCharges'].isnull().sum()}")

# 3. Format SeniorCitizen as categorical string for EDA consistency
df_clean['SeniorCitizen'] = df_clean['SeniorCitizen'].map({0: 'No', 1: 'Yes'})

# 4. Convert target Churn to binary integer (Yes=1, No=0)
df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})

print("\\nCleaned Dataset Shape:", df_clean.shape)
print("Target column unique values:", df_clean['Churn'].unique())
df_clean.head(3)
"""
    cells.append(nbf.v4.new_code_cell(code_p2))
    
    return cells
