import nbformat as nbf

def get_phase7_cells():
    cells = []
    
    md_p7 = """## 8. Phase 7: Model Interpretation & Business Strategy

In this phase, we:
1. Extract feature importances from our final model to identify the primary drivers of customer churn.
2. Formulate 4–5 actionable, revenue-protecting business recommendations.
3. Provide a production-ready single-customer inference function.
"""
    cells.append(nbf.v4.new_markdown_cell(md_p7))
    
    code_feat_imp = """# Extract feature names from preprocessor
preprocessor_fitted = best_rf_pipeline.named_steps['preprocessor']
cat_encoder = preprocessor_fitted.named_transformers_['cat']
cat_encoded_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + cat_encoded_feature_names

# Extract feature importances from the Random Forest classifier
rf_importances = best_rf_pipeline.named_steps['classifier'].feature_importances_

feat_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': rf_importances
}).sort_values(by='Importance', ascending=False)

# Plot top 15 features
fig, ax = plt.subplots(figsize=(10, 7))
top_15_features = feat_imp_df.head(15).sort_values(by='Importance', ascending=True)

ax.barh(top_15_features['Feature'], top_15_features['Importance'], color='#2b5c8f', edgecolor='black', alpha=0.85)
ax.set_title("Top 15 Feature Importances Driving Churn (Final Tuned Model)", pad=15)
ax.set_xlabel("Relative Importance (Gini Impurity Reduction)")
ax.set_ylabel("Feature")

for p in ax.patches:
    width = p.get_width()
    ax.annotate(f"{width:.4f}",
                xy=(width, p.get_y() + p.get_height() / 2),
                xytext=(5, 0), textcoords="offset points",
                ha='left', va='center', fontsize=9, fontweight='semibold')

plt.tight_layout()
fig_path = "reports/figures/feature_importance.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_feat_imp))
    
    md_recs = """### Actionable Business Recommendations

Based on empirical model drivers and bivariate EDA insights:

1. **Incentivize Transition Away from Month-to-Month Contracts**:
   - *Finding*: `Contract_Month-to-month` is the #1 categorical churn driver, with a 42.7% churn rate versus 2.8% for Two-year contracts.
   - *Action*: Offer high-risk month-to-month customers a discounted first 3 months or complimentary premium add-ons if they upgrade to an annual contract.
2. **Launch a Dedicated 'First-90-Days' Onboarding Protocol**:
   - *Finding*: `tenure` is the leading numerical predictor; churn is heavily concentrated in months 1–5.
   - *Action*: Trigger automated check-ins, onboarding guides, and satisfaction surveys at Day 14, 30, and 60. Address early setup friction before dissatisfaction turns into cancellation.
3. **Investigate Fiber Optic Service Satisfaction & Price Sensitivity**:
   - *Finding*: Fiber Optic subscribers exhibit a 41.9% churn rate, and `MonthlyCharges` is a top-3 model feature.
   - *Action*: Audit fiber optic connection stability and pricing tiers. Bundle complimentary TechSupport or offer tiered bandwidth plans to reduce price resistance.
4. **Promote Automated Payment Methods over Electronic Checks**:
   - *Finding*: Electronic check users have a 45.3% churn rate (compared to ~15% for Credit Card / Bank Transfer).
   - *Action*: Provide a $5 monthly statement credit or one-time promotional discount for enrolling in recurring automated Credit Card or Bank Transfer autopay.
5. **Position TechSupport and Security as Essential Value Bundles**:
   - *Finding*: Customers without TechSupport churn at 41.6% vs. 15.2% for those with it.
   - *Action*: Offer complimentary 6-month TechSupport trials to high-risk customers, converting an operational cost into a high-leverage retention anchor.
"""
    cells.append(nbf.v4.new_markdown_cell(md_recs))
    
    code_inference = """# Standalone Single Customer Churn Prediction Function
def predict_customer_churn(customer_profile: dict, pipeline, threshold: float = 0.50):
    \"\"\"
    Takes a raw customer profile dictionary, executes full pipeline preprocessing,
    and returns predicted churn probability, classification, and risk tier.
    \"\"\"
    input_df = pd.DataFrame([customer_profile])
    churn_prob = pipeline.predict_proba(input_df)[0, 1]
    is_churn = int(churn_prob >= threshold)
    
    if churn_prob >= 0.70:
        risk_tier = "Critical Risk"
    elif churn_prob >= 0.40:
        risk_tier = "Moderate Risk"
    else:
        risk_tier = "Low Risk"
        
    return {
        'churn_probability': round(float(churn_prob), 4),
        'churn_prediction': 'Churn' if is_churn == 1 else 'Retained',
        'risk_tier': risk_tier
    }

# Test Example 1: High-Risk Customer Profile
sample_high_risk = {
    'gender': 'Female',
    'SeniorCitizen': 'Yes',
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 2,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'No',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 95.50,
    'TotalCharges': 191.00
}

# Test Example 2: Low-Risk Customer Profile
sample_low_risk = {
    'gender': 'Male',
    'SeniorCitizen': 'No',
    'Partner': 'Yes',
    'Dependents': 'Yes',
    'tenure': 60,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'Yes',
    'TechSupport': 'Yes',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Two year',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Credit card (automatic)',
    'MonthlyCharges': 65.00,
    'TotalCharges': 3900.00
}

result_high = predict_customer_churn(sample_high_risk, loaded_pipeline)
result_low = predict_customer_churn(sample_low_risk, loaded_pipeline)

print("Inference Test 1 (High Risk Profile):")
print(result_high)

print("\\nInference Test 2 (Low Risk Profile):")
print(result_low)
"""
    cells.append(nbf.v4.new_code_cell(code_inference))
    
    return cells
