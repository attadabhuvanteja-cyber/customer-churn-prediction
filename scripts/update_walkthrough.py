with open("C:/Users/dell/.gemini/antigravity/brain/c480fd21-a5b0-4561-856f-8890be0ce40b/walkthrough.md", "r", encoding="utf-8") as f:
    text = f.read()

phase9_section = """### Phase 9: Interactive Streamlit Web Application (`app.py`)
* **Interactive UI**: Built [`app.py`](file:///c:/Users/dell/OneDrive/Desktop/customer%20churn/app.py) featuring:
  - Clean tabbed inputs for Account & Billing, Demographics, and Services with exact dataset category dropdowns.
  - Churn probability prediction gauge and risk tier classification (**Low Risk < 35%**, **Medium/Borderline Risk 35%–50%**, **High Risk ≥ 50%**).
  - Dynamic heuristic retention action plan generated based on the customer's specific risk drivers.
  - "About This Project" sidebar showcasing real test-set performance metrics from the notebook.
  - Zero manual encoding: raw customer dictionary fed directly into `models/churn_model.joblib`.
* **Testing of 3 Realistic Customer Profiles**:
  1. **High-Risk Profile**: 90.9% Churn Probability -> **HIGH RISK** (Month-to-month, Fiber optic, Electronic check, tenure 2 mos).
  2. **Borderline Profile**: 44.3% Churn Probability -> **MEDIUM / BORDERLINE RISK** (Month-to-month, DSL, Mailed check, tenure 8 mos).
  3. **Low-Risk Profile**: 4.6% Churn Probability -> **LOW RISK** (Two-year contract, DSL, Autopay, tenure 60 mos).
* **Dependency Pinning**: Updated [`requirements.txt`](file:///c:/Users/dell/OneDrive/Desktop/customer%20churn/requirements.txt) with `streamlit>=1.30.0` and pinned `scikit-learn==1.9.1`, `pandas==3.0.6`, `numpy==2.5.3`.
* **Deployment Guide**: Added step-by-step instructions for deploying to Streamlit Community Cloud in `README.md`.
"""

if "### Phase 9: Interactive Streamlit Web Application" not in text:
    text = text.replace("## 🔍 Verification & Integrity Results", phase9_section + "\n---\n\n## 🔍 Verification & Integrity Results")
    with open("C:/Users/dell/.gemini/antigravity/brain/c480fd21-a5b0-4561-856f-8890be0ce40b/walkthrough.md", "w", encoding="utf-8") as f:
        f.write(text)
    print("walkthrough.md updated with Phase 9 details.")
