import os
import joblib
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Telco Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for KPI cards and badges
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #2b5c8f;
        margin-bottom: 20px;
    }
    .high-risk {
        color: #d9534f;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .medium-risk {
        color: #f0ad4e;
        font-weight: bold;
        font-size: 1.4rem;
    }
    .low-risk {
        color: #5cb85c;
        font-weight: bold;
        font-size: 1.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Load production pipeline
@st.cache_resource
def load_pipeline():
    model_path = "models/churn_model.joblib"
    if not os.path.exists(model_path):
        model_path = os.path.join(os.path.dirname(__file__), "models", "churn_model.joblib")
    return joblib.load(model_path)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Error loading model pipeline: {e}")
    st.stop()

# Header
st.title("📊 Telco Customer Churn Prediction Engine")
st.markdown("""
Predict customer churn risk in real time using a production-grade machine learning model.
Adjust customer details below to estimate churn probability and view personalized retention recommendations.
""")

# Sidebar: Project Info & Real Performance Metrics
with st.sidebar:
    st.header("ℹ️ About This Project")
    st.markdown("""
    **Dataset**: IBM Telco Customer Churn (7,043 customers, 21 columns)  
    **Architecture**: Scikit-Learn `Pipeline` + `ColumnTransformer` (Zero Data Leakage)  
    **Best Model**: Tuned Random Forest (`class_weight='balanced'`)  
    **Selection Method**: 5-Fold Stratified Cross-Validation on Training Set  
    """)
    
    st.markdown("### 🎯 Real Test-Set Metrics")
    st.markdown("*(Evaluated once on 1,409 untouched test records)*")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("ROC-AUC", "84.25%")
        st.metric("Recall", "78.34%")
        st.metric("F1-Score", "63.21%")
    with col_b:
        st.metric("Accuracy", "75.80%")
        st.metric("Precision", "52.98%")
        st.metric("CV F1-Score", "63.60%")

    st.markdown("---")
    st.markdown("### ⚖️ Decision Threshold")
    st.info("**Selected Threshold: 0.50**  \n*Customers with churn probability ≥ 50% are flagged as high risk. A lower operational cutoff (e.g. 40%) can be used for aggressive retention.*")

# Main Form for Customer Input
st.subheader("📝 Customer Profile Inputs")

with st.form("churn_prediction_form"):
    tab1, tab2, tab3 = st.tabs(["📋 Account & Billing", "👤 Demographics", "🌐 Services & Subscriptions"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        with c1:
            tenure = st.slider("Customer Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months customer has been with company.")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=0)
        with c2:
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=125.0, value=65.0, step=0.5)
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], index=0)
        with c3:
            default_total = round(tenure * monthly_charges, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=9000.0, value=default_total, step=5.0)
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                index=0
            )

    with tab2:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            gender = st.selectbox("Gender", ["Female", "Male"], index=0)
        with d2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"], index=0)
        with d3:
            partner = st.selectbox("Has Partner", ["No", "Yes"], index=0)
        with d4:
            dependents = st.selectbox("Has Dependents", ["No", "Yes"], index=0)

    with tab3:
        s1, s2, s3 = st.columns(3)
        with s1:
            phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], index=0)
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=0)
        with s2:
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], index=0)
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], index=0)
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], index=0)
        with s3:
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], index=0)
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], index=0)
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], index=0)

    submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

# Prediction Execution
if submitted:
    # Assemble input dataframe matching training schema
    customer_dict = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': int(tenure),
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': float(monthly_charges),
        'TotalCharges': float(total_charges)
    }
    
    input_df = pd.DataFrame([customer_dict])
    
    # Predict probabilities via the saved scikit-learn pipeline
    churn_proba = pipeline.predict_proba(input_df)[0, 1]
    churn_pct = churn_proba * 100
    
    st.markdown("---")
    st.subheader("🎯 Prediction Results & Retention Strategy")
    
    col_res1, col_res2 = st.columns([1, 2])
    
    with col_res1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Churn Probability", f"{churn_pct:.1f}%")
        
        if churn_proba >= 0.50:
            st.markdown("Risk Level: <span class='high-risk'>HIGH RISK</span>", unsafe_allow_html=True)
            st.warning("⚠️ High propensity to churn. Immediate retention intervention recommended.")
        elif churn_proba >= 0.35:
            st.markdown("Risk Level: <span class='medium-risk'>MEDIUM / BORDERLINE</span>", unsafe_allow_html=True)
            st.info("ℹ️ Moderate risk. Monitor usage patterns and offer soft engagement incentives.")
        else:
            st.markdown("Risk Level: <span class='low-risk'>LOW RISK</span>", unsafe_allow_html=True)
            st.success("✅ Customer is stable. Standard relationship maintenance is sufficient.")
            
        st.progress(float(churn_proba))
        st.markdown("</div>", unsafe_allow_html=True)

    with col_res2:
        st.markdown("### 💡 Recommended Retention Actions")
        actions = []
        
        # Grounded heuristic recommendations derived from our model feature importances
        if contract == "Month-to-month":
            actions.append("• **Contract Upgrade Offer**: Customer is on a month-to-month plan (the #1 churn driver with a 42.7% historical churn rate). Offer a 15% discount on a 1-year agreement.")
            
        if tenure <= 6:
            actions.append("• **Early Onboarding Concierge**: Customer is in the vulnerable initial 6 months. Schedule a proactive check-in call from Customer Success.")
            
        if internet_service == "Fiber optic" and tech_support == "No":
            actions.append("• **Tech Support Bundle**: Customer has Fiber Optic internet without TechSupport. Provide 6 months of complimentary TechSupport to mitigate technical friction.")
            
        if payment_method == "Electronic check":
            actions.append("• **Autopay Discount Incentive**: Customer pays by Electronic Check (45.3% churn rate). Provide a $5 monthly bill credit for switching to automated Credit Card or Bank Transfer autopay.")
            
        if senior_citizen == "Yes" and monthly_charges > 70:
            actions.append("• **Senior Value Review**: High monthly spend for senior customer. Offer a tailored fixed-income bundle or review plan add-ons to eliminate unused services.")
            
        if not actions:
            actions.append("• Customer profile exhibits strong retention indicators (longer tenure, multi-year contract, automated payments). Continue standard satisfaction check-ins without unnecessary discounting.")
            
        for act in actions:
            st.markdown(act)
