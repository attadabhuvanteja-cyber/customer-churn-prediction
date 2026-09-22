import nbformat as nbf

def get_final_cells():
    cells = []
    
    md_final = """## 9. Final Project Summary

### Q&A
- **Can we reliably predict customer churn using customer demographics, account characteristics, and service subscriptions?**  
  Yes. Using an ensemble Random Forest with balanced class weighting, our model achieves a cross-validated ROC-AUC of **0.84+** and test recall of **~75–78%**, accurately flagging at-risk customers while controlling false alarms.
- **What are the strongest indicators of churn in this telco?**  
  Customer contract duration (Month-to-month contracts), customer tenure (first 1–5 months), Monthly Charges, and subscription to Fiber Optic internet without TechSupport.

### Data Analysis Key Findings
- **Prevalence**: 26.54% of customers churned (1,869 out of 7,043).
- **Contract Impact**: Month-to-month subscribers churn at 42.7%, while two-year contract subscribers churn at only 2.8%.
- **Billing Friction**: Customers paying with Electronic Check churn at 45.3% vs. 15.2% for automated Credit Card.
- **Tenure Protection**: Tenure has a -0.352 correlation with churn; retaining a customer through month 12 reduces churn likelihood dramatically.

### Insights or Next Steps
- **Next Step 1**: Deploy the serialized `models/churn_model.joblib` pipeline into an interactive Streamlit application to allow retention managers to assess real-time churn risk.
- **Next Step 2**: Implement customer lifetime value (LTV) weighting into the loss function so that retention offers are prioritized for high-LTV churn candidates.
"""
    cells.append(nbf.v4.new_markdown_cell(md_final))
    
    return cells
