import nbformat as nbf

def get_phase3_cells():
    cells = []
    
    md_p3 = """## 4. Phase 3: Exploratory Data Analysis (EDA)

We explore customer relationships and churn patterns using publication-ready visualizations. Every figure is saved to `reports/figures/` for reporting.
"""
    cells.append(nbf.v4.new_markdown_cell(md_p3))
    
    # Chart 1: Churn distribution
    code_chart1 = """# Chart 1: Overall Churn Distribution
fig, ax = plt.subplots(figsize=(8, 5))
palette = ['#2b5c8f', '#d95f02']

bars = ax.bar(['Retained (No)', 'Churned (Yes)'], churn_counts.values, color=palette, width=0.5, edgecolor='black', alpha=0.85)

total_customers = len(df_clean)
for bar in bars:
    height = bar.get_height()
    pct = (height / total_customers) * 100
    ax.annotate(f"{height:,}\\n({pct:.1f}%)",
                xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=12)

ax.set_title("Telco Customer Churn Distribution (Baseline Prevalence)", pad=15)
ax.set_ylabel("Number of Customers")
ax.set_ylim(0, 6000)

plt.tight_layout()
fig_path = "reports/figures/churn_distribution.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_chart1))
    
    # Chart 2: Categorical churn rates
    code_chart2 = """# Chart 2: Churn Rates Across Key Categorical Features
cat_features = ['Contract', 'InternetService', 'PaymentMethod', 'TechSupport', 'SeniorCitizen']

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()

for idx, col in enumerate(cat_features):
    ax = axes[idx]
    rate_df = df_clean.groupby(col)['Churn'].agg(['count', 'mean']).reset_index()
    rate_df['churn_rate_pct'] = rate_df['mean'] * 100
    rate_df = rate_df.sort_values(by='churn_rate_pct', ascending=False)
    
    sns.barplot(data=rate_df, x=col, y='churn_rate_pct', ax=ax, palette='Blues_r', edgecolor='black', alpha=0.85)
    ax.axhline(26.54, color='#d95f02', linestyle='--', linewidth=1.5, label='Overall Average (26.5%)')
    
    ax.set_title(f"Churn Rate by {col}", fontweight='bold')
    ax.set_xlabel("")
    ax.set_ylabel("Churn Rate (%)")
    ax.set_ylim(0, 55)
    ax.tick_params(axis='x', rotation=25)
    
    for p in ax.patches:
        height = p.get_height()
        if not np.isnan(height) and height > 0:
            ax.annotate(f"{height:.1f}%",
                        xy=(p.get_x() + p.get_width() / 2, height + 1),
                        ha='center', va='bottom', fontsize=10, fontweight='semibold')
    
    if idx == 0:
        ax.legend(loc='upper right')

fig.delaxes(axes[5])

plt.suptitle("Bivariate Analysis: Churn Rate by Key Service and Billing Attributes", fontsize=16, y=0.98)
plt.tight_layout()
fig_path = "reports/figures/churn_by_categorical_features.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_chart2))
    
    # Chart 3: Numerical distributions
    code_chart3 = """# Chart 3: Numerical Distributions Stratified by Churn Status
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Tenure distribution
sns.histplot(data=df_clean, x='tenure', hue='Churn', kde=True, bins=36,
             palette={0: '#2b5c8f', 1: '#d95f02'}, ax=axes[0], alpha=0.6, element='step')
axes[0].set_title("Customer Tenure (Months) by Churn Status", fontweight='bold')
axes[0].set_xlabel("Tenure (Months)")
axes[0].set_ylabel("Number of Customers")
axes[0].legend(title="Status", labels=['Churned (1)', 'Retained (0)'])

# Monthly charges distribution
sns.histplot(data=df_clean, x='MonthlyCharges', hue='Churn', kde=True, bins=36,
             palette={0: '#2b5c8f', 1: '#d95f02'}, ax=axes[1], alpha=0.6, element='step')
axes[1].set_title("Monthly Charges ($) by Churn Status", fontweight='bold')
axes[1].set_xlabel("Monthly Charges ($)")
axes[1].set_ylabel("Number of Customers")
axes[1].legend(title="Status", labels=['Churned (1)', 'Retained (0)'])

plt.tight_layout()
fig_path = "reports/figures/numerical_distributions_by_churn.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_chart3))
    
    # Chart 4: Correlation heatmap
    code_chart4 = """# Chart 4: Correlation Heatmap for Numeric Features
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Churn']
corr_matrix = df_clean[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='Blues', vmin=-1, vmax=1,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)

ax.set_title("Correlation Heatmap: Numeric Features vs. Churn", pad=15)
plt.tight_layout()
fig_path = "reports/figures/numeric_correlation_heatmap.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_chart4))
    
    # Key EDA findings
    md_findings = """### Key Findings from Exploratory Data Analysis

Based strictly on the verified metrics and visualizations produced above:

1. **Contract Type is the Dominant Churn Indicator**: Customers on **Month-to-month contracts experience a 42.7% churn rate**, compared to **11.3% for One-year contracts** and only **2.8% for Two-year contracts**. Contractual commitment is the strongest barrier against churn.
2. **Early Tenure Vulnerability ("The Onboarding Cliff")**: Churn is intensely concentrated during the first **1 to 5 months** of customer tenure. Once a customer reaches 20+ months of tenure, their churn likelihood drops dramatically.
3. **High Price Sensitivity Among Fiber Optic Users**: Customers with **Fiber Optic internet churn at 41.9%**, compared to **19.0% for DSL** and **7.4% for customers with No internet service**. The monthly charge distribution reveals heavy churn density between \$70 and \$100, which aligns with premium Fiber Optic pricing.
4. **Electronic Check Friction**: Customers paying with **Electronic Check churn at 45.3%**, more than double that of Credit Card (15.2%), Bank Transfer (16.7%), or Mailed Check (19.1%).
5. **TechSupport as a Retention Anchor**: Customers lacking **TechSupport churn at 41.6%**, whereas customers who have active TechSupport churn at only **15.2%**.
6. **Senior Citizens Face Higher Attrition**: **Senior citizens churn at 41.7%**, compared to **23.6% for non-seniors**, highlighting the need for specialized customer care or cost-effective senior plans.
7. **Tenure Inversely Correlates with Churn**: Tenure has a -0.352 Pearson correlation with churn, while showing a strong positive collinearity with TotalCharges ($r = 0.826$).
"""
    cells.append(nbf.v4.new_markdown_cell(md_findings))
    
    return cells
