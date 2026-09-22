import nbformat as nbf

def get_phase6_cells():
    cells = []
    
    md_p6 = """## 7. Phase 6: Hyperparameter Tuning, Threshold Selection & Final Evaluation

In this phase:
1. We run `GridSearchCV` on the training set using 5-fold Stratified CV optimizing for **F1-score**.
2. We evaluate decision probability thresholds using out-of-fold `cross_val_predict` on the training set.
3. We lock the winning pipeline and decision threshold, and **only then** evaluate performance on the untouched test set.
4. We save the final pipeline to `models/churn_model.joblib`.
"""
    cells.append(nbf.v4.new_markdown_cell(md_p6))
    
    code_grid = """# Hyperparameter tuning for Random Forest using GridSearchCV on X_train
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [6, 8, 10],
    'classifier__min_samples_split': [5, 10],
    'classifier__min_samples_leaf': [2, 4]
}

rf_base_pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1))
])

grid_search_rf = GridSearchCV(
    estimator=rf_base_pipe,
    param_grid=param_grid_rf,
    cv=cv,
    scoring='f1',
    n_jobs=-1,
    verbose=0
)

print("Running GridSearchCV for Random Forest on training set...")
grid_search_rf.fit(X_train, y_train)

best_rf_pipeline = grid_search_rf.best_estimator_
print(f"Best CV F1-Score: {grid_search_rf.best_score_:.4f}")
print("Best Hyperparameters:", grid_search_rf.best_params_)
"""
    cells.append(nbf.v4.new_code_cell(code_grid))
    
    code_threshold = """# Cross-Validated Probability Threshold Analysis on Training Set
y_train_probs = cross_val_predict(
    best_rf_pipeline,
    X_train,
    y_train,
    cv=cv,
    method='predict_proba',
    n_jobs=-1
)[:, 1]

thresholds = np.linspace(0.1, 0.9, 81)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    preds = (y_train_probs >= t).astype(int)
    precisions.append(precision_score(y_train, preds, zero_division=0))
    recalls.append(recall_score(y_train, preds))
    f1s.append(f1_score(y_train, preds))

best_idx = np.argmax(f1s)
optimal_threshold = thresholds[best_idx]
print(f"Cross-Validated F1-Optimal Probability Threshold: {optimal_threshold:.2f}")
print(f"At threshold {optimal_threshold:.2f} -> CV Precision: {precisions[best_idx]:.3f}, CV Recall: {recalls[best_idx]:.3f}, CV F1: {f1s[best_idx]:.3f}")

# Plot precision-recall-f1 tradeoff curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(thresholds, precisions, label='Precision', color='#2b5c8f', linewidth=2)
ax.plot(thresholds, recalls, label='Recall', color='#d95f02', linewidth=2)
ax.plot(thresholds, f1s, label='F1-Score', color='#2ca02c', linewidth=2.5, linestyle='--')

ax.axvline(optimal_threshold, color='purple', linestyle=':', label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
ax.axvline(0.50, color='gray', linestyle='-.', alpha=0.7, label='Default Threshold (0.50)')

ax.set_title("Decision Threshold Trade-Off (Cross-Validated on Training Set)", pad=15)
ax.set_xlabel("Churn Probability Threshold")
ax.set_ylabel("Score")
ax.set_xlim(0.1, 0.9)
ax.set_ylim(0, 1.0)
ax.legend(loc='lower center', ncol=2)

plt.tight_layout()
fig_path = "reports/figures/threshold_tradeoff.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_threshold))
    
    code_test_eval = """# Final Evaluation on Untouched Test Set (1,409 customers)
# Models were trained on X_train and selected strictly using CV.

test_eval_models = models.copy()
test_eval_models['Tuned Random Forest (Final)'] = best_rf_pipeline

test_results = []

for name, pipe in test_eval_models.items():
    pipe.fit(X_train, y_train)
    y_test_pred = pipe.predict(X_test)
    y_test_prob = pipe.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred)
    rec = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)
    auc = roc_auc_score(y_test, y_test_prob)
    
    test_results.append({
        'Model': name,
        'Accuracy': round(acc, 4),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'F1-Score': round(f1, 4),
        'ROC-AUC': round(auc, 4)
    })

test_results_df = pd.DataFrame(test_results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)
print("Untouched Test Set Performance (Final Evaluation):")
display(test_results_df)
"""
    cells.append(nbf.v4.new_code_cell(code_test_eval))
    
    code_plots = """# Visual 1: Confusion Matrices for All Models on Test Set
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
axes = axes.flatten()

model_names = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Tuned Random Forest (Final)']

for idx, name in enumerate(model_names):
    pipe = test_eval_models[name]
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Retained (0)', 'Churned (1)'])
    disp.plot(ax=axes[idx], cmap='Blues', colorbar=False)
    axes[idx].set_title(f"{name}\\nRecall: {recall_score(y_test, y_pred):.3f} | F1: {f1_score(y_test, y_pred):.3f}", fontweight='bold')
    axes[idx].grid(False)

plt.suptitle("Confusion Matrix Comparison on Untouched Test Set", fontsize=16, y=0.98)
plt.tight_layout()
fig_path = "reports/figures/confusion_matrices_comparison.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()

# Visual 2: Combined ROC Curves for All Models on Test Set
fig, ax = plt.subplots(figsize=(9, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for idx, (name, pipe) in enumerate(test_eval_models.items()):
    y_prob = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.3f})", color=colors[idx % len(colors)], linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Chance (AUC = 0.500)')
ax.set_title("ROC Curves Comparison on Untouched Test Set", pad=15)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Recall)")
ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
fig_path = "reports/figures/roc_curves_comparison.png"
plt.savefig(fig_path, dpi=300)
print(f"Saved: {fig_path}")
plt.show()
"""
    cells.append(nbf.v4.new_code_cell(code_plots))
    
    code_save = """# Serialize and persist the final winning model pipeline
model_save_path = "models/churn_model.joblib"
joblib.dump(best_rf_pipeline, model_save_path)
print(f"Final production pipeline successfully saved to: {model_save_path}")

# Sanity check: Reload model from disk and confirm prediction equivalence
loaded_pipeline = joblib.load(model_save_path)
loaded_preds = loaded_pipeline.predict(X_test.iloc[:5])
original_preds = best_rf_pipeline.predict(X_test.iloc[:5])
assert np.array_equal(loaded_preds, original_preds), "Reloaded pipeline predictions do not match!"
print("Verification passed: Saved pipeline reloaded and tested successfully.")
"""
    cells.append(nbf.v4.new_code_cell(code_save))
    
    return cells
