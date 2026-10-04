import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, log_loss,
    balanced_accuracy_score, confusion_matrix, classification_report
)

def print_section_header(title):
    """Prints a standardized, aligned section header."""
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)

def section1_dataset_summary(df, feature_cols):
    """Section 1: Dataset Summary & Statistical Description."""
    print_section_header("Section 1 - Dataset Summary")

    total_rows = len(df)
    num_features = len(feature_cols)
    pass_count = (df['result'] == 1).sum()
    fail_count = (df['result'] == 0).sum()
    pass_pct = (pass_count / total_rows) * 100
    fail_pct = (fail_count / total_rows) * 100

    train_size = int(total_rows * 0.8)
    test_size = total_rows - train_size

    print(f" Total Rows        : {total_rows}")
    print(f" Number of Features: {num_features}")
    print(f" Train / Test Split: {train_size} training samples (80.00%) | {test_size} test samples (20.00%)")
    print(f" Class Outcome     : Pass = {pass_count} ({pass_pct:.2f}%) | Fail = {fail_count} ({fail_pct:.2f}%)")

    print("\n Feature Descriptive Statistics:")
    stats_df = df[feature_cols].describe().T[['mean', 'std', 'min', '25%', '50%', '75%', 'max']]
    # Format each metric in the describe table to 4 decimal places
    formatted_stats = stats_df.map(lambda v: f"{v:10.4f}")
    print(formatted_stats.to_string())

def section2_model_info(model):
    """Section 2: Active Model Architecture & Tuned Hyperparameters."""
    print_section_header("Section 2 - Model Info")

    model_name = model.__class__.__name__
    print(f" Model Classifier Type: {model_name}")

    meta_path = os.path.join('models', 'meta.json')
    best_params_path = os.path.join('outputs', 'best_params.json')

    hyperparams = {}
    technique = "Tuned"

    if os.path.exists(best_params_path):
        with open(best_params_path, 'r') as f:
            bp = json.load(f)
            hyperparams = bp.get('best_params', {})
            technique = bp.get('technique', technique)
    elif os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            hyperparams = meta.get('best_params', {})
            technique = meta.get('technique', technique)

    if not hyperparams:
        hyperparams = model.get_params()

    print(f" Optimization Strategy: {technique}")
    print(" Active Hyperparameters:")
    if isinstance(hyperparams, dict):
        for param, val in hyperparams.items():
            print(f"   - {param:<22}: {val}")
    else:
        print(f"   {hyperparams}")

def section3_test_metrics(model, X_test, y_test):
    """Section 3: Comprehensive Test Set Evaluation Metrics."""
    print_section_header("Section 3 - Test Set Metrics")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    roc_auc = roc_auc_score(y_test, y_prob)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    bal_acc = balanced_accuracy_score(y_test, y_pred)

    metrics_table = [
        ("Accuracy", acc, f"{acc * 100:.2f}%"),
        ("Precision", prec, f"{prec * 100:.2f}%"),
        ("Recall (Sensitivity)", rec, f"{rec * 100:.2f}%"),
        ("Specificity", specificity, f"{specificity * 100:.2f}%"),
        ("F1 Score", f1, f"{f1 * 100:.2f}%"),
        ("Balanced Accuracy", bal_acc, f"{bal_acc * 100:.2f}%"),
        ("ROC-AUC Score", roc_auc, f"{roc_auc:.4f}"),
        ("Matthews Correlation (MCC)", mcc, f"{mcc:.4f}"),
        ("Cohen's Kappa Score", kappa, f"{kappa:.4f}"),
        ("Log Loss", logloss, f"{logloss:.4f}")
    ]

    print(f" {'Metric Name':<30} | {'Numeric Value':<15} | {'Percentage / Format'}")
    print(" " + "-" * 65)
    for name, val, fmt in metrics_table:
        print(f" {name:<30} | {val:15.4f} | {fmt}")

    return y_pred, y_prob, cm

def section4_confusion_matrix(cm):
    """Section 4: Confusion Matrix & Error Rates."""
    print_section_header("Section 4 - Confusion Matrix")

    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(" Confusion Matrix Alignment:")
    print("                     Predicted FAIL (0)   Predicted PASS (1)")
    print(f"  Actual FAIL (0) :      TN = {tn:<8}       FP = {fp:<8}")
    print(f"  Actual PASS (1) :      FN = {fn:<8}       TP = {tp:<8}")
    print("\n Detailed Breakdown:")
    print(f"  - True Positives  (TP) : {tp}")
    print(f"  - True Negatives  (TN) : {tn}")
    print(f"  - False Positives (FP) : {fp}")
    print(f"  - False Negatives (FN) : {fn}")
    print(f"  - False Positive Rate (FPR): {fpr:.4f} ({fpr * 100:.2f}%)")
    print(f"  - False Negative Rate (FNR): {fnr:.4f} ({fnr * 100:.2f}%)")

def section5_classification_report(y_test, y_pred):
    """Section 5: Scikit-learn Classification Report."""
    print_section_header("Section 5 - Classification Report")

    target_names = ['FAIL (0)', 'PASS (1)']
    report_str = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    print(report_str)

def section6_train_vs_test(model, X_train, y_train, X_test, y_test):
    """Section 6: Train vs. Test Performance & Overfitting Analysis."""
    print_section_header("Section 6 - Train vs Test Comparison")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    acc_gap = train_acc - test_acc
    f1_gap = train_f1 - test_f1

    print(f" {'Dataset Subset':<18} | {'Accuracy':<12} | {'F1 Score':<12}")
    print(" " + "-" * 50)
    print(f" {'Training Set':<18} | {train_acc:12.4f} | {train_f1:12.4f}")
    print(f" {'Test Set':<18} | {test_acc:12.4f} | {test_f1:12.4f}")
    print(" " + "-" * 50)
    print(f" {'Gap (Train - Test)':<18} | {acc_gap:12.4f} | {f1_gap:12.4f}")

    # Determine Verdict
    if abs(acc_gap) <= 0.05:
        verdict = f"GOOD FIT: Train-Test accuracy gap is {acc_gap * 100:.2f}% (<= 5.00%), demonstrating excellent generalization."
    elif acc_gap > 0.05:
        verdict = f"OVERFITTING: Train accuracy is {acc_gap * 100:.2f}% higher than test accuracy."
    else:
        verdict = f"UNDERFITTING: Test accuracy is {abs(acc_gap) * 100:.2f}% higher than train accuracy."

    print(f"\n Fit Verdict: {verdict}")

def section7_cross_validation(model, X, y):
    """Section 7: 5-Fold Cross-Validation Evaluation."""
    print_section_header("Section 7 - Cross-Validation (5-Fold)")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_acc = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(model, X, y, cv=skf, scoring='f1')

    print(f" {'Fold Number':<15} | {'Accuracy':<12} | {'F1 Score':<12}")
    print(" " + "-" * 45)
    for idx, (acc, f1) in enumerate(zip(cv_acc, cv_f1), 1):
        print(f" Fold {idx:<10} | {acc:12.4f} | {f1:12.4f}")
    print(" " + "-" * 45)
    print(f" {'Mean':<15} | {np.mean(cv_acc):12.4f} | {np.mean(cv_f1):12.4f}")
    print(f" {'Std Dev':<15} | {np.std(cv_acc):12.4f} | {np.std(cv_f1):12.4f}")

def section8_feature_importance(model, feature_cols):
    """Section 8: Feature Importance Breakdown."""
    print_section_header("Section 8 - Feature Importance")

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feat_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': importances,
            'Percentage': importances * 100
        }).sort_values(by='Importance', ascending=False)

        print(f" {'Rank':<6} | {'Feature Name':<35} | {'Importance':<12} | {'Percentage'}")
        print(" " + "-" * 70)
        for rank, (_, row) in enumerate(feat_df.iterrows(), 1):
            print(f" {rank:<6} | {row['Feature']:<35} | {row['Importance']:12.4f} | {row['Percentage']:6.2f}%")
    else:
        print(" Active model does not expose feature_importances_ attribute.")

def section9_tuning_results():
    """Section 9: Hyperparameter Tuning Comparison Table & Baseline vs Tuned Gain."""
    print_section_header("Section 9 - Hyperparameter Tuning Results")

    results_path = os.path.join('outputs', 'tuning_results.csv')
    if os.path.exists(results_path):
        df_tuning = pd.read_csv(results_path)

        # Print formatted table
        print(f" {'Model':<15} | {'Technique':<20} | {'Test Accuracy':<14} | {'Test F1':<12} | {'Search Time'}")
        print(" " + "-" * 75)
        for _, row in df_tuning.iterrows():
            time_str = f"{row['Search_Time_Sec']:.2f}s"
            print(f" {row['Model']:<15} | {row['Technique']:<20} | {row['Test_Accuracy']:14.4f} | {row['Test_F1']:12.4f} | {time_str}")

        # Compute gains
        # Find winning model and baseline of that model family (or Decision Tree baseline)
        winning_row = df_tuning.sort_values(by=['Test_Accuracy', 'Best_CV_Score'], ascending=[False, False]).iloc[0]
        winning_model = winning_row['Model']

        baseline_rows = df_tuning[(df_tuning['Model'] == winning_model) & (df_tuning['Technique'].str.contains('Baseline'))]
        if not baseline_rows.empty:
            base_row = baseline_rows.iloc[0]
        else:
            base_row = df_tuning.iloc[0]

        base_acc = base_row['Test_Accuracy']
        tuned_acc = winning_row['Test_Accuracy']
        abs_acc_gain = tuned_acc - base_acc
        pct_acc_gain = (abs_acc_gain / base_acc) * 100 if base_acc > 0 else 0.0

        base_f1 = base_row['Test_F1']
        tuned_f1 = winning_row['Test_F1']
        abs_f1_gain = tuned_f1 - base_f1
        pct_f1_gain = (abs_f1_gain / base_f1) * 100 if base_f1 > 0 else 0.0

        print("\n Performance Improvement (Baseline vs Best Tuned Model):")
        print(f"  - Winner Model : {winning_model} via {winning_row['Technique']}")
        print(f"  - Accuracy Gain: {abs_acc_gain:+.4f} points ({pct_acc_gain:+.2f}% relative change)")
        print(f"  - F1 Score Gain: {abs_f1_gain:+.4f} points ({pct_f1_gain:+.2f}% relative change)")
    else:
        print(" File 'outputs/tuning_results.csv' not found.")

def main():
    data_path = os.path.join('data', 'students.csv')
    model_path = os.path.join('models', 'model.pkl')

    # 1. Validation check
    missing = []
    if not os.path.exists(data_path):
        missing.append(("data/students.csv", "uv run generate_data.py"))
    if not os.path.exists(model_path):
        missing.append(("models/model.pkl", "uv run tune_model.py"))

    if missing:
        print("[ERROR] Required project files are missing:")
        for file_name, cmd in missing:
            print(f"  - '{file_name}' is missing. Please run: `{cmd}`")
        sys.exit(1)

    # 2. Load dataset and model
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    feature_cols = [
        'attendance_percentage',
        'study_hours_per_day',
        'previous_exam_score',
        'assignments_completed_percentage',
        'internal_marks',
        'sleep_hours'
    ]
    X = df[feature_cols]
    y = df['result']

    # 80/20 train/test split with seed 42
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Print Report Sections
    print("\n" + "#" * 70)
    print("       STUDENT PASS PREDICTOR - COMPREHENSIVE METRICS REPORT")
    print("#" * 70)

    section1_dataset_summary(df, feature_cols)
    section2_model_info(model)
    y_pred, y_prob, cm = section3_test_metrics(model, X_test, y_test)
    section4_confusion_matrix(cm)
    section5_classification_report(y_test, y_pred)
    section6_train_vs_test(model, X_train, y_train, X_test, y_test)
    section7_cross_validation(model, X, y)
    section8_feature_importance(model, feature_cols)
    section9_tuning_results()

    print("\n" + "#" * 70)
    print(" END OF METRICS REPORT")
    print("#" * 70 + "\n")

if __name__ == '__main__':
    main()
