import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Trains the given model, evaluates performance metrics on the test set,
    and returns a dictionary of metrics along with the trained model.
    """
    # Train model
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n================ {name} ================")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"Recall   : {rec * 100:.2f}%")
    print(f"F1 Score : {f1 * 100:.2f}%")
    print("Confusion Matrix:")
    print(f"  [TN={cm[0][0]}, FP={cm[0][1]}]")
    print(f"  [FN={cm[1][0]}, TP={cm[1][1]}]")

    return {
        'name': name,
        'model': model,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm
    }

def main():
    # 1. Load dataset
    data_path = os.path.join('data', 'students.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_data.py first.")

    print(f"Loading dataset from '{data_path}'...")
    df = pd.read_csv(data_path)

    # Separate features (X) and target (y)
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

    # 2. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Dataset split: {len(X_train)} training samples, {len(X_test)} test samples.")

    # 3. Train & Evaluate Models
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_results = evaluate_model("Decision Tree", dt_model, X_train, X_test, y_train, y_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_results = evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test)

    # 4. Compare and select the best model by Accuracy
    if rf_results['accuracy'] >= dt_results['accuracy']:
        best_results = rf_results
    else:
        best_results = dt_results

    print("\n================ Model Selection ================")
    print(f"Best Model: {best_results['name']} (Accuracy: {best_results['accuracy'] * 100:.2f}%)")

    # 5. Print Feature Importances of the best model
    best_model = best_results['model']
    importances = best_model.feature_importances_

    print("\n================ Feature Importances ================")
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    for idx, row in feature_importance_df.iterrows():
        print(f" - {row['Feature']:<35}: {row['Importance']:.4f}")

    # 6. Save the best model
    os.makedirs('models', exist_ok=True)
    model_save_path = os.path.join('models', 'model.pkl')
    joblib.dump(best_model, model_save_path)
    print(f"\nSuccessfully saved the best model ({best_results['name']}) to '{model_save_path}'.")

if __name__ == '__main__':
    main()
