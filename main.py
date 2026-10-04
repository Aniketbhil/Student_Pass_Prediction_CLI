import os
import sys
import json
import subprocess
import joblib
import numpy as np
import pandas as pd

def ensure_model_exists():
    """Checks if model exists; if not, automatically executes data generation and training pipeline."""
    model_path = os.path.join('models', 'model.pkl')
    data_path = os.path.join('data', 'students.csv')

    if not os.path.exists(model_path):
        print("Model file 'models/model.pkl' not found. Executing full pipeline (data -> train -> tune)...")

        if not os.path.exists(data_path):
            print("--> Generating dataset...")
            subprocess.run([sys.executable, 'generate_data.py'], check=True)

        print("--> Training baseline models...")
        subprocess.run([sys.executable, 'train_model.py'], check=True)

        print("--> Running hyperparameter tuning...")
        subprocess.run([sys.executable, 'tune_model.py'], check=True)

    # Load model
    model = joblib.load(model_path)

    # Load metadata if present
    meta_path = os.path.join('models', 'meta.json')
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)

    return model, meta

def load_feature_averages():
    """Loads dataset feature averages to provide feedback explanations."""
    data_path = os.path.join('data', 'students.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return {
            'attendance_percentage': df['attendance_percentage'].mean(),
            'study_hours_per_day': df['study_hours_per_day'].mean(),
            'previous_exam_score': df['previous_exam_score'].mean(),
            'assignments_completed_percentage': df['assignments_completed_percentage'].mean(),
            'internal_marks': df['internal_marks'].mean(),
            'sleep_hours': df['sleep_hours'].mean()
        }
    else:
        # Fallback defaults
        return {
            'attendance_percentage': 70.0,
            'study_hours_per_day': 4.0,
            'previous_exam_score': 50.0,
            'assignments_completed_percentage': 65.0,
            'internal_marks': 25.0,
            'sleep_hours': 6.5
        }

def get_valid_float(prompt_str, min_val, max_val):
    """Prompts for input and validates that it is a float within [min_val, max_val]."""
    while True:
        user_in = input(prompt_str).strip()
        try:
            val = float(user_in)
            if min_val <= val <= max_val:
                return val
            else:
                print(f"  [Invalid Input] Value must be between {min_val} and {max_val}. Please try again.")
        except ValueError:
            print("  [Invalid Input] Please enter a valid numerical value.")

def display_banner(meta, model):
    """Displays terminal welcome banner with model and tuning details."""
    model_name = meta.get('model_name', model.__class__.__name__)
    technique = meta.get('technique', 'Tuned')
    best_params = meta.get('best_params', {})

    print("\n" + "="*60)
    print("        STUDENT PASS PREDICTOR - DMT PROJECT")
    print("="*60)
    print(f" Active Model   : {model_name}")
    print(f" Optimization   : {technique}")
    if best_params:
        params_str = ", ".join([f"{k}={v}" for k, v in best_params.items()])
        print(f" Hyperparameters: {params_str}")
    print("="*60 + "\n")

def analyze_student(student_name, inputs, averages):
    """Compares student values to dataset averages and prints a friendly explanation."""
    weaknesses = []
    strengths = []

    # Checks
    if inputs['attendance_percentage'] < averages['attendance_percentage'] - 10:
        weaknesses.append(f"Attendance ({inputs['attendance_percentage']}%) is below class average ({averages['attendance_percentage']:.1f}%).")
    elif inputs['attendance_percentage'] >= averages['attendance_percentage'] + 10:
        strengths.append(f"Strong attendance ({inputs['attendance_percentage']}%).")

    if inputs['study_hours_per_day'] < averages['study_hours_per_day'] - 1.0:
        weaknesses.append(f"Daily study time ({inputs['study_hours_per_day']} hrs) is below average ({averages['study_hours_per_day']:.1f} hrs).")
    elif inputs['study_hours_per_day'] >= averages['study_hours_per_day'] + 1.5:
        strengths.append(f"Excellent daily study commitment ({inputs['study_hours_per_day']} hrs/day).")

    if inputs['previous_exam_score'] < averages['previous_exam_score'] - 15:
        weaknesses.append(f"Previous exam score ({inputs['previous_exam_score']}) is on the lower side.")
    elif inputs['previous_exam_score'] >= averages['previous_exam_score'] + 15:
        strengths.append(f"High previous exam score ({inputs['previous_exam_score']}).")

    if inputs['internal_marks'] < averages['internal_marks'] - 5:
        weaknesses.append(f"Internal marks ({inputs['internal_marks']}/50) are below average.")
    elif inputs['internal_marks'] >= averages['internal_marks'] + 5:
        strengths.append(f"Good internal marks ({inputs['internal_marks']}/50).")

    if inputs['assignments_completed_percentage'] < averages['assignments_completed_percentage'] - 15:
        weaknesses.append(f"Assignment completion rate ({inputs['assignments_completed_percentage']}%) could be improved.")

    print(f"\n--- Feedback & Performance Analysis for {student_name} ---")
    if strengths:
        print(" Key Strengths:")
        for s in strengths:
            print(f"   ✓ {s}")

    if weaknesses:
        print(" Areas Needing Attention:")
        for w in weaknesses:
            print(f"   ⚠️  {w}")

    if not weaknesses and not strengths:
        print(" Student performance metrics align steadily around class averages.")

def main():
    model, meta = ensure_model_exists()
    averages = load_feature_averages()
    display_banner(meta, model)

    feature_cols = [
        'attendance_percentage',
        'study_hours_per_day',
        'previous_exam_score',
        'assignments_completed_percentage',
        'internal_marks',
        'sleep_hours'
    ]

    while True:
        print("Enter Student Academic & Lifestyle Details:")
        name = input(" Student Name: ").strip()
        if not name:
            name = "Student"

        attendance = get_valid_float(" Attendance Percentage (0 to 100): ", 0.0, 100.0)
        study_hours = get_valid_float(" Daily Study Hours (0 to 8): ", 0.0, 8.0)
        prev_score = get_valid_float(" Previous Exam Score (0 to 100): ", 0.0, 100.0)
        assignments = get_valid_float(" Assignments Completed Percentage (0 to 100): ", 0.0, 100.0)
        internal_marks = get_valid_float(" Internal Marks (0 to 50): ", 0.0, 50.0)
        sleep_hours = get_valid_float(" Daily Sleep Hours (0 to 24): ", 0.0, 24.0)

        input_data = pd.DataFrame([{
            'attendance_percentage': attendance,
            'study_hours_per_day': study_hours,
            'previous_exam_score': prev_score,
            'assignments_completed_percentage': assignments,
            'internal_marks': internal_marks,
            'sleep_hours': sleep_hours
        }])[feature_cols]

        # Predict outcome
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        pass_prob = probabilities[1] * 100
        fail_prob = probabilities[0] * 100

        result_str = "PASS" if prediction == 1 else "FAIL"

        print("\n" + "*"*50)
        print(f" PREDICTION RESULT FOR {name.upper()}: {result_str}")
        print(f" Pass Probability : {pass_prob:.2f}%")
        print(f" Fail Probability : {fail_prob:.2f}%")
        print("*"*50)

        # Print feedback analysis
        student_inputs = {
            'attendance_percentage': attendance,
            'study_hours_per_day': study_hours,
            'previous_exam_score': prev_score,
            'assignments_completed_percentage': assignments,
            'internal_marks': internal_marks,
            'sleep_hours': sleep_hours
        }
        analyze_student(name, student_inputs, averages)

        print("\n" + "-"*50)
        repeat = input("Predict for another student? (y/n): ").strip().lower()
        if repeat not in ['y', 'yes']:
            print("\nThank you for using Student Pass Predictor! Exiting application.\n")
            break
        print("\n")

if __name__ == '__main__':
    main()
