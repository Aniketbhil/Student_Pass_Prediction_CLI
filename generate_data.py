import os
import numpy as np
import pandas as pd

def generate_student_data(num_students=1000, random_seed=42):
    """
    Generates synthetic dataset of students with academic and lifestyle features,
    and calculates pass/fail results based on a realistic weighted formula.
    """
    np.random.seed(random_seed)

    # 1. Generate synthetic feature values within realistic ranges
    attendance = np.random.uniform(40, 100, num_students)  # 40% to 100%
    study_hours = np.random.uniform(0, 8, num_students)     # 0 to 8 hours/day
    previous_score = np.random.uniform(0, 100, num_students) # 0 to 100
    assignments = np.random.uniform(0, 100, num_students)   # 0% to 100%
    internal_marks = np.random.uniform(0, 50, num_students)  # 0 to 50
    sleep_hours = np.random.uniform(4, 9, num_students)     # 4 to 9 hours

    # 2. Compute a weighted score to realistically determine performance
    # Higher weights given to exam scores, study hours, attendance, and internal marks
    weighted_score = (
        0.20 * attendance +
        4.00 * study_hours +
        0.30 * previous_score +
        0.15 * assignments +
        0.60 * internal_marks +
        1.50 * sleep_hours
    )

    # Add Gaussian noise to introduce real-world variability
    noise = np.random.normal(0, 6, num_students)
    final_score = weighted_score + noise

    # 3. Determine pass/fail threshold to achieve roughly 60-70% pass rate
    threshold = np.percentile(final_score, 35)  # 35th percentile cutoff gives 65% pass rate
    result = (final_score >= threshold).astype(int)

    # 4. Create pandas DataFrame
    df = pd.DataFrame({
        'attendance_percentage': np.round(attendance, 2),
        'study_hours_per_day': np.round(study_hours, 2),
        'previous_exam_score': np.round(previous_score, 2),
        'assignments_completed_percentage': np.round(assignments, 2),
        'internal_marks': np.round(internal_marks, 2),
        'sleep_hours': np.round(sleep_hours, 2),
        'result': result
    })

    # 5. Ensure output directory exists and save CSV
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'students.csv')
    df.to_csv(output_path, index=False)

    pass_percentage = (df['result'].sum() / num_students) * 100
    print(f"Dataset generated successfully at '{output_path}'.")
    print(f"Total students: {num_students}")
    print(f"Pass rate: {pass_percentage:.2f}% ({df['result'].sum()} Pass, {num_students - df['result'].sum()} Fail)")

if __name__ == '__main__':
    generate_student_data()
