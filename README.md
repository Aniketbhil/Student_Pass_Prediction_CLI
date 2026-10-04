# Student Pass Predictor

A terminal-based machine learning application built for the **Data Mining Techniques (DMT)** course. The application predicts whether a student will **PASS** or **FAIL** based on academic and lifestyle metrics using Decision Tree and Random Forest classifiers optimized via 5-fold cross-validation hyperparameter tuning.

---

## 📌 Data Mining Techniques (DMT) Concepts Covered

1. **Classification**: Supervised learning technique to categorize student records into binary outcomes: `PASS` (1) or `FAIL` (0).
2. **Train/Test Split**: Partitioning data (80% training, 20% testing) using stratified sampling to preserve class distribution.
3. **Decision Tree Classifier**: A rule-based tree model that splits data according to feature thresholds (evaluated with Gini impurity / Entropy).
4. **Random Forest Classifier**: An ensemble of decision trees combined via bagging to reduce variance and improve prediction stability.
5. **Evaluation Metrics**:
   - **Accuracy**: Overall fraction of correct predictions.
   - **Precision**: Proportion of positive predictions that were true passes.
   - **Recall**: Proportion of actual passing students correctly identified.
   - **F1 Score**: Harmonic mean of Precision and Recall.
   - **Confusion Matrix**: Tabular representation of True Positives, False Positives, True Negatives, and False Negatives.
6. **Feature Importance**: Measuring how much each feature contributes to splitting nodes and reducing impurity.
7. **5-Fold Cross-Validation**: Dividing training data into 5 folds to evaluate model generalization and prevent overfitting.
8. **Hyperparameter Tuning**:
   - **GridSearchCV**: Exhaustive search over specified parameter values.
   - **RandomizedSearchCV**: Randomized sampling over hyperparameter distributions for fast exploration.

---

## 🎯 Overfitting vs. Underfitting

- **Underfitting (High Bias)**: Occurs when a model is too simple (e.g., Decision Tree with `max_depth=1` or `2`) to capture patterns in data, leading to low training and low testing accuracy.
- **Overfitting (High Variance)**: Occurs when a model is overly complex (e.g., unconstrained Decision Tree with `max_depth=None`), learning noise and specific training details. It achieves 100% training accuracy but generalizes poorly to new test data.
- **Optimal Trade-off**: Controlled via hyperparameter tuning (`max_depth=5`, `min_samples_leaf=2`), keeping model complexity balanced for maximum test set accuracy.

---

## 📁 Folder Structure

```text
DMT_project/
├── data/
│   └── students.csv                # Synthetic dataset of 1000 students
├── models/
│   ├── model.pkl                   # Saved best tuned classifier model
│   └── meta.json                   # Winning model metadata
├── outputs/
│   ├── tuning_results.csv          # Comprehensive tuning comparison table
│   ├── best_params.json            # Optimal hyperparameter dictionary
│   ├── baseline_vs_tuned.png       # Poster chart: Baseline vs Tuned performance
│   ├── validation_curve.png        # Poster chart: Decision Tree validation curve
│   ├── top_param_combinations.png  # Poster chart: Top GridSearch parameter sets
│   ├── feature_importance.png      # Poster chart: Feature importance bar plot
│   └── poster_notes.md             # Poster-ready summary bullet points
├── generate_data.py                # Synthetic dataset generator
├── train_model.py                  # Baseline training script
├── tune_model.py                   # Hyperparameter tuning script
├── main.py                         # Interactive terminal prediction app
├── pyproject.toml                  # Project metadata & uv configuration
├── uv.lock                         # Pinned dependency lockfile
└── README.md                       # Project documentation
```

---

## 🚀 How to Run the Project (using `uv`)

Initialize and run all project steps using the `uv` package manager:

### 1. Install Dependencies
```bash
uv sync
```

### 2. Generate Synthetic Dataset (1,000 students)
```bash
uv run generate_data.py
```

### 3. Train Baseline Models (Decision Tree & Random Forest)
```bash
uv run train_model.py
```

### 4. Run Hyperparameter Tuning & Generate Poster Artifacts
```bash
uv run tune_model.py
```

### 5. Launch Interactive Terminal Prediction App
```bash
uv run main.py
```
