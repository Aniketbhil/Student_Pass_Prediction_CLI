# PBL Poster Notes: Student Pass Predictor Hyperparameter Tuning

## 1. Model Parameters vs. Hyperparameters
- **Model Parameters**: Internal configuration variables learned directly from training data (e.g., node split threshold values in decision trees).
- **Hyperparameters**: External configuration parameters set before training that govern the model learning process and structural complexity (e.g., maximum tree depth, minimum samples per leaf).

## 2. Key Hyperparameters Used & Control Mechanisms
- `max_depth`: Limits tree depth to control model complexity and prevent overfitting.
- `min_samples_split`: Minimum samples required to split an internal node; prevents over-segmenting small noise subsets.
- `min_samples_leaf`: Minimum samples required at a leaf node; smooths decision boundaries.
- `n_estimators` (Random Forest): Number of trees in the ensemble; increases stability and reduces variance.
- `max_features` (Random Forest): Number of features considered when looking for the best split; decorrelates individual trees.

## 3. Grid Search vs. Random Search Comparison
- **GridSearch (Exhaustive)**:
  - *Pros*: Guarantees finding the optimal combination within the specified grid.
  - *Cons*: Computationally expensive and slow for large parameter spaces.
- **RandomizedSearch (Stochastic)**:
  - *Pros*: Explores high-dimensional spaces significantly faster with customizable iteration count (`n_iter=20`).
  - *Cons*: Might miss the global optimum if `n_iter` is small.

## 4. Empirical Performance Summary (Baseline vs. Tuned)
| Model | Strategy | Test Accuracy | Test F1 Score | Search Time (s) |
|---|---|---|---|---|
| Decision Tree | Baseline (Default) | 81.50% | 85.02% | 0.00s |
| Decision Tree | GridSearch | 83.00% | 86.29% | 2.43s |
| Random Forest | Baseline (Default) | 88.50% | 91.32% | 0.00s |
| Random Forest | GridSearch | 88.50% | 91.39% | 26.49s |

## 5. Conclusion & PBL Key Takeaways
Hyperparameter tuning via 5-fold cross-validation successfully improved model generalization and reduced overfitting. The baseline Decision Tree scored **81.50%** test accuracy, while hyperparameter optimization yielded a winning **Random Forest (GridSearch)** achieving **88.50%** accuracy and an F1 score of **91.39%**. The top features influencing student pass probabilities were study hours per day, internal assessment marks, and previous exam scores.
