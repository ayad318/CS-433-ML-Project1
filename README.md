# CS-433 Project 1

Heart-attack prediction for the EPFL Machine Learning course (Fall 2026).
From a respondent's answers to the CDC's BRFSS health survey (321 features),
we predict `_MICHD`: whether they have ever had a heart attack or coronary
heart disease (`1`) or not (`-1`).

Only NumPy and the Python standard library are used. No external ML
libraries and no external data.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate` instead of `source .venv/bin/activate`.

Check that the install worked:

```bash
python -c "import numpy; print(numpy.__version__)"
```

## Data

Put these files in `dataset/` and do not rename them:

- `x_train.csv` (328,135 rows)
- `y_train.csv`
- `x_test.csv` (109,379 rows)

## Reproducing our submission

```bash
python run.py
```

This trains the model from scratch (no pretrained weights) and writes
`submission.csv`, the exact file we submitted to AIcrowd. It takes about
20-30 minutes on a laptop (5 cross-validation fits plus the final fit, each
training both models). The run is deterministic: the cross-validation split
and the trees' row/feature sampling use a fixed seed (`SEED = 0`), and
Newton's method has no randomness. It prints:

```
threshold = 0.220
CV F1 = 0.4446 +/- 0.0045
CV accuracy = 0.8830
wrote submission.csv (12.3% predicted positive)
```

## Files

| File | Contents |
|---|---|
| `implementations.py` | The 6 required methods (step 2). Each returns `(w, loss)`. |
| `helpers.py` | Course-provided CSV loading/submission helpers, plus the loss, gradient and sigmoid functions used by `implementations.py`. |
| `preprocessing.py` | Data cleaning and feature generation for logistic regression (`Preprocessor`). |
| `gbdt.py` | Gradient-boosted decision trees written in NumPy (`GradientBoosting`). |
| `run.py` | Full pipeline: both models, ensemble, cross-validation, threshold selection, final training, prediction. |

### Required methods (`implementations.py`)

- `mean_squared_error_gd`: linear regression with gradient descent
- `mean_squared_error_sgd`: linear regression with SGD (batch size 1)
- `least_squares`: normal equations
- `ridge_regression`: normal equations with L2 penalty `lambda_ * ||w||^2`
- `logistic_regression`: gradient descent, labels in `{0, 1}`
- `reg_logistic_regression`: gradient descent with L2 penalty

Every returned loss excludes the penalty term. The MSE uses the course's
`1/(2N)` factor.

## Our pipeline

### 1. Data preparation and feature generation (`preprocessing.py`)

BRFSS answers are coded integers. `7`/`9`, `77`/`99`, `777`/`999`, ... mean
"don't know" or "refused", and `88`/`888` mean "none". `Preprocessor`:

1. **Drops uninformative columns**: interview metadata and IDs (dates,
   `SEQNO`, `_PSU`, questionnaire version, ...) and survey weights. It also
   drops raw answers with mixed-unit encodings (e.g. `ALCDAY5`: `1xx` = days
   per week, `2xx` = days per month) because BRFSS ships cleaned computed
   versions of them (`WEIGHT2` → `WTKG3`, `FRUIT1` → `FRUTDA1_`, `EXEROFT1` →
   `PAFREQ1_`, ...). See `DROP_COLUMNS`.
2. **Categorical columns** (at most 12 distinct values, plus `_STATE`): one-hot
   encoding. Missing values and "don't know"/"refused" codes keep their own
   indicator, because *not answering* can carry information. Categories seen
   fewer than 50 times in training are merged into an "other" level.
3. **Numeric columns** (everything else):
   - refusal codes → missing and "none" codes → `0`, with the codes chosen
     from the column's maximum value (`MISSING_CODES`, `NONE_CODES`);
   - `log1p` for non-negative, heavy-tailed columns (max > 5 × median);
   - median imputation, plus a binary "was missing" feature;
   - standardization to zero mean and unit variance.
4. Removes features that are constant on the training set.

This gives about 1,300 features. The `Preprocessor` learns every statistic
(category lists, medians, means, standard deviations) on the training data
only. Inside cross-validation it is refitted on each training fold, so no
validation data leaks into the features.

### 2. Models

The final prediction averages the probabilities of two models.

**Model A: logistic regression (`run.py`).** L2-regularized logistic regression with `lambda_ = 1e-5`. It minimizes the
same objective as `reg_logistic_regression` (the bias is not penalized), but
uses **Newton's method** instead of gradient descent: each step solves
`H step = grad` and converges in about 10 iterations. Plain gradient descent
with a large step size diverged on the highly correlated one-hot features,
and with a small one it needed thousands of iterations.

**Model B: gradient-boosted trees (`gbdt.py`).** Since no ML library is
allowed, we implemented histogram-based gradient boosting on the logistic
loss (the idea behind XGBoost and LightGBM):

- each raw column (minus `DROP_COLUMNS`) is cut into at most 32 bins, from
  quantiles or one bin per distinct value, with missing values in their own
  bin. Trees work on the raw survey codes, so they need no one-hot encoding
  or scaling;
- 150 trees of depth 5, learning rate 0.1, leaf L2 penalty 1, and at least a
  Hessian sum of 20 per child;
- each tree sees a random 50% of the rows and 50% of the features (fixed
  seed);
- splits maximize the second-order gain
  `G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)`, computed for every bin at once
  from `np.bincount` histograms. Leaves predict `−G/(H+λ)`.

Trees find interactions between answers (e.g. age × diabetes) that the linear
model cannot. Adding hand-made pairwise interaction features to the logistic
regression did not help (see below). With these settings, validation F1 peaks
around 150 trees and then slowly drops as the model overfits.

### 3. Decision threshold

Only 8.8% of training samples are positive, and the competition ranks by F1.
With a 0.5 threshold the model rarely predicts `1`. So we choose the
probability threshold that maximizes F1 on **out-of-fold predictions**:

- the training set is shuffled (seed 0) and split into 5 folds;
- for each fold, the preprocessor and model are fitted on the other 4 folds
  and predict probabilities for the held-out fold;
- the threshold (grid 0.01 to 0.99, step 0.005) that maximizes F1 over all
  out-of-fold predictions is kept (0.215);
- the reported CV F1 is the mean ± std over the 5 folds at that threshold.

Finally, both models are refitted on all training data, and test samples with probability ≥ threshold are predicted `1`.

### Cross-validation results

All experiments use the first 3 of the 5 folds above.

| Features | Model | F1 | Accuracy |
|---|---|---|---|
| Raw (mean-imputed, standardized) | Ridge regression | 0.409 | 0.864 |
| Raw (mean-imputed, standardized) | Logistic regression | 0.414 | 0.868 |
| `Preprocessor` | Ridge regression | 0.430 | 0.875 |
| `Preprocessor` | Logistic regression (Newton) | 0.439 | 0.881 |
| `Preprocessor` + 190 pairwise products of the top-20 features | Logistic regression | 0.440 | 0.882 |
| `Preprocessor` | MLP (128 ReLU units, dropout 0.3, Adam) | 0.438 | 0.874 |
| Raw codes, binned | Gradient-boosted trees | 0.445 | 0.883 |
| both | **Logistic regression + trees (average)** | **0.445** | **0.884** |

Rows 1-4 tune the threshold on each training fold. The other rows tune one
threshold on the pooled validation predictions. The ensemble improves on
logistic regression in every fold (+0.005 to +0.007 F1). Adding the MLP to
the ensemble did not help further. F1 changes by less than 0.005 for
`lambda_` between `1e-6` and `1e-3`.
