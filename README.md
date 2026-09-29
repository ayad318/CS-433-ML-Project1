# CS-433 Project 1

Heart-attack prediction for the EPFL Machine Learning course (Fall 2026). The graded code lives in `implementations.py`.

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

## Libraries

The six methods in `implementations.py` need NumPy and the Python standard library. `import numpy as np` is already in that file.

## Data

Keep these files in `dataset/` and do not rename them:

- `x_train.csv`
- `y_train.csv`
- `x_test.csv`

## Code

`implementations.py` must define:

- `mean_squared_error_gd`
- `mean_squared_error_sgd`
- `least_squares`
- `ridge_regression`
- `logistic_regression`
- `reg_logistic_regression`

Each function returns `(w, loss)`.
