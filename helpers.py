"""Some helper functions for project 1."""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


# ---------------------------------------------------------------------------
# Logistic regression helpers
# ---------------------------------------------------------------------------


def sigmoid(t):
    """Apply the sigmoid function 1 / (1 + exp(-t)) element-wise.

    Uses the identity sigmoid(t) = 0.5 * (1 + tanh(t / 2)), which gives the
    same values as 1 / (1 + exp(-t)) but never overflows for large |t| and
    is about twice as fast in numpy.

    Args:
        t: numpy array of any shape.

    Returns:
        numpy array of the same shape with values in (0, 1).
    """
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def compute_logistic_loss(y, tx, w):
    """Compute the negative log-likelihood loss of logistic regression.

    The loss is averaged over the N samples:
        L(w) = (1/N) * sum_n [ log(1 + exp(x_n^T w)) - y_n * x_n^T w ]

    np.logaddexp(0, z) computes log(1 + exp(z)) in a numerically stable way,
    avoiding overflow for large |z|.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        loss: scalar, the average negative log-likelihood.
    """
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def compute_logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic regression loss w.r.t. w.

        grad(w) = (1/N) * X^T (sigmoid(X w) - y)

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        gradient: numpy array of shape (D,).
    """
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


def compute_mse_loss(y, tx, w):
    """Compute the mean squared error loss with the 1/2 factor of the course.

        L(w) = 1/(2N) * sum_n (y_n - x_n^T w)^2

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        loss: scalar, the MSE loss.
    """
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss w.r.t. w.

        grad(w) = -(1/N) * X^T (y - X w)

    Works for the full data (gradient descent) and for a single sample
    (stochastic gradient descent, where N = 1).

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the feature matrix.
        w: numpy array of shape (D,), the weight vector.

    Returns:
        gradient: numpy array of shape (D,).
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)
