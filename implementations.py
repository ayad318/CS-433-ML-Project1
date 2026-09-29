import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using the normal equations."""
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations."""
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent."""
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent."""
    raise NotImplementedError
