import numpy as np

from helpers import (
    sigmoid,
    compute_logistic_loss,
    compute_logistic_gradient,
    compute_mse_loss,
    compute_mse_gradient,
)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Minimizes the MSE loss 1/(2N) * ||y - X w||^2 by taking max_iters
    full-gradient steps of size gamma starting from initial_w.

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, number of gradient descent steps to run.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the MSE loss evaluated at w.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_mse_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    At each of the max_iters steps, one sample n is drawn uniformly at random
    and w is updated with the gradient computed on that sample only.

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, number of SGD steps to run.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the MSE loss on the FULL dataset evaluated at w.
    """
    w = initial_w
    n_samples = len(y)
    for _ in range(max_iters):
        n = np.random.randint(n_samples)
        # Slicing n:n+1 keeps the 2D shape (1, D) so the same gradient
        # function can be reused.
        gradient = compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
        w = w - gamma * gradient
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)

    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)

    N = len(y)

    mse = 1 / (2 * N) * (((y - (np.matmul(tx, w))) ** 2).sum())

    return (w, mse)


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar, the MSE loss at w, without the penalty term.
    """

    N = tx.shape[0]

    # Identity matrix
    I = np.eye(tx.shape[1])

    A = tx.T @ tx + N * lambda_ * I
    b = tx.T @ y

    w = np.linalg.solve(A, b)
    loss = compute_mse_loss(y, tx, w)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (labels y in {0, 1}).

    Minimizes the average negative log-likelihood by taking max_iters
    gradient steps of size gamma starting from initial_w.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, number of gradient descent steps to run.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the logistic loss evaluated at w.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Minimizes the average negative log-likelihood plus the penalty
    lambda_ * ||w||^2 with max_iters gradient steps of size gamma. The
    gradient of the penalty term is 2 * lambda_ * w. The returned loss does
    NOT include the penalty term.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the feature matrix.
        lambda_: float, the regularization parameter.
        initial_w: numpy array of shape (D,), the initial weight vector.
        max_iters: int, number of gradient descent steps to run.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the logistic loss at w, without the penalty term.
    """
    w = initial_w
    for _ in range(max_iters):
        gradient = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
