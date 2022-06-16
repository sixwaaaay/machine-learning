#  Author: [sixwaaaay](https://github.com/sixwaaaay)
#  Description: ridge regression

from sklearn.linear_model import Ridge
import numpy as np


def ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    """
    Ridge regression of single variable
    :param x: input data
    :param y: output data
    :param alpha: regularization parameter
    :return: RidgeRegression object that trained on x and y, and can predict y
    """
    reg = Ridge(alpha=alpha)
    reg.fit(x, y)
    return reg
