#  Author: [sixwaaaay](https://github.com/sixwaaaay)
#  Description: trivail linear regression
import numpy as np
from sklearn.linear_model import LinearRegression


def linear_regression(x: np.ndarray, y: np.ndarray) -> LinearRegression:
    """
    Linear regression of single variable
    :param x: input data
    :param y: output data
    :return: LinearRegression object that trained on x and y, and can predict y
    """
    reg = LinearRegression()
    reg.fit(x, y)
    return reg
