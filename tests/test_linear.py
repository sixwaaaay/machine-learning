import pytest

from regression.linear import linear_regression
import numpy as np


def test_linear_single_variable():
    """
    Test linear regression of single variable
    assume that x is a 1D array and y is a 1D array
    in this test, y = x + 4
    """
    x = np.array([[1], [2], [3], [4], [5]])
    # x were organized with row first, which means each row is a sample
    # therefore, x is a 2D array
    y = np.array([5, 6, 7, 8, 9])
    reg = linear_regression(x, y)
    assert reg.coef_[0] == pytest.approx(1)
    assert reg.intercept_ == pytest.approx(4)
