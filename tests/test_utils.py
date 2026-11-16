# -*- coding: utf-8 -*-

import numpy as np
import pytest

from pyHSP._utils import check_X, check_y, check_X_y

# tests for check_X


def test_check_X():
    X = np.arange(4 * 3).reshape(4, 3) + 10
    assert check_X(X, n_features=3, min_samples=4) is X


def test_check_X_invalid_features():
    X = np.arange(4 * 2).reshape(4, 2) + 10
    with pytest.raises(ValueError, match="exactly 3 features"):
        check_X(X, n_features=3, min_samples=4)


# tests for check_y


def test_check_y():
    np.testing.assert_equal(check_y([0, 1], graded=False), np.array([0, 1]))


@pytest.mark.parametrize("y", ([1, 2], [0, 0.5]))
def test_check_y_invalid(y):
    with pytest.raises(ValueError, match="0 or 1"):
        check_y(y, graded=False)


def test_check_y_graded():
    y = np.array([0, 0.5, 0.6, 1])
    assert check_y(y, graded=True) is y


@pytest.mark.parametrize("y", ([0, 2], [-1, 1]))
def test_check_y_graded_invalid(y):
    with pytest.raises(ValueError, match="between 0 and 1"):
        check_y(np.array(y), graded=True)


# tests for check_X_y


def test_check_X_y():
    # main custom testing is on scores, so this is a quick test on valid inputs
    X = [[1, 2, 3]]
    y = [0.6]
    X_, y_ = check_X_y(X, y, n_features=3, min_samples=1, graded=True)
    np.testing.assert_equal(X_, np.array(X))
    np.testing.assert_equal(y_, np.array(y))


def test_check_X_y_inconsistent_length():
    X = np.arange(4 * 3).reshape(4, 3) + 10
    y = [0, 0.5, 0.6]
    with pytest.raises(ValueError, match="minimum of 4"):
        check_X_y(X, y, n_features=3, min_samples=4, graded=True)
