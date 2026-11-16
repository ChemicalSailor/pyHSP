# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 10:14:28 2026

@author: tfletcher
"""

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import os
# import argparse

from typing import Tuple, Callable
from numpy.typing import ArrayLike
from sklearn.utils import check_array as _skl_check_array, check_consistent_length
from functools import wraps

from ._conf import BETA_MAX


def check_X(
    X: ArrayLike, *, n_features: int = 3, min_samples: int = 1, estimator=None
) -> np.ndarray:
    """Check that X is a valid array with the specified minimum features and samples."""
    X_: np.ndarray = _skl_check_array(
        X,
        ensure_2d=True,
        allow_nd=False,
        ensure_all_finite=True,
        accept_sparse=False,
        ensure_min_samples=min_samples,
        ensure_non_negative=True,
        estimator=estimator,
        input_name="X",
    )
    if X_.shape[1] != n_features:
        raise ValueError(f"X must have exactly {n_features} features.")
    return X_


def check_y(
    y: ArrayLike, *, graded: bool = False, min_samples: int = 1, estimator=None
) -> np.ndarray:
    """Check scores are either 0 or 1, or between 0 and 1 (inclusive) for graded scoring."""
    y_: np.ndarray = _skl_check_array(
        y,
        ensure_2d=False,
        allow_nd=False,
        ensure_all_finite=True,
        accept_sparse=False,
        ensure_min_samples=min_samples,
        estimator=estimator,
        input_name="y",
    )
    if y_.ndim != 1:
        raise ValueError("y must be a 1-dimensional array.")
    if not graded and not np.all((y_ == 1) + (y_ == 0)):
        raise ValueError(
            "Invalid score values for non-graded fit. All score "
            "values must be 0 or 1."
        )
    if np.any((y_ > 1) + (y_ < 0)):
        raise ValueError(
            "Invalid score values. All scores must be between 0 and 1 inclusive."
        )
    return y_
    # if np.std(y_) == 0:
    #     raise ValueError(
    #         "Scores appear to be the same for all samples, which is impossible to fit."
    #     )


def check_X_y(
    X: ArrayLike,
    y: ArrayLike,
    *,
    n_features: int = 3,
    min_samples: int = 1,
    graded: bool = False,
    estimator=None,
) -> tuple[np.ndarray, np.ndarray]:

    X_ = check_X(X, n_features=n_features, min_samples=min_samples, estimator=estimator)
    y_ = check_y(y, graded=graded, min_samples=min_samples, estimator=estimator)
    check_consistent_length(X_, y_)

    return X_, y_


def check_beta(beta: float) -> float:
    """Check that beta is a valid float."""
    if not isinstance(beta, (int, float)):
        raise TypeError("beta must be a float.")
    if beta <= 0 or beta > BETA_MAX:
        raise ValueError(
            f"beta must be a float greater than 0 and less than or equal to {BETA_MAX}."
        )
    return float(beta)


# def validates_dataframe(func) -> Callable:
#     """Decorator to validate that the input is a pandas dataframe."""

#     @wraps(func)
#     def wrapper(self, X, *args, **kwargs) -> pd.DataFrame | np.ndarray:
#         if isinstance(X, pd.DataFrame):
#             X = validate_dataframe(X)
#             return func(self, X, *args, **kwargs)

# return wrapper


def _as_series(func) -> Callable:
    """Decorator to output a series if input is a dataframe."""

    @wraps(func)
    def wrapper(self, X, *args, **kwargs) -> pd.Series | ArrayLike:
        result = func(self, np.asarray(X), *args, **kwargs)
        if isinstance(X, pd.DataFrame):
            return pd.Series(result, index=X.index)
        return result

    return wrapper
