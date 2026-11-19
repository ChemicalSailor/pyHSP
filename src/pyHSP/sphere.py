# -*- coding: utf-8 -*-


from __future__ import annotations
import numpy as np
import pandas as pd
import json
from warnings import warn
from typing import (
    Callable,
    Sequence,
    Union,
    Optional,
    Tuple,
    List,
    TypeVar,
    Literal,
    Self,
    overload,
)
from abc import abstractmethod, ABC
from sklearn.base import BaseEstimator, clone, ClassifierMixin
from sklearn.utils import (
    check_X_y,
    check_scalar,
    check_array,
    check_consistent_length,
    check_random_state,
)
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    log_loss,
    f1_score,
    r2_score,
    get_scorer,
    mean_squared_error,
    mean_absolute_error,
)
from scipy.special import expit
from scipy.optimize import minimize
from functools import wraps, partial
from pathlib import Path

# try:
#     import plotly.graph_objects as go  # noqa

#     plotly_installed = True
# except ModuleNotFoundError:
#     plotly_installed = False

# if plotly_installed:
#     from ._plotly_accessor import PlotlyAccessor

from . import __version__
from .types import SeriesOrDataFrame, TPathLike
from .pandas import validate_HSP_dataframe, _get_labels, prepare_dataframe
from .functional import (
    calculate_distance,
    RED,
    conc_wt_to_mol,
    conc_correction,
    mole_fraction,
    inverse_sum,
    solvent_molarity,
)
from .io import read_hsdx_result
from ._utils import check_X, check_X_y, keep_index, check_beta
from ._conf import BETA_MAX
from numpy.typing import ArrayLike, NDArray

# # TODO: docstring
# def fit_sphere(HSP, graded=False, classic=False, Mw=None, score_col="score"):
#     if classic:
#         sph = ClassicSphere(graded=graded)
#     else:
#         sph = Sphere(Mw, graded=graded)
#     return sph.prepare_fit(HSP, scores=score_col)


# # TODO: docstring
# def sphere_report(estimator, X, y_true=None, *, sample_weight=None, score_col="score"):
#     o = estimator.coef_.copy()
#     y_true = X[score_col] if y_true is None else y_true
#     o["F1"] = estimator.score(
#         X, y_true.round(0), sample_weight=sample_weight, scoring="f1"
#     )
#     o["R2"] = estimator.score(X, y_true, sample_weight=sample_weight, scoring="r2")
#     return o


# TODO: add dHA and dHD scaling as proposed by Louwerse
def _decision_function(
    coefs: ArrayLike,
    X: ArrayLike,
    *,
    da: bool = False,
    # conc: bool = False,
    solvent_radii: bool = False,
    # mvol_factor: float = 1520,
    dD_correction: bool = True,
    squared_dist: bool = False,
    # eps: float = 1e-12,
) -> NDArray[np.floating]:
    """
    Calculate the signed distance from the solubility-sphere boundary.

    Positive values indicate points inside the sphere; negative values indicate
    points outside it. When solvent radii are enabled, the effective radius is
    adjusted for solvent molar volume and concentration.

    Parameters
    ----------
    coefs : ndarray of shape (4,) or (5,)
        Sphere center coordinates followed by its radius. The vector has four
        elements for the combined hydrogen-bonding term and five for separate
        donor and acceptor terms.
    X : ndarray of shape (n_samples, n_features)
        Solvent HSPs.
    da : bool, default=False
        Whether to use separate donor and acceptor HSPs.
    solvent_radii : bool, default=False
        Whether to combine the solute and solvent radii and apply the
        concentration correction.
    dD_correction : bool, default=True
        Whether to use the classic dispersion-distance correction.
    squared_dist : bool, default=False
        Whether to compare squared radii and distances.

    Returns
    -------
    ndarray of shape (n_samples,)
        Signed distance from the sphere boundary.
    """
    X, coefs = np.asarray(X, dtype=np.float64), np.asarray(coefs, dtype=np.float64)

    assert X.ndim == 2
    assert coefs.ndim == 1

    solv_HSP: NDArray[np.floating] = X[:, :4] if da else X[:, :3]
    solute_HSP: NDArray[np.floating] = coefs[:4] if da else coefs[:3]
    solute_radius: float = coefs[4] if da else coefs[3]

    benchmark_conc = 0.5

    # set mvol_factor according to values in published matlab code
    mvol_factor = 1545.0 if dD_correction else 1520.0

    if solvent_radii:
        # Solvent radii are defined for the benchmark concentration of x=0.5,
        # therefore the solute radius is also defined at x=0.5.
        # We find the effective radius of solute + solvent, and then use the
        # scaling rule to adjust to the actual concentration.
        # We only use the concentration correction to apply this scaling;
        # applying the fit to different concentrations is discouraged as it is
        # unlikely to yield sensible results.

        mvol = X[:, 4] if da else X[:, 3]
        solvent_radius = mvol_factor / mvol
        r_eff_benchmark = inverse_sum(solute_radius, solvent_radius)

        molar_conc = X[:, 5] if da else X[:, 4]
        conc_molfrac = mole_fraction(molar_conc, solvent_molarity(mvol))
        r_eff = conc_correction(conc_molfrac, benchmark_conc) * r_eff_benchmark

        # In this treatment, the radius variables are equal to a squared
        # distance and may be negative. We then use the squared solute-solvent
        # distance to calculate our decision score.
        squared_dist = True
    else:
        # For the classic algorithm either the squared or unsquared distance can
        # be used and this choice is left up to the user; the result from fitting
        # will reflect this choice. The default is to use the unsquared distance
        # to align with the classic method.
        r_eff = solute_radius

    if squared_dist:
        r_eff = r_eff**2

    solv_distance = calculate_distance(
        solv_HSP,
        solute_HSP,
        root=not squared_dist,
        classic=dD_correction,
        da=da,
    )
    in_out_score = r_eff - solv_distance

    return in_out_score


def _log_loss(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sample_weight: np.ndarray | None = None,
    eps: float = 1e-12,
):
    """
    Calculate binary log loss, including for graded target values.

    Unlike scikit-learn's binary log-loss, this function accepts target values
    between zero and one and returns the sum of per-sample losses.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        Target values in the interval [0, 1].
    y_proba : ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.
    sample_weight : ndarray of shape (n_samples,) or None, default=None
        Per-sample weights. If provided, the weighted losses are summed.
    eps : float, default=1e-12
        Minimum distance of predicted probabilities from zero and one.

    Returns
    -------
    float
        Sum of the per-sample log losses, weighted when ``sample_weight`` is
        provided.
    """
    assert y_true.ndim == 1
    assert y_proba.ndim == 1

    y_proba = np.clip(y_proba, eps, 1 - eps)[:, np.newaxis]
    y_proba = np.append(1 - y_proba, y_proba, axis=1)

    y_true = y_true[:, np.newaxis]
    y_true = np.append(1 - y_true, y_true, axis=1)

    loss = -(y_true * np.log(y_proba)).sum(axis=1)
    if sample_weight is not None:
        return np.dot(loss, sample_weight)
    return loss.sum()


class HSPEstimator(BaseEstimator, ClassifierMixin):
    """Base class for solubility sphere fitting."""

    da: bool
    graded: bool
    squared_dist: bool
    coef_: pd.Series
    _dfunc_kwargs: dict
    pd: _HSPDataFrameAccessor

    @property
    def _n_hsp(self) -> int:
        """The number of HSP features used by this estimator."""
        return 4 if self.da else 3

    def set_coef(self, coef_: ArrayLike, beta_: float = BETA_MAX) -> Self:
        """
        Set the sphere coefficients and decision-function parameters.

        Use this method to set the coefficients when constructing an instance
        via method chaining.

        Parameters
        ----------
        coef_ : array-like of shape (4,) or (5,)
            Sphere center coordinates followed by the radius. The center has
            three coordinates for combined hydrogen bonding, or four for
            separate donor and acceptor terms.
        beta_ : float, default=10.0
            Coefficient defining the sharpness of the sphere boundary.

        Returns
        -------
        HSPEstimator
            The estimator with its coefficients set.
        """
        labels = _get_labels(self.da) + ["radius"]
        coef = np.asarray(coef_)

        if (coef.ndim) != 1:
            raise ValueError(
                f"`coef` is expected to be 1-dimensional array-like, found {coef.ndim} dimensions."
            )

        if len(labels) != len(coef):
            raise ValueError(
                f"`coef` expected to be sequence with {len(labels)} elements;"
                f" got {type(coef)} with {len(coef)} elements."
            )

        self.coef_ = pd.Series(coef, index=labels)
        self.beta_ = check_beta(beta_)

        # save parameters for the decision function used during fitting for making predictions
        self.fit_params_ = {"squared_dist": self.squared_dist, "da": self.da}
        return self

    def to_dict(self) -> dict:
        """
        Serialize the fitted estimator as a dictionary.

        Returns
        -------
        dict
            Estimator class name, constructor parameters and fitted coefficients.
        """
        d = {
            "name": self.__class__.__name__,
            "params": self.get_params(),
            "coef_": self.coef_.to_dict(),
            "beta_": self.beta_,
        }
        return d

    def to_json(self, path: TPathLike | None = None, pretty: bool = True) -> str | None:
        """
        Serialize the fitted estimator as JSON.

        Parameters
        ----------
        path : path-like or None, default=None
            Destination file. If ``None``, return the JSON string. When a path
            is provided, it is opened in exclusive-create mode and must not
            already exist.
        pretty : bool, default=True
            Whether to indent the JSON output for readability.

        Returns
        -------
        str or None
            The JSON string when ``path`` is ``None``; otherwise, ``None``
            after writing the file.
        """
        d = self.to_dict()
        d["pyhsp_version"] = __version__

        indent = 2 if pretty else None

        if not path:
            return json.dumps(d, indent=indent)
        with open(path, "x") as f:
            return json.dump(d, f, indent=indent)

    @classmethod
    def from_dict(
        cls: type[Self],
        dict_: dict,
        cls_error: Literal["ignore", "warn", "raise"] = "ignore",
    ) -> Self:
        """
        Create from serialized definition.

        Parameters
        ----------
        dict_ : dict
            Serialized dict created by `to_dict()`.
        cls_error : {"ignore", "warn", "raise"}, default="ignore"
            Action to take if serialised class name does not match this class.

        Notes
        -----
        Mismatched class names are ignored by default because loading a
        compatible estimator under a different class can be intentional.

        Returns
        -------
        HSPEstimator
            An estimator reconstructed from the serialized definition.
        """
        valid_cls_error = ("ignore", "warn", "raise")
        if cls_error not in valid_cls_error:  # pragma: no cover
            raise ValueError(
                f"`cls_error` must be one of {', '.join(valid_cls_error)}; got {cls_error}"
            )

        d = dict_

        if cls.__name__ != d["name"]:
            err = (
                f"Data was saved from a {d['name']} object which may "
                "not be compatible with this class."
            )
            if cls_error == "warn":
                warn(err, UserWarning)
            elif cls_error == "raise":
                raise TypeError(err)

        # Being lazy and relying on duck typing for import; provides some
        # portability but the class needs to support all of the initialisation
        # args that were serialised
        o = cls().set_params(**d["params"])
        o.set_coef(pd.Series(d["coef_"]), d["beta_"])
        return o

    @classmethod
    def from_json(
        cls,
        path: TPathLike,
        cls_error: Literal["ignore", "warn", "raise"] = "warn",
    ) -> Self:
        """
        Create from JSON serialized definition.

        Parameters
        ----------
        path : path-like
            Location of the JSON file.
        cls_error : {"ignore", "warn", "raise"}, default="warn"
            Action to take if serialised class name does not match this class.

        Returns
        -------
        HSPEstimator
            An estimator reconstructed from the JSON definition.
        """
        with open(path) as fp:
            d = json.load(fp)
        return cls.from_dict(d, cls_error)

    # def __eq__(self, other) -> bool:
    #     d_self = self.to_dict()
    #     try:
    #         d_other = other.to_dict()
    #     except AttributeError:
    #         return False
    #     return d_self == d_other

    @keep_index
    def distance(self, X: ArrayLike) -> NDArray[np.floating]:
        """
        Calculate distances from the fitted sphere center to solvents.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.

        Returns
        -------
        ndarray of shape (n_samples,) or pandas.Series
            Distances in Hansen solubility space. If ``X`` is a DataFrame, the
            returned Series retains its index.
        """

        check_is_fitted(self, "coef_")
        X = check_X(X, n_features=self._n_hsp, estimator=self)
        coef = np.asarray(self.coef_)[: self._n_hsp]
        return calculate_distance(coef, X, da=self.da)

    @keep_index
    def decision_function(self, X: ArrayLike) -> NDArray[np.floating]:
        """
        Calculate the signed distance of solvents from the sphere boundary.

        Positive values indicate points inside the solubility sphere and
        negative values indicate points outside it.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.

        Returns
        -------
        ndarray of shape (n_samples,) or pandas.Series
            Signed distance metric. If ``X`` is a DataFrame, the returned Series
            retains its index.
        """

        check_is_fitted(self, ["coef_", "fit_params_"])
        X = check_X(X, n_features=self._n_hsp, estimator=self)
        coef = np.asarray(self.coef_)
        kws = self._dfunc_kwargs.copy()
        kws.update(self.fit_params_)
        return _decision_function(coef, X, **kws)

    # predict_proba is defined privately without keep_index decorator to explictly return an ndarray for typing purposes
    def _predict_proba(self, X: ArrayLike) -> NDArray[np.floating]:
        check_is_fitted(self, "beta_")
        X = check_X(X, n_features=self._n_hsp, estimator=self)
        return expit(self.beta_ * self.decision_function(X))

    @keep_index
    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]:
        """
        Estimate the probability that each solvent dissolves the solute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.

        Returns
        -------
        ndarray of shape (n_samples,) or pandas.Series
            Predicted probabilities for dissolution. If ``X`` is a DataFrame,
            the returned Series retains its index.
        """
        return self._predict_proba(X)

    @keep_index
    def predict(self, X: ArrayLike) -> NDArray[np.uint8]:
        """
        Predict whether each solvent dissolves the solute.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.

        Returns
        -------
        ndarray of shape (n_samples,) or pandas.Series
            Binary dissolution predictions (0 or 1). If ``X`` is a DataFrame,
            the returned Series retains its index.
        """
        X = check_X(X, n_features=self._n_hsp, estimator=self)
        return self._predict_proba(X).round().astype(np.uint8)

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """
        Abstract method to be implemented for fitting.
        """
        raise NotImplementedError(  # pragma: no-cover
            "BaseSphere is an abstract class and cannot be fitted."
        )

    def fit_predict(self, X: ArrayLike, y: ArrayLike) -> NDArray[np.uint8] | pd.Series:
        """
        Fit the estimator and predict the supplied samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.
        y : array-like of shape (n_samples,)
            Solubility scores.

        Returns
        -------
        ndarray of shape (n_samples,) or pandas.Series
            Binary dissolution predictions (0 or 1). If ``X`` is a DataFrame,
            the returned Series retains its index.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(
        self,
        X: ArrayLike,
        y_true: ArrayLike,
        *,
        sample_weight: ArrayLike | None = None,
        scoring: Callable | str = "f1",
    ) -> float:
        """
        Evaluate the fitted sphere with a scikit-learn scorer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.
        y_true : array-like of shape (n_samples,)
            True solubility scores for ``X``.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Per-sample weights passed to the scorer.
        scoring : str or callable, default="f1"
            Scikit-learn scorer name or scorer callable.

        Returns
        -------
        float
            The selected score. The default F1 score is useful for
            classification with imbalanced classes.
        """
        X, y = check_X_y(
            X, y_true, n_features=self._n_hsp, graded=self.graded, estimator=self
        )
        scorer = get_scorer(scoring)
        return scorer(self, X, y_true, sample_weight=sample_weight)

    # if plotly_installed:
    #     plot = CachedAccessor("plot", PlotlyAccessor)


class _HSPDataFrameAccessor:

    def __init__(
        self,
        sphere: HSPEstimator,
        *,
        inc_mvol: bool | str = False,
        inc_conc: bool | str = False,
    ):
        """
        Initialize the DataFrame adapter for an HSP estimator.

        Parameters
        ----------
        sphere : HSPEstimator
            Estimator used to for sphere fitting.
        inc_mvol : bool or str, default=False
            Whether to include the molar volume feature when preparing data.
            A string specifies the concentration column name.
        inc_conc : bool or str, default=False
            Whether to include the concentration feature. A string specifies
            the concentration column name.
        """

        self._sphere = sphere
        self._inc_mvol = inc_mvol
        self._inc_conc = inc_conc

    @overload
    def __call__(
        self,
        HSP: pd.DataFrame,
        return_scores: Literal[True] = ...,
        score_col: str = "Score",
        drop_na: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]: ...

    @overload
    def __call__(
        self,
        HSP: pd.DataFrame,
        return_scores: Literal[False] = ...,
        score_col: str = "Score",
        drop_na: bool = True,
    ) -> pd.DataFrame: ...

    def __call__(
        self,
        HSP: pd.DataFrame,
        return_scores: bool = True,
        score_col: str = "Score",
        drop_na: bool = True,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
        """
        Return `X` and optionally `y` from a dataframe for use with the estimator.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input HSP data.
        return_scores : bool, default=True
            Whether to extract scores from ``score_col``.
        score_col : str, default="Score"
            Name of the column with scores.
        drop_na : bool, default=True
            Whether to drop rows with missing values.

        Returns
        -------
        pandas.DataFrame or tuple of (pandas.DataFrame, pandas.Series)
            Prepared features, and (when ``return_scores=True``) the solubility
            scores.
        """
        scores = score_col if return_scores else False
        scores_trf = None if self._sphere.graded else np.round
        # Mw = self._sphere.Mw if self._inc_conc else None
        return prepare_dataframe(
            HSP,
            scores=scores,
            da=self._sphere.da,
            drop_na=drop_na,
            mvol=self._inc_mvol,
            conc=self._inc_conc,
            # convert_to_molar=True,
            # Mw=Mw,
            scores_transformer=scores_trf,
        )

    def fit(self, HSP: pd.DataFrame, *, score_col: str = "Score") -> HSPEstimator:
        """
        Fit the Sphere using the provided dataframe.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data, including solubility scores.
        score_col : str, default="Score"
            Name of the column with scores.

        Returns
        -------
        HSPEstimator
            The fitted estimator.
        """
        X, y = self(HSP, return_scores=True, score_col=score_col, drop_na=True)
        return self._sphere.fit(X, y)

    def fit_predict(self, HSP: pd.DataFrame, *, score_col: str = "Score") -> pd.Series:
        """
        Fit the estimator using a DataFrame and return predictions for the same samples.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data, including solubility scores.
        score_col : str, default="Score"
            Name of the column with scores.

        Returns
        -------
        pandas.Series
            Binary dissolution predictions, indexed like the input.
        """
        X, y = self(HSP, return_scores=True, score_col=score_col, drop_na=True)
        return self._sphere.fit_predict(X, y)  # pyright: ignore[reportReturnType]

    def predict(self, HSP: pd.DataFrame) -> pd.Series:
        """
        Predict dissolution for solvents in a DataFrame.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data.

        Returns
        -------
        pandas.Series
            Binary dissolution predictions, indexed like the input.
        """
        X = self(HSP, return_scores=False, drop_na=False)
        return self._sphere.predict(X)  # pyright: ignore[reportReturnType]

    def predict_proba(self, HSP: pd.DataFrame) -> pd.Series:
        """
        Estimate dissolution probabilities for solvents in a DataFrame.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data.

        Returns
        -------
        pandas.Series
            Predicted probabilities, indexed like the input.
        """
        X = self(HSP, return_scores=False, drop_na=False)
        return self._sphere.predict_proba(X)  # pyright: ignore[reportReturnType]

    def decision_function(self, HSP: pd.DataFrame) -> pd.Series:
        """
        Calculate the signed distance of solvents from the sphere boundary.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data.

        Returns
        -------
        pandas.Series
            Signed distance metric, indexed like the input.
        """
        X = self(HSP, return_scores=False, drop_na=False)
        return self._sphere.decision_function(X)  # pyright: ignore[reportReturnType]

    def distance(self, HSP: pd.DataFrame) -> pd.Series:
        """
        Calculate distances from the fitted sphere center to solvents.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data.

        Returns
        -------
        pandas.Series
            Distances in Hansen solubility space, indexed like the input.
        """
        X = self(HSP, return_scores=False, drop_na=False)
        return self._sphere.distance(X)  # pyright: ignore[reportReturnType]

    def score(
        self, HSP: pd.DataFrame, *, scoring: str = "f1", score_col: str = "Score"
    ) -> float:
        """
        Evaluate the estimator on scores from an HSP DataFrame.

        Parameters
        ----------
        HSP : pandas.DataFrame
            Input solvent HSP data, including solubility scores.
        scoring : str, default="f1"
            Name of the scikit-learn scorer to use.
        score_col : str, default="Score"
            Name of the column with scores.

        Returns
        -------
        float
            The score returned by the estimator's ``score`` method.
        """
        X, y = self(HSP, return_scores=True, score_col=score_col, drop_na=True)
        return self._sphere.score(X, y, scoring=scoring)


class ClassicSphere(HSPEstimator):
    """
    HSP estimator using the classic distance metric.

    Parameters
    ----------
    alpha : float, default=0
        Regularization factor that controls the sphere radius. Negative values
        favor a smaller sphere; positive values favor a larger sphere.
    beta : float or "fit", default=10.0
        Factor controlling the sharpness of the sphere boundary. High values
        give a hard boundary where misclassification is strongly penalised,
        whereas a low value gives a soft boundary which may be more relevant for
        samples with marginal solubility in several solvents. This parameter can be
        estimated during fitting by passing a value of 'fit'; as this
        increases the number of model parameters, it is recommended to only use
        this option with a large number of samples.
    initialisation : {"mean", "random"}, default="mean"
        Method to generate initial points. Mean initialisation generates initial
        conditions based on good solvents. Random initialisation generates initial
        points by sampling uniformly across the Hansen space.
    init_sigma : float, default=3.0
        Sigma value to use in Gaussian sampling when ``initialisation="mean"``.
        Increasing will introduce more variability in initial HSP values,
        exploring a wider solubility space with the risk of settling into local
        minima.
    loss : callable or None, default=None
        Custom loss function accepting ``y_true``, ``y_proba``, and
        ``sample_weight``. If not callable, the built-in log loss is used.
    squared_dist : bool, default=False
        Whether to use squared distances in fitting. The default uses
        unsquared distances.
    eps : float, default=1e-12
        Small value used to clip probabilities in the built-in log loss.
    bounds : list of (min, max) tuples, default=None
        Coefficient bounds to use during fitting. If supplied, this should be a
        list of (min, max) pairs of equal length to the number of coefficients,
        i.e. 4 for a standard fit and 5 for a donor/acceptor split; `None` may
        be specified to indicate no bound. See SciPy optimization documentation
        for additional details. By default the solubility parameters and radius
        are bounded to be strictly positive.
    method : str, default="L-BFGS-B"
        Algorithm passed to SciPy's ``minimize`` function.
    n_fits : int, default=20
        Number of optimization attempts.
    random_state : int or None, default=None
        Seed used to initialize the random number generator for fit
        initializations.

    Attributes
    ----------
    coef_ : pandas.Series
        Fit HSP coefficients, including sphere radius.
    beta_ : float
        Coefficient for sharpness of the sphere boundary used or estimated during fitting.
    fit_results_ : pandas.DataFrame
        HSP coefficients and value of loss function for each fit in `n_fits`.
    fit_success_ : int
        Number of successful optimization attempts.
    pd : accessor
        See support for pandas dataframes in the notes.

    Examples
    --------
    Fit a sphere using example data

    >>> from pyHSP.examples import load_pla
    >>> pla = load_pla()
    >>> X = pla[['dD', 'dP', 'dH']]
    >>> y = pla['Score']
    >>> sphere = ClassicSphere().fit(X, y)  # Series with same index as X

    Make predictions

    >>> predicted_solubility = sphere.predict(X)

    Use the `pd` accessor

    >>> sphere = ClassicSphere().pd.fit(pla)
    >>> predicted_solubility = sphere.pd.predict(pla)

    Notes
    -----
    **API**
        The estimator API follows scikit-learn conventions: solvent HSPs are
        features ``X`` and solubility scores are targets ``y``, respectively passed as 2d
        and 1d array-like objects; return values are numpy arrays. `X` is assumed to have three
        columns in the order dD, dP, dH; an error will be raised if an array with incorrect shape is
        passed. An HSPEstimator is a scikit-learn classifier, implementing `fit`, `predict`, `predict_proba`,
        `decision_function` and `score` methods. The default scorer used is the F1 metric.
        Methods to calculate solute-solvent distance and RED are additionally provided.

    **Support for pandas dataframes**
        DataFrames and Series can be used as 2d and 1d array-like objects respectively, but
        dataframes must still adhere to the three column convention. If a method receives dataframe
        input the output will be a Series with the same index. Enhanced support for dataframes is
        provided via the `pd` accessor: This implements the same methods which accept a (variable size) dataframe as a
        single paramter; the appropriate columns are extracted for the `X` and `y` variables and passed
        to the estimator.
    """

    def __init__(
        self,
        *,
        # graded: bool = False,
        da: bool = False,
        alpha: float = 0,
        beta: float | Literal["fit"] = BETA_MAX,
        initialisation: Literal["mean", "random"] = "mean",
        init_sigma: float = 3.0,
        loss: Callable | None = None,
        squared_dist: bool = False,
        eps: float = 1e-12,
        bounds: list[tuple[float | None, float | None]] | None = None,
        method: str = "L-BFGS-B",
        n_fits: int = 20,
        random_state: int | None = None,
    ):
        self.graded = False
        self.da = da
        self.bounds = bounds
        self.alpha = alpha
        self.beta = beta
        self.method = method
        self.loss = loss
        self.squared_dist = squared_dist
        self.eps = eps
        self.n_fits = n_fits
        self.initialisation = initialisation
        self.init_sigma = init_sigma
        self.random_state = random_state

        self.pd = _HSPDataFrameAccessor(self, inc_mvol=False, inc_conc=False)

    @classmethod
    def from_hsdx(cls, path: TPathLike, *, beta_: float = BETA_MAX) -> Self:
        """
        Create an estimator from fitted results in an HSDX file.

        Parameters
        ----------
        path : path-like
            Path to the HSDX file.
        beta_ : float, default=10.0
            Sharpness of the sphere boundary.

        Returns
        -------
        ClassicSphere
            Estimator initialized with the coefficients read from the file.
        """

        coef = read_hsdx_result(path)
        if list(coef.index) == _get_labels(da=True, radius=True):
            da = True
        else:
            da = False
        return cls(da=da, squared_dist=False).set_coef(coef, beta_=beta_)

    def _get_initial_coefs(
        self, X: np.ndarray, y: np.ndarray, rng: np.random.RandomState
    ):
        """
        Generate an initial coefficient vector for an optimization attempt.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Validated solvent HSP features.
        y : ndarray of shape (n_samples,)
            Validated solubility scores.
        rng : RandomState
            Random number generator used to sample initial values.

        Returns
        -------
        ndarray
            Initial sphere coefficients, with beta included when
            ``self.beta == "fit"``.
        """
        n_hsp = self._n_hsp
        if self.initialisation == "mean":
            # take the average of good solvents as initial guess, with some noise
            good_solvents = y >= 0.5
            initial_HSP = X[good_solvents, :n_hsp].mean(axis=0)
            initial_r = calculate_distance(initial_HSP, X[good_solvents, :n_hsp]).mean(
                axis=0
            )
            initial_coefs = np.append(initial_HSP, initial_r)
            initial_coefs += rng.normal(scale=self.init_sigma, size=n_hsp + 1)
        elif self.initialisation == "random":
            # pick values uniformly from a range
            low, high = [10, 0], [30, 30]
            nH = 2 if self.da else 1
            low, high = low + [0] * nH, high + [40] * nH
            low, high = low + [0], high + [20]
            initial_coefs = rng.uniform(low, high)
        else:
            raise ValueError(
                f"Invalid value '{self.initialisation}' for parameter `initialisation`;"
                " expected one of 'mean' or 'random'"
            )

        if self.beta == "fit":
            initial_coefs = np.append(initial_coefs, [1.0])

        return initial_coefs

    @property
    def _dfunc_kwargs(self):
        """Return keyword arguments configuring the decision function."""
        return dict(
            da=self.da,
            solvent_radii=False,
            dD_correction=True,
            squared_dist=self.squared_dist,
        )

    def _loss_fn(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ):
        """
        Evaluate the configured fitting loss.

        Parameters
        ----------
        y_true : ndarray of shape (n_samples,)
            Target solubility scores.
        y_pred : ndarray of shape (n_samples,)
            Predicted dissolution probabilities.
        sample_weight : ndarray of shape (n_samples,) or None, default=None
            Per-sample weights passed to the loss function.

        Returns
        -------
        float
            Loss value returned by the configured loss function.
        """
        if callable(self.loss):
            fn = self.loss
        else:
            # use _log_loss by default
            fn = partial(_log_loss, eps=self.eps)
            # if self.graded:
            #     # allow MAE or MSE for graded fitting
            #     if self.loss == "mae":
            #         fn = mean_absolute_error
            #     elif self.loss == "mse":
            #         fn = mean_squared_error

        return fn(y_true, y_pred, sample_weight=sample_weight)

    def fit(
        self, X: ArrayLike, y: ArrayLike, *, sample_weight: ArrayLike | None = None
    ):
        """
        Fit the sphere to solvent HSPs and solubility scores.

        The best result from ``n_fits`` optimization attempts is stored as the
        fitted coefficients. At least three samples with varying target values
        are required.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.
        y : array-like of shape (n_samples,)
            Solubility scores. Values must be binary unless ``graded=True``,
            in which case values in [0, 1] are supported.
        sample_weight : array-like of shape (n_samples,) or None, default=None
            Per-sample weights used by the loss function.

        Returns
        -------
        ClassicSphere
            The fitted estimator.

        Raises
        ------
        ValueError
            If the target scores have no variation, ``beta`` is invalid, or
            the bounds have the wrong number of entries.
        """
        X, y = check_X_y(
            X,
            y,
            n_features=self._n_hsp,
            min_samples=3,
            graded=self.graded,
            estimator=self,
        )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            check_consistent_length(sample_weight, y)

        if np.var(y) == 0:
            raise ValueError(
                "Cannot fit a sphere to data with no variation in the target values."
            )

        n_coef = self._n_hsp + 1

        # validate and set up parameter beta
        fit_beta = self.beta == "fit"

        e = ValueError(
            f"Invalid value {self.beta} for `beta`. Must be 'fit' or a positive number"
        )
        if fit_beta:
            beta = 0.0  # placeholder, beta will be fitted
        else:
            if isinstance(self.beta, str):
                raise e
            beta = float(self.beta)
            if beta <= 0:
                raise e

        # function to be minimised
        # we add -alpha * radius as a regularisation term to control sphere size
        # using -alpha means a larger number favours a larger sphere
        def loss(coefs: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
            """Return the weighted data loss with radius regularization."""
            if fit_beta:
                beta: float = coefs[-1]
                coefs = coefs[:-1]
            else:
                beta = float(self.beta)
            radius = coefs[-1]
            in_out_dist = _decision_function(coefs, X, **self._dfunc_kwargs)
            proba = expit(beta * in_out_dist)
            return (
                self._loss_fn(y, proba, sample_weight=sample_weight)
                - self.alpha * radius
            )

        bounds: list[tuple[float | None, float | None]]

        if not self.bounds:
            # limit coefs to strictly positive
            bounds = [(0.0, None)] * n_coef
            # if self.da:
            #     # Allow the radius to be -ve for a D/A fit; a negative radius
            #     # corresponds to only dissolving with strong hydrogen bonding.
            #     # Note that in terms of mixing a -ve radius is illogical, a
            #     # species should always mix with itself (distance = 0), but in
            #     # terms of dissolving there can be an additional melting
            #     # enthalpy that needs to be overcome, and this would also
            #     # influence the radius of the sphere. As interpretation depends
            #     # on the problem we make the bounds a user-adjustable parameter.
            #     bounds[-1] = (None, None)
        else:
            if len(self.bounds) != n_coef:
                raise ValueError(
                    f"Invalid number of bounds; expected {n_coef} but got {len(self.bounds)}"
                )
            bounds = self.bounds  # pragma: no cover

        # hard set bounds for beta
        if fit_beta:
            bounds.append((0.0, BETA_MAX))

        result_labels = _get_labels(self.da) + ["radius"]
        if fit_beta:
            result_labels.append("beta")

        # seed a random state that can propagate across multiple fits
        rng = check_random_state(self.random_state)

        def single_fit(X: np.ndarray, y: np.ndarray):
            """Run one bounded optimization attempt and return its result."""
            initial_coefs = self._get_initial_coefs(X, y, rng)
            # minimise loss() by adjusting coefs
            opt_res = minimize(
                loss,
                initial_coefs,
                args=(X, y),
                method=self.method,
                bounds=bounds,
            )
            if opt_res.success:
                result = dict(zip(result_labels, opt_res.x))
                result["loss"] = opt_res.fun
                return result
            return None  # pragma: no cover

        fit_results = [single_fit(X, y) for i in range(self.n_fits)]
        fit_results = pd.DataFrame(filter(None, fit_results)).sort_values(
            "loss", ignore_index=True
        )

        self.fit_success_ = len(fit_results)
        self.fit_results_ = fit_results

        labels = _get_labels(self.da) + ["radius"]
        coef_: pd.Series = fit_results[labels].iloc[0]
        if fit_beta:
            beta_: float = fit_results["beta"].iloc[0]
        else:
            beta_: float = beta

        # use set_coef to additionally set fit_params_
        self.set_coef(coef_, beta_=beta_)

        return self

    @keep_index
    def RED(self, X: ArrayLike) -> NDArray[np.floating]:
        """
        Calculate the Relative Energy Difference (RED) for each solvent.

        RED is the ratio of distance from the sphere center to its radius.
        Values below one are inside the fitted sphere; values above one are
        outside it. This method is provided for compatibility with HSPiP.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Solvent HSPs with three features ('dD', 'dP', 'dH') or four ('dD', 'dP', 'dH_donor', 'dH_acceptor') when donor and acceptor
            HSPs are separate.

        Returns
        -------
        ndarray of shape (n_samples,) or pandas.Series
            Relative Energy Differences. If ``X`` is a DataFrame, the returned
            Series retains its index.
        """
        check_is_fitted(self, "coef_")
        X = check_X(X, n_features=self._n_hsp, estimator=self)
        return RED(self.coef_, X)


class BatchedEstimator:
    """
    Estimator for fitting a series of spheres.

    This estimator is a glorified for-loop that handles cloning and fitting of
    a new estimator for each given dataset. A naive implementation would refit
    the same estimator each time. Each fitted estimator is stored in a list in
    the `fitted_estimators_` attribute. Predictions and scores are returned as
    a list matching the number of given datasets.

    Parameters
    ----------
    estimator : estimator object
        The estimator to use for fitting. During fitting this is cloned and
        fit to each given dataset.

    """

    def __init__(self, estimator: HSPEstimator):
        """
        Initialize a batched estimator.

        Parameters
        ----------
        estimator : HSPEstimator
            Estimator to clone and fit once for each dataset.
        """
        self.estimator = estimator

    def fit(self, X: Sequence[ArrayLike], y: Sequence[ArrayLike]):
        """
        Fit a separate estimator to each dataset.

        Parameters
        ----------
        X : sequence of array-like
            Feature datasets, one per estimator to fit.
        y : sequence of array-like
            Target arrays corresponding to each feature dataset.

        Returns
        -------
        BatchedEstimator
            The batched estimator with its fitted estimators stored in
            ``fitted_estimators_``.
        """
        self.fitted_estimators_ = []
        for single_X, single_y in zip(X, y):
            new_estimator: HSPEstimator = clone(self.estimator)  # type: ignore
            new_estimator.fit(single_X, single_y)
            self.fitted_estimators_.append(new_estimator)
        return self

    def predict(self, X: Sequence[ArrayLike]):
        """
        Predict for each dataset using its corresponding fitted estimator.

        Parameters
        ----------
        X : sequence of array-like
            Feature datasets, one for each fitted estimator.

        Returns
        -------
        list of ndarray or pandas.Series
            Predictions, one result per dataset.

        Raises
        ------
        ValueError
            If the number of datasets does not match the number of fitted
            estimators.
        """
        if len(X) != len(self.fitted_estimators_):
            raise ValueError(
                "The length of given data does not match the number of fit estimators"
            )
        predictions_list = []
        for estimator, single_X in zip(self.fitted_estimators_, X):
            predictions_list.append(estimator.predict(single_X))
        return predictions_list

    def fit_predict(self, X: Sequence[ArrayLike], y: Sequence[ArrayLike]):
        """
        Fit one estimator per dataset and return predictions for those datasets.

        Parameters
        ----------
        X : sequence of array-like
            Feature datasets, one per estimator to fit.
        y : sequence of array-like
            Target arrays corresponding to each feature dataset.

        Returns
        -------
        list of ndarray or pandas.Series
            Predictions, one result per dataset.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(
        self,
        X: Sequence[ArrayLike],
        y_true: Sequence[ArrayLike],
        *,
        scorer: str | None = None,
    ):
        """
        Score each dataset with its corresponding fitted estimator.

        Parameters
        ----------
        X : sequence of array-like
            Feature datasets, one per fitted estimator.
        y_true : sequence of array-like
            Target arrays corresponding to the feature datasets.
        scorer : str or None, default=None
            Scikit-learn scorer name to use for every estimator. If ``None``,
            each estimator's default ``score`` method is used.

        Returns
        -------
        list of float
            One score per dataset.
        """
        if scorer is None:
            scores = [
                est.score(single_X, single_y)
                for est, single_X, single_y in zip(self.fitted_estimators_, X, y_true)
            ]
        else:
            score = get_scorer(scorer)
            scores = [
                score(est, single_X, single_y)
                for est, single_X, single_y in zip(self.fitted_estimators_, X, y_true)
            ]
        return scores
