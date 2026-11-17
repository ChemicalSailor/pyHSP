# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from typing import Callable, List, Union, Literal, Sequence
from numpy.typing import ArrayLike

from .types import SeriesOrDataFrame
from .functional import conc_wt_to_mol
from ._conf import HSP_COLS, HSP_DA_COLS


def _get_labels(da=False, radius=False) -> List[str]:
    labels = list(HSP_DA_COLS) if da else list(HSP_COLS)
    if radius:
        labels += ["radius"]
    return labels


def validate_HSP_dataframe(
    df: pd.DataFrame,
    da: bool = False,
    check_mvol: bool = False,
    check_index: bool = True,
) -> pd.DataFrame:
    """
    Check a dataframe complies with expected HSP format.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe to check.
    da : bool, default=False
        Check for donor/accepter dH columns.
    check_mvol : bool, default=False
        Check for an MVol column.
    check_index : bool, default=True
        Check the index is a string dtype, e.g solvent names.

    Returns
    -------
    pandas.DataFrame
        The unmodified dataframe.

    Raises
    ------
    TypeError
        If column dtypes are not numeric and if the index does not have a string dtype and ``check_index`` is True.

    """
    required_cols = _get_labels(da)
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Passed data object must be a pandas DataFrame")
    if check_mvol:
        required_cols += ["MVol"]
    # try accessing each key and let pandas raise exception for missing keys
    for key in required_cols:
        # check dtypes are numeric
        numeric = pd.api.types.is_numeric_dtype(df[key])
        if not numeric:
            raise TypeError(f"DType of column {key} is non-numeric.")
    if check_index and not pd.api.types.is_string_dtype(df.index):
        raise TypeError("DataFrame index must be of string dtype.")
    return df


def HSP(
    data: ArrayLike,
    da: bool = False,
    mvol: bool = False,
    index: list[str] | None = None,
    validate: Literal[False, "no-index", "strict"] = "no-index",
) -> pd.DataFrame:
    """
    Format an array as a dataframe with columns labelled according to pyHSP convention.

    Parameters
    ----------
    data : array-like of shape (n_features,) or (n_samples, n_features)
        Numeric Hansen solubility parameters. 1d data is converted to 2d to
        match the dataframe format. Features are assumed to be in the order dD, dP, dH.
        If ``da=True`` then features are assumed to be dD, dP, dH_donor, dH_acceptor.
    da : bool, default=False
        Treat `data` as having dH_donor and dH_acceptor columns.
    mvol : bool, default=False
        Treat `data` as having an 'MVol' column. This is expected as the last
        column.
    index : list[str], default=None
        The index to use for the dataframe, typically a list of solvent names.
    validate : {False, 'no-index', 'strict'}, default='no-index'
        Validate the dataframe structure after creation. 'strict' mode checks
        the dataframe index is a string dtype, 'no-index' disables the index
        dtype check, `False` disables all validation.

    Returns
    -------
    pandas.DataFrame
        The converted array.

    Raises
    ------
    ValueError
        When the input array does not have the same number of features defined by the ``da`` and ``mvol`` parameters.

    """
    data = np.atleast_2d(data)
    cols = _get_labels(da)
    if mvol:
        cols += ["MVol"]

    if data.shape[1] != len(cols):
        raise ValueError(
            f"Shape {data.shape} of the passed array does not "
            f"match the expected number of columns {len(cols)}"
        )

    df = pd.DataFrame(data, columns=cols, index=index)
    if validate:
        check_idx = validate == "strict"
        return validate_HSP_dataframe(df, da=da, check_mvol=mvol, check_index=check_idx)
    return df


# TODO: add an example to docs
def index_contains(
    series_or_dataframe: SeriesOrDataFrame,
    entries: str | Sequence[str],
    case_sensitive: bool = False,
    return_mask: bool = False,
):
    """
    Search for entries in the index of a series or dataframe.

    Parameters
    ----------
    series_or_dataframe : pandas.Series or pandas.DataFrame
        The series or dataframe object to search. The search is for strings
        containing the given text.
    entries : str or sequence of str
        The string or strings to search for.
    case_sensitive : bool, default=False
        Apply case-sensitive searching.
    return_mask : bool, default=False
        Return a boolean array indicating found entries.

    Returns
    -------
    pandas.Series or pandas.DataFrame or ndarray
        A new series or dataframe with any found entries. If `return_mask` is truthy and ndarray is returned.

    """

    idx = series_or_dataframe.index
    if isinstance(entries, str):
        entries = [entries]
    ma = (
        np.stack([idx.str.contains(text, case=case_sensitive) for text in entries])
        .sum(axis=0)
        .astype(bool)
    )
    if return_mask:
        return ma
    return series_or_dataframe.iloc[ma].copy()


def prepare_dataframe(
    HSP_data: pd.DataFrame,
    *,
    scores: bool | str = True,
    drop_na: bool = True,
    da: bool = False,
    inc_mvol: bool = False,
    inc_conc: bool | str = False,
    convert_to_molar: bool = False,
    Mw: float | None = None,
    scores_transformer: Callable | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """
    Prepare solubility data in a dataframe for fitting.

    Returns an estimator compatible dataframe and (optionally) solubility scores as a series.
    Columns are matched case-insensitively in the provided dataframe and only the required
    columns are returned.

    Parameters
    ----------
    HSP_data : pandas.DataFrame
        Solubility data.
    scores : bool or str, default=True
        Whether to return scores data. If True then scores are assumed to be
        in the column name 'score'. Alternatively the column name can be
        provided as a string; string matching is not case-sensitive.
    drop_na : bool, default=True
        Whether to drop rows which do not have a score. This parameter is ignored when `scores=False`.
    da : bool, default=False
        Use split hydrogen bonding term. Expects and returns dH_donor and dH_acceptor columns.
    inc_mvol : bool, default=False
        Whether to include the molar volume (mvol) column.
    inc_conc : bool or str, default=False
        Whether to include the concentration column. If `True` the column is assumed
        to be labelled 'conc'. Alternatively the column name can be provided as a
        string; string matching is not case-sensitive. The returned column in the return dataframe will always be labelled 'conc'.
    convert_to_molar : bool, default=False
        Convert the concentration (assumed to be given in mg/ml)
        to a molar concentration.
    Mw : float, default=None
        The solute molecular weight. Required when `convert_to_molar` is `True`.
    scores_transformer : callable, default=None
        Function to apply to scores. The function recieves and is expected to
        return a 1D array-like object.

    Raises
    ------
    ValueError
        When `convert_to_molar` is True and a molecular weight is not given.

    Returns
    -------
    X: pandas.DataFrame, y: pandas.Series (optional)

    """
    df = HSP_data.copy()
    df.columns = df.columns.str.lower()
    required_cols = _get_labels(da)
    if scores:
        if isinstance(scores, str):
            df["Score"] = df[scores.lower()]
        required_cols += ["Score"]
    if inc_mvol:
        required_cols += ["MVol"]
    if inc_conc:
        if isinstance(inc_conc, str):
            df["Conc"] = df[inc_conc]
        required_cols += ["Conc"]
    required_cols_lowercase = [s.lower() for s in required_cols]
    df = df.loc[:, required_cols_lowercase]
    df.columns = required_cols
    if inc_conc and convert_to_molar:
        if not Mw:
            raise ValueError(
                "Cannot convert to molar concentration without molecular weight. Provide parameter `Mw`."
            )
        df["Conc"] = conc_wt_to_mol(df["Conc"], Mw)
    if not scores:
        return df
    if drop_na:
        df = df.dropna(axis=0, subset=["Score"])
    if scores_transformer:
        df["Score"] = scores_transformer(df["Score"])
    y = df["Score"]
    X = df.drop(columns="Score")
    return X, y
