# -*- coding: utf-8 -*-

from pathlib import Path
from typing import Optional, overload
import xml.etree.ElementTree as ET
import re
from functools import partial
import pandas as pd

from .types import TPathLike, SeriesOrDataFrame
from .pandas import _get_labels


def read_hsdx(
    file: TPathLike,
    na_values: list[str] = ["-", " ", "na", "nan", "-/-", "- / -", "*"],
) -> pd.DataFrame:
    """
    Read solvent data from .hsdx file into a dataframe.

    Parameters
    ----------
    file : path-like
        The location of the .hsdx file.
    na_values : list[str]
        Values interpreted as null.

    Returns
    -------
    pandas.DataFrame
        Solvent data in pyHSP dataframe format.

    Notes
    -----
    The current implementation is not secure against malicious XML and should
    not be used on untrusted files.

    """
    tree = ET.parse(file)
    chemicals = tree.iter("Chemical")

    # filter out entries that don't have a value for dD, i.e. the result entry
    def check_dD_value(e):
        dD = e.find("δD")
        if dD is None:
            return False
        return bool(dD.text)

    chemicals = filter(check_dD_value, chemicals)

    cols = {
        "Name": "Solvent",
        "dD": "δD",
        "dP": "δP",
        "dH": "δH",
        "dHD/A": "δHD_x002F_A",
        "MVol": "MVol",
        "CAS": "CAS",
        "SMILES": "SMILES",
        "Score": "Score",
    }

    def to_dict(xml_element):
        data = {}
        for key, tag in cols.items():
            for el in xml_element.findall(tag):
                data[key] = el.text
        return data

    data = list(map(to_dict, chemicals))
    df = pd.DataFrame(data)

    # parse null values
    df = df.map(lambda val: None if val in na_values else val)

    # split donor/accepter terms
    has_DA = pd.notna(df["dHD/A"])
    split_DA = df.loc[has_DA, "dHD/A"].apply(str.split, args=("/",))
    df.loc[has_DA, "dH_donor"] = [el[0] for el in split_DA]
    df.loc[has_DA, "dH_acceptor"] = [el[1] for el in split_DA]

    # make numeric
    numeric_cols = [
        "dD",
        "dP",
        "dH",
        "MVol",
        "Score",
        "dH_donor",
        "dH_acceptor",
    ]
    df = df.astype({n: "float64" for n in numeric_cols})

    # final formatting
    # str_cols = [ 'Name', 'CAS', 'SMILES' ]
    df = df.drop(columns=["dHD/A"]).set_index("Name")

    return df


def _parse_fit_result_param(
    result: str, pattern: str, err_value: float | None = None
) -> float:
    # this regex is using positive lookbehind assertion
    # the first part in brackets matches the preceeding part
    # the second part is the pattern to return, a number optionally preceeded by a space
    _pattern = f"(?<={pattern}=) ?[0-9.]*"
    r = re.search(_pattern, result)
    if r:
        return float(r[0])
    if err_value is None:
        raise ValueError(f"Error matching pattern '{pattern}='")
    return float(err_value)


def read_hsdx_result(file: TPathLike, extended: bool = False) -> pd.Series:
    """
    Read a fit result saved in HSDX format.

    Parameters
    ----------
    file : path-like
        Location of the HSDX file.
    extended : bool, default=False
        Additionally return fit quality metrics, if present.

    Returns
    -------
    pandas.Series
        The solubility parameters in order dD, dP, dH, radius.
        If dH donor/acceptor parameters are found these are returned in place of
        dH parameter. Fit quality metrics are included as addtional entries if `extended=True`.

    """

    tree = ET.parse(file)
    result = list(tree.iter("Results"))[-1].text
    if not result or result == "-":
        raise ValueError("This file does not seem to have any fit results.")
    # result = result or ""

    parse_result = partial(_parse_fit_result_param, result)

    dD, dP, dH, R = [parse_result(s) for s in ("D", "P", "H", "R")]

    try:
        dH_donor, dH_acceptor = [parse_result(s) for s in ("HDon", "HAcc")]
        da = True
    except ValueError:
        da = False

    labels = _get_labels(da, True)
    if da:
        s = pd.Series([dD, dP, dH_donor, dH_acceptor, R], index=labels)
    else:
        s = pd.Series([dD, dP, dH, R], index=labels)

    if extended:
        for pat in ("In", "Out", "Wrong In", "Wrong Out", "Fit"):
            s[pat] = parse_result(pat, 0)

    return s


@overload
def to_json(data: SeriesOrDataFrame, path: None = None) -> str: ...


@overload
def to_json(data: SeriesOrDataFrame, path: TPathLike = ...) -> None: ...


def to_json(data: SeriesOrDataFrame, path: TPathLike | None = None):
    """
    Save data in JSON format.

    This is a convinience function wrapping the built in pandas function with
    preset parameters.

    Parameters
    ----------
    data : pandas.Series or pandas.DataFrame
        The data to save.
    path : path-like, default=None
        The location to save to. If None is passed the JSON string is returned.

    Returns
    -------
    None or str
        If path is None, returns the resulting JSON string; otherwise returns
        None.

    """
    return data.to_json(path, orient="index", indent=2)


def read_json(path: TPathLike) -> pd.DataFrame:
    """
    Read HSP data saved in JSON format.

    Parameters
    ----------
    path : path-like
        Location of the data file.

    Returns
    -------
    pandas.DataFrame
    """
    return pd.read_json(path, orient="index")


def hsdx_to_json(file: TPathLike, out: TPathLike | None = None) -> None:
    """
    Convert an HSDX file to JSON format.

    This does not preserve fit results embedded in an HSDX file. To serialize
    fit results see `Sphere`.

    Parameters
    ----------
    file : path-like
        Location of HSDX file.
    out : path-like, default=None
        Output path for the generated JSON. If not supplied the original path is
        used with '.json' file extension.

    Returns
    -------
    None
    """
    file = Path(file)
    df = read_hsdx(file)

    if out:
        to_json(df, out)
        return

    to_json(df, file.with_suffix(".json"))
    return


def hsdx_to_csv(file: TPathLike, out: TPathLike | None = None, **kws) -> None:
    """
    Convert an HSDX file to CSV format.

    Parameters
    ----------
    file : path-like
        Location of the HSDX file.
    out : path-like, default=None
        Output path for the generated CSV. If not supplied the original path is
        used with '.csv' file extension.
    **kws
        Keyword arguments passed to `DataFrame.to_csv` method.

    Returns
    -------
    None
    """
    file = Path(file)
    df = read_hsdx(file)

    if out:
        df.to_csv(out, **kws)
        return

    df.to_csv(file.with_suffix(".csv"), **kws)
    return
