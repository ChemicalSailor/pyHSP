# -*- coding: utf-8 -*-

from typing import Optional

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
    import xml.etree.ElementTree as ET

    tree = ET.parse(file)
    chemicals = tree.iter("Chemical")
    # idx = [ solv.text for solv in tree.iter( 'Solvent' ) if solv.text ]
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

    data = [
        to_dict(chemical) for chemical in chemicals if chemical.find("Solvent").text
    ]
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
    import xml.etree.ElementTree as ET
    import re

    tree = ET.parse(file)
    result = list(tree.iter("Results"))[-1].text

    try:
        # this regex is using positive lookbehind assertion
        # the first part in brackets matches the preceeding part
        # the second part is the pattern to return, a number optionally preceeded by a space
        dD = float(re.search("(?<=D=) ?[0-9.]*", result)[0])
        dP = float(re.search("(?<=P=) ?[0-9.]*", result)[0])
        dH = float(re.search("(?<=H=) ?[0-9.]*", result)[0])
        R = float(re.search("(?<=R=) ?[0-9.]*", result)[0])

        dH_donor = re.search("(?<=HDon=) ?[0-9.]*", result)
        if dH_donor:
            dH_donor = float(dH_donor[0])
            dH_acceptor = float(re.search("(?<=HAcc=) ?[0-9.]*", result)[0])

            labels = _get_labels(da=True) + ["radius"]
            s = pd.Series([dD, dP, dH_donor, dH_acceptor, R], index=labels)
        else:
            labels = _get_labels(da=False) + ["radius"]
            s = pd.Series([dD, dP, dH, R], index=labels)

    except TypeError as e:
        if "NoneType" in e.args[0]:
            raise ValueError("This file does not seem to have a fit result.")
        else:  # pragma: no cover
            raise e

    if extended:
        s["In"] = int(re.search("(?<=In=) ?[0-9]*", result)[0])
        s["Out"] = float(re.search("(?<=Out=) ?[0-9.]*", result)[0])
        wi = re.search("(?<=Wrong In=) ?[0-9]*", result)
        s["Wrong In"] = int(wi[0]) if wi else 0
        wo = re.search("(?<=Wrong Out=) ?[0-9.]*", result)
        s["Wrong Out"] = int(wo[0]) if wo else 0
        s["Fit"] = float(re.search("(?<=Fit=) ?[0-9.]*", result)[0])

    return s


def to_json(data: SeriesOrDataFrame, path: TPathLike = None) -> Optional[str]:
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


def hsdx_to_json(file: TPathLike, out: TPathLike = None) -> None:
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
    df = read_hsdx(file)

    if out:
        to_json(df, out)
        return

    to_json(df, file.with_suffix(".json"))
    return


def hsdx_to_csv(file: TPathLike, out: TPathLike = None, **kws) -> None:
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
    df = read_hsdx(file)

    if out:
        df.to_csv(out, **kws)
        return

    df.to_csv(file.with_suffix(".csv"), **kws)
    return
