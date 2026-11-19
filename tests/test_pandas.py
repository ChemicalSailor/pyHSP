# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from pyHSP.pandas import (
    validate_HSP_dataframe,
    HSP,
    index_contains,
    prepare_dataframe,
)


@pytest.fixture
def df():
    df = pd.DataFrame(
        np.empty((3, 5)),
        columns=["dD", "dP", "dH", "Mvol", "extra"],
        index=[
            "2-methyl tetrahydrofuran",
            "tetrahydrofuran (THF)",
            "xylene",
        ],
    )
    df.index.name = "Name"
    return df


class TestValidateDataFrame:

    @staticmethod
    def test_not_a_dataframe():
        obj = np.empty((5, 3))
        with pytest.raises(TypeError, match="pandas DataFrame"):
            validate_HSP_dataframe(obj)  # type: ignore

    @staticmethod
    def test_returns_valid(df):
        retval = validate_HSP_dataframe(df)
        assert retval is df

    @staticmethod
    def test_non_numeric(df):
        df["dP"] = list(map(str, df["dP"]))
        with pytest.raises(TypeError, match="non-numeric"):
            validate_HSP_dataframe(df)

    @staticmethod
    def test_missing_keys(df):
        with pytest.raises(KeyError, match="dH"):
            validate_HSP_dataframe(df.drop(columns="dH"))

    @staticmethod
    def test_missing_da(df):
        with pytest.raises(KeyError, match="dH_donor"):
            validate_HSP_dataframe(df.drop(columns="dH_donor"), da=True)

    @staticmethod
    def test_missing_mvol(df):
        with pytest.raises(KeyError, match="MVol"):
            validate_HSP_dataframe(df.drop(columns="MVol"), check_mvol=True)

    @pytest.mark.parametrize("check", [True, False])
    @staticmethod
    def test_numeric_index(df, check):
        df = df.reset_index()
        if check:
            with pytest.raises(TypeError, match="string dtype"):
                validate_HSP_dataframe(df, check_index=True)
        else:
            assert validate_HSP_dataframe(df, check_index=False) is df


class TestHSP:

    @pytest.mark.parametrize(
        "param",
        [dict(), dict(da=True), dict(mvol=True), dict(da=True, mvol=True)],
    )
    @staticmethod
    def test_passes_validation(param: dict):
        param.setdefault("da", False)
        param.setdefault("mvol", False)
        ncol = 3
        if param["da"]:
            ncol += 1
        if param["mvol"]:
            ncol += 1
        arr = np.empty((5, ncol))

        df = HSP(arr, validate=False, **param)
        assert (
            validate_HSP_dataframe(
                df, da=param["da"], check_mvol=param["mvol"], check_index=False
            )
            is df
        )

    @staticmethod
    def test_1d_array():
        arr = [10.0, 5.0, 1.0]
        df = HSP(arr, validate=False)
        assert validate_HSP_dataframe(df, check_index=False) is df

    @staticmethod
    def test_shape_mismatch():
        with pytest.raises(ValueError, match="number of columns"):
            HSP(np.empty((5, 5)))

    @staticmethod
    def test_index():
        idx = list("abcd")
        df = HSP(np.empty((4, 3)), index=idx)
        assert all(df.index == idx)


@pytest.mark.parametrize(
    ("string", "n"),
    [("thf", 1), ("tetrahydro", 2), (["methyl", "xylene"], 2), ("benzoate", 0)],
)
def test_index_contains(df, string, n):
    res = index_contains(df, string)
    assert len(res) == n


def test_index_contains_returns_mask(df):
    res = index_contains(df, "thf", return_mask=True)
    assert isinstance(res, np.ndarray)
    assert res.dtype == np.bool


def test_case_sensitive(df):
    res = index_contains(df, "thf", case_sensitive=True)
    assert len(res) == 0


class TestPrepareDataframe:

    df = pd.DataFrame(
        [
            [18.0, 7.0, 6.0, 102.0, 0.8, 10.0, 1.3, 5.7],
            [19.5, 3.0, 5.0, 124.0, 0.3, 15.0, 0.3, 4.9],
            [17.1, 0.6, 0.3, 98.0, pd.NA, 10.0, 0.1, 0.3],
        ],
        columns=[
            "dD",
            "dP",
            "dH",
            "MVol",
            "Score",
            "conc",
            "dH_donor",
            "dH_acceptor",
        ],
        index=["A", "B", "C"],
    )

    cols = {"dD", "dP", "dH"}

    def test_returns_df_and_series(self):
        X, y = prepare_dataframe(self.df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    @pytest.mark.parametrize(
        ("param", "expected_cols"),
        [
            (dict(), cols),
            (dict(mvol=True), cols | {"MVol"}),
            (dict(conc=True), cols | {"Conc"}),
            (dict(da=True), {"dD", "dP", "dH_donor", "dH_acceptor"}),
        ],
    )
    def test_expected_cols(self, param, expected_cols):
        X = prepare_dataframe(self.df, scores=False, **param)
        assert set(X.columns) == expected_cols

    @pytest.mark.parametrize("drop", [True, False])
    def test_drop_na(self, drop):
        X, y = prepare_dataframe(self.df, scores=True, drop_na=drop)
        expected_len = 2 if drop else 3
        assert len(X) == expected_len
        assert len(y) == expected_len

    @pytest.mark.parametrize("scores", (True, "conc"))
    def test_score_param(self, scores):
        if scores == "conc":
            X, y = prepare_dataframe(self.df, scores="conc")
            assert all(y == [10.0, 15.0, 10.0])
        else:
            X, y = prepare_dataframe(self.df, scores=True, drop_na=True)
            assert all(y == [0.8, 0.3])

    @pytest.mark.parametrize("conc", (True, "mvol"))
    def test_conc_param(self, conc):
        X = prepare_dataframe(self.df, scores=False, conc=conc)
        if conc == "mvol":
            assert all(X["Conc"] == [102.0, 124.0, 98.0])
        else:
            assert all(X["Conc"] == [10.0, 15.0, 10.0])

    @pytest.mark.parametrize("conc", (True, "conc"))
    def test_mvol_param(self, mvol):
        X = prepare_dataframe(self.df, scores=False, mvol=mvol)
        if mvol == "conc":
            assert all(X["MVol"] == [10.0, 15.0, 10.0])
        else:
            assert all(X["MVol"] == [102.0, 124.0, 98.0])

    def test_no_Mw_error(self):
        with pytest.raises(ValueError, match="Cannot convert"):
            prepare_dataframe(self.df, convert_to_molar=True, Mw=None, conc=True)

    def test_score_transform(self):
        X, y = prepare_dataframe(
            self.df, scores=True, drop_na=True, scores_transformer=np.round
        )
        assert all(y == [1.0, 0.0])
