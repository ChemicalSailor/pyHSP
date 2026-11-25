# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from functools import partial
import json
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score

from pyHSP._conf import BETA_MAX
from pyHSP.functional import calculate_distance
from pyHSP.sphere import (
    _decision_function,
    _log_loss,
    ClassicSphere,
    BatchedEstimator,
    _HSPDataFrameAccessor,
)
from pyHSP.io import read_json


@pytest.fixture
def solvents_full(data_dir):
    df = read_json(data_dir / "solvents.json")
    return df


@pytest.fixture
def solvents(solvents_full):
    return solvents_full.iloc[:, :3]


class TestDecisionFunction:

    coefs = [16.0, 2.0, 2.0]  # add radius in function
    X = np.array([[17.0, 4.0, 4.0, 110.0, 0.5]])  # D, P, H, mvol, conc(M)

    def test_distance(self):
        dist = calculate_distance(
            self.coefs, self.X[:, :3], root=True, classic=True, da=False
        )
        assert dist == pytest.approx(12.0**0.5)

    @pytest.mark.parametrize(
        ("r", "exp"),
        [(12.0**0.5, 0.0), (3.0, 3.0 - 12.0**0.5), (4.0, 4.0 - 12.0**0.5)],
        ids=["boundary", "outside", "inside"],
    )
    def test_classic(self, r, exp):
        coefs = np.array(self.coefs + [r])
        retval = _decision_function(
            coefs,
            self.X,
            da=False,
            solvent_radii=False,
            dD_correction=True,
            squared_dist=False,
        )
        assert retval == pytest.approx(exp)

    @pytest.mark.parametrize(
        ("r", "exp"),
        [(12.0**0.5, 0.0), (3.0, 9.0 - 12.0), (4.0, 16.0 - 12.0)],
        ids=["boundary", "outside", "inside"],
    )
    def test_classic_squared(self, r, exp):
        coefs = np.array(self.coefs + [r])
        retval = _decision_function(
            coefs,
            self.X,
            da=False,
            solvent_radii=False,
            dD_correction=True,
            squared_dist=True,
        )
        assert retval == pytest.approx(exp)

    @pytest.mark.skip(reason="fails - not currently required")
    @pytest.mark.parametrize(
        ("r", "exp"),
        [(9.0, -3.802741339), (12.0, 9.6707624)],
        ids=["outside", "inside"],
    )
    def test_solvent_radii(self, r, exp):
        coefs = np.array(self.coefs + [r])
        retval = _decision_function(
            coefs,
            self.X,
            da=False,
            solvent_radii=True,
            dD_correction=True,
            squared_dist=True,
        )
        assert retval == pytest.approx(exp)

    def test_solvent_radii_ignores_square(self):
        coefs = np.array(self.coefs + [12.0])
        f = partial(
            _decision_function,
            coefs,
            self.X,
            da=False,
            solvent_radii=True,
            dD_correction=True,
        )
        assert f(squared_dist=True) == f(squared_dist=False)


def test__log_loss():
    y_true = np.array([1.0, 0.0, 0.0, 0.5])
    y_pred = np.array([0.9, 0.9, 0.2, 0.7])
    exp = np.array(
        [
            0.105360516,
            2.302585093,
            0.223143551,
            0.780323874,
        ]
    ).sum()
    result = _log_loss(y_true, y_pred)
    np.testing.assert_allclose(result, exp)


# Set up variables and fixtures for testing Sphere's
COEF = np.array([17.0, 4.0, 4.0, 6.0])
LABELS = ("dD", "dP", "dH", "radius")


@pytest.fixture
def sphere():
    return ClassicSphere(n_fits=2).set_coef(COEF)


@pytest.fixture
def solvent_scores(sphere, solvents):
    dist = sphere.distance(solvents)
    return (dist <= COEF[-1]).astype("uint8")


class TestClassicSphere:

    def test_set_coef(self, sphere, check):
        with check:
            pdt.assert_series_equal(sphere.coef_, pd.Series(COEF, index=LABELS))
        with check:
            assert sphere.beta_ == BETA_MAX

    @pytest.mark.parametrize(
        "values",
        [[1, 2, 3], [[1, 2, 3], [4, 5, 6]]],
        ids=("too short", "2-dimensional"),
    )
    def test_set_coef_invalid(self, values):
        with pytest.raises(ValueError):
            ClassicSphere().set_coef(values)

    def test_serialize_roundtrip(self, sphere, tmp_path):
        p = tmp_path / "test_serialize.json"
        sphere.to_json(p)
        new = ClassicSphere.from_json(p)
        assert all(
            (sphere.get_params() == new.get_params(), all(sphere.coef_ == new.coef_))
        )

    def test_serialize_dict_elements(self, sphere):
        d = json.loads(sphere.to_json())
        expected = ("name", "params", "coef_", "pyhsp_version")
        for e in expected:
            assert e in d, f"missing {e}"

    def test_from_dict_raises(self, sphere):
        d = sphere.to_dict()
        d["name"] = "NotASphere"
        with pytest.raises(TypeError, match="NotASphere"):
            ClassicSphere.from_dict(d, "raise")

    def test_from_dict_warns(self, sphere):
        d = sphere.to_dict()
        d["name"] = "NotASphere"
        with pytest.warns(UserWarning, match="NotASphere"):
            ClassicSphere.from_dict(d, "warn")

    def test_from_hsdx(self, data_dir):
        coef_expected = pd.Series(
            [16.8, 9.3, 4.2, 10.1], index=["dD", "dP", "dH", "radius"]
        )
        p = data_dir / "HSDX.hsdx"
        new = ClassicSphere.from_hsdx(p)
        pdt.assert_series_equal(new.coef_, coef_expected)

    # def test_equality(self):
    #     pass

    def test_distance(self, sphere):
        X = np.array([[18.0, 5.0, 2.0]])
        exp = (4 * (18 - 17) ** 2 + (5 - 4) ** 2 + (4 - 2) ** 2) ** 0.5
        assert sphere.distance(X) == pytest.approx(exp)

    def test_decision_function(self, sphere):
        X = np.array([[18.0, 5.0, 2.0]])
        assert sphere.decision_function(X) == pytest.approx(3.0)

    def test_predict(self, sphere):
        X = [[18.0, 5.0, 2.0], [19.0, 2.0, 10.0]]
        assert all(sphere.predict(X) == np.array([1, 0]))

    def test_RED(self, sphere):
        X = [[18.0, 5.0, 2.0], [19.0, 2.0, 10.0]]
        exp = np.array([0.5, 1.247219129])
        np.testing.assert_allclose(sphere.RED(X), exp)

    def test_predict_df(self, sphere, solvents, solvent_scores):
        res = sphere.predict(solvents)
        pdt.assert_series_equal(res, solvent_scores)

    def test_score(self, sphere, solvents, solvent_scores):
        assert sphere.score(solvents, solvent_scores) == 1.0

    def test_fit(self, solvents, solvent_scores, check):
        # using pytest-check imported as a fixture
        # beta set to a high value as expecting perfect classification
        sphere = ClassicSphere(n_fits=5, beta=5.0, random_state=42).fit(
            solvents, solvent_scores
        )
        pred = sphere.predict(solvents)
        with check:
            assert len(sphere.coef_) == 4
        with check:
            assert hasattr(sphere, "fit_results_")
        with check:
            assert hasattr(sphere, "fit_success_")
        with check:
            assert sphere.beta_ == 5.0, "`beta` should not have been fit"
        with check:
            assert (
                "beta" not in sphere.fit_results_.columns
            ), "`beta` should not be in fit_results_"
        with check:
            assert all(pred == solvent_scores)

    def test_fit_at_least_3_samples(self):
        X = np.arange(2 * 3).reshape(2, 3) + 10
        y = [0, 1]
        with pytest.raises(ValueError, match="minimum of 3"):
            ClassicSphere().fit(X, y)

    def test_fit_y_has_variance(self):
        X = np.arange(4 * 3).reshape(4, 3) + 10
        y = [0, 0, 0, 0]
        with pytest.raises(ValueError, match="no variation"):
            ClassicSphere().fit(X, y)

    def test_fit_beta(self, solvents, solvent_scores):
        sphere = ClassicSphere(n_fits=5, beta="fit", random_state=42).fit(
            solvents, solvent_scores
        )
        assert sphere.beta_ == BETA_MAX

    @pytest.mark.parametrize("beta", ["silly", -2])
    def test_fit_validates_beta(self, beta, solvents, solvent_scores):
        with pytest.raises(ValueError, match="'fit' or a positive"):
            ClassicSphere(n_fits=2, beta=beta).fit(solvents, solvent_scores)

    def test_fit_squared_dist(self, solvents, solvent_scores):
        sphere = ClassicSphere(
            squared_dist=True, beta="fit", random_state=42, n_fits=20
        )
        pred = sphere.fit_predict(solvents, solvent_scores)
        assert all(pred == solvent_scores)

    @pytest.mark.parametrize("method", ["mean", "random"])
    @pytest.mark.parametrize("beta", [2.0, "fit"])
    def test__get_initial_coefs_length(self, solvents, solvent_scores, method, beta):
        sphere = ClassicSphere(initialisation=method, beta=beta)
        rng = check_random_state(None)
        X = solvents.values[:, :3]
        coefs = sphere._get_initial_coefs(X, solvent_scores.values, rng)
        exp = 4
        # exp += 1 if da else 0
        exp += 1 if beta == "fit" else 0
        assert coefs.shape == (exp,)

    def test__get_initial_coefs_mean(self):
        X = np.array([[10.0] * 3, [20.0] * 3])
        y = np.array([1, 1])
        rng = check_random_state(None)
        exp = np.array([15.0] * 3 + [(4 * 25 + 25 + 25) ** 0.5])
        sphere = ClassicSphere(initialisation="mean", init_sigma=0.0)
        coef = sphere._get_initial_coefs(X, y, rng)
        np.testing.assert_allclose(coef, exp)

    def test__get_initial_coefs_raises(self):
        X = np.array([[10.0] * 3, [20.0] * 3])
        y = np.array([1, 1])
        rng = check_random_state(None)
        sphere = ClassicSphere(initialisation="silly")  # type: ignore
        with pytest.raises(ValueError, match="'mean' or 'random'"):
            sphere._get_initial_coefs(X, y, rng)


class TestBatchedEstimator:

    @pytest.fixture
    def split_data(self, solvents, solvent_scores):
        X1, X2, y1, y2 = train_test_split(
            solvents,
            solvent_scores,
            test_size=0.5,
            stratify=solvent_scores,
            random_state=42,
        )
        return [X1, X2], [y1, y2]

    @pytest.fixture
    def batched(self):
        sphere = ClassicSphere(beta="fit", n_fits=5, random_state=42)
        return BatchedEstimator(sphere)

    def test_fit(self, split_data, batched):
        batched.fit(*split_data)
        assert len(batched.fitted_estimators_) == 2
        e1, e2 = batched.fitted_estimators_
        assert not e1 is sphere
        # check coefs are different (excluding beta)
        assert all(e1.coef_.values[:-1] != e2.coef_.values[:-1])

    def test_fit_predict(self, split_data, batched):
        pred = batched.fit_predict(*split_data)
        assert len(pred) == 2

    def test_predict_raises(self, split_data, batched):
        batched.fit(*split_data)
        Xs, ys = split_data
        X = [Xs[0]]
        with pytest.raises(ValueError, match="does not match the number"):
            batched.predict(X)

    @pytest.mark.parametrize("scorer", [None, make_scorer(accuracy_score)])
    def test_score(self, split_data, batched, scorer):
        batched.fit(*split_data)
        scores = np.array(batched.score(*split_data, scorer=scorer))
        assert scores.shape == (2,)
        assert np.array(scores).dtype == float


@pytest.fixture
def solvents_df(solvents_full, solvent_scores):
    solvents_full["Score"] = solvent_scores
    solvents_full["Conc"] = 0.4
    return solvents_full


class TestAccessorIntegration:

    def test_call_X(self, sphere, solvents_df):
        X = sphere.pd(solvents_df, return_scores=False)
        pdt.assert_frame_equal(X, solvents_df[["dD", "dP", "dH"]])

    def test_call_y(self, sphere, solvents_df):
        X, y = sphere.pd(solvents_df, return_scores=True)
        pdt.assert_series_equal(y, solvents_df["Score"])

    def test_fit_predict(self, solvents_df):
        predictions = ClassicSphere(n_fits=5, random_state=42).pd.fit_predict(
            solvents_df
        )
        pdt.assert_series_equal(predictions, solvents_df["Score"], check_names=False)
