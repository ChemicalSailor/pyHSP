# test_functional.py

import numpy as np
import pytest

from pyHSP.functional import (
    RED,
    calculate_distance,
    conc_correction,
    conc_mol_to_wt,
    conc_wt_to_mol,
    inverse_sum,
    molar_volume,
    mole_fraction,
    solvent_molarity,
    _da_dist_euclid,
    _da_dist_beerbower,
    _da_dist_louwerse,
)

# -----------------------------------------------------------------------------
# Simple conversion and utility functions
# -----------------------------------------------------------------------------


def test_inverse_sum():
    assert inverse_sum(2.0, 2.0) == pytest.approx(1.0)
    assert inverse_sum(4.0, 4.0) == pytest.approx(2.0)


def test_conc_wt_to_mol():
    assert conc_wt_to_mol(10.0, 100.0) == 0.1


def test_conc_mol_to_wt():
    assert conc_mol_to_wt(2.0, 100.0) == 200.0


def test_molar_volume():
    assert molar_volume(100.0, 2.0) == 50.0


def test_solvent_molarity():
    assert solvent_molarity(100.0) == 10.0


def test_mole_fraction():
    assert mole_fraction(1.0, 9.0) == 0.1


# -----------------------------------------------------------------------------
# Concentration correction
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("param", "expected"),
    [
        ((0.5, 0.5), 1.0),
        ((0.1, 0.1), 1.0),
        ((0.1, 0.5), 1.302765538),
        ((0.1, 0.3), 1.241727394),
    ],
)
def test_conc_correction(param, expected):
    assert conc_correction(*param) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("param", "expected_shape"),
    [
        (([0.5], 0.5), (1,)),
        (([0.1, 0.2], 0.5), (2,)),
        (([0.1, 0.2], [0.5, 0.6]), (2,)),
    ],
)
def test_conc_correction_vectorised(param, expected_shape):
    conc, bench = [np.asarray(p) for p in param]
    result = conc_correction(conc, bench)
    assert isinstance(result, np.ndarray)
    assert result.shape == expected_shape


@pytest.mark.parametrize(
    ("a", "b", "expected_shape"),
    [
        ([[0.5, 1.5]], [[1.5, 0.5]], (1,)),
        ([[0.5, 1.5], [0.7, 2.0]], [[1.5, 0.5]], (2,)),
        ([[0.5, 1.5], [0.7, 2.0]], [[1.5, 0.5], [1.1, 2.0]], (2,)),
    ],
)
class TestDaDistVectorised:

    @staticmethod
    def test_euclid(a, b, expected_shape):
        a, b = np.asarray(a), np.asarray(b)
        result = _da_dist_euclid(a, b)
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape

    @staticmethod
    def test_beerbower(a, b, expected_shape):
        a, b = np.asarray(a), np.asarray(b)
        result = _da_dist_beerbower(a, b)
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape

    @staticmethod
    def test_louwerse(a, b, expected_shape):
        a, b = np.asarray(a), np.asarray(b)
        result = _da_dist_louwerse(a, b)
        assert isinstance(result, np.ndarray)
        assert result.shape == expected_shape


# @pytest.fixture
# def da_points():

#     # w/w - w/w
#     # w/s - w/w
#     # s/s - w/w
#     # s/s - s/s
#     # w/s - w/s
#     # w/s - s/s
#     a = np.array([[0.3, 5.6], [0.3, 5.6], [5.6, 0.3], [5.6, 0.3], [2.0, 2.3]])

#     b = np.array([[0.5, 5.4], [5.4, 0.5], [0.5, 5.4], [5.4, 0.5], [2.0, 2.3]])
#     return a, b


class TestDaDistBehaviour:

    CASES = [
        {
            # same a and b gives 0
            "a": [[1.0, 6.0]],
            "b": [[1.0, 6.0]],
            "euclid": 0.0,
            "beerbower": 0.0,
            "louwerse": 0.0,
        },
        {
            # weak/strong & strong/strong
            # expect favourable interaction
            "a": [[1.0, 6.0]],
            "b": [[6.0, 6.0]],
            "euclid": 25.0,
            "beerbower": 0.0,
            "louwerse": 0.0,
        },
        {
            # weak/strong & strong/moderate
            # mixing is favourable for a but not for b --> moderate
            "a": [[1.0, 6.0]],
            "b": [[6.0, 3.0]],
            "euclid": 34.0,
            "beerbower": -30.0,
            "louwerse": -15.0,
        },
        {
            # weak/weak & strong/strong
            # mixing more favourable for a but not for b
            "a": [[1.0, 1.0]],
            "b": [[6.0, 6.0]],
            "euclid": 50.0,
            "beerbower": 50.0,
            "louwerse": 25.0,
        },
        {
            # weak/strong & strong/weak
            # should be strongest interaction
            "a": [[1.0, 6.0]],
            "b": [[6.0, 1.0]],
            "euclid": 50.0,
            "beerbower": -50.0,
            "louwerse": -25.0,
        },
    ]

    @pytest.mark.parametrize("case", CASES)
    def test_euclid(self, case):
        a, b = np.asarray(case["a"]), np.asarray(case["b"])
        result = _da_dist_euclid(a, b)
        expected = case["euclid"]
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("case", CASES)
    def test_beerbower(self, case):
        a, b = np.asarray(case["a"]), np.asarray(case["b"])
        result = _da_dist_beerbower(a, b)
        expected = case["beerbower"]
        np.testing.assert_allclose(result, expected)

    @pytest.mark.parametrize("case", CASES)
    def test_louwerse(self, case):
        a, b = np.asarray(case["a"]), np.asarray(case["b"])
        result = _da_dist_louwerse(a, b)
        expected = case["louwerse"]
        np.testing.assert_allclose(result, expected)


# @pytest.mark.parametrize(
#     ("hsp", "da"),
#     [
#         ([20.0, 10.0, 5.0], False),
#         ([20.0, 10.0, 0.3, 4.8], True),
#         ([20.0, 10.0, 0.3, 4.8], "euclid"),
#         ([20.0, 10.0, 0.3, 4.8], "beerbower"),
#         ([20.0, 10.0, 0.3, 4.8], "louwerse"),
#     ],
# )


class TestCalculateDistance:

    def test_distance_zero(self):
        hsp = [20.0, 7.0, 5.0]
        assert calculate_distance(hsp, hsp, da=False, root=False) == 0.0

    @pytest.mark.parametrize("classic", [True, False])
    def test_classic_param(self, classic):
        a = [20.0, 10.0, 5.0]
        b = [18.0, 10.0, 5.0]
        result = calculate_distance(a, b, classic=classic, da=False, root=True)

        expected = 4.0 if classic else 2.0
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize("root", [True, False])
    def test_root_param(self, root):
        a = [20.0, 10.0, 5.0]
        b = [18.0, 10.0, 5.0]
        result = calculate_distance(a, b, classic=False, da=False, root=root)

        expected = 2.0 if root else 4.0
        assert result == pytest.approx(expected)

    def test_symmetric(self):
        a = [18.0, 5.0, 3.0]
        b = [19.0, 7.0, 1.0]
        assert calculate_distance(a, b) == calculate_distance(b, a)

    @pytest.mark.parametrize("mode", ["euclid", "beerbower", "louwerse"])
    def test_symmetric_da(self, mode):
        a = [18.0, 5.0, 1.0, 5.0]
        b = [20.0, 2.0, 5.0, 7.0]
        assert calculate_distance(
            a, b, da=mode, root=False
        ) == calculate_distance(b, a, da=mode, root=False)

    # -----------------------------------------------------------------------------
    # calculate_distance - broadcasting behaviour
    # -----------------------------------------------------------------------------

    def test_broadcast_single_to_many(self):
        solute = [20.0, 10.0, 5.0]

        solvents = np.array(
            [
                [20.0, 10.0, 5.0],
                [19.0, 10.0, 5.0],
                [18.0, 10.0, 5.0],
            ]
        )

        result = calculate_distance(solute, solvents, classic=True)
        expected = np.array([0.0, 2.0, 4.0])
        np.testing.assert_allclose(result, expected)

    def test_pairwise_many_to_many(self):
        a = np.array(
            [
                [20.0, 10.0, 5.0],
                [21.0, 10.0, 5.0],
            ]
        )

        b = np.array(
            [
                [20.0, 10.0, 5.0],
                [20.0, 10.0, 5.0],
            ]
        )

        result = calculate_distance(a, b, classic=True)
        expected = np.array([0.0, 2.0])
        np.testing.assert_allclose(result, expected)

    def test_broadcast_symmetric(self):
        solute = [20.0, 10.0, 5.0]

        solvents = np.array(
            [
                [20.0, 10.0, 5.0],
                [19.0, 10.0, 5.0],
                [18.0, 10.0, 5.0],
            ]
        )

        np.testing.assert_allclose(
            calculate_distance(solute, solvents, classic=True),
            calculate_distance(solvents, solute, classic=True),
        )

    def test_invalid_da_mode(self):
        with pytest.raises(ValueError, match="Accepted strings"):
            calculate_distance(
                [1, 2, 3, 4],
                [1, 2, 3, 4],
                root=False,
                da="not-a-mode",
            )

    @pytest.mark.parametrize("mode", ["beerbower", "louwerse"])
    def test_da_disallows_root(self, mode):
        with pytest.raises(ValueError, match="root=True"):
            calculate_distance([1, 2, 3, 4], [1, 2, 3, 4], da=mode, root=True)


# -----------------------------------------------------------------------------
# RED
# -----------------------------------------------------------------------------


class TestRED:

    def test_zero_at_centre(self):
        solute = [20.0, 10.0, 5.0, 4.0]
        solvent = [20.0, 10.0, 5.0]
        assert RED(solute, solvent) == pytest.approx(0.0)

    def test_correct_value(self):
        solute = [20.0, 10.0, 5.0, 4.0]
        solvent = [19.0, 10.0, 5.0]
        assert RED(solute, solvent) == pytest.approx(0.5)

    def test_one_at_boundary(self):
        solute = [20.0, 10.0, 5.0, 4.0]
        solvent = [18.0, 10.0, 5.0]
        assert RED(solute, solvent) == pytest.approx(1.0)
