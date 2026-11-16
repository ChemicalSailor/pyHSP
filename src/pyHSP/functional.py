# -*- coding: utf-8 -*-

import numpy as np

# import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import argparse

from typing import Literal
from numpy.typing import ArrayLike


def inverse_sum(a: float, b: float) -> float:
    r"""
    Returns the inverse summation, or harmonic mean, of a and b.

    .. math::
        inverse\_sum(a,b) = \dfrac{1}{\dfrac{1}{a} + \dfrac{1}{b}} = \dfrac{ab}{a+b}

    """
    r_eff = a * b / (a + b)
    return r_eff


def conc_mol_to_wt(conc_mol: float, Mw: float) -> float:
    """
    Convert mol/L to mg/mL.

    Parameters
    ----------
    conc_mol : float
        The concentration in mol/L.
    Mw : float
        The solute molecular weight.

    Returns
    -------
    conc_wt : float
        Concentration in mg/mL.

    """
    return conc_mol * Mw


def conc_wt_to_mol(conc_wt: float, Mw: float) -> float:
    """
    Convert mg/mL to molar (mol/L) concentration.

    Parameters
    ----------
    conc_wt : float
        The concentration in mg/mL.
    Mw : float
        The solute molecular weight.

    Returns
    -------
    conc_mol : float
        Concentration in mol/L.

    """
    return conc_wt / Mw


def conc_correction(x: float, benchmark: float = 0.5) -> float:
    r"""
    Calculate the radius concentration scaling factor.

    .. math::
        conc\_correction( x ) = \dfrac{[x ln(x) + (1-x) ln (1-x)][y(1-y)]}{[y ln(y) + (1-y) ln (1-y)][x(1-x)]}
    Where y is the benchmark concentration.

    Parameters
    ----------
    x : float
        The solution concentration as a mole fraction.
    benchmark : float, default=0.5
        The solution concentration (as a mole fraction) for which the benchmark
        radius is defined.

    Returns
    -------
    float
        A factor to scale solubilty radius between desired and benchmark
        concentrations.

    """
    y = benchmark
    num = (x * np.log(x) + (1 - x) * np.log(1 - x)) * y * (1 - y)
    den = (y * np.log(y) + (1 - y) * np.log(1 - y)) * x * (1 - x)
    return num / den


def molar_volume(Mw: float, density: float):
    """
    The volume occupied by one mole (MVol).

    Parameters
    ----------
    Mw : float
        The molecular weight of the substance. By convention given in g/mol.
    density : float
        The density of the substance. By convention given in g/cm3.

    Returns
    -------
    float
        The molar volume. By convention in cm3/mol.

    """
    return Mw / density


def solvent_molarity(molar_volume: float):
    """
    Calculate molarity (moles per litre) of a solvent.

    Parameters
    ----------
    molar_volume : float
        The molar volume of the solvent given in mL/mol. See `molar_volume()`.

    Returns
    -------
    float
        Molarity in moles per litre.

    """
    return 1000 / molar_volume


def mole_fraction(molar_conc: float, solvent_molarity: float) -> float:
    """
    Calculate solute concentration as mole ratio of solute to solvent.

    Parameters
    ----------
    molar_conc : float
        Solute concentration in mol/L.
    solvent_molarity : float
        Molaraity of solvent in mol/L.

    Returns
    -------
    float
    """
    total_moles = solvent_molarity + molar_conc
    return molar_conc / total_moles


def _da_dist_euclid(da_a: np.ndarray, da_b: np.ndarray) -> np.ndarray:
    """Return the squared euclidean distance of donor/acceptor interaction."""
    assert da_a.ndim == 2
    assert da_b.ndim == 2
    diff = da_a - da_b
    return (diff**2).sum(axis=1)


def _da_dist_beerbower(da_a: np.ndarray, da_b: np.ndarray) -> np.ndarray:
    """
    Return beerbower distance for donor/acceptor interaction

    For use as an alternative to the squared distance.
    """
    assert da_a.ndim == 2
    assert da_b.ndim == 2
    diff = da_a - da_b
    return 2 * diff[:, 0] * diff[:, 1]


def _da_dist_louwerse(da_a: np.ndarray, da_b: np.ndarray) -> np.ndarray:
    """
    Return distance for donor/acceptor interaction as proposed by Louwerse et al.

    Use as an alternative to squared distance in tandem with rescaled
    donor/acceptor parameters.
    """
    assert da_a.ndim == 2
    assert da_b.ndim == 2
    dist = (
        da_a[:, 0] * da_b[:, 1]
        + da_b[:, 0] * da_a[:, 1]
        - da_a[:, 0] * da_a[:, 1]
        - da_b[:, 0] * da_b[:, 1]
    )
    return -dist


def calculate_distance(
    HSP_A: ArrayLike,
    HSP_B: ArrayLike,
    *,
    root: bool = True,
    classic: bool = True,
    da: Literal[True, False, "euclid", "beerbower", "louwerse"] = False,
) -> np.ndarray:
    r"""
    Calculate distance between points in Hansen solubility space.

    Parameters
    ----------
    HSP_A, HSP_B : array-like of shape (n_features,) or (n, n_features)
        Points in Hansen solubility space. n_features is expected to be 3 (dD, dP, dH) or 4 when using donor/acceptor terms (dD, dP, dH_donor, dH_acceptor).
        When n>1 the distance calculation is broadcast
        on the first axis.
    root : bool, default=True
        Return the distance squared if False.
    classic : bool, default=True
        Include factor of 4 in dD part of distance as per classic Hansen formula.
    da : bool or str, default=False
        Use donor/acceptor interaction instead of dH. With this mode only the
        squared distance can be returned. The HSP arrays must be of shape (4,)
        or (n,4) in the order dD, dP, dH_donor, dH_acceptor. Valid arguments are
        'euclid', 'beerbower' or 'louwerse'. A value of `True` is the same as
        'euclid'. See notes for a description of each option.

    Returns
    -------
    ndarray
        The distance between the given points. When more than one entry is given in either
        `HSP_A` or `HSP_B` an array of distances is calculuated following numpy
        broadcasting rules.

    Notes
    -----
    Various methods have been proposed for including hydrogen bonding
    donor/acceptor interatcions in the distance metric, but no single method has
    yet been determined to be 'correct'. The result of all methods will depend
    on how the dH parameter is split into donor and acceptor parts and it is the
    responsibility of the user to provide suitable data. It should be
    noted that for a vast majority of problems, the 3 term HSP (dD, dP, dH) is
    sufficient. for these reasons the support for donor/acceptor interactions
    should be considered *experimental*.

    Three methods are included here:

    'euclid'
        Extends the Euclidean distance metric across 4 dimensions, such that
        :math:`dist_H^2 = (\delta HD_1 - \delta HD_2)^2
        + (\delta HA_1 - \delta HA_2)^2`. This (as far as I can tell) is the
        method used by HSPiP so is most appropriate to use when using
        donor/acceptor parameters from HSPiP.
    'beerbower'
        Uses the rule proposed by Beerbower: :math:`dist_H^2 =
        2 * (\delta HD_1 - \delta HD_2)* (\delta HA_1 - \delta HA_2)`.
        This form can result in a negative distance.
    'louwerse'
        Uses the rule proposed by Louwerse et al: :math:`dist_H^2 = -
        (\delta HD_1 \delta HA_2 + \delta HD_2 \delta HA_1 -
        \delta HD_1 \delta HA_1 - \delta HD_2 \delta HA_2)`. This form can result
        in a negative distance. Louwerse et al propose an alternative method for
        splitting dH into donor/acceptor parameters, therefore using another
        form of parameter splitting with this mehtod is unlikely to yield
        useable results.

    """
    HSP_A = np.atleast_2d(HSP_A)
    HSP_B = np.atleast_2d(HSP_B)
    diff = np.subtract(HSP_A, HSP_B)
    if classic:
        diff[:, 0] *= 2
    D_squared = diff**2
    if da:
        valid_str = ("euclid", "beerbower", "louwerse")
        if isinstance(da, str) and da not in valid_str:
            raise ValueError(
                "Accepted strings for parameter `da` are " + ", ".join(valid_str)
            )
        if root and da in ("beerbower", "louwerse"):
            raise ValueError(
                f"Specifying `root=True` and `da='{da}'` is not supported."
            )

        da_a, da_b = HSP_A[:, [2, 3]], HSP_B[:, [2, 3]]
        if da == "beerbower":
            D_squared[:, 2] = _da_dist_beerbower(da_a, da_b)
        elif da == "louwerse":
            D_squared[:, 2] = _da_dist_louwerse(da_a, da_b)
        else:
            D_squared[:, 2] = _da_dist_euclid(da_a, da_b)

        # index to eliminate last parameter in d/a case
        D_squared = D_squared[:, :3]

    dist = D_squared.sum(axis=1)
    if root:
        # root=True and da=True handled above
        dist = np.sqrt(dist)

    dist = dist.squeeze()
    # if dist.size == 1:
    #     return dist.item()
    return dist


def RED(solute_HSP: ArrayLike, solvent_HSP: ArrayLike):
    """
    Calculate the Relative Energy Difference.

    The Relative Energy Difference parametrizes the distance of a solvent
    inside or outside the solute solubility sphere. It is simply the ratio of
    the solute-solvent distance to the solute radius. Solvents with a high
    probability of dissolving the solute have RED < 0, marginal solvents
    have RED ~ 1 and poor solvents have RED > 1.

    Parameters
    ----------
    solute_HSP : array-like, shape (4,) or (n,4)
        The solubility parameters of one or many solutes, including radius.
    solvent_HSP : array-like, shape (3,) or (n,3)
        The solubility parameters of one or many solvents.

    Returns
    -------
    array of shape (n,)
        The Relative Energy Difference. When a 2D array is given for either
        solute or solvent the result is broadcast following numpy rules.

    Notes
    -----
    Calculating RED is only supported for the "classic" HSP using dD, dP and dH
    as a matter of compatibility.
    """
    solute_HSP = np.atleast_2d(solute_HSP)
    solvent_HSP = np.atleast_2d(solvent_HSP)
    radius = solute_HSP[:, 3]
    dist = calculate_distance(solute_HSP[:, :3], solvent_HSP)
    red = (dist / radius).squeeze()
    return red


# def solvent_radius(molar_volume: float):
#     """
#     Approximate the radius of the solvent sphere from molar volume.

#     Note this is the radius of the 'solubility sphere' for the solvent in
#     parameter space not the radius of the molecule (or any other physical
#     dimension). The radius is given based on 0.5 mole fraction.
#     """
#     return 1520 / molar_volume  # value given in SI for no dD correction factor
