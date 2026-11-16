# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 16:03:10 2026

@author: tfletcher
"""

import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import os
# import argparse
from scipy.optimize import minimize

from .pyHSP import calculate_distance, validate_HSP_dataframe
from .pandas import _get_labels


def calculate_blend_HSP(
        mixture_table: pd.DataFrame,
        da: bool = False,
        check_vol: bool = True,
        volume_col = 'vol'
        ) -> pd.Series:
    """
    Calculate HSP of a mixture.

    Each row of a dataframe is interpreted as a component. An additonal column
    is expected (default 'vol') containing the ratio of the component in the
    mixture.

    Parameters
    ----------
    mixture_table : DataFrame
        Dataframe of mixture components HSP plus 'vol' column.
    da : bool, optional
        DataFrame has dH donor and accepter columns.
    check_vol : bool, optional
        Validate that the given component ratios sum to unity.

    Returns
    -------
    Series
        The blend HSP.

    """
    mixture_table = validate_HSP_dataframe(mixture_table, da=da, check_mvol=False, check_index=False)
    labels = _get_labels( da ) + [volume_col]
    HSP = mixture_table[labels]
    ratios = mixture_table.loc[:,volume_col].to_numpy().reshape(-1, 1)
    if check_vol and not np.isclose( ratios.sum(), 1 ):
        raise ValueError( 'The given component ratios do not sum to unity.' )
    return (HSP * ratios).sum(axis='rows')


def create_grid( pairs:list ):
    """
    Create a grid of binary solvent blends from given pairs.

    Using a grid of solvent blends is a good way to experimentally test solubility,
    especially when mixing known good and bad solvents.

    Parameters
    ----------
    pairs : sequence of tuple
        Pairs of solvents given in the form `(solvent1, solvent2, n, bounds)`. The solvents
        must be given as a Series with keys `dD, dP, dH, MVol`. `n` is the number of
        blends, the default is 5 if not given. `bounds` is the min/max percentage of each
        solvent and must sum to 100, the default is `(100,0)` if not given.

    Raises
    ------
    ValueError
        When tuple in pairs sequence does not contain at least two items.
    ValueError
        When bounds do not sum to 100

    Returns
    -------
    DataFrame
        HSP of calculated solvent blends.

    """
    table = []
    for spec in pairs:
        if len( spec ) < 2:
            raise ValueError( 'Expected tuple with length of at least 2, got length %d.' % len(spec) )
        if len(spec) == 2:
            solv1, solv2 = spec
            n, bounds = 5, (100,0)
        elif len(spec) == 3:
            solv1, solv2, n = spec
            if type(n) is tuple:
                bounds = n
                n = 5
            else:
                bounds = (100, 0)
        else:
            solv1, solv2, n, bounds = spec
        if sum( bounds ) != 100:
            raise ValueError( 'The bounds must sum to 100. Received {}'.format( str(bounds) ) )
        mixture_master = pd.DataFrame( [solv1, solv2] )
        blends = np.linspace( *bounds, n )
        for solv1_ratio in blends:
            mix = mixture_master.copy()
            mix['vol%'] = [ solv1_ratio, 100 - solv1_ratio ]
            mix_hsp = calculate_blend_HSP( mix )
            mix_hsp.name = '{solv1}:{solv2} {r1}:{r2}'.format( solv1=solv1.name, solv2=solv2.name, r1=solv1_ratio, r2=100-solv1_ratio )
            mix_hsp['solv1'] = solv1.name
            mix_hsp['solv2'] = solv2.name
            mix_hsp['solv1_%'] = solv1_ratio
            mix_hsp['solv2_%'] = 100 - solv1_ratio
            table.append( mix_hsp )
    table = pd.DataFrame( table )
    return table.drop_duplicates( table.columns[:4] )


def optimize_blend(
        solvents,
        target,
        *,
        mode='optimize',
        da=False,
        # reg=None,
        bounds=None,
        target_n=None,
        alpha=0.05,
        tol=1e-5,
        # eps=1e-12
        ):
    """
    Create a solvent blend matching target HSP.

    Parameters
    ----------
    solvents : dataframe
        Library of solvents used to create blend.
    target : array-like
        HSP values to match blend to.
    mode : str, 'optimize' or 'match', optional
        Algorithm to adjust solvent ratios: 'match' adjusts solvent ratios with
        equal weighting to acheive a value close to `target`; this is useful
        for finding replacement solvents. 'optimize' minimises the HSP distance
        to `target` taking into account dD and donor-acceptor corrections; this
        is useful when trying to find a solvent blend for a given solute. The
        default is 'optimize'.
    da : bool, optional
        Whether to use donor-acceptor HSP values. The default is False.
    bounds : Tuple[ Tuple, length 2 ], optional
        Bounds for solvent weights. By default the limits for all solvents are
        (0, 1).
    target_n : int, optional
        Regularisation parameter setting the number of desired solvents in the
        blend. This is a target value and the optimised blend may include more
        or less than the target number of solvents depending on the setting of
        `alpha`. The default is None.
    alpha : float, optional
        Regularisation multiplier; increasing adds stronger regularisation, but
        may lead to optimization errors if set too high. The default is 0.05.
    tol : float, optional
        Tolerance value for considering a solvent to have zero weight in a
        blend when counting solvents. The default is 1e-5.

    Raises
    ------
    ValueError
        When an unknown value is given for `mode`.
    RuntimeError
        When an optimisation fails.

    Returns
    -------
    ratios: Series, HSP: Series
        Component ratios matching `solvents`.
        HSP of the resulting blend.

    """
    mixture_table = solvents.copy()

    # function to minimise
    def blend_distance( weights:np.ndarray ) -> float:
        mixture_table['vol'] = weights
        blend_HSP = calculate_blend_HSP( mixture_table, da=da, check_vol=False )
        if mode == 'optimize':
            # Use the actual distance equation including the relative importance
            # of dD parameter and donor-acceptor interactions
            _da = da
            _classic = True
        elif mode == 'match':
            # Modifying distance arguments to return a simple squared error
            _da = False
            _classic = False
        else:
            raise ValueError( f'Unknown value `{mode}` given for `mode` argument.' )
        dist = calculate_distance( blend_HSP[:-1], target, root=False, classic=_classic, da=_da )
        # if reg=='l1':
        #     # sum of weights will always be 1, so use inverse sum where small
        #     # weights add the most to regularisation term
        #     reg_term = 1 / ( 1 / ( weights + eps ) ).sum()
        # elif reg=='l2':
        #     reg_term = (weights**2).sum()
        # else:
        #     reg_term = 0
        reg_term = 0
        if target_n:
            # maths trick to approx count how many solvents are in use whilst
            # keeping function continuous; if weight is >> threshold then
            # value ~1 and if weight << threshold then value ~0
            n_solv = np.sum( weights / ( weights + tol ) )
            reg_term += ( target_n - n_solv )**2 / mixture_table.shape[0]**2
        # if reg=='l2':
        #     reg_term += (weights**2).sum()
        return dist + alpha * reg_term

    n_solvents = solvents.shape[0]
    weights_init = np.random.random( n_solvents )
    weights_init /= weights_init.sum()

    if bounds is None:
        bounds = ( (0, 1), ) * n_solvents

    constraints = [
        { # components must sum to 100%
        'type': 'eq',
        'fun': lambda weights: 1 - np.sum( weights ),
        },
    ]
    opt_res = minimize( blend_distance, weights_init, method='SLSQP', bounds=bounds, constraints=constraints )
    if not opt_res.success:
        raise RuntimeError( "Error during optimisation", opt_res, solvents, target )

    mixture_table['vol'] = opt_res.x
    blend_HSP = calculate_blend_HSP( mixture_table, da=da )
    ratios = pd.Series( opt_res.x, index=solvents.index, name='vol' )
    return ratios, blend_HSP


# class SolventOptimizer:

#     def __init__( self, solvents, *, da=False, reg='l1', target_n=None, alpha=1, decimals=1, eps=1e-12 ):
#         self.solvents = solvents
#         self.da = da
#         self.reg = reg
#         self.target_n = target_n
#         self.aplha = alpha
#         self.decimals = decimals
#         self.eps = eps

#     # def match( self, target ):
#     #     """Create a solvent blend that matches the target HSP"""
#     #     pass

#     def optimize( self, target, *, da_match=False ):
#         """Create a solvent blend that minimizes the distance to the target."""
#         pass