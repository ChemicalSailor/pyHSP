# -*- coding: utf-8 -*-

# prgma: no cover

from pathlib import Path
from .io import read_json

DATA_DIR = Path(__file__).parent / "data"


def load_pla():
    """
    Load polylatic acid (PLA) example.

    Returns a dataframe with 'dD', 'dP' and 'dH' columns and a series of binary scores
    (1 for soluble, 0 for insoluble).

    This data was published on the `HSPiP website
    <https://hansen-solubility.com/HSP-science/sphere.php>`__ (accessed Nov 2026).

    """
    return read_json(DATA_DIR / "pla.json")
