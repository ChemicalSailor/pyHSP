# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 15:10:26 2026

@author: tfletcher
"""

import numpy as np
import pandas as pd

# import matplotlib.pyplot as plt
# import os
# import argparse
from typing import Union
from pathlib import Path
from numpy.typing import NDArray

SeriesOrDataFrame = Union[pd.Series, pd.DataFrame]
TPathLike = Union[Path, str]
Numeric = float | int | NDArray[np.floating]
