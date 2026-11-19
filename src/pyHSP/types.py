# -*- coding: utf-8 -*-

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
