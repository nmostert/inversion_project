# %%
"""
# Voronoi TGSD estimation method implimentation
"""

# %%
import copy
from tabulate import tabulate
# from pandas import ExcelWriter
from functools import reduce
from time import process_time
from scipy.optimize import minimize
from scipy.sparse.linalg import lsqr
from scipy.stats import beta, expon, truncnorm, norm, uniform
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("..")
import project.vis_utils as vis
import project.io_utils as io
import project.inversion as inv
import matplotlib.colors as colors

pd.options.display.float_format = '{:,g}'.format
plt.style.use(['ggplot'])

import logging

logging.basicConfig(level=logging.INFO)


# %%
""" ### Observation Data

Reading in Colima observation dataset and extracting grid
"""

# %%
filename = "../data/colima/colima_real_data.csv"

raw_df, grid = io.import_colima(filename)


# grid.to_csv("../data/colima/colima_grid.csv",
#             sep=" ", header=False, index=False)
grid = io.read_grid("../data/colima/colima_grid.csv")

io.print_table(raw_df.head())

io.print_table(grid)



