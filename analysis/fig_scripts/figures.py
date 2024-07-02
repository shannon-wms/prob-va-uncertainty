import os
import sys
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import iris
import iris.plot as iplt
import iris.analysis.maths as maths
import numpy as np
import cartopy.crs as ccrs
import datetime as dt
from matplotlib import colors
from seaborn import color_palette
from matplotlib.colors import LinearSegmentedColormap
from merph.utilities.plot_tools import plotTitle

from pvauncertainty.plotting import *
from pvauncertainty.plotting import _get_datestamps, _set_grid, _get_title, _constrain_cube_for_plot
from pvauncertainty.ph_sampling import *

cubes_dir = "../data/name_out_cubes/"
csv_file = "../data/chunks_10_14km.csv"
fig_dir = "../../figures/"

volcano_height = 1725

# From postproc_toolbox
thresholds = [2E-4, 2E-3, 5E-3, 10E-3]
colours = ['darkorange', 'deeppink', 'blueviolet', 'mediumblue']

import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = RuntimeWarning)