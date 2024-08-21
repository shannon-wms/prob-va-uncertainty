"""
Script containing functions for plotting results of exceedance probability 
evaluation and estimation.

Author: Shannon Williams

Date: 25/09/2023
"""

import sys
import os
import iris
import numpy as np
import pandas as pd
import iris.plot as iplt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as mticker
import datetime 
import img2pdf
from iris.cube import Cube, CubeList
from iris import Constraint

from pvauncertainty.cubes import get_cube, get_log_cube, member_prob_cube

# Colour maps
bupu_cmap = cm.get_cmap("brewer_BuPu_09")
bupu_colors = [colors.rgb2hex(bupu_cmap(i)) for i in (2, 4, 6, 7, 8)]
bupu_colors.insert(0, "#ffffff")

magma_cmap = cm.get_cmap("magma", 7)
magma_colors = [colors.rgb2hex(magma_cmap.reversed()(i)) 
                for i in range(magma_cmap.N)]

ylorbr_cmap = cm.get_cmap("brewer_YlOrBr_09")
ylorbr_colors = [colors.rgb2hex(ylorbr_cmap(i)) for i in range(ylorbr_cmap.N)]

def plot_cube(
    cube: Cube, 
    title: str,
    quantity: str = None,
    contour: bool = True,
    levels: tuple = (0, 1E-4, 1E-3, 1E-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1), 
    ticklabels: tuple = None,
    colors: list = bupu_colors, 
    save_file: str = None, 
    verbose: bool = True,
    **kwargs):
    """Plot a two-dimensional cube on a map with colourbar.

    Args:
        cube (Cube): Two-dimensional (latitude, longitude) slice of an iris cube.
        title (str): Figure title.
        quantity (str, optional): Quantity to plot, to label colourbar. If None, takes the Quantity attribute of cube. Defaults to None.
        contour (bool, optional): Whether to plot as contours or colormesh. Defaults to True.
        levels (tuple, optional): If contour, colorbar levels. Defaults to (0, 1E-4, 1E-3, 1E-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1).
        ticklabels (tuple, optional): Tick labels to correspond to levels. Defaults to None.
        colors (list, optional): Discrete colors to correspond to levels if contour. Defaults to bupu_colors.
        save_file (str, optional): Path to save figure as png to. If None, figure is not saved. Defaults to None.
        verbose (bool, optional): Whether to print to console. Defaults to True.
    """
    res = kwargs.pop("res", "50m")
    extent = kwargs.pop("extent", (-40, 40, 30, 80))
    cbar_axes = kwargs.pop("cbar_axes", [0.15, 0.22, 0.7, 0.02])
    cbar_format = kwargs.pop("cbar_format", "%.1e")
    rotation = kwargs.pop("rotation", 0)
    extend = kwargs.pop("extend", "neither")
    
    if quantity is None:
        quantity = cube.attributes["Quantity"]
              
    fig = plt.figure(figsize = (8, 10))
    ax = plt.subplot(1, 1, 1, projection = ccrs.PlateCarree())
    ax = _set_grid(ax, res, extent)

    # Plot contours
    if contour:
        cf = iplt.contourf(cube, levels = levels, colors = colors,
                           extend = extend) 
    else:
        cf = iplt.pcolormesh(cube, **kwargs)

    # Colour bar
    cbaxes = fig.add_axes(cbar_axes)
    cbartext = quantity
    cbar = plt.colorbar(cf,
                    cax = cbaxes,
                    orientation = "horizontal",
                    format = cbar_format, 
                    spacing = "uniform",
                    extend = extend)
    cbar.ax.tick_params(rotation = rotation)
    cbar.ax.set_xlabel(cbartext, fontsize =10) 
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)

    ax.set_title(title)

    if save_file is not None:
        fig.savefig(save_file)
        if verbose:
            print("Saved figure as " + save_file)
        fig.clf()
        plt.close()
    else:
        plt.show() 

def plot_excprobs(
    prob_cube: Cube, 
    threshold: float, 
    h_km: float = None, 
    fl_index: int = None, 
    fl: str = None, 
    time_index: int = None, 
    date_stamp: str = None, 
    levels: tuple = (0, 1E-3, 1E-2, 0.1, 0.25, 0.5, 1),
    ticklabels: tuple = (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.01$", 
                          r"$0.25$", r"$0.5$", r"$1$"),
    save_file: str = None, 
    **kwargs):
    """Plot exceedance probability for a given threshold at a specified time 
    and FL.

    Args:
        prob_cube (Cube): Iris cube of exceedance probability estimates. 
        threshold (float): Ash concentration threshold.
        h_km (float, optional): Plume height to be included in title. Defaults to None.
        fl_index (int, optional): Index of flight level to constrain by. Defaults to None.
        fl (str, optional): FL range to be included in title if cube does not need to be constrained, e.g. "FL000 to FL050". If fl_index is specified this will instead be generated automatically. Defaults to None.
        time_index (int, optional): Index of time point to constrain by. Defaults to None.
        date_stamp (str, optional): Date stamp to be included in title if cube does not need to be constrained. If time_index is specified this will instead be generated automatically.. Defaults to None.
        levels (tuple, optional): If contour, colorbar levels. Defaults to (0, 1E-4, 1E-3, 1E-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1).
        ticklabels (tuple, optional): Tick labels to correspond to levels. Defaults to (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.01$", r"$0.25$", r"$0.5$", r"$1$").
        save_file (str, optional): Path to save figure as png to. If None, figure is not saved. Defaults to None.
    """
    colors = kwargs.pop("colors", bupu_colors)
    # Slice by time, FL, threshold
    prob_cube, title = _constrain_cube_for_plot(
        prob_cube, threshold, quantity = "Exceedance Probability", h_km = h_km,
        time_index = time_index, date_stamp = date_stamp, 
        fl_index = fl_index, fl = fl)
    
    plot_cube(prob_cube, title, levels = levels, ticklabels = ticklabels, 
              colors = colors, save_file = save_file, **kwargs)

def plot_mult_excprobs(
    probs: Cube | str, 
    workdir: str, 
    h_km: float = None, 
    z_level: list = None, 
    time_inds: list = None,
    thresholds: tuple = None, 
    **kwargs):
    """Plot exceedance probability maps for each threshold, time and flight level, given output cube with these as coordinates.

    Args:
        probs (Cube | str): Cube of exceedance probabilities, or path of netCDF file to load this cube.
        workdir (str): Directory to save png files to.
        h_km (float, optional): Plume height to be added to title. Defaults to None.
        z_level (list, optional): List of flight levels to plot. If None, all are plotted and saved. Defaults to None.
        time_inds (list, optional): List of time indexes to plot. If None, all are plotted and saved. Defaults to None.
        thresholds (tuple, optional): List of thresholds to plot. If None, all are plotted and saved. Defaults to None.
    """
    # Create directory to save outputs
    exc_dir = workdir + "/excprobs"
    if not os.path.exists(exc_dir):
        os.makedirs(exc_dir)
        print("Created directory " + exc_dir)
    
    if isinstance(probs, str):
        probs = iris.load(probs)[0]

    # Time coordinates to datestamps
    t_coord = probs.coord("time")
    date_stamps = _get_datestamps(t_coord = t_coord)

    # Subset by flight level
    if z_level is not None:
        if isinstance(z_level, int):
            z_level = [z_level]
        probs = probs.extract(
            Constraint(flight_level = lambda cell: cell in z_level))

    fl_coord = probs.coord("flight_level")
    fl_bounds = fl_coord.bounds
    fl_points = fl_coord.points

    # Subset by time 
    if time_inds is not None:
        if isinstance(time_inds, int):
            time_inds = [time_inds]
        times = [t_coord.units.num2date(t_coord.points[i]) for i in time_inds]
        probs = probs.extract(
            Constraint(time = lambda t: t.point in times))
    
    # Subset thresholds
    if thresholds is not None and isinstance(thresholds, float):
            thresholds = tuple([thresholds])
    elif thresholds is None:
        thresholds = probs.coord("threshold").points

    # For each threshold
    for threshold in thresholds: 
        # For each time instance
        for j in range(len(probs.coord("time").points)): 
            date_stamp = date_stamps[j]
            prob_cube = probs.extract(Constraint(
                time = lambda cell:
                    cell.point == t_coord.units.num2date(t_coord.points[j])))
            
            # For each flight level
            for k, fl_bound in enumerate(fl_bounds): 
                fl_title = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
                fl_file = f"FL{int(fl_bound[0]):03d}-{int(fl_bound[1]):03d}"
                save_file = (exc_dir + f"/PROB_EXC_{threshold:.0E}_" + 
                             fl_file + "_" + date_stamp + ".png")
                
                plot_cube = prob_cube.extract(
                    Constraint(
                        flight_level = lambda cell: cell.point == fl_points[k]))

                plot_excprobs(plot_cube, threshold = threshold, h_km = h_km, 
                              fl = fl_title, date_stamp = date_stamp, 
                              save_file = save_file, **kwargs)

def plot_ppfs(
    ppf_cubes: Cube, 
    h_km: float = None, 
    fl_index: int = None,
    fl: str = None, 
    time_index: int = None,
    date_stamp: str = None, 
    levels: tuple = (-5, -4, -3, -2, -1, 0, 1), 
    save_file: str = None, 
    verbose: bool = True,
    **kwargs):
    """Plot ash concentration percentile cube.

    Args:
        ppf_cubes (Cube): Iris cube with percentile coordinate.
        h_km (float, optional): Plume height to be included in title. Defaults to None.
        fl_index (int, optional): Index of flight level to constrain by. Defaults to None.
        fl (str, optional): FL range to be included in title if cube does not need to be constrained, e.g. "FL000 to FL050". If fl_index is specified this will instead be generated automatically. Defaults to None.
        time_index (int, optional): Index of time point to constrain by. Defaults to None.
        date_stamp (str, optional): Date stamp to be included in title if cube does not need to be constrained. If time_index is specified this will instead be generated automatically.. Defaults to None.
        levels (tuple, optional): If contour, colorbar levels. Defaults to (-5, -4, -3, -2, -1, 0, 1).
        save_file (str, optional): Path to save figure as png to. If None, figure is not saved. Defaults to None.
        verbose (bool, optional): Whether to print to console. Defaults to True.
    """

    colors = kwargs.pop("colors", magma_colors)
    ticklabels = kwargs.pop("ticklabels", None)
    cbar_axes = kwargs.pop("cbar_axes", [0.38, 0.1, 0.27, 0.02])

    percentiles = ppf_cubes.coord("percentile").points
    n_percentiles = len(percentiles)

    ppf_cubes, title = _constrain_cube_for_plot(
        ppf_cubes, threshold = None, 
        quantity = "Ash concentration percentiles", 
        h_km = h_km, time_index = time_index, date_stamp = date_stamp,
        fl_index = fl_index, fl = fl)

    fig = plt.figure(figsize = (15, 5))
    for i in range(n_percentiles):
        ppf_cube = ppf_cubes.extract(Constraint(
            percentile = lambda cell: cell == percentiles[i]))
        ppf_cube.data[ppf_cube.data == -np.inf] = np.nan

        ax = plt.subplot(1, n_percentiles, i+1, projection = ccrs.PlateCarree())
        ax = _set_grid(ax)
        
        # Plot contours
        pwr_cube = ppf_cube.copy()
        pwr_cube.data = np.power(10, ppf_cube.data) * 1E+6 # convert to ug/m^3
        log_levels = np.power(10, np.array(levels).astype(float)) * 1E+6 
        cf = iplt.contourf(pwr_cube, 
                           levels = log_levels, 
                           extend = "both", 
                           colors = colors, 
                           **kwargs) 

        ax.set_title(r"{}th percentile".format(int(percentiles[i] * 100)))

    # Colour bar
    cbaxes = fig.add_axes(cbar_axes)
    cbartext = "Ash concentration" + r" (ug m$^{-3}$)"
    cbar = plt.colorbar(cf,
                        cax = cbaxes, 
                        orientation = "horizontal",
                        spacing = "uniform")
    cbar.ax.set_xlabel(cbartext, fontsize =10) 
    cbar.ax.set_xscale("log")

    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)
    else:
        cbar.ax.xaxis.set_major_locator(
            mticker.LogLocator(10, numticks = len(levels)))
    cbar.ax.minorticks_off()
    
    fig.suptitle(title)

    if save_file is not None:
        fig.savefig(save_file)
        if verbose:
            print("Saved figure as " + save_file)
        fig.clf()
        plt.close()
    else:
        plt.show() 

def plot_mult_ppfs(
    ppfs: Cube | str, 
    workdir: str, 
    h_km: float = None, 
    z_level: int = None, 
    time_inds: list = None,
    **kwargs):
    """Plot ash concentration percentile maps for time and flight level, given output cube with these as coordinates.

    Args:
        probs (Cube | str): Cube of ash concentration percentiles, or path of netCDF file to load this cube.
        workdir (str): Directory to save png files to.
        h_km (float, optional): Plume height to be added to title. Defaults to None.
        z_level (list, optional): List of flight levels to plot. If None, all are plotted and saved. Defaults to None.
        time_inds (list, optional): List of time indexes to plot. If None, all are plotted and saved. Defaults to None.
    """
    ppf_dir = workdir + "/percentiles"
    if not os.path.exists(ppf_dir):
        os.makedirs(ppf_dir)
        print("Created directory " + ppf_dir)
    
    if isinstance(ppfs, str):
        ppfs = iris.load(ppfs)[0]
        
    # Time coordinates to datestamps
    t_coord = ppfs[0].coord("time")
    date_stamps = _get_datestamps(t_coord = t_coord) 

    # Subset by flight level
    if z_level is not None:
        if isinstance(z_level, float):
            z_level = [z_level]
        ppfs = ppfs.extract(
            Constraint(flight_level = lambda cell: cell in z_level))

    fl_coord = ppfs.coord("flight_level")
    fl_bounds = fl_coord.bounds
    fl_points = fl_coord.points

    # Subset by time 
    if time_inds is not None:
        times = [t_coord.units.num2date(t_coord.points[i]) for i in time_inds]
        ppfs = ppfs.extract(Constraint(time = lambda t: t.point in times))        

    for j in range(len(ppfs.coord("time").points)): # For each time instance
        date_stamp = date_stamps[j]
        ppf_cube = ppfs.extract(Constraint(
            time = lambda cell:
                cell.point == t_coord.units.num2date(t_coord.points[j])))
        
        for k, fl_bound in enumerate(fl_bounds): # For each flight level
            fl_title = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
            fl_file = f"FL{int(fl_bound[0]):03d}-{int(fl_bound[1]):03d}"
            save_file =  (ppf_dir + f"/PERCENTILE_PLOT_" + fl_file + "_" + 
                          date_stamp + ".png")
            
            plot_cube = ppf_cube.extract(
                Constraint(
                    flight_level = lambda cell: cell.point == fl_points[k]))

            plot_ppfs(plot_cube, h_km = h_km, fl = fl_title, 
                      date_stamp = date_stamp, save_file = save_file, **kwargs)

def plot_cis(
    ci_cubes: Cube, 
    threshold: float,
    log: bool = False,
    h_km: float = None,
    fl_index: int = None,
    fl: str = None, 
    time_index: int = None,
    date_stamp: str = None, 
    save_file: str = None, 
    verbose: bool = True, 
    **kwargs):
    """Plot confidence intervals for exceedance probability estimates.

    Args:
        ci_cubes (Cube): Cube with confidence_level coordinate containing 3 points: lower limit, exceedance probability estimate, and upper limit.
        threshold (float): Threshold to constrain cube by.
        log (bool, optional): Whether ci_cube is . Defaults to False.
        h_km (float, optional): Plume height to be included in title. Defaults to None.
        fl_index (int, optional): Index of flight level to constrain by. Defaults to None.
        fl (str, optional): FL range to be included in title if cube does not need to be constrained, e.g. "FL000 to FL050". If fl_index is specified this will instead be generated automatically. Defaults to None.
        time_index (int, optional): Index of time point to constrain by. Defaults to None.
        date_stamp (str, optional): Date stamp to be included in title if cube does not need to be constrained. If time_index is specified this will instead be generated automatically.. Defaults to None.
        save_file (str, optional): Path to save figure as png to. If None, figure is not saved. Defaults to None.
        verbose (bool, optional): Whether to print to console. Defaults to True.
    """

    ticklabels = kwargs.pop("ticklabels", None)
    cmap = kwargs.pop("cmap", "BuPu")
    cbar_axes = kwargs.pop("cbar_axes", [0.38, 0.1, 0.27, 0.04])
    extend = kwargs.pop("extend", "min" if log else "neither")
    vmin = kwargs.pop("vmin", -3 if log else 0)
    vmax = kwargs.pop("vmax", 0 if log else 1)
    # cbar_format = kwargs.pop("cbar_format", "%.1f")
     
    # Get confidence levels
    levels = list(ci_cubes.coord("confidence_level").points)
    levels.sort()
    n_levels = len(levels)
    ci_level = ci_cubes.attributes["Confidence Level"]
    quantity = (f"{ci_level} CI for " + 
                f"{'log-' if log else ''}exceedance probability")

    # Construct plot titles
    ci_cubes, title = _constrain_cube_for_plot(
        ci_cubes, threshold = threshold, quantity = quantity, 
        h_km = h_km, time_index = time_index, date_stamp = date_stamp,
        fl_index = fl_index, fl = fl)
    titles = (f"Lower {ci_level} limit", "Exceedance probability estimate", 
              f"Upper {ci_level} limit")

    fig = plt.figure(figsize = (15, 5))
    
    for i, level in enumerate(levels):
        ci_cube = ci_cubes.extract(Constraint(
            confidence_level = lambda cell: cell == level))

        ax = plt.subplot(1, n_levels, i+1, projection = ccrs.PlateCarree())
        ax = _set_grid(ax)
        
        # Plot contours
        cf = iplt.pcolormesh(ci_cube, vmin = vmin, vmax = vmax, cmap = cmap,
                             **kwargs) 

        ax.set_title(titles[i])

    # Colour bar
    cbaxes = fig.add_axes(cbar_axes)
    cbartext = f"{'Log-e' if log else 'E'}xceedance Probability"
    cbar = plt.colorbar(cf,
                        cax = cbaxes, 
                        format = "%.1f",
                        orientation = "horizontal",
                        spacing = "uniform",
                        extend = extend)
    cbar.ax.set_xlabel(cbartext, fontsize = 10) 

    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)

    fig.suptitle(title)

    if save_file is not None:
        fig.savefig(save_file)
        if verbose:
            print("Saved figure as " + save_file)
        fig.clf()
        plt.close()
    else:
        plt.show() 

def load_and_plot(out_dir: str, esp_csv: str = None, **kwargs):
    """Carry out plot_mult_excprobs for all samples in a NAME output directory.

    Args:
        out_dir (str): Directory of NAME outputs containing subdirectories of samples.
        esp_csv (str): Csv file containing eruption source parameters for each point. Defaults to None.
    """
    sample_dirs = [f.path for f in os.scandir(out_dir) if f.is_dir() 
                   and f.name.startswith("sample")]
    if esp_csv is not None:
        data = pd.read_csv(esp_csv)
        h_km = data["H (km asl)"]
    else:
        h_km = [None] * len(sample_dirs)

    for i, dir in enumerate(sample_dirs):
        probs = dir + "/exc_prob_cube.nc"
        # Load iris cube
        try: 
            plot_mult_excprobs(probs, dir, h_km[i], **kwargs)
        except OSError:
            print(f"Failed to load {probs}. Moving to next file")
            continue
        
def plot_member_probs(work_dir: str, 
                      fl_index: int, 
                      time_index: int, 
                      threshold: float, 
                      df: int,
                      sigma: float,
                      h_km: float = None,
                      levels: tuple = (0, 1E-3, 1E-2, 0.1, 0.25, 0.5, 1), 
                      ticklabels: tuple = (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.01$", 
                                           r"$0.25$", r"$0.5$", r"$1$"),
                      save_file: str = None, 
                      verbose: bool = True,
                      **kwargs):
    """Plot exceedance probabilities for each ensemble member on a grid, for a threshold, time and FL combination.

    Args:
        work_dir (str): Directory containing NAME output files.
        fl_index (int): Index of flight level to constrain by. Defaults to None.
        time_index (int): Index of time point to constrain by. Defaults to None.
        threshold (float): Ash concentration threshold.
        df (int): Degrees of freedom in t-distribution.
        sigma (float): Scale parameter in t-distribution.
        h_km (float, optional): Plume height in km a.s.l. Defaults to None.
        levels (tuple, optional): If contour, colorbar levels. Defaults to (0, 1E-4, 1E-3, 1E-2, 0.05, 0.1, 0.25, 0.5, 0.75, 1).
        ticklabels (tuple, optional): Tick labels to correspond to levels. Defaults to None.
        save_file (str, optional): Path to save figure as png to. If None, figure is not saved. Defaults to None.
        verbose (bool, optional): Whether to print to console. Defaults to True.
    """
    colors = kwargs.pop("colors", bupu_colors)
                      
    member_dirs = [f.path for f in os.scandir(work_dir) if f.is_dir() 
                   and f.name.startswith("member")]
    
    fig = plt.figure(figsize = (19, 11))
    thr_constraint = Constraint(threshold = lambda cell: cell == threshold)

    for i, member_dir in enumerate(member_dirs):
        cube = get_cube(member_dir, **kwargs)
        log_cube = get_log_cube(cube)
        prob_cube = member_prob_cube(log_cube, df, sigma, h_km, [threshold])

        if i == 0:
            t_coord = prob_cube.coord("time")
            if len(t_coord.points) > 1:
                t_constraint = Constraint(time = lambda cell:
                    cell.point == t_coord.units.num2date(
                        t_coord.points[time_index]))
            date_stamp = _get_datestamps(t_coord = t_coord)[time_index]

            # Get FL constraint
            fl_coord = prob_cube.coord("flight_level")
            fl_bounds = fl_coord.bounds
            fl_bound = fl_bounds[fl_index]
            fl_points = fl_coord.points
            fl_constraint = Constraint(flight_level = lambda cell: 
                cell.point == fl_points[fl_index])
            fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"

        prob_cube = prob_cube.extract(thr_constraint & fl_constraint & 
                                      t_constraint)
        
        # Assumes 18 ensemble members
        ax = plt.subplot(4, 5, i+1, projection = ccrs.PlateCarree())
        ax = _set_grid(ax)

        cf = iplt.contourf(prob_cube, levels = levels, colors = colors) 

        ax.set_title("Member " + str(i))
        print(f"Member {i} done.")

    quantity = prob_cube.attributes["Quantity"]

    cbaxes = fig.add_axes([0.6, 0.2, 0.3, 0.02])

    cbar = plt.colorbar(cf,
                    cax = cbaxes,
                    orientation = "horizontal",
                    format = "%.1e", 
                    spacing = "uniform")
    cbar.ax.tick_params(rotation = 90)
    cbar.ax.set_xlabel(quantity, fontsize = 10) 
    if ticklabels is not None:
        cbar.ax.set_xticklabels(ticklabels)

    title = _get_title(quantity, fl, date_stamp, h_km, threshold)
    fig.suptitle(title)

    if save_file is not None:
        fig.savefig(save_file)
        if verbose:
            print("Saved figure as " + save_file)
        fig.clf()
        plt.close()
    else:
        plt.show() 

def png_to_pdf(work_dir: str, fls: list = None, file_root: str = "PROB_EXC", 
               out_dir: str = None, fl_height: float = 50, 
               thresholds: tuple = (2E-4, 2E-3, 5E-3, 1E-2)):
    """For a directory containing png files created by plot_mult_excprobs or plot_mult_ppfs, create pdf files for the evolution of the measure through time for a specified threshold and FL.

    Args:
        work_dir (str): Directory containing NAME output files.
        fl (list): List of FLs.
        file_root (str, optional): A text string common to all files to be loaded. Defaults to "Fields_grid99*". Defaults to "PROB_EXC".
        out_dir (str, optional): Directory for .pdf files to be saved to. If None, defaults to work_dir. Defaults to None.
        fl_height (float, optional): Height of each FL. Defaults to 50.
        thresholds (tuple, optional): Ash concentration thresholds. Defaults to (2E-4, 2E-3, 5E-3, 1E-2).
    """
    if fls is not None:
        if not isinstance(fls, list):
            fls = [fls]
        
    if thresholds is None:
        thresholds = range(1)
                       
    if file_root[-1] != "_":
        file_root += "_"

    for fl in fls:
        for threshold in thresholds:
            fl_range = (f"FL{int(fl - fl_height / 2):03d}-" +
            f"{int(fl + fl_height / 2):03d}")
                
            if file_root == "PROB_EXC_":
                file_name = file_root + f"{threshold:.0E}_" + fl_range
            else:
                file_name = file_root + fl_range
            
            try:
                files = [f.path for f in os.scandir(work_dir) 
                         if f.name.startswith(file_name) 
                         and f.name.endswith(".png")]
            except FileNotFoundError:
                print(f"Files beginning with {file_name} not found in " + 
                      f"{work_dir}.")
                continue
                
            files.sort()
            
            if len(files) > 0:
                if out_dir is None:
                    out_dir = work_dir
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                # Generate filename
                pdf_file = out_dir + "/" + file_name + ".pdf"
                # Save all images to pdf
                with open(pdf_file, "wb") as pdf:
                    pdf.write(img2pdf.convert(files))
                print(f"Saved images to {pdf_file}.")
            else: 
                print(f"Could not find .png files in {work_dir} beginning " +
                      f"with {file_name}.")

def _set_grid(ax, res: str = "50m", extent: tuple = (-40, 40, 30, 80),
              top = False, right = False, bottom = True, left = True,
              alpha = 0.9, lty = "solid"):
    """Add latitude-longitude grid and outline of countries to axis.
    """

    gl = ax.gridlines(draw_labels = True, linewidth = 0.8, alpha = alpha, 
                      linestyle = lty)
    gl.top_labels = top
    gl.right_labels = right
    gl.bottom_labels = bottom
    gl.left_labels = left   

    gl.xlocator = mticker.FixedLocator(
        list(np.linspace(extent[0], extent[1], num = 9))) 
    gl.ylocator = mticker.FixedLocator(
        list(np.linspace(extent[2], extent[3], num = 6))) 
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    countries = cfeature.NaturalEarthFeature(
        category = "cultural",
        name = "admin_0_countries_lakes",
        scale = res,
        facecolor = "none")
    ax.add_feature(countries, edgecolor = "black", #zorder = 2, 
                   linewidth = 0.7)
    ax.set_extent(extent)
    
    return ax

def _constrain_cube_for_plot(cube: Cube, threshold: float = None, 
                             quantity: str = None, h_km: float = None,
                             time_index: int = None, date_stamp: str = None, 
                             fl_index: int = None, fl: str = None):
    """Constrain cube by time, FL and threshold, and construct title for plot.
    """
    
    if time_index is not None:
        t_coord = cube.coord("time")
        t_points = t_coord.points
        # Subset cube by this coordinate
        if len(t_points) > 1:
            cube = cube.extract(
                Constraint(
                    time = lambda cell: 
                        cell.point == t_coord.units.num2date(
                            t_points[time_index])))
        date_stamp = _get_datestamps(t_coord = t_coord)[time_index]
    elif date_stamp is None:
        raise ValueError("Provide one of time_index or date_stamp.")

    # Slice by FL
    if fl_index is not None:
        fl_coord = cube.coord("flight_level")
        fl_bounds = fl_coord.bounds
        fl_bound = fl_bounds[fl_index]
        fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
        # Subset by FL
        fl_points = fl_coord.points
        if len(fl_points) > 1:
            cube = cube.extract(
                Constraint(
                    flight_level = lambda cell: 
                        cell.point == fl_points[fl_index]))
    elif fl is None:
        raise ValueError("Provide one of fl_index or fl.")
    
    if threshold is not None and len(cube.coord("threshold").points) > 1:
        cube = cube.extract(
            Constraint(threshold = lambda cell: cell == threshold))
    
    # Generate plot title
    if quantity is None:
        quantity = cube.attributes["Quantity"]

    title = _get_title(quantity, fl, date_stamp, h_km, threshold)

    return cube, title

def _get_title(quantity: str, fl: str, date_stamp = str, 
               h_km: float = None, threshold: float = None, 
               hours_since: int = None):
    """Generate title string for plot."""
    h_title = (r" given  $H={h}$ km asl".format(h = h_km) if h_km is not None 
               else "")
    th_title = (r" of {t}".format(t = threshold * 1E+6) + r" ug m$^{-3}$" 
                if threshold is not None else "")
    hours_title = (f" (T+{hours_since})" if hours_since is not None else "")
    title = (f"{quantity}" + th_title + f" at {fl}" + h_title +
             f"\nValid for {date_stamp}" + hours_title)
             
    return title

def _get_datestamps(cube = None, t_coord = None, time_since_start = False):
    """Generate list of date stamps for plot."""
    if cube is not None:
        t_coord = cube.coord("time")
    elif t_coord is None:
        raise ValueError("Provide one of cube or t_coord.")
    
    dts = [datetime.datetime.utcfromtimestamp(time * 60 ** 2) 
           for time in t_coord.points]
    date_stamps = [dt.strftime("%Y%m%d%H%M") for dt in dts]

    if time_since_start:
        start_time = datetime.datetime.strptime(
            cube.attributes["Start of release"], "%H%MUTC %d/%m/%Y")
        hours = [int((dt - start_time) / datetime.timedelta(hours = 1)) 
                 for dt in dts]
        return date_stamps, hours 
    else:
        return date_stamps
