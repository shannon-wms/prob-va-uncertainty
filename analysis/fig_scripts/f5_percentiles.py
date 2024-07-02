import iris
from figures import *
from pvauncertainty.plotting import _get_datestamps, _set_grid, _constrain_cube_for_plot

volcano_height = 1725
height_asl = 12
height_avl = height_asl - (volcano_height / 1000)

# Get params from MERPH model
ivespa = set_ivespa_obs(height_avl * 1000, volcano_height)
df = ivespa.df
mu = ivespa.mu[0]
sigma = ivespa.Sigma[0]

time_index = 6
fl_index = 0
levels = (-6, -5, -4, -3, -2, -1, 0)
log_levels = np.power(10, np.array(levels).astype(float)) * 1E+6 

ppf_cube = iris.load("../data/control_ppf.nc")[0]
ensemble_ppf = iris.load("../data/ensemble_ppf.nc")[0]

fl_coord = ppf_cube.coord("flight_level")
fl_bounds = fl_coord.bounds
fl_bound = fl_bounds[fl_index]
fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
t_coord = ppf_cube[0].coord("time")
date_stamps, hours = _get_datestamps(ppf_cube[0], time_since_start = True)
date_stamp = date_stamps[time_index]
hours_since = hours[time_index]
title = (f"Ash Concentration Percentile Point at {fl}" +
         r" given $H={h}$ km asl ".format(h = height_asl) + 
         f"\nValid for {date_stamp} (T+{hours_since})")
# labels = ["a", "b"]

fig = plt.figure(figsize = (20, 9))
j = 0
for i in range(6):
    # Evaluate percentile
    if i < 3:
        plot_cube, _ = _constrain_cube_for_plot(
            ppf_cube, threshold = None, 
            h_km = height_asl, time_index = time_index, 
            fl_index = fl_index)
        plot_cube = plot_cube.extract(iris.Constraint(
                percentile = lambda cell: 
                    cell.point == plot_cube.coord("percentile").points[i]))
    else:
        plot_cube = ensemble_ppf.extract(iris.Constraint(
                percentile = lambda cell: 
                    cell.point == ensemble_ppf.coord("percentile").points[i-3]))
        plot_cube.data[plot_cube.data == -np.inf] = np.nan
        plot_cube.data = np.power(10, plot_cube.data) * 1E+6


    ax = plt.subplot(2, 3, i+1, projection = ccrs.PlateCarree())
    top = True if i in range(3) else False
    right = True if i in [2, 5] else False
    bottom = True if i in range(3, 6) else False
    left = True if i in [0, 3] else False

    ax = _set_grid(ax, top = top, right = right, bottom = bottom, left = left)

    # Plot contours
    cf = iplt.contourf(plot_cube, levels = log_levels, extend = "both",
                        colors = magma_colors) 

    if i < 3:
        ax.set_title(r"{}th percentile".format(int(ppf_cube.coord("percentile").points[i] * 100)),
                     fontsize = 16)
    if left:
        label = "Control met" if i < 3 else "Ensemble met"
        ax.text(-0.15, 0.5, label, va = "bottom", ha = "center", rotation = "vertical",
                rotation_mode = "anchor", transform = ax.transAxes, size = 16)
        # ax.text(-0.15, 1, labels[j], transform = ax.transAxes, 
        #         fontweight = "bold", fontsize = 12)
                
        j += 1

# Colour bar
cbaxes = fig.add_axes([0.38, 0.02, 0.27, 0.02])
cbartext = "Ash concentration" + u" (\u03bc"  +r"g m$^{-3}$)"
cbar = plt.colorbar(cf,
                    cax = cbaxes, 
                    orientation = "horizontal",
                    spacing = "uniform",
                    fraction = .05, shrink = .9, pad = 0.1)
cbar.ax.set_xlabel(cbartext, fontsize =14) 
cbar.ax.set_xscale("log")
cbar.ax.xaxis.set_major_locator(
    mticker.LogLocator(10, numticks = len(levels)))
cbar.ax.minorticks_off()
# cbar.ax.set_xticklabels(("0", "10", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$"))

fig.suptitle(title, y = 1, fontsize = 18)

fig.savefig(fig_dir + "control-met-percentiles.png", format = "png",
            bbox_inches = "tight")
