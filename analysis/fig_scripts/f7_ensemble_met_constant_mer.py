from figures import *
from iris.cube import CubeList
from merphuncertainty.plotting import _get_datestamps, _set_grid, _get_title, _constrain_cube_for_plot
              
volcano_height = 1725
height_asl = 12
height_avl = height_asl - (volcano_height / 1000)

levels = (0, 1E-3, 1E-2, 0.1, 0.25, 0.5, 1)
ticklabels = (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.1$", r"$0.25$", r"$0.5$", r"$1$")

# Get params from MERPH model
ivespa = set_ivespa_obs(height_avl * 1000, volcano_height)
df = ivespa.df
mu = ivespa.mu[0]
sigma = ivespa.Sigma[0]

# Evaluate MER for this height
mer_gs = 10 ** mu * 1000
mer_gs_ht = mer_gs / (height_avl * 1000)

time_index = 6
fl_index = 0

save_files = ["../data/ensemble_exc_prob.nc", 
              "../data/control_prob_cube.nc",
              "../data/ensemble_merph_exc_prob.nc",
              "../data/ensemble_merph_pop_sd.nc"]
cube_list = CubeList([iris.load(save_file)[0] for save_file in save_files])
n_rows = len(save_files)
labels = ["a", "b", "c", "d"]

fl_coord = cube_list[0].coord("flight_level")
fl_bounds = fl_coord.bounds
fl_bound = fl_bounds[fl_index]
fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
t_coord = cube_list[0].coord("time")
date_stamps, hours = _get_datestamps(cube_list[0], time_since_start = True)
date_stamp = date_stamps[time_index]
hours_since = hours[time_index]

title = (f"Exceedance Probability of Aviation Ash Concentration Thresholds and Population Standard Deviation at {fl}" +
         r" given $H={h}$ km asl ".format(h = height_asl) + 
         f"\nValid for {date_stamp} (T+{hours_since})")

fig = plt.figure(figsize = (20, 14.5))
outer = gridspec.GridSpec(6, 4, wspace = 0.0, hspace = 0.2, 
                          height_ratios = [1, 1, 1, .25, 1, .15])
n = 0
for i, cube in enumerate(cube_list):
    m = 0
    for j, threshold in enumerate(thresholds):
        # ax = plt.Subplot(fig, (inner_rows[rows[i]])[i])
        plot_cube, _ = _constrain_cube_for_plot(
            cube, threshold = threshold, 
            h_km = height_asl, time_index = time_index, 
            fl_index = fl_index)
        # Plot contours
        if i == n_rows-1 and j == 0:
            n += 1
        top = True if n == 0 else False
        right = True if m == 3 else False
        bottom = True if n == 4 else False
        left = True if m == 0 else False
        ax = fig.add_subplot(outer[n, m], projection = ccrs.PlateCarree())
        ax = _set_grid(ax, top = top, right = right, bottom = bottom, left = left,
                    alpha = 0.7)
        if i < n_rows-1:
            # Contours for exceedance probs
            cf = iplt.contourf(plot_cube, levels = levels, colors = bupu_colors,
                                extend = "neither") 
            if i == 0:
                ax.set_title(r"{t}".format(t = int(threshold * 1E+6)) + u" \u03bc"  +r"g m$^{-3}$", fontsize = 16)
        else:
            # Colourmesh for population standard deviation
            cf1 = iplt.pcolormesh(plot_cube, vmin = 0, vmax = 0.5, cmap = "bone_r")# levels = levels,
        if m == 0:
            ax.text(-0.2, 1, labels[i], transform = ax.transAxes, 
                    fontweight = "bold", fontsize = 12)

        m += 1
    n+= 1
cbaxes1 = fig.add_axes([0.3825, 0.35, 0.25, 0.015])
cbar1 = plt.colorbar(cf, cax = cbaxes1,
                orientation = "horizontal",
                spacing = "uniform",
                fraction = .05, shrink = .9, pad = 0.1)
cbar1.ax.set_xlabel("Exceedance Probability", fontsize = 14) 
cbar1.ax.set_xticklabels(ticklabels)
cbaxes2 = fig.add_axes([0.3825, 0.11, 0.25, 0.015])
cbar2 = plt.colorbar(cf1, cax = cbaxes2,
                orientation = "horizontal",
                # format = "%.1e", 
                spacing = "uniform",
                fraction = .05, shrink = .9, pad = 0.1)
cbar2.ax.set_xlabel("Population Standard Deviation", fontsize = 14) 
fig.subplots_adjust(wspace = 0.0, hspace = 0.15)
fig.suptitle(title, y = .97, fontsize = 16)

fig.savefig(fig_dir + "comparison-plot.png", format = format,
            bbox_inches = "tight")