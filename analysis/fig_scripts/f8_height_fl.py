from figures import *
from merphuncertainty.plotting import _constrain_cube_for_plot, _set_grid, _get_datestamps

volcano_height = 1725
chunks = SourceChunks(volcano_height, csv_file)
chunks.name_output = VolcanicNAME("output", True, "/", csv_file)
chunks.name_output.n_members = 18
chunks.list_member_chunks = []
for member in range(18):
    ash_chunks_filename = cubes_dir + f"ash_chunks_member_{member}.nc"
    # Load ash cube list and constrain
    chunks.list_member_chunks.append(
        chunks.name_output.load_ash_cube_list(ash_chunks_filename, member))

heights_asl = [10, 12, 14]
cubes = iris.cube.CubeList([1 - eval_t_cdf(height_asl = h, chunks = chunks) for h in heights_asl])
quad_cube = iris.load(cubes_dir + "quad_exc_prob_12_sd1.nc")[0]
cubes.append(quad_cube)

time_index = 6
threshold = 0.0002
levels = (0, 1E-3, 1E-2, 0.1, 0.25, 0.5, 1)
ticklabels = (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.1$", r"$0.25$", r"$0.5$", r"$1$")
quantity = cubes[0].attributes["Quantity"]

fl_coord = cubes[0].coord("flight_level")
fl_bounds = fl_coord.bounds

t_coord = cubes[0].coord("time")
date_stamps, hours = _get_datestamps(cubes[0], time_since_start = True)
date_stamp = date_stamps[time_index]
hours_since = hours[time_index]

fig = plt.figure(figsize = (20, 17))
i = 1
for row, fl_index in enumerate([2, 4, 6, 8, 10]):
    for col, cube in enumerate(cubes):
        height = heights_asl[col] if col < 3 else ""
        plot_cube, title = _constrain_cube_for_plot(
            cube, threshold = threshold, 
            h_km = height, time_index = time_index, 
            fl_index = fl_index)

        ax = plt.subplot(5, 4, i, projection = ccrs.PlateCarree())
        top = True if row == 0 else False
        right = True if col == 3 else False
        bottom = True if row == 4 else False
        left = True if col == 0 else False
        ax = _set_grid(ax, top = top, right = right, bottom = bottom, left = left)
        
        if left:
            fl_bound = fl_bounds[fl_index]
            fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
            ax.text(-0.2, 0.5, fl, va = "bottom", ha = "center", rotation = "vertical",
                    rotation_mode = "anchor", transform = ax.transAxes, size = 16)

        if top:
            title = r"$H={h}$ km asl".format(h = height) if col < 3 else r"$H \sim TN(12, 1, 10, 14)$"
            ax.set_title(title, fontsize = 16)


        # Plot contours
        cf = iplt.contourf(plot_cube, levels = levels,
                           colors = bupu_colors) 

        # plt.subplots_adjust(wspace = 0, hspace = 0)
        i += 1

# Colour bar
cbaxes = fig.add_axes([0.38, 0.05, 0.27, 0.015])
cbar = plt.colorbar(cf,
                    cax = cbaxes, 
                    orientation = "horizontal",
                    spacing = "uniform",
                fraction = .05, shrink = .9, pad = 0.1)
cbar.ax.set_xlabel(quantity, fontsize =14) 
cbar.ax.set_xticklabels(ticklabels)

title = ("Exceedance Probability" + 
         r" of {t}".format(t = int(threshold * 1E+6)) + u" \u03bc"  +r"g m$^{-3}$" +
         f"\nValid for {date_stamp} (T+{hours_since})")
fig.suptitle(title, y = .95, fontsize = 18)

fig.savefig(fig_dir + "ensemble-height-fl.png", format = format,
            bbox_inches = "tight")
