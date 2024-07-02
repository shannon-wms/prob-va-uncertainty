from figures import *
from pvauncertainty.plotting import _get_datestamps, _set_grid, _constrain_cube_for_plot
from PIL import Image

sds = [0.5, 1, 2]
loc = 12
fl_inds = [ 7, 8, 9, 10] 
time_index = 6
threshold = 0.0002
levels = (0, 1E-3, 1E-2, 0.1, 0.25, 0.5, 1)
ticklabels = (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.1$", r"$0.25$", r"$0.5$", r"$1$")
colorscale = bupu_colors
n_int = 1

fig = plt.figure(figsize = (20, 17))
outer = gridspec.GridSpec(4, 3, wspace = 0.2, hspace = 0.25, 
                          height_ratios = [1, 1, 1, 1])

m = 0
for i, sd in enumerate(sds):
    file_name = f"../data/quadrature_results/quad_exc_prob_{n_int}_sd{str(sd).replace('.', '_')}.nc"
    cube = iris.load(file_name)[0]
    for j, fl_index in enumerate(fl_inds):

        if j == 0 and m == 0:
            fl_coord = cube[0].coord("flight_level")
            fl_bounds = fl_coord.bounds
            t_coord = cube.coord("time")
            date_stamps, hours = _get_datestamps(cube, time_since_start = True)
            date_stamp = date_stamps[time_index]
            hours_since = hours[time_index]
            title = (r"Exceedance Probability of {t}".format(t = int(threshold * 1E+6)) + u" \u03bc" + r"g m$^{-3}$" 
                     f"\nValid for {date_stamp} (T+{hours_since})")

        plot_cube, _ = _constrain_cube_for_plot(
            cube, threshold = threshold, time_index = time_index, 
            fl_index = fl_index)
        # plot_cube.data = np.log10(plot_cube.data)
        # Plot contours
        top = True if j == 0 else False
        right = True if i == 2 else False
        bottom = True if j == 3 else False
        left = True if i == 0 else False
        ax = fig.add_subplot(outer[j, i], projection = ccrs.PlateCarree())
        ax = _set_grid(ax, top = top, right = right, bottom = bottom, left = left,
                    alpha = 0.7)
        # ax.set_aspect("auto")
        if left:
            fl_bound = fl_bounds[fl_index]
            fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
            ax.text(-0.125, 0.55, fl, va = "bottom", ha = "center", rotation = "vertical",
                    rotation_mode = "anchor", transform = ax.transAxes, size = 16)
        # Contours for exceedance probs
        cf = iplt.contourf(plot_cube, levels = levels, colors = colorscale,
                            extend = "neither") 
        if j == 0 and m == 0:
            ax.text(-0.2, 1.35, "b", transform = ax.transAxes, 
                    fontweight = "bold", fontsize = 16)
    m += 1

fig.suptitle(title, y =0.95, fontsize = 18)
plt.tight_layout()
cbaxes1 = fig.add_axes([0.3825, 0.07, 0.25, 0.015])
cbar1 = plt.colorbar(cf, cax = cbaxes1,
                orientation = "horizontal",
                spacing = "uniform",
                fraction = .05, shrink = .9, pad = 0.1)
cbar1.ax.set_xlabel("Exceedance Probability", fontsize = 14) 
cbar1.ax.set_xticklabels(ticklabels)

fig.savefig(fig_dir + "h-sd-comparison.png", format = "png",
            bbox_inches = "tight")

# resize and combine fig a and b
im1 = Image.open(fig_dir + "mer-h-jointplot.png")
width1, height1 = im1.size
im2 = im1.crop((0, 0, width1-125, height1-20))
width2, height2 = im2.size

im3 = Image.open(fig_dir + "h-sd-comparison.png")
width3, height3 = im3.size
im4 = im3.resize((width2, int(height3 * (width2 / width3))),
                 Image.Resampling.LANCZOS)
_, height4 = im4.size

total_height = height2 + height4
new_im =  Image.new("RGB", (width2, total_height))
new_im.paste(im2, (0, 0))
new_im.paste(im4, (0, height2))
new_im.save(fig_dir + "jointplot-comparison.png")