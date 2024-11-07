from figures import *
from pvauncertainty.plotting import _get_datestamps, _set_grid

chunks = SourceChunks(volcano_height, csv_file)
chunks.name_output = VolcanicNAME("output", True, "/", csv_file)
chunks.name_output.n_members = 18
chunks.list_member_chunks = []
for member in range(18):
    ash_chunks_filename = cubes_dir + f"ash_chunks_member_{member}.nc"
    # Load ash cube list and constrain
    chunks.list_member_chunks.append(
        chunks.name_output.load_ash_cube_list(ash_chunks_filename, member))

height_asl = 12
height_avl = height_asl - (chunks.volcano_height / 1000)
    
time_index = 6
fl_index = 0
threshold = 0.0002
    
# Get params from MERPH model
ivespa = set_ivespa_obs(height_avl * 1000, chunks.volcano_height)
df = ivespa.df
mu = ivespa.mu[0]
sigma = ivespa.Sigma[0]

# Evaluate MER for this height
mer_gs = 10 ** mu * 1000
mer_gs_ht = mer_gs / (height_avl * 1000)

levels = (0, 1E-3, 1E-2, 0.1, 0.25, 0.5, 1)
ticklabels = (r"$0$", r"$10^{-3}$", r"$10^{-2}$", r"$0.1$", r"$0.25$", r"$0.5$", r"$1$")

thresholds = [0.2E-3, 2E-3, 5E-3, 10E-3]
n_members = len(chunks.list_member_chunks)

fig = plt.figure(figsize = (21.5, 14))
for i in range(n_members):
    chunks_member_cube_list = chunks.list_member_chunks[i]
    
    # Get rescaled cube
    rescaled_cube = reconstruct_cube_from_list(
        cube_list = chunks_member_cube_list, 
        plume_height = height_asl * 1000,
        new_mer_gs_ht = mer_gs_ht)
    
    # Get log-ash concentration cube
    log_cube = rescaled_cube.copy()
    log_cube.data = np.log10(rescaled_cube.data) 

    # Deal with masked cube for NaN values
    if isinstance(log_cube.data, np.ma.core.MaskedArray):
        log_cube.data = log_cube.data.filled(-np.inf)
                            
    # Evaluate exceedance probability cube
    prob_cube = member_prob_cube(
        log_cube, 
        df = df, 
        scale = sigma,
        h_km = height_asl, 
        exceed = True,
        thresholds = [thresholds[0]])

    if i == 0:
        t_coord = prob_cube.coord("time")
        date_stamps, hours = _get_datestamps(rescaled_cube, 
                                             time_since_start = True)
        date_stamp = date_stamps[time_index]
        hours_since = hours[time_index]
        if len(t_coord.points) > 1:
            t_constraint = iris.Constraint(time = lambda cell:
                cell.point == t_coord.units.num2date(
                    t_coord.points[time_index]))

        # Get FL constraint
        fl_coord = prob_cube.coord("flight_level")
        fl_bounds = fl_coord.bounds
        fl_bound = fl_bounds[fl_index]
        fl_points = fl_coord.points
        fl_constraint = iris.Constraint(flight_level = lambda cell: 
            cell.point == fl_points[fl_index])
        fl = f"FL{int(fl_bound[0]):03d} to FL{int(fl_bound[1]):03d}"
        sum_prob_cubes = prob_cube.copy()
    else:
        sum_prob_cubes = maths.add(sum_prob_cubes, prob_cube)

    prob_cube = prob_cube.extract(fl_constraint & t_constraint)
    
    # Assumes 18 ensemble members
    ax = plt.subplot(4, 5, i+1, projection = ccrs.PlateCarree())
    top = True if i in range(5) else False
    right = True if i in [4, 9, 14, 17] else False
    bottom = True if i in range(13, 18) else False
    left = True if i in [0, 5, 10, 15] else False
    ax = _set_grid(ax, top = top, right = right, bottom = bottom, left = left)

    cf = iplt.contourf(prob_cube, levels = levels, colors = ylgnbu_colors) 

    ax.set_title("Member " + str(i), fontsize = 16)
    print(f"Member {i} done.")

save_file = "../data/ensemble_merph_exc_prob.nc"
if not os.path.exists(save_file):
    mean_prob_cube = sum_prob_cubes / n_members
    iris.save(mean_prob_cube, save_file)
else:
    mean_prob_cube = iris.load(save_file)[0]

quantity = prob_cube.attributes["Quantity"]

cbaxes = fig.add_axes([0.63, 0.2, 0.25, 0.02])

cbar = plt.colorbar(cf,
                cax = cbaxes,
                orientation = "horizontal",
                format = "%.1e", 
                spacing = "uniform",
                    fraction = .05, shrink = .9, pad = 0.1)
cbar.ax.set_xlabel(quantity, fontsize = 14) 
cbar.ax.set_xticklabels(ticklabels)

title = ("Exceedance Probability" + 
         r" of {t}".format(t = int(threshold * 1E+6)) + u" \u03bc" + r"g m$^{-3}$" +
         r" given $H={h}$ km asl ".format(h = height_asl) +
         f"\nValid for {date_stamp} (T+{hours_since})")
fig.suptitle(title, fontsize = 18)
# fig.subplots_adjust(wspace = 0.05, hspace = 0.05)

fig.savefig(fig_dir + "members-plot.png", format = "png",
            bbox_inches = "tight")