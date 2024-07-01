from figures import *
import matplotlib.colors as colors
import matplotlib.patches as mpatches

spectral_cmap = cm.get_cmap("Paired_r") 
spectral_colors = [colors.rgb2hex(spectral_cmap(i)) for i in range(spectral_cmap.N)]
rdbu_cmap = cm.get_cmap("brewer_RdBu_11")
rdbu_colors = [colors.rgb2hex(rdbu_cmap(i)) for i in [0, 2, 4, 5, 6, 8, 10]]

n_intervals = [1, 2, 4, 6, 12]

full_df = pd.read_csv("../data/quad_diff.csv", index_col = 0)
# melt data frame
full_df_melt = pd.melt(full_df, id_vars=['time_since', 'threshold', 'n_int'], value_vars=full_df.columns[3:-1], var_name='flight_level', value_name='l2')
full_df_melt['l2'] = full_df_melt['l2'].astype(float)

time_avg_df = full_df_melt.groupby(['flight_level', 'threshold', 'n_int']).mean().reset_index()
fl_avg_df = full_df_melt.groupby(['time_since', 'threshold', 'n_int']).mean().reset_index()

fig = plt.figure(figsize = (18, 8))
fs = 12
log = False
for i, threshold in enumerate(thresholds):
    ax1 = plt.subplot(2, 4, i+1)
    ax2 = plt.subplot(2, 4, i+5)
    for j, n_int in enumerate(n_intervals):
        df1 = fl_avg_df[(fl_avg_df['threshold'] == threshold) & (fl_avg_df['n_int'] == n_int)]
        df2 = time_avg_df[(time_avg_df['threshold'] == threshold) & (time_avg_df['n_int'] == n_int)]
        df2 = df2.reset_index(drop = True)
        fls = [df2["flight_level"][i].split("-") for i in range(len(df2))]
        fl_ints = [int(fl[0].split("FL")[1]) for fl in fls]
        fl_ints.append(int(fls[-1][1]))
        avgs = [(fl_ints[i] + fl_ints[i-1]) / 2 for i in range(1, len(fl_ints))]        
        if log:
            y1 = np.log10(df1['l2'])
            y2 = np.log10(df2['l2'])
            ylim1 = [-10, -4]
            ylim2 = [-10, 0]
        else:
            y1 = df1['l2']
            y2 = df2['l2']
            ylim1 = [0, 7E-5]
            ylim2 = [0, 1.2E-4]
        ax1.plot(df1['time_since'], y1, label=f'{n_int}', color = spectral_colors[2*j+2],
                 linewidth = 2)
        ax1.plot(df1['time_since'], y1, label=f'{n_int}', color = spectral_colors[2*j+2],
                 marker = ".", markersize = 10)
        ax2.plot(avgs, y2, label=f'{n_int}', color = spectral_colors[2*j+2],
                 linewidth = 2)
        ax2.plot(avgs, y2, label=f'{n_int}', color = spectral_colors[2*j+2],
                 marker = ".", markersize = 10)

    ax1.set_title(r"{t}".format(t = int(threshold * 1E+6)) + u" \u03bc"  +r"g m$^{-3}$", fontsize = 14)
    fig.text(0.5, 0.48, "Time since eruption (hours)", ha = "center", 
             va = "center",fontsize = fs)
    # ax1.set_xlabel('Time since eruption (hours)')
    ax1.set_ylim(ylim1)
    fig.text(0.5, 0.04, "Flight level (FL)", ha = "center", 
             va = "center",fontsize = fs)
    if not log:
        ax1.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
        ax2.ticklabel_format(axis = "y", style = "sci", scilimits = (0,0))
    if i == 0:
        ax1.set_ylabel('Error', fontsize = fs)
        ax2.set_ylabel('Error', fontsize = fs)
        # scientific notation for y axis ticklabels
    elif i == 3:
        # y axis tick labels on right side
        ax1.yaxis.tick_right()
        ax2.yaxis.tick_right()
    else:
        ax1.set_yticklabels([])
        ax2.set_yticklabels([])
        # hide ticks on y axis
        ax1.yaxis.set_tick_params(size = 0)
        ax2.yaxis.set_tick_params(size = 0)

    ax1.set_xticks(np.arange(12, 49, 6))
    ax2.set_ylim(ylim2)
    ax2.set_xticks(fl_ints)
    ax2.set_xticklabels(["{:03d}".format(fl) for fl in fl_ints], rotation=90)
    ax1.grid(alpha = 0.3, linestyle = "solid")
    ax2.grid(alpha = 0.3, linestyle = "solid")

# set common legend for all subplots on bottom under the second row of plots
fig.text(0.08, .9, "a", fontweight = "bold", fontsize = 14)
fig.text(0.08, .48, "b", fontweight = "bold", fontsize = 14)

legend_axes = fig.add_axes([0.2, -.07, 0.6, 0.1])
legend_axes.set_axis_off()
legend_axes.legend(handles = [mpatches.Patch(color = spectral_colors[2*j+2], alpha = 0.8, 
                                     label = f"{n_int}")
                      for j, n_int in enumerate(n_intervals)],
           title = "Number of quadrature intervals", loc = "center", fontsize = 11,
           title_fontsize = fs, ncol = 5)

fig.savefig(fig_dir + "quad-sensitivity-fl", format = format,
            bbox_inches = "tight")
