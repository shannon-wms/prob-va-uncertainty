from figures import *
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from seaborn import color_palette
import numpy as np

ivespa = merph.IVESPA
ivespa.set_vars(xvar = "H", yvar = "Q")
ivespa.mle(plot = False)
height = 10
ivespa.set_obs(height)
pts = ivespa.posterior_point([0.025, 0.975])
mu = ivespa.mu[0]
mer = 10 ** mu

alphas = np.array([.005, 0.025, .05, .125, .25])
alphas = np.concatenate([alphas, 1 - alphas])
alphas = np.sort(alphas)
n_alphas = alphas.size
n_ints = n_alphas // 2

n_steps: int = 100
h0: np.typing.NDArray[np.float64]
h0 = np.linspace(-1, np.log10(50), n_steps)  # type: ignore
q_intervals = np.zeros((n_steps, n_alphas))

q0 = np.zeros((n_steps))

for j, hh in enumerate(h0):
    ivespa.set_obs(hh, logscale=True)
    q0[j] = ivespa.mu
    q_intervals[j, :] = ivespa.posterior_point(alphas)

colormap = color_palette("flare", n_ints)
sm = LinearSegmentedColormap.from_list("p", colormap, N=n_ints)

alpha = 1
alpha_lines = 0.2
ylim =[10, 10 ** 9] 
xlim = [1, 50]
xticks1 = (1, 2, 5, 10, 20, 50)
xticks2 = (0, 10, 20, 30, 40, 50)
hlines = [10 ** q for q in range(2, 9)]
fs = 12

fig = plt.figure(figsize = (12, 5))
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
ax1.set_ylim(ylim)
ax1.set_xlim(xlim)
ax1.hlines(hlines, xlim[0], xlim[1], 
           color = "gray", linestyles = "solid", alpha = alpha_lines)
ax1.vlines(xticks1[1:-1], ylim[0], ylim[1], 
           color = "gray", linestyles = "solid", alpha = alpha_lines)   
ax2.hlines(hlines, 0, xlim[1], 
           color = "gray", linestyles = "solid", alpha = alpha_lines)
ax2.vlines(xticks2[1:-1], ylim[0], ylim[1], 
           color = "gray", linestyles = "solid", alpha = alpha_lines)   
ax2.set_ylim(ylim)
ax2.set_xlim([0,50])

for j in range(n_alphas-1):
    k = j if j < n_ints else n_ints - j - 2
    ax1.fill_between(
        10 ** h0,
        10 ** q_intervals[:, j],
        10 ** q_intervals[:, j + 1],
        facecolor=colormap[k],
        edgecolor=colormap[k],
        alpha=alpha,
        linestyle = "dashed")
    ax2.fill_between(
        10 ** h0,
        10 ** q_intervals[:, j],
        10 ** q_intervals[:, j + 1],
        facecolor=colormap[k],
        edgecolor=colormap[k],
        alpha=alpha,
        linestyle = "dashed")

alpha_pts = 0.5    

ax1.plot(10**h0, 10**q0, color="black")
ax1.scatter(ivespa.height, ivespa.mer, color = "black", alpha =alpha_pts, zorder = 2)
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_ylabel(r"MER (kg s$^{-1}$)", fontsize = fs)
ax1.set_xticks(xticks1)
ax1.set_xticklabels(("1", "2", "5", "10", "20", "50"))
ax1.xaxis.set_tick_params(which = "minor", bottom = False)
# add text label to top left of axes
ax1.text(0.6, 10 ** 9, "a", fontweight = "bold", fontsize = 12)
# ax1.minorticks_off()


ax2.plot(10**h0, 10**q0, color="black")
ax2.scatter(ivespa.height, ivespa.mer, color = "black", alpha =alpha_pts, zorder = 2)
ax2.set_yscale("log")
ax2.set_xticks(xticks2)
ax2.text(-6, 10 ** 9, "b", fontweight = "bold", fontsize = 12)

# set common x label
fig.text(0.5, 0.02, "Plume height (km avl)", ha = "center", va = "center",
         fontsize = fs)

fig2, ax3 = plt.subplots()

sc = ax3.scatter(
    np.arange(n_ints),
    np.arange(n_ints),
    c=np.arange(n_ints),
    cmap=sm,
)

# cbar_ax = ax2_divider.append_axes("left", size="7%", pad="2%")
cbar_ax = fig.add_axes([.935, 0.11, 0.03, .77])
cbar = plt.colorbar(sc, cax = cbar_ax,
                orientation = "vertical",
                spacing = "uniform",
                fraction = .05, shrink = .9, pad = 0)
# fig.colorbar(sc, cax=cbar_ax)
# tick_locs = (np.arange(n_ints) + 0.5) * (n_ints - 1) / n_ints
tick_locs = np.arange(n_ints+1) * (n_ints-1) / n_ints
cbar_ax.set_yticks(tick_locs)
cbar_ax.set_ylabel("Interpercentile range (%)", fontsize = fs)
clabels = [100 - int(200*p) for p in alphas[:n_ints]]
clabels.append(0)
cbar_ax.set_yticklabels(clabels)
plt.close(fig2)

# Save with lines
fig.savefig(fig_dir + f"ivespa-posterior-ppf.pdf", format = "pdf",
            bbox_inches = "tight")