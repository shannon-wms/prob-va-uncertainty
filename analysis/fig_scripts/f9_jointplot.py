from figures import fig_dir
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import t, truncnorm
import matplotlib.pyplot as plt
from pvauncertainty.utils import set_ivespa_obs

# Set the size of the sample
size = 10000

volcano_height = 1725

sds = [0.5, 1, 2]
loc = 12

fig, axs = plt.subplots(1, len(sds), figsize=(15, 5.5))

for i, sd in enumerate(sds):
    sd = sds[i]
    tn = truncnorm((10 - loc) / sd, (14 - loc) / sd, loc = loc, scale = sd)
    Hs = tn.rvs(size = size)
    ivespa = set_ivespa_obs(Hs * 1000, volcano_height)
    df = ivespa.df
    mus = ivespa.mu
    sigmas = ivespa.Sigma
    logQs = t.rvs(df, loc = mus, scale = sigmas, size = size)
    data = pd.DataFrame({'Hs': Hs, 'logQs': logQs})

    # Create a jointplot
    p = sns.JointGrid(data = data, x = "Hs", y = "logQs", space = 0.05)
    cax = p.plot_joint(sns.kdeplot, fill = True, kind='kde', 
                       cmap = "GnBu", bw_adjust=2, cut = 0, 
                       color = "black")
    p.plot_marginals(sns.kdeplot, color = "black")
    p.ax_joint.set_xlim(9, 15)
    p.ax_joint.set_ylim(4, 9)
    p.ax_joint.set_aspect("equal")
    if i == 1:
        p.ax_joint.set_xlabel("Plume Height (km)")
    else:
        p.ax_joint.set_xlabel("")
    if i == 0:
        p.ax_joint.set_ylabel("MER (kg/s)")
    else:
        p.ax_joint.set_ylabel("")

    p.ax_marg_x.cla()
    xs = np.linspace(9, 15, 1000)
    sns.lineplot(x = xs, y = tn.pdf(xs), color = "black", ax = p.ax_marg_x)
    p.ax_marg_x.set_ylim(0, 0.85)
    p.ax_marg_x.set_yticks([])
    p.ax_marg_y.set_visible(False)
    p.ax_marg_x.set_xticklabels([])
    p.ax_marg_x.xaxis.set_ticklabels(range(9, 16))
    plt.setp(p.ax_joint.get_xticklabels(), visible=True)
    plt.setp(p.ax_marg_x.get_xticklabels(), visible=False)
    p.ax_joint.set_yticklabels([r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$", r"$10^9$"])
    p.ax_marg_x.set_title(r"$H \sim $" + f"TN(12, {sd}" + r"$^2$" + ", 10, 14)")
    # cbar = plt.colorbar(cax, ax = p.ax_joint, label = "Density")

    if i == 0:
        p.ax_marg_x.text(1, 1, "a", fontweight = "bold", fontsize = 12)
    # Move the subplot to the correct location
    p.figure.set_figwidth(4.5)
    p.figure.set_figheight(4.5)
    p.figure.subplots_adjust(left=0.15, right=0.86, top=0.9, bottom=0.1)
    p.figure.canvas.draw()
    bbox = p.figure.bbox_inches.from_bounds(5, 0, 5, 5)
    axs[i].axis('off')
    fig.figimage(p.figure.canvas.buffer_rgba(), xo=72*5*i, yo=0, origin='upper')

fig.subplots_adjust(wspace = 0.5)
fig.text(0.1, 0.85, "a", horizontalalignment = "left", 
         verticalalignment = "top",
         fontweight = "bold", fontsize = 10)
plt.tight_layout()
plt.show()

fig.savefig(fig_dir + "mer-h-jointplot.png", format = "png",
            bbox_inches = "tight")