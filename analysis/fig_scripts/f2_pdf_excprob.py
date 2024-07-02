from figures import *
from scipy.stats import t
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ivespa = merph.IVESPA
ivespa.set_vars(xvar = "H", yvar = "Q")
ivespa.mle(plot = False)
height = 10
ivespa.set_obs(height)
mu = ivespa.mu[0]
mer = 10 ** mu
alpha, beta = ivespa.beta_vec
df = ivespa.df
sigma = ivespa.Sigma

thresholds = [t * 1E6 for t in thresholds]
fs = 12

fig = plt.figure(figsize = (16, 6))
outer = gridspec.GridSpec(1, 4, wspace = 0.4, hspace = 0.15, 
                          width_ratios = [1, .5, .75, .15])

inner0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[0])
inner1 = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec = outer[1], hspace = .1)
inner2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[2], hspace = .1)
inner3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[3], hspace = .1)

ax1 = plt.Subplot(fig, inner0[0])
ax1.set_xscale("log")
ax1.set_ylim(0, .55)
ax1.set_xlim(10 ** 3, 10 ** 10)

ax3 = plt.Subplot(fig, inner2[0])
ax3.set_xscale("log")
ax3.set_ylim(0, 1)
ax3.set_xlim(1, 10 ** 6)

ax4 = plt.Subplot(fig, inner2[1])
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_ylim(10 ** -5, 1.05)
ax4.set_xlim(1, 10 ** 6)

yp = 10 ** np.linspace(3, 10, num = 1000)
yp1 = 10 ** np.linspace(0, 6, num = 1000)
loc_c = np.array([1, 2, 2.5])

axes = list

for i, h in enumerate(np.array([9.0, 10.0, 11.0])):
    ax = plt.Subplot(fig, inner1[i])
    ax.set_xscale("log")
    ax.set_ylim(0, 0.6)
    ax.set_xlim(1, 10 ** 6)
    if i == 0:
        ax.text(0.01, 0.6, "b", fontweight = "bold", fontsize = 14)

    loc_h = alpha + beta * np.log10(h)
    
    lty = ["dashed", "solid", "dotted"][i]
    ax1.plot(yp, t.pdf(np.log10(yp), df = df, loc = loc_h, scale = sigma),
             color = "black", linestyle = lty)
    
    
    for j, threshold in enumerate(thresholds):
        colour = colours[j]
        
        ymax = t.pdf(np.log10(threshold), df = df, loc = loc_c[i], scale = sigma)

        ax.vlines(x = threshold, ymin = 0, ymax = ymax, color = colour)
        ax.fill_between(x = yp1, y1 = 0, 
                        y2 = t.pdf(np.log10(yp1), df = df, loc = loc_c[i], 
                                    scale = sigma), 
                        where = (yp1 >= threshold), 
                        color = colour, alpha = 0.5)
    if i < 2:
        ax.get_xaxis().set_visible(False)

    if i == 1:
        ax.set_ylabel("Probability density", fontsize = fs)
    if i == 2:
        ax.set_xlabel("Ash concentration" + u" (\u03bc"  +r"g m$^{-3}$)", fontsize = fs)

    ax.plot(yp1, t.pdf(np.log10(yp1), df = df, loc = loc_c[i], scale = sigma),
            color = "black", linestyle = lty)
    ax3.plot(yp1, 1 - t.cdf(np.log10(yp1), df = df, loc = loc_c[i], scale = sigma),
             color = "black", linestyle = lty)
    ax4.plot(yp1, 1 - t.cdf(np.log10(yp1), df = df, loc = loc_c[i], scale = sigma),
             color = "black", linestyle = lty)
    
    fig.add_subplot(ax)

ax1.text(10 ** 2, 0.55, "a", fontweight = "bold", fontsize = 14)
ax3.text(0.05, 1, "c", fontweight = "bold", fontsize = 14)
ax4.text(0.05, 1, "d", fontweight = "bold", fontsize = 14)

for j, threshold in enumerate(thresholds):
    colour = colours[j]
    tmax = 1 - t.cdf(np.log10(threshold), df = df, loc = loc_c[-1], scale = sigma)
    tmin = 1 - t.cdf(np.log10(threshold), df = df, loc = loc_c[0], scale = sigma)

    whisker_len = .1
    ax3.vlines(x = threshold, ymin = tmin, ymax = tmax, color = colour, alpha = 0.7)
    ax3.hlines(y = np.array([tmin, tmax]), 
               xmin = 10 ** (np.log10(threshold) - whisker_len), 
               xmax = 10 ** (np.log10(threshold) + whisker_len), 
               color = colour, alpha = 0.8)
    ax4.vlines(x = threshold, ymin = tmin, ymax = tmax, color = colour, alpha = 0.7)
    ax4.hlines(y = np.array([tmin, tmax]), 
               xmin = 10 ** (np.log10(threshold) - whisker_len), 
               xmax = 10 ** (np.log10(threshold) + whisker_len), 
               color = colour, alpha = 0.8)

ax1.set_xlabel(r"MER (kg s$^{-1}$)", fontsize = fs)
ax1.set_ylabel("Probability density", fontsize = fs)

ax3.get_xaxis().set_visible(False)
ax3.set_ylabel("Exceedance probability", fontsize = fs)

ax4.set_ylabel("Exceedance probability", fontsize = fs)
ax4.set_xlabel("Ash concentration" + u" (\u03bc"  +r"g m$^{-3}$)", fontsize = fs)

ax5 = plt.Subplot(fig, inner3[0])
ax5.set_axis_off()
ax5.legend(handles = [mlines.Line2D([], [], color = "black", linestyle = "dashed", 
                                    label = r"$H=9$ km"),
                      mlines.Line2D([], [], color = "black", linestyle = "solid",
                                     label = r"$H=10$ km"),
                      mlines.Line2D([], [], color = "black", linestyle = "dotted",
                                     label = r"$H=11$ km")],
           title = "Plume height", loc = "center", fontsize = 11, 
           title_fontsize = fs)

ax6 = plt.Subplot(fig, inner3[1])
ax6.set_axis_off()
ax6.legend(handles = [mpatches.Patch(color = colours[i], alpha = 0.8, 
                                     label = f"{int(thresholds[i])} " + u"\u03bc" + r"g m$^{-3}$")
                      for i in range(len(thresholds))],
           title = "Ash concentration \n threshold", loc = "center", fontsize = 11,
           title_fontsize = fs)


fig.add_subplot(ax1)
fig.add_subplot(ax3)
fig.add_subplot(ax4)
fig.add_subplot(ax5)
fig.add_subplot(ax6)

fig.savefig(fig_dir + "pdf-plots.pdf", format = "pdf", bbox_inches = "tight")