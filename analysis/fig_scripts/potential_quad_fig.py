import os
from figures import *
from iris.cube import CubeList
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

cubes_dir = "../data/name_out_cubes/"
csv_file = "../data/chunks_10_14km.csv"
fig_dir = "figures/"

# Gauss-Kronrod points
x = (0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0,
    -0.148874338981631210884826001129720,
    -0.294392862701460198131126603103866,
    -0.433395394129247190799265943165784,
    -0.562757134668604683339000099272694,
    -0.679409568299024406234327365114874,
    -0.780817726586416897063717578345042,
    -0.865063366688984510732096688423493,
    -0.930157491355708226001207180059508,
    -0.973906528517171720077964012084452,
    -0.995657163025808080735527280689003)

w = (0.066671344308688137593568809893332,
     0.149451349150580593145776339657697,
     0.219086362515982043995534934228163,
     0.269266719309996355091226921569469,
     0.295524224714752870173892994651338,
     0.295524224714752870173892994651338,
     0.269266719309996355091226921569469,
     0.219086362515982043995534934228163,
     0.149451349150580593145776339657697,
     0.066671344308688137593568809893332)

a = 10
b = 14
c = 0.5 * (a+b)
h = 0.5 * (b-a)
y = [c + h * x[i] for i in range(len(x))]

thresholds = [2E-4, 2E-3, 5E-3, 10E-3]

h_lower = 10
h_upper = 14
scale = 1
heights = np.linspace(h_lower, h_upper, num = 50)

loc = (h_lower + h_upper) / 2
tn = truncnorm((h_lower - loc) / scale, (h_upper - loc) / scale, loc, scale)

pdf_h = tn.pdf(heights)

if not os.path.exists("cdf_given_h.npy"):
    volcano_height = 1725
    height_asl = 12
    height_avl = height_asl - (volcano_height / 1000)

    ivespa = set_ivespa_obs(height_avl * 1000, volcano_height)
    df = ivespa.df
    mu = ivespa.mu[0]
    sigma = ivespa.Sigma[0]

    # Evaluate MER for this height
    mer_gs = 10 ** mu * 1000
    mer_gs_ht = mer_gs / (height_avl * 1000)

    # Ensemble exceedance probabilities
    chunks = SourceChunks(volcano_height, csv_file)
    chunks.name_output = VolcanicNAME("output", True, "/", csv_file)
    chunks.name_output.n_members = 18
    chunks.list_member_chunks = []

    for member in range(18):
        ash_chunks_filename = cubes_dir + f"ash_chunks_member_{member}.nc"
        # Load ash cube list and constrain
        chunks.list_member_chunks.append(
            chunks.name_output.load_ash_cube_list(ash_chunks_filename, member))

    lat_points = chunks.list_member_chunks[0][0].coord("latitude").points
    lon_points = chunks.list_member_chunks[0][0].coord("longitude").points

    # Example usage
    latitude = 50
    longitude = 10

    lat_index = np.argmin(np.abs(lat_points - latitude))
    lon_index = np.argmin(np.abs(lon_points - longitude))

    sliced_cubes = []
    for member_chunks in chunks.list_member_chunks:
        member_slices = []
        for cube in member_chunks:
            # Slice the cube at the desired latitude and longitude
            sliced_cube = cube[6, 0, lat_index, lon_index]
            member_slices.append(sliced_cube)
        sliced_cubes.append(member_slices)

    chunks.list_member_chunks = sliced_cubes

    cdf_given_h = np.array([eval_t_cdf(h, chunks, thresholds).data for h in heights]).T
    # save cdf_given_h to a file
    np.save("cdf_given_h.npy", cdf_given_h)


    # integrand = [pdf_h[i] * cdf_given_h.T[0][i] for i in range(len(heights))]

    plt.show()

    gk_cdf_given_h = np.array([
        eval_t_cdf(y[2 * i + 1], chunks, thresholds).data for i in range(int(len(y)/2))])

    np.save("gk_cdf_given_h.npy", gk_cdf_given_h.T)
else:
    cdf_given_h = np.load("cdf_given_h.npy")
    gk_cdf_given_h = np.load("gk_cdf_given_h.npy")

integrand = [x * y for x, y in zip(pdf_h, cdf_given_h.T)]

gk_pdf_h = [tn.pdf(y[2 * i + 1]) for i in range(int(len(y)/2))]

gk_integrand = [x * y for x, y in zip(gk_pdf_h,  gk_cdf_given_h)]

fig = plt.figure(figsize = (13, 9))
outer = gridspec.GridSpec(2, 2, wspace = 0.2, hspace = 0.15, 
                          width_ratios = [1, 1.3], height_ratios = [1, .25])
inner0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = outer[0], hspace = 0.25)
inner1 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[1])
inner2 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec = outer[2], wspace = 0.1, hspace = -0.1)

ax1 = plt.Subplot(fig, inner0[0])
ax1.set_xlim(h_lower, h_upper)
ax1.set_ylim(0, 0.45)
ax1.plot(heights, pdf_h, color = "black")
for j in range(int(len(y)/2)):
    ax1.scatter(y[2*j + 1], gk_pdf_h[j], color = "black", s = 15)
    ax1.vlines(x = y[2 * j + 1], ymin = 0, ymax = gk_pdf_h[j], color = "black", 
               linestyle = "--", alpha = .3, zorder = 0)
ax1.set_xticks(np.arange(h_lower, h_upper+1, 1))
ax1.set_ylabel(r"$p_H(h)$", fontsize = 12)

ax2 = plt.Subplot(fig, inner0[1])
ax2.set_xlim(h_lower, h_upper)
ax2.set_ylim(0, 1)
# ax2.set_ylabel("Conditional CDF", fontsize = 12)
ax2.set_ylabel(r"$F_x(\log v | h)$", fontsize = 12)

for i, cdf in enumerate(cdf_given_h):
    ax2.plot(heights, cdf, color = colours[i])
    for j in range(int(len(y)/2)):
        ax2.scatter(y[2*j + 1], gk_cdf_given_h[j][i], color = colours[i], s = 15)
for j in range(int(len(y)/2)):
    ax2.vlines(x = y[2 * j + 1], ymin = 0, ymax = gk_cdf_given_h[j][-1], color = "black", 
               linestyle = "--", alpha = .3, zorder = 0)
ax2.set_xticks(np.arange(h_lower, h_upper+1, 1))
ax2.set_xlabel(r"Height $h$ (km asl)", fontsize = 12)

ax3 = plt.Subplot(fig, inner1[0])
ax3.set_xlim(h_lower, h_upper)
ax3.set_ylim(0, 0.32)

for j in range(int(len(y)/2)):
    ax3.vlines(x = y[2 * j + 1], ymin = 0, ymax = gk_integrand[j][-1], color = "black", 
            linestyle = "--", alpha = .3, zorder = 1)

for i in reversed(range(len(thresholds))):
    this_integrand = [integrand[k][i] for k in range(len(integrand))]
    ax3.plot(heights, this_integrand, color = colours[i])
    ax3.fill_between(heights, this_integrand, color = colours[i], 
                     alpha = 0.3, zorder = 0)
    for j in range(int(len(y)/2)):
        ax3.scatter(y[2*j + 1], gk_integrand[j][i], color = colours[i], s = 100 * w[j]) 

ax3.set_xlabel(r"Height $h$ (km asl)", fontsize = 12)
# ax3.set_ylabel("Integrand", fontsize = 12)
ax3.set_ylabel(r"$F_x(\log v | h) \cdot p_H(h)$", fontsize = 12)

ax1.text(9.3, .45, "a", fontweight = "bold", fontsize = 14)
ax2.text(9.3, 1, "b", fontweight = "bold", fontsize = 14)
ax3.text(9.5, .32, "c", fontweight = "bold", fontsize = 14)

ax4 = plt.Subplot(fig, inner2[0])
ax4.set_axis_off()
ax4.legend(handles = [mpatches.Patch(color = colours[i], alpha = 0.8, 
                                     label = f"{int(thresholds[i] * 1E6)} " + u"\u03bc" + r"g m$^{-3}$")
                      for i in range(len(thresholds))],
           title = r"Ash concentration threshold $v$", loc = "center", fontsize = 11,
           title_fontsize = 12, ncol = 4, bbox_to_anchor = (1.2, 0.5))

fig.add_subplot(ax1)
fig.add_subplot(ax2)
fig.add_subplot(ax3)
fig.add_subplot(ax4)

plt.show()

fig.savefig(fig_dir + "quad-plots.pdf", format = "pdf", bbox_inches = "tight")


