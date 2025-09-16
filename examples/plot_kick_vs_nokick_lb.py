"""
Sky positions of disrupted binaries with and without supernova kicks
====================================================================

This example script shows the sky positions of disrupted binaries, with and without supernova kicks.
The primary and secondary stars are shown as squares and pluses, respectively, connected by a dashed line.
The final positions of the binary had no kick occurred is shown as a circle.
In the background we use the Wagg2022 SFH to show the number of stars in each region of the sky.
"""

import cogsworth
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import numpy as np
import astropy.units as u
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D


# sphinx_gallery_start_ignore
plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': 0.7*fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4,
          'savefig.dpi': 300}
plt.rcParams.update(params)
# sphinx_gallery_end_ignore

# set a specific seed for reproducibility
seed = 53480
np.random.seed(seed)

# sample a full population
p_full = cogsworth.pop.Population(70, final_kstar1=[13, 14], processes=4, BSE_settings={"binfrac": 1.0},
                                  use_default_BSE_settings=True)
p_full.create_population(with_timing=True)

# subselect 5 disrupted binaries
p = p_full[p_full.disrupted]
p = p[p.bin_nums[[1, 2, 3, 5, 6]]]

# integrate the orbits again without kicks
no_kick_orbits = [p.galactic_potential.integrate_orbit(p.primary_orbits[i][0],
                                                       t1=p.primary_orbits[i].t[0],
                                                       t2=p.primary_orbits[i].t[-1],
                                                       dt=1 * u.Myr)
                  for i in range(len(p))]

# convert the final positions to galactic coordinates
no_kick_coords = [SkyCoord(x=no_kick_orbits[i][-1].x, y=no_kick_orbits[i][-1].y, z=no_kick_orbits[i][-1].z,
                           representation_type="cartesian", unit="kpc", frame="galactocentric").galactic
                  for i in range(len(no_kick_orbits))]

# get the true final position and shift the longitudes to put 0 in the middle
final_coords = p.get_final_mw_skycoord().galactic
final_l = final_coords.l.value + 180
final_l[final_l > 360] -= 360

# set the limits for the latitude
b_lims = [-45, 45]

fig, ax = plt.subplots(figsize=(12, 10))

# sample the background and turn into galactic coordinates (shifting l in the same way)
background = cogsworth.sfh.Wagg2022(size=500000)
background_coords = SkyCoord(x=background.x, y=background.y, z=background.z,
                             unit="kpc", frame="galactocentric").galactic
background_l = background_coords.l.value + 180
background_l[background_l > 360] -= 360

# calculate a 2D histogram of the background and plot it
l_edges = np.linspace(0, 360, 201)
b_edges = np.linspace(*b_lims, 201)
H, l_edges, b_edges = np.histogram2d(background_l, background_coords.b.value,
                                     bins=(l_edges, b_edges))
H[H < 3] = np.nan
im = ax.imshow(H.T, origin="lower", extent=(0, 360, -90, 90), cmap="binary", aspect="auto",
               norm=LogNorm(), zorder=-1000, alpha=0.7)

# create an inset colourbar with smaller labels and ticks
cax = ax.inset_axes([0.02, 0.02, 0.4, 0.03])
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
cbar.set_label("Number of stars", fontsize=0.6*fs)
cbar.ax.xaxis.set_tick_params(labelsize=0.6*fs)
cbar.ax.xaxis.set_tick_params(which="major", size=4)
cbar.ax.xaxis.set_tick_params(which="minor", size=2)
cbar.ax.xaxis.set_ticks_position('top')
cbar.ax.xaxis.set_label_position('top')

# use some fixed colours that go well with the background
colours = ["dodgerblue", "tab:orange", "limegreen", "tab:red", "tab:pink"]

# go through each binary
for i in range(len(p)):
    # final primary position
    ax.scatter(final_l[:len(p)][i], final_coords[:len(p)].b.value[i],
               color=colours[i], marker="s", s=100, edgecolor="black")

    # final secondary position
    ax.scatter(final_l[len(p):][i], final_coords[len(p):].b.value[i],
               color=colours[i], marker="P", s=150, edgecolor="black")

    # connect the two with a dashed line
    ax.plot([final_l[:len(p)][i],
             final_l[len(p):][i]],
            [final_coords[:len(p)].b.value[i],
             final_coords[len(p):].b.value[i]],
            color=colours[i], linestyle="--", zorder=-1, label="Final positions")

    # final position without kick
    fake_l = no_kick_coords[i].l.value + 180
    fake_l = fake_l - 360 if fake_l > 360 else fake_l
    ax.scatter(fake_l, no_kick_coords[i].b.value, s=100, color=colours[i],
               label="Final positions (no SNe)", edgecolor="black")


# manual legend info
handles = [Line2D([0], [0], marker='s', color='black', linestyle="--",
                  markersize=10, markerfacecolor="white"),
           Line2D([0], [0], marker='P', color='black', linestyle="--",
                  markersize=10, markerfacecolor="white"),
           Line2D([0], [0], marker='o', color='black', linestyle="none",
                  markersize=10, markerfacecolor="white")]
labels = ["Primary", "Secondary", "Binary (no SNe)"]

# set the limits and labels
ax.set(xlim=[0, 360], ylim=b_lims, xlabel=r"Galactic longitude, $b$ [deg]", ylabel=r"Galactic latitude, $\ell$ [deg]")
leg = ax.legend(handles, labels, loc="upper right", fontsize=0.8 * fs)
leg.set_title("Final positions", prop={"size": 0.8 * fs})

# set the ticks to be with 0 in the middle
ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
ax.set_xticklabels(["180", "225", "270", "315", "0", "45", "90", "135", "180"])

# plt.savefig("galactic_positions.pdf", bbox_inches="tight", format="pdf")

plt.show()


# uncomment the below to get more information and plots about each binary
# -----------------------------------------------------------------------
# for bin_num in p.bin_nums:
#     p.plot_cartoon_binary(bin_num)
#     p.plot_orbit(bin_num, t_max=100 * u.Myr)

# print(p.bpp[p.bpp["evol_type"].isin([15, 16])])
# print(p.final_bpp)
# print(p.kick_info[["star", "disrupted", "natal_kick", "vsys_1_total", "vsys_2_total"]])