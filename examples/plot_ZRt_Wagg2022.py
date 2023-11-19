"""
Metallicity-radius-time relations
=================================

Plots showing how metallicity, birth radius and birth time are correlated in the
:class:`~cogsworth.galaxy.Wagg2022` star formation history model.

In the first plot we can see that the highest metallicities are confined to the centre of the galaxy. For each
radius we also see a range of metallicities - this is a result of systems having different formation times.
When colouring by age and doing some stacked histograms with age bins, one can see this trend.
In the marginals one can also note the inside-out growth of the galaxy (more recent times have larger radii)
and metal enrichment of the galaxy (more recent times have high metallicity).

This is all discussed in more detail in the
`star formation history tutorial <../tutorials/pop_settings/initial_galaxy.ipynb>`_.
"""

import cogsworth
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# sphinx_gallery_start_ignore
plt.style.use("dark_background")
plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)
# sphinx_gallery_end_ignore

g = cogsworth.galaxy.Wagg2022(size=500000)

# create a 2x2 plot with no space between axes
fig, axes = plt.subplots(2, 2, figsize=(15, 15),
                         gridspec_kw={"width_ratios": [5, 1],
                                      "height_ratios": [1, 5]})
fig.subplots_adjust(hspace=0.0, wspace=0.0)

# hide the top right panel
axes[0, 1].axis("off")

# fix the x and y lims
xlims = (1e-2, 50)
ylims = (3e-5, 1e-1)

# add histograms to each axis
axes[0, 0].hist(g.rho.value, np.geomspace(*xlims, 500))
hexbin = axes[1, 0].hexbin(x=g.rho.value, y=g.Z.value, bins="log",
                           xscale="log", yscale="log",
                           extent=np.log10((*xlims, *ylims)))
axes[1, 1].hist(g.Z.value, np.geomspace(*ylims, 500), orientation="horizontal")

# create an inset axis and add a colourbar
inset_ax = axes[1, 0].inset_axes([0.07, 0.15, 0.6, 0.1])
fig.colorbar(hexbin, cax=inset_ax, orientation="horizontal",
             label="Number of systems")

# set the axes limits, scales and labels
axes[0, 0].set(xlim=xlims, xscale="log")
axes[1, 0].set(xlim=xlims, ylim=ylims, xlabel="Galactocentric radius [kpc]", ylabel="Metallicity")
axes[1, 1].set(ylim=ylims, yscale="log")

# for the marginals hide the ticks and spines
for ax in [axes[0, 0], axes[1, 1]]:
    ax.set(yticks=[], xticks=[])
    ax.spines[:].set_visible(False)

plt.show()

# --------------------------------------------------------------
# Second plot starts here

# create a 2x2 plot with no space between axes
fig, axes = plt.subplots(2, 2, figsize=(15, 15),
                         gridspec_kw={"width_ratios": [5, 1],
                                      "height_ratios": [1, 5]})
fig.subplots_adjust(hspace=0.0, wspace=0.0)

# hide the top right panel
axes[0, 1].axis("off")

# fix the x and y lims
xlims = (1e-2, 50)
ylims = (3e-5, 1e-1)

time_bins = [(i, i + 3) for i in range(0, 12, 3)] * u.Gyr

# add histograms to each axis
axes[0, 0].hist([g.rho.value[(g.tau >= low) & (g.tau < high)]
                 for low, high in time_bins],
                np.geomspace(*xlims, 500), stacked=True,
                label=[f'{low.value:1.0f}' + r'$< \tau <$' + f'{high.value:1.0f}'
                       for low, high in time_bins])
axes[0, 0].legend(loc="center left", ncol=2, fontsize=0.7*fs)
hexbin = axes[1, 0].hexbin(x=g.rho.value, y=g.Z.value, C=g.tau.value,
                           xscale="log", yscale="log", vmin=0, vmax=12,
                           extent=np.log10((*xlims, *ylims)), cmap="plasma")
axes[1, 1].hist([g.Z.value[(g.tau >= low) & (g.tau < high)]
                 for low, high in time_bins],
                np.geomspace(*ylims, 500), orientation="horizontal", stacked=True)

# create an inset axis and add a colourbar
inset_ax = axes[1, 0].inset_axes([0.07, 0.15, 0.6, 0.1])
fig.colorbar(hexbin, cax=inset_ax, orientation="horizontal",
             label="Lookback time [Gyr]")

# set the axes limits, scales and labels
axes[0, 0].set(xlim=xlims, xscale="log")
axes[1, 0].set(xlim=xlims, ylim=ylims, xlabel="Galactocentric radius [kpc]", ylabel="Metallicity")
axes[1, 1].set(ylim=ylims, yscale="log")

# for the marginals hide the ticks and spines
for ax in [axes[0, 0], axes[1, 1]]:
    ax.set(yticks=[], xticks=[])
    ax.spines[:].set_visible(False)

plt.show()
