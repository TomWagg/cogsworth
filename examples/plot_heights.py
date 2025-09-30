"""
Galactic plane distances
========================

Plot the distance from the galactic plane for each binary in the population, separating by whether a binary
experienced a kick from a supernova or not.

As one would expect, kicked systems are more likely to be found further from the galactic plane.
"""

import cogsworth
import matplotlib.pyplot as plt
import numpy as np

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

p = cogsworth.pop.Population(500, final_kstar1=[13, 14], processes=1,
                             BSE_settings={"binfrac": 1.0}, use_default_BSE_settings=True)
p.create_population(with_timing=False)

# find any binary that experienced a kick from a supernova
kicked = np.isin(p.bin_nums, p.bpp[(p.bpp["evol_type"] == 15) | (p.bpp["evol_type"] == 16)]["bin_num"])

# split the population
p_kicked = p[kicked]
p_unkicked = p[~kicked]

fig, ax = plt.subplots()

for pop, label, colour in zip([p_unkicked, p_kicked], ["Unkicked", "Kicked"], ["grey", "C2"]):
    ax.hist(abs(pop.final_pos[:, 2].value), bins=np.geomspace(1e-3, 1e3, 20),
            histtype="step", linewidth=3, color=colour)
    ax.hist(abs(pop.final_pos[:, 2].value), bins=np.geomspace(1e-3, 1e3, 20),
            alpha=0.5, color=colour, label=label)

ax.set(xscale="log", xlabel="Distance from Galactic Plane [kpc]", ylabel="Number of systems")
ax.legend()
plt.show()
