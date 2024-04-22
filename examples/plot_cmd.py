"""
Gaia colour-magnitude diagram
=============================

A colour-magnitude diagram using simulated photometry for Gaia.

Each point represents a star in the population, coloured by its stellar type. Circles are used for bound
binaries, and triangles for unbound binaries (upwards indicating the primary star, downwards indicating
the secondary star).
"""

import cogsworth
import matplotlib.pyplot as plt

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

p = cogsworth.pop.Population(2000, processes=1, BSE_settings={"binfrac": 1.0})
p.create_population()

p.get_observables(filters=["G", "BP", "RP"],
                  assume_mw_galactocentric=True, ignore_extinction=True)

cogsworth.plot.plot_cmd(p, show=False)
plt.tight_layout()          # <-- this is just to get rid of weird padding in online docs
plt.show()
