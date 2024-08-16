"""
Main sequence lifetimes
=======================

Show how the main sequence ages of each star in a binary are different.

For primary stars, the main sequence lifetime of a star is directly related to its initial mass, with more
massive stars living shorter lives (with some dependence on metallicity). For secondary stars, the same
relation doesn't hold, as the secondary star may accrete mass during its main sequence when its companion
initiates mass transfer. This will increase the mass of the secondary star,
shortening its main sequence lifetime.
"""

import cogsworth
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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


p = cogsworth.pop.Population(10000, processes=1, BSE_settings={"binfrac": 1.0})
p.sample_initial_binaries()
p.perform_stellar_evolution()


for suffix, label in zip(["_1", "_2"], ["Primary stars", "Secondary stars"]):
    prev_kstar = p.bpp[f"kstar{suffix}"].shift(1)
    end_ms_rows = p.bpp[(p.bpp["evol_type"] == 2) & (p.bpp[f"kstar{suffix}"] == 2) & (prev_kstar == 1)]
    initC_rows = p.initC.loc[end_ms_rows["bin_num"]]

    fig, ax = plt.subplots()

    scatter = ax.scatter(initC_rows[f"mass{suffix}"], end_ms_rows["tphys"], c=initC_rows["metallicity"],
                         s=10, norm=LogNorm(), rasterized=True)
    fig.colorbar(scatter, label="Metallicity")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(r"Initial mass $[{\rm M_\odot}]$")
    ax.set_ylabel(r"Main sequence lifetime $[{\rm Myr}]$")

    ax.set(xscale="log", yscale="log",
           xlabel="Initial mass [M$_\odot$]", ylabel="Main sequence lifetime [Myr]",
           xlim=(2e-1, 1e2), ylim=(3e0, 2e4))

    ax.annotate(label, xy=(0.95, 0.95), xycoords="axes fraction", ha="right", va="top", fontsize=fs)

    plt.show()
