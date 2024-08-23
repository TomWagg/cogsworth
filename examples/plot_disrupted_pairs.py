"""
Disrupted binary pairs
======================

Plot each disrupted binary on the sky and draw a line between each pair.
"""

import cogsworth
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
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


p = cogsworth.pop.Population(100, final_kstar1=[13, 14], processes=1, BSE_settings={"binfrac": 1.0})
p.create_population(with_timing=False)

galactic_plane = SkyCoord(l=np.linspace(1e-10, 2 * np.pi, 10000), b=np.zeros(10000),
                          unit="rad", frame="galactic").transform_to("icrs")

final_coords = p.get_final_mw_skycoord().icrs

fig, ax = plt.subplots()

in_order = np.argsort(galactic_plane.ra.value)
ax.plot(galactic_plane.ra.value[in_order], galactic_plane.dec.value[in_order],
        label="Galactic Plane", color="grey", zorder=-1, lw=2)

ax.plot([final_coords[:len(p)][p.disrupted].ra.value,
         final_coords[len(p):].ra.value],
        [final_coords[:len(p)][p.disrupted].dec.value,
         final_coords[len(p):].dec.value],
        color="grey", linestyle="--", alpha=1,
        marker="o", markerfacecolor="tab:green", markeredgecolor="none",
        markersize=10, label="Disrupted binaries", rasterized=True)

handles, labels = plt.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]

ax.set(xlim=[0, 360], ylim=[-90, 90], xlabel="Right Ascension [deg]", ylabel="Declination [deg]")
ax.legend(handles, labels, loc="upper center", fontsize=0.8 * fs)

# plt.savefig("disrupted_pairs.pdf", bbox_inches="tight", format="pdf")

plt.show()
