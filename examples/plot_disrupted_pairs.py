"""
Disrupted binary pairs
======================

Plot each disrupted binary and draw a line between each pair.
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
          'ytick.minor.size': 4}
plt.rcParams.update(params)
# sphinx_gallery_end_ignore


p = cogsworth.pop.Population(100, final_kstar1=[13, 14])
p.create_population(with_timing=False)

galactic_plane = SkyCoord(l=np.linspace(1e-10, 2 * np.pi, 10000), b=np.zeros(10000),
                          unit="rad", frame="galactic").transform_to("icrs")

final_coords = SkyCoord(x=p.final_pos[:, 0], y=p.final_pos[:, 1], z=p.final_pos[:, 2],
                        frame="galactocentric", representation_type="cartesian").icrs

fig, ax = plt.subplots()

in_order = np.argsort(galactic_plane.ra.value)
ax.plot(galactic_plane.ra.value[in_order], galactic_plane.dec.value[in_order], label="Galactic Plane")

ax.plot([final_coords[:len(p)][p.disrupted].ra.value, final_coords[len(p):].ra.value],
        [final_coords[:len(p)][p.disrupted].dec.value, final_coords[len(p):].dec.value],
        color="grey", linestyle="--", alpha=0.5)
ax.scatter([final_coords[:len(p)][p.disrupted].ra.value, final_coords[len(p):].ra.value],
           [final_coords[:len(p)][p.disrupted].dec.value, final_coords[len(p):].dec.value],
           color="C2")

ax.set_xlabel("Right Ascension [deg]")
ax.set_ylabel("Declination [deg]")

plt.show()
