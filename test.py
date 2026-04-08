import cogsworth
import astropy.units as u
import matplotlib as mpl
import gala.potential as gp
import os
import numpy as np
from gala.units import galactic

g = cogsworth.sfh.Wagg2022()
g.sample(10_000)
g._Z = np.logspace(-4, 0, len(g)) * u.Msun
print(g.Z)

print(np.diff(np.log10(np.percentile(g.tau.value, [5, 95]))))
print(np.diff(np.log10(np.percentile(g.Z.value, [5, 95]))))

g.plot(show=True, colour_by="Z")

# plt.close('all')
