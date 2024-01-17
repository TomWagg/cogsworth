"""
Initial periods and mass transfer
=================================

Show how initial periods are different for mass transferring binaries.

Mass transferring binaries generally started with shorter periods.
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


p = cogsworth.pop.Population(5000)
p.sample_initial_binaries()
p.perform_stellar_evolution()

fig, ax = plt.subplots()

mt_bin_nums = p.bpp[p.bpp["evol_type"] == 3]["bin_num"].unique()
no_mt_bin_nums = p.bin_nums[~np.isin(p.bin_nums, mt_bin_nums)]
porb_init = np.log10(p.initC["porb"])

bins = np.linspace(0, 5, 20)

ax.hist(porb_init, bins=bins, density=True, label="All binaries")
ax.hist(porb_init.loc[mt_bin_nums], bins=bins, alpha=0.5, density=True, label="Mass transfer binaries")

ax.legend()

ax.set_xlabel("Initial period [log days]")
ax.set_ylabel(r"${\rm d}N/{\rm d}({\rm log} P)$")

plt.show()
plt.close()
