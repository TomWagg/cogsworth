"""
Binary evolution cartoon
========================

Show an example of binary evolution in a simple cartoon.
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

p = cogsworth.pop.Population(100, final_kstar1=[13, 14], processes=1, BSE_settings={"binfrac": 1.0})
p.sample_initial_binaries()
p.perform_stellar_evolution()

# find something that experiences mass transfer, common envelope, and undergoes a supernova
mt_bin_nums = p.bpp[p.bpp["evol_type"] == 3]["bin_num"].values
ce_bin_nums = p.bpp[p.bpp["evol_type"] == 7]["bin_num"].values
sn_bin_nums = p.bpp[p.bpp["evol_type"] == 15]["bin_num"].values
uni, counts = np.unique(np.concatenate((mt_bin_nums, ce_bin_nums, sn_bin_nums)), return_counts=True)
bin_num = uni[np.argmax(counts)]

p.plot_cartoon_binary(bin_num, show=False)
plt.tight_layout()          # <-- this is just to get rid of weird padding in online docs
plt.show()
