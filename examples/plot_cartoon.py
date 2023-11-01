"""
Binary evolution cartoon
========================

Show an example of binary evolution in a simple cartoon.
"""

import cogsworth
import matplotlib.pyplot as plt
import numpy as np

p = cogsworth.pop.Population(100, final_kstar1=[13, 14])
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
