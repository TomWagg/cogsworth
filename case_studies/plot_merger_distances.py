from astropy.visualization import quantity_support
quantity_support()
import matplotlib.pyplot as plt

import numpy as np

future_merger_distances = np.load("../data/future_merger_distances.npy")
past_merger_distances = np.load("../data/past_merger_distances.npy")

distances = np.concatenate((past_merger_distances, future_merger_distances))

print(len(distances))

plt.hist(np.log10(distances), bins="fd")
plt.xlabel(r"Distance, $\log_{10} (D / {\rm kpc})$")
plt.ylabel(r"$\mathrm{d} N / \mathrm{d} \log_{10} (D / {\rm kpc})$")
plt.show()
