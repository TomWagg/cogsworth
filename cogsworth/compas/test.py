# import matplotlib.pyplot as plt
# import cogsworth
# from utils import create_bpp_from_COMPAS_files

# bpp = create_bpp_from_COMPAS_files("COMPAS_Output_9/COMPAS_Output.h5")
# cogsworth.plot.plot_cartoon_evolution(bpp, 1751172534, show=False)
# # plt.close()
# plt.show()

import cogsworth

p = cogsworth.compas.pop.COMPASPopulation(10, "compas_config.yaml")
p.perform_stellar_evolution()

print(p.initial_binaries)
