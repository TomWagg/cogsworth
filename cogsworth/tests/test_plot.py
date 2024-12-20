import numpy as np
import pandas as pd
import unittest
import cogsworth


class Test(unittest.TestCase):
    def test_plot_cmd(self):
        p = cogsworth.pop.Population(2)
        p.sample_initial_binaries()
        p.perform_stellar_evolution()

        p._observables = pd.DataFrame({"G_abs_1": np.ones(len(p)), "BP_app_1": np.ones(len(p)),
                                       "RP_app_1": np.ones(len(p)),
                                       "G_abs_2": np.ones(len(p)), "BP_app_2": np.ones(len(p)),
                                       "RP_app_2": np.ones(len(p)),
                                       "secondary_brighter": np.zeros(len(p)).astype(bool)})
        cogsworth.plot.plot_cmd(p, show=False)

    def test_plot_orbit(self):
        """Test you can plot a galactic orbit of a binary"""
        p = cogsworth.pop.Population(10, final_kstar1=[13, 14])
        p.create_population()
        while not any(p.disrupted):
            p.create_population()

        p.plot_orbit(p.bin_nums[p.disrupted][0], show=False)

        try:
            p.plot_orbit(-1, show=False)
        except ValueError:
            pass
