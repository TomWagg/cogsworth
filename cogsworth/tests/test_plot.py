import numpy as np
import pandas as pd
import astropy.units as u
import unittest
import cogsworth
import matplotlib.pyplot as plt


class Test(unittest.TestCase):
    def test_plot_cmd(self):
        p = cogsworth.pop.Population(2, use_default_BSE_settings=True)
        p.sample_initial_binaries()
        p.perform_stellar_evolution()

        p._observables = pd.DataFrame({
            "Gaia_G_EDR3_abs_1": np.ones(len(p)),
            "Gaia_BP_EDR3_app_1": np.ones(len(p)),
            "Gaia_RP_EDR3_app_1": np.ones(len(p)),
            "Gaia_G_EDR3_abs_2": np.ones(len(p)),
            "Gaia_BP_EDR3_app_2": np.ones(len(p)),
            "Gaia_RP_EDR3_app_2": np.ones(len(p)),
            "secondary_brighter": np.zeros(len(p)).astype(bool)
        })
        cogsworth.plot.plot_cmd(p, show=False)
        plt.close('all')

    def test_plot_orbit(self):
        """Test you can plot a galactic orbit of a binary"""
        p = cogsworth.pop.Population(50, final_kstar1=[13, 14], use_default_BSE_settings=True)
        p.create_population()
        while not any(p.disrupted):
            p.create_population()

        p.plot_orbit(p.bin_nums[p.disrupted][0], show=False)

        # test plotting an orbit with a non-existent binary fails
        should_fail = False
        try:
            p.plot_orbit(-1, show=False)
        except ValueError:
            should_fail = True
        self.assertTrue(should_fail, "Plotting an orbit with a non-existent binary should fail")

        # test plotting an orbit with garbage time limits fails
        should_fail = False
        try:
            p.plot_orbit(0, t_min=10 * u.Myr, t_max=0 * u.Myr, show=False)
        except ValueError:
            should_fail = True
        self.assertTrue(should_fail, "Plotting an orbit with t_min > t_max should fail")

        plt.close('all')
