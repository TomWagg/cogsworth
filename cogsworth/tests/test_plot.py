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
