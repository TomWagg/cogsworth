import numpy as np
import unittest
import kicker.galaxy as galaxy
import os


class Test(unittest.TestCase):
    def test_basic_class(self):
        """Check that base class can't be used alone"""

        g = galaxy.Galaxy(size=10000, components=[""], component_masses=[1],
                          immediately_sample=False)
        it_broke = False
        try:
            g.sample()
        except NotImplementedError:
            it_broke = True

        self.assertTrue(it_broke)

    def test_bad_inputs(self):
        """Ensure the classes fail with bad input"""
        g = galaxy.Frankel2018(size=None, immediately_sample=False)
        it_broke = False
        try:
            g.sample()
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        g = galaxy.Frankel2018(size=100, components=None, component_masses=None,
                               immediately_sample=False)
        it_broke = False
        try:
            g.sample()
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_valid_ranges(self):
        """Check that the drawn variables have valid ranges"""
        g = galaxy.Frankel2018(size=10000)

        self.assertTrue((g.tau.min() >= 0) & (g.tau.max() <= g.galaxy_age))
        self.assertTrue(g.Z.min() >= 0.0)

    def test_io(self):
        """Check that a galaxy can be saved and re-loaded"""
        g = galaxy.Frankel2018(size=10000)

        g.save("testing-galaxy-io")

        g_loaded = galaxy.load("testing-galaxy-io")

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.positions.icrs.distance == g_loaded.positions.icrs.distance))

        os.remove("testing-galaxy-io.h5")
        os.remove("testing-galaxy-io-galaxy-params.txt")
