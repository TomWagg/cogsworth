import numpy as np
import unittest
import cogsworth.sfh as sfh
import os


class Test(unittest.TestCase):
    def test_basic_class(self):
        """Check that base class can't be used alone"""

        g = sfh.StarFormationHistory(size=10000, components=[""], component_masses=[1],
                                     immediately_sample=False)
        it_broke = False
        try:
            g.draw_lookback_times()
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.draw_radii()
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.draw_phi()
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.draw_heights()
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.get_metallicity()
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_bad_inputs(self):
        """Ensure the classes fail with bad input"""
        g = sfh.Wagg2022(size=None, immediately_sample=False)
        it_broke = False
        try:
            g.sample()
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        g = sfh.Wagg2022(size=100, components=None, component_masses=None, immediately_sample=False)
        it_broke = False
        try:
            g.sample()
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_valid_ranges(self):
        """Check that the drawn variables have valid ranges"""
        g = sfh.Wagg2022(size=10000)

        self.assertTrue((g.tau.min() >= 0) & (g.tau.max() <= g.galaxy_age))
        self.assertTrue(g.Z.min() >= 0.0)

    def test_io(self):
        """Check that a galaxy can be saved and re-loaded"""
        g = sfh.Wagg2022(size=10000)

        g.save("testing-galaxy-io")

        g_loaded = sfh.load("testing-galaxy-io")

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

        os.remove("testing-galaxy-io.h5")

    def test_getters(self):
        """Test getting attributes"""
        it_broke = False
        try:
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            len(g)
            g.components
            g.component_masses
            g.tau
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            g.Z
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            g.x
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            g.y
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            g.z
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            g.positions
            g = sfh.Wagg2022(size=10, immediately_sample=False)
            g.which_comp
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)

    def test_setters(self):
        """Test setting attributes"""
        g = sfh.Wagg2022(size=10000, immediately_sample=False)
        g.size = 100
        self.assertTrue(g.size == 100)

        # make sure it crashes for invalid inputs
        it_broke = False
        try:
            g.size = -1
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.size = "not an int"
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_custom_galaxy(self):
        """Test saving a custom galaxy class"""

        class Custom(sfh.Wagg2022):
            def __init__(self, size, components=["low_alpha_disc"], component_masses=[1], **kwargs):
                super().__init__(size, components, component_masses, **kwargs)

        g = Custom(size=100)

        g.save("testing-galaxy-custom")

        g_loaded = sfh.load("testing-galaxy-custom")

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

        os.remove("testing-galaxy-custom.h5")

    def test_plot(self):
        """Test plotting capabilities"""
        g = sfh.Wagg2022(size=1000)

        it_broke = False
        try:
            g.plot(component="low_alpha_disc", xlim=(-20, 20), ylim=(-20, 20), zlim=(-7, 7), show=False)
            g.plot(coordinates="cylindrical", component="low_alpha_disc", show=False)
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)

        it_broke = False
        try:
            g.plot(coordinates="nonsense")
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_indexing(self):
        """Ensure that indexing works correctly (reprs too)"""
        g = sfh.Wagg2022(size=10)
        print(g)

        # make sure it fails for strings
        it_worked = True
        try:
            g["absolute nonsense mate"]
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        inds = [np.random.randint(g.size),
                np.random.randint(g.size, size=4),
                list(np.random.randint(g.size, size=2)),
                slice(0, 7, 3)]

        for ind in inds:
            g_ind = g[ind]
            if isinstance(ind, slice):
                ind = list(range(ind.stop)[ind])
            self.assertTrue(np.all(g.tau[ind] == g_ind.tau))
