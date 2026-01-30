import numpy as np
import unittest
import cogsworth.sfh as sfh
import os
import astropy.units as u
import gala.potential as gp
from gala.units import galactic
import matplotlib.pyplot as plt
import tempfile

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

    def test_burst_uniform_disc(self):
        """Ensure the burst uniform disc class works"""
        g = sfh.BurstUniformDisc(size=10000,
                                 t_burst=5 * u.Gyr,
                                 R_max=20 * u.kpc,
                                 z_max=1 * u.kpc,
                                 Z_all=0.02)
        self.assertTrue(np.all(g.tau == 5 * u.Gyr))
        self.assertTrue(np.all(g.z <= 1 * u.kpc))
        self.assertTrue(np.all(g.rho <= 20 * u.kpc))
        self.assertTrue(np.all(g.Z == 0.02))

    def test_constant_uniform_disc(self):
        """Ensure the constant uniform disc class works"""
        g = sfh.ConstantUniformDisc(size=10000,
                                    t_burst=5 * u.Gyr,
                                    R_max=20 * u.kpc,
                                    z_max=1 * u.kpc,
                                    Z_all=0.02)
        self.assertTrue(np.all(g.tau <= 5 * u.Gyr))
        self.assertTrue(np.all(g.z <= 1 * u.kpc))
        self.assertTrue(np.all(g.rho <= 20 * u.kpc))
        self.assertTrue(np.all(g.Z == 0.02))

    def test_constant_plummer_sphere(self):
        """Ensure the constant Plummer sphere class works"""
        g = sfh.ConstantPlummerSphere(size=10000,
                                      tau_min=0 * u.Gyr,
                                      tau_max=10 * u.Gyr,
                                      Z_all=0.02,
                                      M=1e10*u.Msun,
                                      a=5.0*u.kpc,
                                      r_trunc=30 * u.kpc)
        self.assertTrue(np.all(g.tau >= 0 * u.Gyr))
        self.assertTrue(np.all(g.tau <= 10 * u.Gyr))
        self.assertTrue(np.all(g.Z == 0.02))
        self.assertTrue(np.all(np.sqrt(g.x**2 + g.y**2 + g.z**2) <= 30 * u.kpc))

    def test_plummer_indexing(self):
        """Ensure the constant Plummer sphere class works"""
        g = sfh.ConstantPlummerSphere(size=100,
                                      tau_min=0 * u.Gyr,
                                      tau_max=10 * u.Gyr,
                                      Z_all=0.02,
                                      M=1e10*u.Msun,
                                      a=5.0*u.kpc,
                                      r_trunc=30 * u.kpc)
        g_subset = g[:5]
        self.assertTrue(g_subset.size == 5)
        self.assertTrue(np.all(g.tau[:5] == g_subset.tau))

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

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-io"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-io"))

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

    def test_io_custom_sfh(self):
        """Check you can save a custom SFH class"""
        class Fixed_Z(sfh.Wagg2022):
            def get_metallicity(self):
                return 0.02

        g = Fixed_Z(size=10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-io-custom"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-io-custom"))

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

    def test_io_custom_sfh_with_params(self):
        """Check you can save a custom SFH class with its own parameters"""
        class Fixed_Z(sfh.Wagg2022):
            def __init__(self, fixed_Z, **kwargs):
                self.fixed_Z = fixed_Z
                super().__init__(**kwargs)

            def get_metallicity(self):
                return self.fixed_Z

        g = Fixed_Z(size=10000, fixed_Z=0.02)

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-io-custom-params"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-io-custom-params"))    

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

    def test_io_SB15(self):
        """Check you can save and load the SB15 class"""
        g = sfh.SandersBinney2015(size=1000, time_bins=1, potential=gp.MilkyWayPotential(version='v2'))

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-io-sb15"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-io-sb15"))

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))
        self.assertTrue(g.potential == g_loaded.potential)

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

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-custom"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-custom"))

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

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

        plt.close('all')

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

    def test_spheroidal_dwarf(self):
        """Test the Carina Dwarf SFH class"""
        # first make a bad one
        it_broke = False
        try:
            s = sfh.SpheroidalDwarf(size=1000, J_0_star=1, alpha=1, eta=1, fixed_Z=0.02,
                                    tau_min=5 * u.Gyr, galaxy_age=4 * u.Gyr)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        # then a carina dwarf
        s = sfh.CarinaDwarf(size=1000, fixed_Z=0.02, tau_min=0 * u.Gyr, galaxy_age=13.5 * u.Gyr)
        self.assertTrue(np.all(s.Z == 0.02))
        self.assertTrue(np.all(s.tau >= 0 * u.Gyr))
        self.assertTrue(np.all(s.tau <= 13.5 * u.Gyr))

    def test_sb15(self):
        """Test the Sanders & Binney (2015) SFH class"""
        s = sfh.SandersBinney2015(size=100, time_bins=1, potential=gp.MilkyWayPotential(version='v2'),
                                  verbose=True)
        # ensure all thick disc stars are older than thin disc stars
        self.assertTrue(np.all(s.tau[s.which_comp == "thick_disc"] >= np.max(s.tau[s.which_comp == "thin_disc"])))

    def test_custom_df(self):
        """Test a custom DF-based SFH class"""
        class SimpleDF(sfh.DistributionFunctionBasedSFH):
            def __init__(self, size, **kwargs):
                super().__init__(size=size, components=["simple"], component_masses=[1], **kwargs)

            def draw_lookback_times(self):
                self._tau = np.full(self.size, 5.0) * u.Gyr
                return self._tau

            def get_metallicity(self):
                self._Z = np.full(self.size, 0.01)
                return self._Z
            
        s = SimpleDF(size=100, potential=gp.MilkyWayPotential(version='v2'), df={
                'type': 'QuasiIsothermal',
                'Rdisk': 3.45,
                'Rsigmar': 7.8,
                'Rsigmaz': 7.8,
                'sigmar0': (48.3*u.km/u.s).decompose(galactic).value,
                'sigmaz0': (30.7*u.km/u.s).decompose(galactic).value,
                'Sigma0': 1.0,
            }, immediately_sample=False)

        s = SimpleDF(size=100, potential=gp.MilkyWayPotential(version='v2'), df=[{
                'type': 'QuasiIsothermal',
                'Rdisk': 3.45,
                'Rsigmar': 7.8,
                'Rsigmaz': 7.8,
                'sigmar0': (48.3*u.km/u.s).decompose(galactic).value,
                'sigmaz0': (30.7*u.km/u.s).decompose(galactic).value,
                'Sigma0': 1.0,
            }], immediately_sample=False)
        s._which_comp = np.array(["simple"] * s.size)
        s.sample()

        self.assertTrue(np.all(s.tau == 5.0 * u.Gyr))
