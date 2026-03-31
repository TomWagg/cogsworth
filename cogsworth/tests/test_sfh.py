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

        g = sfh.StarFormationHistory()
        it_broke = False
        try:
            g.draw_lookback_times(10)
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.draw_radii(10)
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.draw_phi(10)
        except NotImplementedError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            g.draw_heights(10)
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
        g = sfh.BurstUniformDisc(t_burst=5 * u.Gyr,
                                 R_max=20 * u.kpc,
                                 z_max=1 * u.kpc,
                                 Z_all=0.02)
        g.sample(10000)
        self.assertTrue(np.all(g.tau == 5 * u.Gyr))
        self.assertTrue(np.all(g.z <= 1 * u.kpc))
        self.assertTrue(np.all(g.rho <= 20 * u.kpc))
        self.assertTrue(np.all(g.Z == 0.02))

    def test_constant_uniform_disc(self):
        """Ensure the constant uniform disc class works"""
        g = sfh.ConstantUniformDisc(t_burst=5 * u.Gyr,
                                    R_max=20 * u.kpc,
                                    z_max=1 * u.kpc,
                                    Z_all=0.02)
        g.sample(10000)
        self.assertTrue(np.all(g.tau <= 5 * u.Gyr))
        self.assertTrue(np.all(g.z <= 1 * u.kpc))
        self.assertTrue(np.all(g.rho <= 20 * u.kpc))
        self.assertTrue(np.all(g.Z == 0.02))

    def test_constant_plummer_sphere(self):
        """Ensure the constant Plummer sphere class works"""
        g = sfh.ConstantPlummerSphere(tau_min=0 * u.Gyr,
                                      tau_max=10 * u.Gyr,
                                      Z_all=0.02,
                                      M=1e10*u.Msun,
                                      a=5.0*u.kpc,
                                      r_trunc=30 * u.kpc)
        g.sample(10000)
        self.assertTrue(np.all(g.tau >= 0 * u.Gyr))
        self.assertTrue(np.all(g.tau <= 10 * u.Gyr))
        self.assertTrue(np.all(g.Z == 0.02))
        self.assertTrue(np.all(np.sqrt(g.x**2 + g.y**2 + g.z**2) <= 30 * u.kpc))

    def test_plummer_indexing(self):
        """Ensure the constant Plummer sphere class works"""
        g = sfh.ConstantPlummerSphere(tau_min=0 * u.Gyr,
                                      tau_max=10 * u.Gyr,
                                      Z_all=0.02,
                                      M=1e10*u.Msun,
                                      a=5.0*u.kpc,
                                      r_trunc=30 * u.kpc)
        g.sample(100)
        g_subset = g[:5]
        self.assertTrue(len(g_subset) == 5)
        self.assertTrue(np.all(g.tau[:5] == g_subset.tau))

    def test_valid_ranges(self):
        """Check that the drawn variables have valid ranges"""
        g = sfh.Wagg2022()
        g.sample(10000)

        self.assertTrue((g.tau.min() >= 0) & (g.tau.max() <= g.galaxy_age))
        self.assertTrue(g.Z.min() >= 0.0)

    def test_io(self):
        """Check that a galaxy can be saved and re-loaded"""
        g = sfh.Wagg2022()
        g.sample(10000)

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

        g = Fixed_Z()
        g.sample(10000)

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

        g = Fixed_Z(fixed_Z=0.02)
        g.sample(10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-io-custom-params"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-io-custom-params"))    

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

    def test_io_SB15(self):
        """Check you can save and load the SB15 class"""
        g = sfh.SandersBinney2015(time_bins=1, potential=gp.MilkyWayPotential(version='v2'))
        g.sample(1000)

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

        g = sfh.SandersBinney2015(potential=gp.MilkyWayPotential(version='v2'))
        len(g)
        g.components

        attrs = ["tau", "Z", "x", "y", "z", "positions", "phi", "rho", "v_x", "v_y", "v_z", "v_R", "v_T", "v_z", "v_phi"]

        for attr in attrs:
            it_broke = False
            try:
                getattr(g, attr)
            except Exception as e:
                it_broke = True
            self.assertTrue(it_broke)

        g.sample(1000)
        for attr in attrs:
            it_broke = False
            try:
                getattr(g, attr)
            except Exception as e:
                it_broke = True
            self.assertFalse(it_broke)
        
    def test_custom_galaxy(self):
        """Test saving a custom galaxy class"""

        class Custom(sfh.BurstUniformDisc):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        g = Custom()
        g.sample(10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "testing-galaxy-custom"))
            g_loaded = sfh.load(os.path.join(tmpdir, "testing-galaxy-custom"))

        self.assertTrue(np.all(g.tau == g_loaded.tau))
        self.assertTrue(np.all(g.rho == g_loaded.rho))
        self.assertTrue(np.all(g.z == g_loaded.z))

    def test_plot(self):
        """Test plotting capabilities"""
        g = sfh.Wagg2022()
        g.sample(1000)

        it_broke = False
        try:
            g.plot(xlim=(-20, 20), ylim=(-20, 20), zlim=(-7, 7), show=False)
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)

        plt.close('all')

    def test_indexing(self):
        """Ensure that indexing works correctly (reprs too)"""
        g = sfh.Wagg2022()
        g.sample(10)
        print(g)

        # make sure it fails for strings
        it_worked = True
        try:
            g["absolute nonsense mate"]
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        inds = [np.random.randint(len(g)),
                np.random.randint(len(g), size=4),
                list(np.random.randint(len(g), size=2)),
                slice(0, 7, 3)]

        for ind in inds:
            g_ind = g[ind]
            if isinstance(ind, slice):
                ind = list(range(ind.stop)[ind])
            self.assertTrue(np.all(g.tau[ind] == g_ind.tau), msg=f"tau mismatch for ind={ind}")

    def test_spheroidal_dwarf(self):
        """Test the Carina Dwarf SFH class"""
        # first make a bad one
        it_broke = False
        try:
            s = sfh.SpheroidalDwarf(J_0_star=1, alpha=1, eta=1, fixed_Z=0.02,
                                    tau_min=5 * u.Gyr, galaxy_age=4 * u.Gyr)
            s.sample(1000)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        # then a carina dwarf
        s = sfh.CarinaDwarf(fixed_Z=0.02, tau_min=0 * u.Gyr, galaxy_age=13.5 * u.Gyr)
        s.sample(1000)
        self.assertTrue(np.all(s.Z == 0.02))
        self.assertTrue(np.all(s.tau >= 0 * u.Gyr))
        self.assertTrue(np.all(s.tau <= 13.5 * u.Gyr))

    def test_sb15(self):
        """Test the Sanders & Binney (2015) SFH class"""
        s = sfh.SandersBinney2015(time_bins=1, potential=gp.MilkyWayPotential(version='v2'),
                                  verbose=True)
        s.sample(100)

    def test_custom_df(self):
        """Test a custom DF-based SFH class"""
        class SimpleDF(sfh.DistributionFunctionBasedSFH):
            def __init__(self, **kwargs):
                super().__init__(components=["simple"], component_masses=[1], **kwargs)

            def draw_lookback_times(self, size):
                self._tau = np.full(size, 5.0) * u.Gyr
                return self._tau

            def get_metallicity(self):
                self._Z = np.full(len(self._tau), 0.01)
                return self._Z
            
        s = SimpleDF(potential=gp.MilkyWayPotential(version='v2'), df={
                'type': 'QuasiIsothermal',
                'Rdisk': 3.45,
                'Rsigmar': 7.8,
                'Rsigmaz': 7.8,
                'sigmar0': (48.3*u.km/u.s).decompose(galactic).value,
                'sigmaz0': (30.7*u.km/u.s).decompose(galactic).value,
                'Sigma0': 1.0,
            })
        s.sample(1000)

        self.assertTrue(np.all(s.tau == 5.0 * u.Gyr))
