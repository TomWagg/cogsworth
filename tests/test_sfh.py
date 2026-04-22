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

        attrs = ["tau", "Z", "x", "y", "z", "positions", "velocities", "phi", "rho",
                 "v_x", "v_y", "v_z", "v_R", "v_T", "v_z", "v_phi"]

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
                np.random.choice(len(g), size=4),
                list(np.random.choice(len(g), size=2)),
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

    def test_sormani_bar(self):
        """Test the Sormani & Binney (2022) SFH class"""
        # sample with a gala potential
        s = sfh.MilkyWayBarSormani2022(potential=gp.MilkyWayPotential(version='v2'))
        s.sample(100)

        # same with an agama potential
        s = sfh.MilkyWayBarSormani2022(potential=gp.MilkyWayPotential(version='v2').as_interop("agama"))
        s.sample(100)

    def test_custom_df(self):
        """Test a custom DF-based SFH class"""
        class SimpleDF(sfh.DistributionFunctionBasedSFH):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

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
        s.sample(size=1000)

        self.assertTrue(np.all(s.tau == 5.0 * u.Gyr))

    def test_sfh_repr(self):
        """Test repr of a StarFormationHistory before and after sampling"""
        g = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)
        self.assertIn("not yet sampled", repr(g))
        g.sample(10)
        self.assertIn("size=10", repr(g))

    def test_sfh_arithmetic(self):
        """Test __add__, __mul__, __rmul__, and copy on StarFormationHistory"""
        g1 = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)
        g2 = sfh.BurstUniformDisc(t_burst=10 * u.Gyr)

        # sfh + sfh → CompositeStarFormationHistory
        comp = g1 + g2
        self.assertIsInstance(comp, sfh.CompositeStarFormationHistory)
        self.assertEqual(len(comp.components), 2)

        # sfh + composite → composite gains one component
        g3 = sfh.BurstUniformDisc(t_burst=8 * u.Gyr)
        comp2 = g3 + comp
        self.assertEqual(len(comp2.components), 3)

        # unsupported type
        self.assertEqual(g1.__add__("not an sfh"), NotImplemented)

        # scalar multiply
        g1_scaled = g1 * 2.0
        self.assertAlmostEqual(g1_scaled._composite_weight, g1._composite_weight * 2.0)
        # right multiply
        self.assertAlmostEqual((3.0 * g1)._composite_weight, g1._composite_weight * 3.0)
        # unsupported type
        self.assertEqual(g1.__mul__("not a number"), NotImplemented)

        # copy returns a fresh instance of the same class
        g1.sample(10)
        g1_copy = g1.copy()
        self.assertIsInstance(g1_copy, sfh.BurstUniformDisc)
        self.assertTrue(np.all(g1_copy.tau == g1.tau))

    def test_sfh_getitem_errors(self):
        """Test error and edge-case paths in StarFormationHistory.__getitem__"""
        g = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)

        # bad index type on a plain StarFormationHistory subclass
        with self.assertRaises(ValueError):
            g["bad index"]

        # indexing before sampling returns a new unsampled instance
        g_sub = g[0]
        self.assertIsNone(g_sub._tau)

    def test_sfh_plot(self):
        """Test plot on a plain StarFormationHistory subclass"""
        g = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)
        g.sample(100)
        g.plot(show=False)
        plt.close('all')

    def test_citations(self):
        """Test citation output paths for both base and composite SFH classes"""
        # empty citations list → early return
        sfh.StarFormationHistory.sfh_citation_statement([], "")

        # get_citations on a StarFormationHistory with citations → prints bibtex
        g = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)
        g.get_citations(filename="")

        # write bibtex to file
        with tempfile.TemporaryDirectory() as tmpdir:
            bib_path = os.path.join(tmpdir, "refs.bib")
            g.get_citations(filename=bib_path)
            self.assertTrue(os.path.exists(bib_path))

        # get_citations on a CompositeStarFormationHistory
        comp = sfh.Wagg2022()
        comp.get_citations(filename="")

    def test_composite_indexing_advanced(self):
        """Test boolean-array indexing, chained indexing, and repeated indices"""
        g = sfh.Wagg2022()
        g.sample(50)

        # boolean array indexing
        mask = np.zeros(len(g), dtype=bool)
        mask[:10] = True
        g_bool = g[mask]
        self.assertEqual(len(g_bool), 10)
        self.assertTrue(np.allclose(g.tau[mask].value, g_bool.tau.value))

        # chained indexing: second index applied to a result that already has _sort_order
        ind1 = np.array([40, 5, 20])   # unsorted → result has _sort_order
        ind2 = np.array([2, 0])
        g_sub2 = g[ind1][ind2]
        self.assertTrue(np.allclose(g.tau[ind1][ind2].value, g_sub2.tau.value))

        # repeated indices
        ind_rep = np.array([3, 1, 3])
        g_rep = g[ind_rep]
        self.assertEqual(len(g_rep), 3)
        self.assertTrue(np.allclose(g.tau[ind_rep].value, g_rep.tau.value))

    def test_composite_getattr_setattr(self):
        """Test __getattr__ AttributeError, __setattr__ for COMBINE_ATTRS, and unsampled __len__"""
        g = sfh.Wagg2022()

        # __len__ on unsampled composite is 0
        self.assertEqual(len(g), 0)

        g.sample(20)

        # accessing a non-existent attribute raises AttributeError
        with self.assertRaises(AttributeError):
            _ = g.not_a_real_attribute

        # setting a COMBINE_ATTR distributes values to components and reads back correctly
        new_tau = np.linspace(1, 10, 20) * u.Gyr
        g._tau = new_tau
        self.assertTrue(np.allclose(g.tau.value, new_tau.value))

        # setting with wrong type raises ValueError
        with self.assertRaises(ValueError):
            g._tau = "not an array"

        # setattr round-trip through sort_order: index with unsorted ind, then overwrite _tau
        g.sample(20)
        ind = np.array([15, 3, 10])   # unsorted → sub has _sort_order
        g_sub = g[ind]
        new_sub_tau = np.array([1.0, 2.0, 3.0]) * u.Gyr
        g_sub._tau = new_sub_tau
        self.assertTrue(np.allclose(g_sub.tau.value, new_sub_tau.value))

    def test_composite_setattr_list(self):
        """Test __setattr__ for COMBINE_ATTRS when the value is a list"""
        g = sfh.Wagg2022()
        g.sample(20)

        # setting a COMBINE_ATTR distributes values to components and reads back correctly
        new_tau = [1] * len(g)
        g._tau = new_tau

        self.assertTrue(np.allclose(g.tau, new_tau))

    def test_composite_add(self):
        """Test CompositeStarFormationHistory.__add__ with all operand types"""
        comp = sfh.Wagg2022()
        disc = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)

        # composite + sfh
        combined = comp + disc
        self.assertEqual(len(combined.components), len(comp.components) + 1)

        # composite + composite
        comp2 = sfh.Wagg2022()
        combined2 = comp + comp2
        self.assertEqual(len(combined2.components), len(comp.components) + len(comp2.components))

        # unsupported type
        self.assertEqual(comp.__add__("not an sfh"), NotImplemented)

    def test_concat(self):
        """Test the concat() function for various input cases"""
        g1 = sfh.BurstUniformDisc(t_burst=5 * u.Gyr)
        g1.sample(30)
        g2 = sfh.BurstUniformDisc(t_burst=10 * u.Gyr)
        g2.sample(30)

        # single object is returned as-is
        self.assertIs(sfh.concat(g1), g1)

        # empty raises ValueError
        with self.assertRaises(ValueError):
            sfh.concat()

        # two StarFormationHistorys are concatenated
        combined = sfh.concat(g1, g2)
        self.assertEqual(len(combined), 60)

        # two CompositeStarFormationHistorys are concatenated
        c1 = sfh.Wagg2022()
        c1.sample(20)
        c2 = sfh.Wagg2022()
        c2.sample(20)
        combined_c = sfh.concat(c1, c2)
        self.assertEqual(len(combined_c), 40)

        # mismatched component counts raises ValueError
        c_extra = c1 + sfh.BurstUniformDisc(t_burst=5 * u.Gyr)
        with self.assertRaises(ValueError):
            sfh.concat(c1, c_extra)

        # mixed types raises ValueError
        with self.assertRaises(ValueError):
            sfh.concat(g1, c1)

    def test_load_bad_key(self):
        """Test that loading from a nonexistent key raises ValueError"""
        g = sfh.Wagg2022()
        g.sample(100)

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "test.h5")
            g.save(fname)
            with self.assertRaises(ValueError):
                sfh.load(fname, key="nonexistent_key")
