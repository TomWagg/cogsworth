import importlib.util
import os
import unittest
from unittest import mock
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as coords

import cogsworth.obs.gaia as gaia
import cogsworth.obs.observables as obs
import cogsworth.pop as pop

HAS_GAIAMOCK = importlib.util.find_spec("gaiamock") is not None

# the full scanning law is ~1 GB, so we only ship the healpix files for the sky positions used in these tests
GAIAMOCK_TEST_DATA = os.path.join(os.path.dirname(__file__), "test_data", "gaiamock")


class FakePopulation():
    """Just enough of a Population to select gaiamock sources"""
    def __init__(self):
        self.processes = 1
        self.__citations__ = []

        # BH+MS, MS+NS, BH+BH, disrupted BH+MS, faint BH+MS, WD+MS, wide MS+MS, merger
        self.final_bpp = pd.DataFrame({
            "kstar_1": [14, 1, 14, 14, 14, 11, 1, 1],
            "kstar_2": [1, 13, 14, 1, 1, 1, 1, 15],
            "sep": [100.0, 200.0, 50.0, -1.0, 100.0, 100.0, 1e6, 0.0],
            "mass_1": [10.0, 1.0, 10.0, 10.0, 10.0, 0.6, 1.0, 2.0],
            "mass_2": [1.0, 1.4, 10.0, 1.0, 1.0, 1.0, 0.8, 0.0],
            "porb": [300.0, 500.0, 100.0, -1.0, 300.0, 300.0, 1e8, 0.0],
            "ecc": [0.1, 0.2, 0.0, -1.0, 0.1, 0.1, 0.5, 0.0],
        }, index=pd.Index([10, 11, 12, 13, 14, 15, 16, 17], name="bin_num"))
        self.disrupted = self.final_bpp["sep"].values < 0

        # apparent magnitudes of each star, with A_G = 1 at 1 kpc
        G_1 = np.array([np.inf, 16.0, np.inf, 15.0, np.inf, 20.0, 15.0, 14.0])
        G_2 = np.array([15.0, np.inf, np.inf, 17.0, 22.0, 15.0, 16.0, np.inf])
        G = obs.add_mags(G_1, G_2)
        G[self.disrupted] = np.nan
        self.observables = pd.DataFrame({
            "Gaia_G_EDR3_app": G, "Gaia_G_EDR3_abs": G - 11,
            "Gaia_G_EDR3_app_1": G_1, "Gaia_G_EDR3_abs_1": G_1 - 11,
            "Gaia_G_EDR3_app_2": G_2, "Gaia_G_EDR3_abs_2": G_2 - 11,
        }, index=self.final_bpp.index)

        # place everything ~1 kpc from the Sun (plus an extra row for the disrupted secondary)
        n = len(self.final_bpp) + self.disrupted.sum()
        self.final_pos = np.tile([-7.122, 0.0, 0.0], (n, 1)) * u.kpc
        self.final_vel = np.tile([10.0, 230.0, 5.0], (n, 1)) * u.km / u.s

    def __len__(self):
        return len(self.final_bpp)

    def get_final_mw_skycoord(self):
        return coords.SkyCoord(x=self.final_pos[:, 0], y=self.final_pos[:, 1], z=self.final_pos[:, 2],
                               v_x=self.final_vel[:, 0], v_y=self.final_vel[:, 1], v_z=self.final_vel[:, 2],
                               representation_type="cartesian", unit=u.kpc, frame="galactocentric")


class Test(unittest.TestCase):
    def test_source_selection(self):
        """Test that bound binaries in the magnitude range are selected"""
        p = FakePopulation()
        with self.assertWarns(UserWarning):
            sources = gaia.get_gaiamock_sources(p, seed=42)

        self.assertEqual(sources.index.tolist(), [10, 11, 15, 16])
        self.assertTrue(set(gaia.CASCADE_ARGS) <= set(sources.columns))

        # brighter star should always be m1
        self.assertTrue(np.allclose(sources["m1"], [1.0, 1.0, 1.0, 1.0]))
        self.assertTrue(np.allclose(sources["m2"], [10.0, 1.4, 0.6, 0.8]))
        self.assertTrue(np.allclose(sources["f"], [0.0, 0.0, 0.01, 10**-0.4]))
        self.assertTrue(np.allclose(sources["period"], [300.0, 500.0, 300.0, 1e8]))
        G = p.observables["Gaia_G_EDR3_app"].loc[sources.index]
        self.assertTrue(np.allclose(sources["phot_g_mean_mag"], G))

        # sources are ~1 kpc away, so parallax ~1 mas and A_G ~ 1
        self.assertTrue(np.allclose(sources["parallax"], 1.0, atol=0.05))
        self.assertTrue(np.allclose(sources["A_G"], 1.0, atol=0.1))

        # random orientations should be in range
        self.assertTrue(np.all((sources["inc_deg"] >= 0) & (sources["inc_deg"] <= 180)))
        self.assertTrue(np.all((sources["Tp"] >= 0) & (sources["Tp"] <= sources["period"])))

        # seeds should make orientations reproducible
        first = gaia.get_gaiamock_sources(p, seed=1, remove_wide_binaries=True)
        second = gaia.get_gaiamock_sources(p, seed=1, remove_wide_binaries=True)
        self.assertTrue(first.equals(second))

    def test_wide_binaries(self):
        """Test that wide binaries are warned about or removed"""
        p = FakePopulation()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            sources = gaia.get_gaiamock_sources(p, remove_wide_binaries=True)
            self.assertEqual(sources.index.tolist(), [10, 11, 15])

            # a larger threshold means nothing is wide
            sources = gaia.get_gaiamock_sources(p, wide_separation=10 * u.arcsec)
            self.assertEqual(sources.index.tolist(), [10, 11, 15, 16])

        # binaries with dark companions are never wide
        p.final_bpp.loc[10, "sep"] = 1e6
        sources = gaia.get_gaiamock_sources(p, remove_wide_binaries=True)
        self.assertEqual(sources.index.tolist(), [10, 11, 15])

    def test_bad_input(self):
        """Test that it breaks when it should"""
        p = FakePopulation()
        with self.assertRaises(ValueError):
            gaia.get_gaiamock_sources(p, G_filter="Gaia_BP_EDR3")
        with self.assertRaises(ValueError):
            gaia.run_gaiamock(gaia.get_gaiamock_sources(p, remove_wide_binaries=True), data_release="dr6")

    def test_decode(self):
        """Test decoding of raw gaiamock results"""
        rows = [[0] * 23,
                [-1, 1.1] + [-1] * 21,
                [-9, 30] + [-9] * 10 + [2.5, 3] + [-9] * 9,
                [-7, 20] + [-7] * 6 + [1.9, 4] + [-7] * 13,
                [2.0, 0.05, *[0.0] * 8, 300.0, 1.0, 0, 0, 0.3, 0.01, 60, 1.5, 0.05, 20, 60, 1.0, 3.0]]
        d = gaia.decode_cascade(rows)
        self.assertEqual(d["solution_type"].tolist(),
                         ["unobservable", "single_star", "accel9", "accel7", "orbital"])
        self.assertTrue(np.isnan(d["ruwe"][0]))
        self.assertTrue(np.allclose(d["ruwe"].values[1:], [1.1, 2.5, 1.9, 3.0]))
        self.assertTrue(np.allclose(d["accel_significance"].values[2:4], [30, 20]))

        # a0/sigma = 30, parallax/sigma = 40 so only the orbital solution passes
        self.assertEqual(d["published_dr4_loose"].tolist(), [False] * 4 + [True])

    def test_empty(self):
        """Test that no sources gives an empty table"""
        empty = gaia.get_gaiamock_sources(FakePopulation(), G_range=(0, 1))
        self.assertEqual(len(empty), 0)

        d = gaia.run_gaiamock(empty)
        self.assertEqual(len(d), 0)
        self.assertIn("solution_type", d.columns)

    @unittest.skipUnless(HAS_GAIAMOCK, "gaiamock not installed")
    @mock.patch.dict(os.environ, {"GAIAMOCK_DATA_DIR": GAIAMOCK_TEST_DATA})
    def test_end_to_end(self):
        """A 15 Msun BH on a 300 d orbit around a G = 14 star at 1 kpc should be a clear DR4 orbit"""
        sources = pd.DataFrame([dict(ra=120.0, dec=-30.0, parallax=1.0, pmra=-5.0, pmdec=3.0, m1=1.0, m2=15.0,
                                     period=300.0, Tp=50.0, ecc=0.3, omega=1.0, inc_deg=60.0, w=2.0,
                                     phot_g_mean_mag=14.0, f=0.0)])

        # global random state should not be affected
        state = np.random.get_state()
        d = gaia.run_gaiamock(sources, seed=0)
        self.assertTrue(np.all(state[1] == np.random.get_state()[1]))

        self.assertTrue(d["published_dr4_loose"][0])
        self.assertLess(abs(d["fit_period"][0] / 300 - 1), 0.02)

        # results shouldn't depend on the number of processes
        sources = pd.concat([sources] * 2, ignore_index=True)
        serial = gaia.run_gaiamock(sources, seed=0)
        self.assertTrue(serial.equals(gaia.run_gaiamock(sources, seed=0, processes=2)))

    @unittest.skipUnless(HAS_GAIAMOCK, "gaiamock not installed")
    @mock.patch.dict(os.environ, {"GAIAMOCK_DATA_DIR": GAIAMOCK_TEST_DATA})
    def test_population_astrometry(self):
        """Test the Population wrapper combines sources and cascade results"""
        p = FakePopulation()
        astrometry = pop.Population.get_gaia_astrometry(p, seed=42, remove_wide_binaries=True)

        # kwargs should be passed through to source selection
        self.assertEqual(astrometry.index.tolist(), [10, 11, 15])
        self.assertTrue(set(gaia.CASCADE_ARGS) <= set(astrometry.columns))
        self.assertIn("solution_type", astrometry.columns)
        self.assertIn("gaiamock", p.__citations__)

        # same seed, same result
        again = pop.Population.get_gaia_astrometry(p, seed=42, remove_wide_binaries=True)
        self.assertTrue(astrometry.equals(again))
