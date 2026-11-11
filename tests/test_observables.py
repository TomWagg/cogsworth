import numpy as np
import pandas as pd
import pytest
import unittest
from astropy.coordinates import SkyCoord
import cogsworth.obs.observables as obs
import astropy.units as u


class Test(unittest.TestCase):
    def test_imports(self):
        """Test that the imports work"""
        try:
            from cogsworth import obs
            from cogsworth.obs import mist
            from cogsworth.obs import observables
        except ImportError:
            self.fail("Failed to import mist or observables")

        try:
            from cogsworth.obs import nonsense
        except ImportError:
            pass
        else:
            self.fail("Should have failed to import nonsense")

    def test_magnitudes(self):
        """Test the magnitude functions"""
        # check using ints and lists is the same as floats and arrays
        from_int_list = obs.add_mags(1, 2, [1, 2])
        from_float_array = obs.add_mags(1.0, 2.0, np.array([1, 2]))
        self.assertTrue(np.all(from_int_list == from_float_array))

        # catch the case of strings
        it_broke = False
        try:
            obs.add_mags("not a mag", "NOPE")
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        # check NaNs
        with_nan = obs.add_mags(1, 2, np.nan, remove_nans=True)
        without_nan = obs.add_mags(1, 2, remove_nans=True)
        self.assertTrue(with_nan == without_nan)

        # check mag conversion
        m_app = np.array([20, 21, 22])
        distance = [2, 2, 2] * u.kpc
        M_abs = obs.get_absolute_mag(m_app=m_app, distance=distance)
        m_app_converted = obs.get_apparent_mag(M_abs=M_abs, distance=distance)
        print(m_app)
        print(m_app_converted)
        self.assertTrue(np.all(m_app == m_app_converted))

    def test_bad_input(self):
        """Test that it breaks when it should"""
        it_worked = True
        try:
            obs.get_photometry(filters=["Gaia_G_EDR3"])
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        it_worked = True
        try:
            obs.get_photometry(filters=["Gaia_G_EDR3"], population="dummy")
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        it_worked = True
        try:
            obs.get_photometry(filters=["Gaia_G_EDR3"], population="dummy", distances="dummy")
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

    def test_bad_filters(self):
        """Test that unknown filters and main filters are caught before any computation"""
        with self.assertRaises(KeyError):
            obs.get_photometry(filters=["NOT A FILTER"], population="dummy", distances="dummy",
                               ignore_extinction=True)
        with self.assertRaises(ValueError):
            obs.get_photometry(filters=["Gaia_G_EDR3"], population="dummy", distances="dummy",
                               ignore_extinction=True, main_filter="Gaia_BP_EDR3")


# --- extinction (using fake dust maps from conftest.py so that the large dust map files aren't needed) ---
def galactic(lon, lat, dist):
    return SkyCoord(l=np.atleast_1d(lon) * u.deg, b=np.atleast_1d(lat) * u.deg,
                    distance=np.atleast_1d(dist) * u.kpc, frame="galactic")


def test_extinction_bad_input(fake_dustmaps):
    """Bad dust maps, kwargs or coordinates should be caught"""
    coords = galactic(30, 5, 1)
    with pytest.raises(ValueError, match="Unknown dust map"):
        obs.get_extinction(coords, dustmaps=["north", "not_a_map"])
    with pytest.raises(ValueError, match="not in `dustmaps`"):
        obs.get_extinction(coords, dustmaps="north", dustmap_kwargs={"south": {}})
    with pytest.raises(ValueError, match="'init' and 'query'"):
        obs.get_extinction(coords, dustmaps="north", dustmap_kwargs={"north": {"mode": "median"}})
    with pytest.raises(ValueError, match="must include distances"):
        obs.get_extinction(SkyCoord(l=[30] * u.deg, b=[5] * u.deg, frame="galactic"), dustmaps="north")
    with pytest.raises(RuntimeError, match="fetch_missing"):
        obs.get_extinction(coords, dustmaps="missing")


def test_extinction_conversion_and_priority(fake_dustmaps):
    """A_V should be R_V times E(B-V) from the highest priority map with coverage"""
    coords = galactic([1, 300, 90], [0.5, 0, 45], [2, 2, 2])
    Av = obs.get_extinction(coords, dustmaps=["north", "south"], R_V=3.0, reflect=False)
    assert np.allclose(Av, 3.0 * np.array([0.2, 2 * 0.5, 0.2]))


def test_extinction_kwargs(fake_dustmaps):
    """User kwargs should override the defaults when loading and querying maps"""
    obs.get_extinction(galactic(90, 45, 1), dustmaps="north",
                       dustmap_kwargs={"north": {"init": {"scale": 1.0}, "query": {"mode": "median"}}})
    instance = fake_dustmaps.FakeNorthQuery.instances[-1]
    assert instance.scale == 1.0
    assert instance.query_kwargs == [{"mode": "median"}]


@pytest.mark.parametrize("dustmap", ["north", "everywhere"])
def test_extinction_reflection(fake_dustmaps, dustmap):
    """Every position should be covered once reflections are allowed, whether a map has no data (north) or
    has data that is outside of its footprint (everywhere)"""
    rng = np.random.default_rng(42)
    n = 5000
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    coords = galactic(lon, lat, np.ones(n))

    no_reflect = obs.get_extinction(coords, dustmaps=dustmap, R_V=1.0, reflect=False)
    too_low = coords.icrs.dec.deg < -30
    assert np.isnan(no_reflect[too_low]).all() and np.allclose(no_reflect[~too_low], 0.1)

    reflected = obs.get_extinction(coords, dustmaps=dustmap, R_V=1.0, reflect=True)
    assert np.allclose(reflected, 0.1)
    assert all((c.icrs.dec.deg >= -30).all() for c in fake_dustmaps.EverywhereQuery.queried)


def test_extinction_shapes_and_missing(fake_dustmaps):
    """Scalar coordinates and non-finite positions (e.g. missing orbits) should be handled"""
    scalar = SkyCoord(l=90 * u.deg, b=45 * u.deg, distance=1 * u.kpc, frame="galactic")
    assert np.isclose(obs.get_extinction(scalar, dustmaps="north", R_V=1.0), 0.1)

    gc = SkyCoord(x=[np.inf, 1.0] * u.kpc, y=[np.inf, 1.0] * u.kpc, z=[np.inf, 1.0] * u.kpc,
                  frame="galactocentric", representation_type="cartesian")
    Av = obs.get_extinction(gc, dustmaps="north")
    assert np.isnan(Av[0]) and np.isfinite(Av[1])

    # no maps are loaded if there's nothing to query
    fake_dustmaps.FakeNorthQuery.instances = []
    assert np.isnan(obs.get_extinction(gc[:1], dustmaps="north")).all()
    assert len(fake_dustmaps.FakeNorthQuery.instances) == 0


def test_photometry_with_dustmaps(fake_dustmaps):
    """Extinction should be calculated for primaries and disrupted secondaries"""
    final_bpp = pd.DataFrame({
        "sep": [10.0, -1.0, 5.0], "metallicity": [0.0142] * 3,
        "mass_1": [1.0] * 3, "mass_2": [0.8] * 3, "rad_1": [1.0] * 3, "rad_2": [0.8] * 3,
        "teff_1": [5800.0] * 3, "teff_2": [5000.0] * 3, "lum_1": [1.0] * 3, "lum_2": [0.5] * 3,
        "kstar_1": [1] * 3, "kstar_2": [1] * 3,
    }, index=[10, 20, 30])

    # 3 primaries/bound binaries, then 1 disrupted secondary, all in the northern sky
    pos = galactic([90, 100, 120, 150], [30, 20, 40, 50], [1, 2, 3, 4]).galactocentric.cartesian.xyz.T
    phot = obs.get_photometry(filters="Gaia_G_EDR3", final_bpp=final_bpp, final_pos=pos.to(u.kpc),
                              assume_mw_galactocentric=True, dustmaps="north",
                              extinction_kwargs={"R_V": 1.0})
    assert np.allclose(phot["Av_1"], [0.1, 0.2, 0.3])
    assert np.allclose(phot["Av_2"], [0.1, 0.4, 0.3])
    # only loaded once for all stars
    assert len(fake_dustmaps.FakeNorthQuery.instances) == 1
