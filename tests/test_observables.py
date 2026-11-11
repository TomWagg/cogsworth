import sys
import types
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


# --- dust maps (using fake maps so that the large dust map files aren't needed) ---


class FakeNorthQuery:
    """Covers dec > -30 deg (like Bayestar), returns 0.1 * distance in kpc"""
    instances = []

    def __init__(self, scale=0.1):
        self.scale = scale
        self.query_kwargs = []
        FakeNorthQuery.instances.append(self)

    def __call__(self, coords, mode="random_sample"):
        self.query_kwargs.append({"mode": mode})
        values = self.scale * coords.distance.kpc
        if mode == "samples":
            return np.stack([values, values], axis=-1)
        return np.where(coords.icrs.dec.deg > -30, values, np.nan)


class FakeSouthPlaneQuery:
    """Covers the southern galactic plane (like DECaPS), returns a constant 0.5"""
    def __init__(self):
        pass

    def __call__(self, coords, mode="random_sample"):
        lon, lat = coords.l.deg, coords.b.deg
        return np.where(((lon > 239) | (lon < 6)) & (np.abs(lat) < 10), 0.5, np.nan)


class MissingQuery:
    def __init__(self):
        raise FileNotFoundError("no map file here")


@pytest.fixture
def fake_dustmaps(monkeypatch):
    module = types.ModuleType("fake_dustmaps")
    module.FakeNorthQuery = FakeNorthQuery
    module.FakeSouthPlaneQuery = FakeSouthPlaneQuery
    module.MissingQuery = MissingQuery
    monkeypatch.setitem(sys.modules, "fake_dustmaps", module)
    FakeNorthQuery.instances = []

    for name, cls, to_ebv in [("north", "FakeNorthQuery", 1.0), ("south", "FakeSouthPlaneQuery", 2.0),
                              ("missing", "MissingQuery", 1.0)]:
        monkeypatch.setitem(obs.DUSTMAPS, name, obs.DustMapSpec(module="fake_dustmaps", query_class=cls,
                                                                to_ebv=to_ebv, citation=name,
                                                                fetch_hint=f"fetch_{name}()"))
    return module


def galactic(lon, lat, dist):
    return SkyCoord(l=np.atleast_1d(lon) * u.deg, b=np.atleast_1d(lat) * u.deg,
                    distance=np.atleast_1d(dist) * u.kpc, frame="galactic")


def test_dustmaps_bad_input(fake_dustmaps):
    """Unknown, duplicate or missing dust maps (and bad kwargs) should be caught"""
    coords = galactic(30, 5, 1)
    with pytest.raises(ValueError, match="Unknown dust map"):
        obs.get_extinction(coords, dustmaps=["north", "not_a_map"])
    with pytest.raises(ValueError, match="only be supplied once"):
        obs.get_extinction(coords, dustmaps=["north", "north"])
    with pytest.raises(ValueError, match="At least one"):
        obs.get_extinction(coords, dustmaps=[])
    with pytest.raises(ValueError, match="not in `dustmaps`"):
        obs.get_extinction(coords, dustmaps="north", dustmap_kwargs={"south": {}})
    with pytest.raises(ValueError, match="'init' and 'query'"):
        obs.get_extinction(coords, dustmaps="north", dustmap_kwargs={"north": {"mode": "median"}})
    with pytest.raises(ValueError, match="must include distances"):
        obs.get_extinction(SkyCoord(l=[30] * u.deg, b=[5] * u.deg, frame="galactic"), dustmaps="north")
    with pytest.raises(RuntimeError, match="fetch_missing"):
        obs.get_extinction(coords, dustmaps="missing")
    with pytest.raises(ValueError, match="one value per coordinate"):
        obs.get_extinction(coords, dustmaps="north", dustmap_kwargs={"north": {"query": {"mode": "samples"}}})


def test_dustmaps_priority_and_conversion(fake_dustmaps):
    """Earlier maps take priority, later maps fill gaps and each map's E(B-V) conversion is applied"""
    # galactic centre-ish (dec ~ -29, in both footprints), southern plane (only south) and northern sky
    coords = galactic([1, 300, 90], [0.5, 0, 45], [2, 2, 2])
    north_first = obs.get_extinction(coords, dustmaps=["north", "south"], R_V=3.0, reflect=False)
    assert np.allclose(north_first, 3.0 * np.array([0.2, 2 * 0.5, 0.2]))

    south_first = obs.get_extinction(coords, dustmaps=["south", "north"], R_V=3.0, reflect=False)
    assert np.allclose(south_first, 3.0 * np.array([2 * 0.5, 2 * 0.5, 0.2]))

    # southern plane point is not covered by "north" alone
    north_only = obs.get_extinction(coords, dustmaps="north", R_V=3.0, reflect=False)
    assert np.isnan(north_only[1]) and np.allclose(north_only[[0, 2]], 0.6)


def test_dustmaps_kwargs(fake_dustmaps):
    """User kwargs should override the defaults when loading and querying maps"""
    obs.get_extinction(galactic(90, 45, 1), dustmaps="north",
                       dustmap_kwargs={"north": {"init": {"scale": 1.0}, "query": {"mode": "median"}}})
    instance = FakeNorthQuery.instances[-1]
    assert instance.scale == 1.0
    assert instance.query_kwargs == [{"mode": "median"}]


def test_dustmaps_reflection(fake_dustmaps):
    """Every position on the sky should be covered once reflections are allowed"""
    rng = np.random.default_rng(42)
    n = 5000
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    coords = galactic(lon, lat, np.ones(n))

    no_reflect = obs.get_extinction(coords, dustmaps="north", R_V=1.0, reflect=False)
    too_low = coords.icrs.dec.deg < -30
    assert np.isnan(no_reflect[too_low]).all() and np.allclose(no_reflect[~too_low], 0.1)

    reflected = obs.get_extinction(coords, dustmaps="north", R_V=1.0, reflect=True)
    assert np.allclose(reflected, 0.1)


def test_dustmaps_shapes_and_missing(fake_dustmaps):
    """Scalar coordinates and non-finite positions (e.g. missing orbits) should be handled"""
    scalar = SkyCoord(l=90 * u.deg, b=45 * u.deg, distance=1 * u.kpc, frame="galactic")
    assert np.isclose(obs.get_extinction(scalar, dustmaps="north", R_V=1.0), 0.1)

    gc = SkyCoord(x=[np.inf, 1.0] * u.kpc, y=[np.inf, 1.0] * u.kpc, z=[np.inf, 1.0] * u.kpc,
                  frame="galactocentric", representation_type="cartesian")
    Av = obs.get_extinction(gc, dustmaps="north")
    assert np.isnan(Av[0]) and np.isfinite(Av[1])

    # no maps are loaded if there's nothing to query
    FakeNorthQuery.instances = []
    assert np.isnan(obs.get_extinction(gc[:1], dustmaps="north")).all()
    assert len(FakeNorthQuery.instances) == 0


def test_list_dustmaps(capsys):
    obs.list_dustmaps()
    out = capsys.readouterr().out
    for name in obs.DUSTMAPS:
        assert name in out


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
    assert len(FakeNorthQuery.instances) == 1


class EverywhereQuery:
    """Returns 0.1 * distance in kpc everywhere, recording the coordinates it was queried with"""
    queried = []

    def __init__(self):
        pass

    def __call__(self, coords, mode="random_sample"):
        EverywhereQuery.queried.append(coords)
        return 0.1 * coords.distance.kpc


def test_dustmaps_footprint(fake_dustmaps, monkeypatch):
    """Positions outside a map's footprint should never be queried, even if the map has data there"""
    fake_dustmaps.EverywhereQuery = EverywhereQuery
    EverywhereQuery.queried = []
    monkeypatch.setitem(obs.DUSTMAPS, "everywhere", obs.DustMapSpec(
        module="fake_dustmaps", query_class="EverywhereQuery", footprint=obs._pan_starrs_footprint))

    rng = np.random.default_rng(7)
    n = 2000
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    coords = galactic(lon, lat, np.ones(n))
    too_low = coords.icrs.dec.deg < -30

    no_reflect = obs.get_extinction(coords, dustmaps="everywhere", R_V=1.0, reflect=False)
    assert np.isnan(no_reflect[too_low]).all() and np.allclose(no_reflect[~too_low], 0.1)
    assert all((c.icrs.dec.deg >= -30).all() for c in EverywhereQuery.queried)

    # reflections should then fill everything, still without querying outside the footprint
    EverywhereQuery.queried = []
    reflected = obs.get_extinction(coords, dustmaps="everywhere", R_V=1.0, reflect=True)
    assert np.allclose(reflected, 0.1)
    assert all((c.icrs.dec.deg >= -30).all() for c in EverywhereQuery.queried)

    # a later map should still be able to cover positions outside an earlier map's footprint
    chained = obs.get_extinction(coords, dustmaps=["everywhere", "south"], R_V=1.0, reflect=False)
    in_south = ((lon > 239) | (lon < 6)) & (np.abs(lat) < 10)
    assert np.allclose(chained[too_low & in_south], 2 * 0.5)


def test_bayestar_footprint():
    """The Bayestar footprint should be a strict cut at dec = -30 deg"""
    coords = SkyCoord(ra=[0, 0, 0] * u.deg, dec=[-29.9, -30.1, 45] * u.deg, distance=[1, 1, 1] * u.kpc)
    assert obs.DUSTMAPS["bayestar2019"].footprint(coords.galactic).tolist() == [True, False, True]
