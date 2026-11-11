import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

import cogsworth.obs.dust as dust


def galactic(lon, lat, dist):
    return SkyCoord(l=np.atleast_1d(lon) * u.deg, b=np.atleast_1d(lat) * u.deg,
                    distance=np.atleast_1d(dist) * u.kpc, frame="galactic")


def test_parse_dustmaps(fake_dustmaps):
    """Unknown, duplicate or missing dust maps should be caught"""
    assert dust._parse_dustmaps("north") == ["north"]
    assert dust._parse_dustmaps(("north", "south")) == ["north", "south"]
    with pytest.raises(ValueError, match="Unknown dust map"):
        dust._parse_dustmaps(["north", "not_a_map"])
    with pytest.raises(ValueError, match="only be supplied once"):
        dust._parse_dustmaps(["north", "north"])
    with pytest.raises(ValueError, match="At least one"):
        dust._parse_dustmaps([])


def test_default_dustmaps_registered():
    """Default dust maps should all be registered and have citations"""
    assert dust._parse_dustmaps(dust.DEFAULT_DUSTMAPS) == list(dust.DEFAULT_DUSTMAPS)
    from cogsworth.citations import CITATIONS
    for spec in dust.DUSTMAPS.values():
        assert spec.citation in CITATIONS["observables"]


def test_list_dustmaps(capsys):
    dust.list_dustmaps()
    out = capsys.readouterr().out
    for name in dust.DUSTMAPS:
        assert name in out


def test_load_dustmap(fake_dustmaps):
    """Loading should merge init kwargs with the defaults and give a helpful error on failure"""
    assert dust.load_dustmap("north").scale == 0.1
    assert dust.load_dustmap("north", {"scale": 1.0}).scale == 1.0
    with pytest.raises(RuntimeError, match="fetch_missing"):
        dust.load_dustmap("missing")
    with pytest.raises(ValueError, match="Unknown dust map"):
        dust.load_dustmap("not_a_map")


def test_query_dustmaps_priority_and_conversion(fake_dustmaps):
    """Earlier maps take priority, later maps fill gaps and each map's E(B-V) conversion is applied"""
    north, south = dust.load_dustmap("north"), dust.load_dustmap("south")
    # galactic centre-ish (dec ~ -29, in both footprints), southern plane (only south) and northern sky
    lon, lat, dist = np.array([1., 300., 90.]), np.array([0.5, 0., 45.]), np.array([2., 2., 2.])

    north_first = dust.query_dustmaps([("north", north, {}), ("south", south, {})], lon, lat, dist)
    assert np.allclose(north_first, [0.2, 2 * 0.5, 0.2])

    south_first = dust.query_dustmaps([("south", south, {}), ("north", north, {})], lon, lat, dist)
    assert np.allclose(south_first, [2 * 0.5, 2 * 0.5, 0.2])

    # southern plane point is not covered by "north" alone
    north_only = dust.query_dustmaps([("north", north, {"mode": "median"})], lon, lat, dist)
    assert np.isnan(north_only[1]) and np.allclose(north_only[[0, 2]], 0.2)
    assert north.query_kwargs[-1] == {"mode": "median"}


def test_query_dustmaps_multiple_values(fake_dustmaps):
    """Query modes giving more than one value per coordinate should be rejected"""
    north = dust.load_dustmap("north")
    with pytest.raises(ValueError, match="one value per coordinate"):
        dust.query_dustmaps([("north", north, {"mode": "samples"})], np.array([90.]), np.array([45.]),
                            np.array([1.]))


def test_query_dustmaps_footprint(fake_dustmaps):
    """Positions outside a map's footprint should never be queried, even if the map has data there"""
    everywhere, south = dust.load_dustmap("everywhere"), dust.load_dustmap("south")
    rng = np.random.default_rng(7)
    n = 2000
    lon = rng.uniform(0, 360, n)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
    dist = np.ones(n)
    too_low = galactic(lon, lat, dist).icrs.dec.deg < -30

    ebv = dust.query_dustmaps([("everywhere", everywhere, {})], lon, lat, dist)
    assert np.isnan(ebv[too_low]).all() and np.allclose(ebv[~too_low], 0.1)
    assert all((c.icrs.dec.deg >= -30).all() for c in fake_dustmaps.EverywhereQuery.queried)

    # a later map should still be able to cover positions outside an earlier map's footprint
    chained = dust.query_dustmaps([("everywhere", everywhere, {}), ("south", south, {})], lon, lat, dist)
    in_south = ((lon > 239) | (lon < 6)) & (np.abs(lat) < 10)
    assert np.allclose(chained[too_low & in_south], 2 * 0.5)

    # nothing in the footprint means no queries at all
    fake_dustmaps.EverywhereQuery.queried = []
    assert np.isnan(dust.query_dustmaps([("everywhere", everywhere, {})], lon[too_low], lat[too_low],
                                        dist[too_low])).all()
    assert fake_dustmaps.EverywhereQuery.queried == []


def test_bayestar_footprint():
    """The Bayestar footprint should be a strict cut at dec = -30 deg"""
    coords = SkyCoord(ra=[0, 0, 0] * u.deg, dec=[-29.9, -30.1, 45] * u.deg, distance=[1, 1, 1] * u.kpc)
    assert dust.DUSTMAPS["bayestar2019"].footprint(coords.galactic).tolist() == [True, False, True]


def test_query_dustmaps_skips_later_maps(fake_dustmaps):
    """Later maps shouldn't be queried at all if earlier maps already cover every position"""
    north, everywhere = dust.load_dustmap("north"), dust.load_dustmap("everywhere")
    ebv = dust.query_dustmaps([("north", north, {}), ("everywhere", everywhere, {})],
                              np.array([90., 120.]), np.array([45., 30.]), np.array([1., 2.]))
    assert np.allclose(ebv, [0.1, 0.2])
    assert fake_dustmaps.EverywhereQuery.queried == []


def test_load_real_dustmap_missing_file(tmp_path):
    """A real dustmaps map that fails to load should point users to how to download it"""
    with pytest.raises(RuntimeError, match=r"dustmaps\.bayestar\.fetch") as exc_info:
        dust.load_dustmap("bayestar2019", {"map_fname": str(tmp_path / "missing.h5")})
    assert isinstance(exc_info.value.__cause__, OSError)
