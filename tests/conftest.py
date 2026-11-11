import sys
import types

import numpy as np
import pytest

import cogsworth.obs.dust as dust


# --- fake dust maps (so that the large dust map files aren't needed for tests) ---
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
    def __call__(self, coords, mode="random_sample"):
        lon, lat = coords.l.deg, coords.b.deg
        return np.where(((lon > 239) | (lon < 6)) & (np.abs(lat) < 10), 0.5, np.nan)


class EverywhereQuery:
    """Returns 0.1 * distance in kpc everywhere, recording the coordinates it was queried with"""
    queried = []

    def __call__(self, coords, mode="random_sample"):
        EverywhereQuery.queried.append(coords)
        return 0.1 * coords.distance.kpc


class MissingQuery:
    """Fails to load, like a dust map whose files haven't been downloaded"""
    def __init__(self):
        raise FileNotFoundError("no map file here")


@pytest.fixture
def fake_dustmaps(monkeypatch):
    """Register fake dust maps: "north" (to_ebv=1), "south" (to_ebv=2), "everywhere" (Pan-STARRS footprint)
    and "missing" (fails to load). Returns the fake module containing the query classes."""
    module = types.ModuleType("fake_dustmaps")
    for cls in [FakeNorthQuery, FakeSouthPlaneQuery, EverywhereQuery, MissingQuery]:
        setattr(module, cls.__name__, cls)
    monkeypatch.setitem(sys.modules, "fake_dustmaps", module)
    FakeNorthQuery.instances = []
    EverywhereQuery.queried = []

    specs = {
        "north": dict(query_class="FakeNorthQuery", to_ebv=1.0),
        "south": dict(query_class="FakeSouthPlaneQuery", to_ebv=2.0),
        "everywhere": dict(query_class="EverywhereQuery", footprint=dust._pan_starrs_footprint),
        "missing": dict(query_class="MissingQuery"),
    }
    for name, kwargs in specs.items():
        monkeypatch.setitem(dust.DUSTMAPS, name, dust.DustMapSpec(module="fake_dustmaps", citation=name,
                                                                  fetch_hint=f"fetch_{name}()", **kwargs))
    return module
