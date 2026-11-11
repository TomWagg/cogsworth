"""Specifications and helpers for the 3D dust maps from :mod:`dustmaps` used to calculate extinction.

Each available dust map is described by a :class:`DustMapSpec` in :data:`DUSTMAPS`. The functions here load
and query the maps to give E(B-V), whilst :func:`~cogsworth.obs.observables.get_extinction` converts this to
a visual extinction.
"""
import contextlib
import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

from cogsworth.utils import check_dependencies

__all__ = ["DustMapSpec", "DUSTMAPS", "DEFAULT_DUSTMAPS", "list_dustmaps", "load_dustmap", "query_dustmaps"]


def _pan_starrs_footprint(coords):
    """Pan-STARRS 1 footprint used by the Bayestar maps (dec >= -30 deg)"""
    return coords.icrs.dec >= -30 * u.deg


@dataclass(frozen=True)
class DustMapSpec:
    """Specification of a 3D dust map from :mod:`dustmaps` that can be used by
    :func:`~cogsworth.obs.observables.get_extinction`

    Parameters
    ----------
    module : `str`
        Module containing the query class (e.g. ``"dustmaps.bayestar"``)
    query_class : `str`
        Name of the query class within ``module`` (e.g. ``"BayestarQuery"``)
    init_kwargs : `dict`
        Default keyword arguments used when instantiating the query class
    query_kwargs : `dict`
        Default keyword arguments used when querying the map
    to_ebv : `float`
        Factor that converts the map's native units to E(B-V)
    citation : `str`
        Key of the citation for this map in :mod:`cogsworth.citations`
    fetch_hint : `str`
        Code snippet for downloading the map files. This is only ever displayed in error messages (it is never
        executed)
    coverage : `str`
        Human-readable description of the map footprint
    footprint : `callable`, optional
        Function that takes a :class:`~astropy.coordinates.SkyCoord` (in the galactic frame) and returns a
        boolean mask of which positions are within the map's footprint. Positions outside of the footprint
        are never queried. If ``None`` then every position is queried (and the map itself returns NaN outside
        its coverage).
    """
    module: str
    query_class: str
    init_kwargs: dict = field(default_factory=dict)
    query_kwargs: dict = field(default_factory=dict)
    to_ebv: float = 1.0
    citation: str = ""
    fetch_hint: str = ""
    coverage: str = ""
    footprint: Optional[Callable] = None


DUSTMAPS = {
    "bayestar2019": DustMapSpec(
        module="dustmaps.bayestar",
        query_class="BayestarQuery",
        init_kwargs={"max_samples": 2, "version": "bayestar2019"},
        query_kwargs={"mode": "random_sample"},
        # TODO: Bayestar19 reports reddening in an arbitrary unit (see Green+2019, Table 1). For now this is
        # treated as E(B-V) to match previous cogsworth behaviour, but the conversion should be revisited
        to_ebv=1.0,
        citation="bayestar2019",
        fetch_hint="import dustmaps.bayestar; dustmaps.bayestar.fetch(version='bayestar2019')",
        coverage="dec >= -30 deg (Pan-STARRS 1 footprint)",
        # the map has some pixels slightly below -30 deg, but we follow the documented footprint
        footprint=_pan_starrs_footprint,
    ),
    "decaps": DustMapSpec(
        module="dustmaps.decaps",
        # memory-mapped version of the query, the full DECaPSQuery loads ~30 GB into RAM
        query_class="DECaPSQueryLite",
        init_kwargs={},
        query_kwargs={"mode": "random_sample"},
        # DECaPS reports reddening directly in E(B-V) (Zucker+2025)
        to_ebv=1.0,
        citation="decaps",
        fetch_hint="import dustmaps.decaps; dustmaps.decaps.fetch()",
        coverage="southern Galactic plane (239 < l < 6 deg, |b| < 10 deg)",
    ),
}
"""Available 3D dust maps, keyed by the names accepted by :func:`~cogsworth.obs.observables.get_extinction`"""

DEFAULT_DUSTMAPS = ("bayestar2019",)
"""Dust maps used by default in :func:`~cogsworth.obs.observables.get_extinction`"""


def list_dustmaps():
    """Print the dust maps that are available and their coverage"""
    for name, spec in DUSTMAPS.items():
        print(f"{name}: {spec.coverage}")


def _parse_dustmaps(dustmaps):
    """Validate a dust map name (or list of names) and return them as a list"""
    if isinstance(dustmaps, str):
        dustmaps = [dustmaps]
    dustmaps = list(dustmaps)
    if len(dustmaps) == 0:
        raise ValueError("At least one dust map must be supplied")
    unknown = [name for name in dustmaps if name not in DUSTMAPS]
    if len(unknown) > 0:
        raise ValueError(f"Unknown dust map(s) {unknown}, available options are {list(DUSTMAPS)}")
    if len(set(dustmaps)) != len(dustmaps):
        raise ValueError(f"Each dust map can only be supplied once, got {dustmaps}")
    return dustmaps


def load_dustmap(name, init_kwargs=None):
    """Load a dust map (i.e. instantiate its query class), hiding any print statements from :mod:`dustmaps`

    Parameters
    ----------
    name : `str`
        Name of the dust map (see :func:`list_dustmaps` for options)
    init_kwargs : `dict`, optional
        Keyword arguments for the query class, which override the defaults in :data:`DUSTMAPS`

    Returns
    -------
    query : :class:`dustmaps.map_base.DustMap`
        The loaded dust map query object

    Raises
    ------
    RuntimeError
        If the dust map fails to load (e.g. because the files haven't been downloaded)
    """
    spec = DUSTMAPS[_parse_dustmaps(name)[0]]
    init_kwargs = {**spec.init_kwargs, **({} if init_kwargs is None else init_kwargs)}
    if spec.module.split(".")[0] == "dustmaps":
        assert check_dependencies("dustmaps")
    query_class = getattr(importlib.import_module(spec.module), spec.query_class)
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            return query_class(**init_kwargs)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Failed to load the '{name}' dust map. If you haven't downloaded it yet, "
                           f"you can do so with `{spec.fetch_hint}`") from e


def query_dustmaps(dustmaps, lon, lat, dist):
    """Get E(B-V) for each position, querying each map in turn for positions that are still missing

    Each map is only queried for positions within its footprint (if it has one), and its values are converted
    to E(B-V) using its ``to_ebv`` factor (see :class:`DustMapSpec`).

    Parameters
    ----------
    dustmaps : `list` of `tuple`
        Each element is (name, query object, query kwargs), in order of priority. The query objects can be
        created with :func:`load_dustmap`.
    lon, lat : :class:`~numpy.ndarray`
        Galactic longitude and latitude in degrees
    dist : :class:`~numpy.ndarray`
        Distances in kpc

    Returns
    -------
    ebv : :class:`~numpy.ndarray`
        E(B-V) at each position, NaN where no map has coverage
    """
    ebv = np.full(len(lon), np.nan)
    for name, query, query_kwargs in dustmaps:
        todo = np.isnan(ebv)
        if not todo.any():
            break
        coords = SkyCoord(l=lon[todo] * u.deg, b=lat[todo] * u.deg, distance=dist[todo] * u.kpc,
                          frame="galactic")

        # only query positions within the map's footprint
        footprint = DUSTMAPS[name].footprint
        if footprint is not None:
            inside = np.asarray(footprint(coords), dtype=bool)
            todo[todo] = inside
            if not inside.any():
                continue
            coords = coords[inside]

        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            values = np.asarray(query(coords, **query_kwargs), dtype=float)
        if values.shape != (todo.sum(),):
            raise ValueError(f"Querying the '{name}' dust map with {query_kwargs} returned an array of shape "
                             f"{values.shape} rather than one value per coordinate. Only query modes that "
                             "return a single value per coordinate (e.g. 'random_sample', 'median', 'mean') "
                             "are supported")
        ebv[todo] = DUSTMAPS[name].to_ebv * values
    return ebv
