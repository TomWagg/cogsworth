import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from copy import copy

import contextlib
import importlib
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

from cogsworth.utils import check_dependencies
from cogsworth.obs.mist import MISTBolometricCorrectionGrid, check_filters

__all__ = ["get_log_g", "get_absolute_bol_mag", "get_apparent_mag", "get_absolute_mag", "add_mags",
           "DustMapSpec", "list_dustmaps", "get_extinction", "get_photometry"]


def get_log_g(mass, radius):
    """Computes log of the surface gravity in cgs

    Parameters
    ----------
    mass : :class:`~astropy.units.Quantity` [mass]
        Mass of the star
    radius : :class:`~astropy.units.Quantity` [radius]
        Radius of the star

    Returns
    -------
    log g : :class:`~numpy.ndarray`
        Log of the surface gravity in cgs
    """
    g = const.G * mass / radius**2

    # avoid division by zero errors (for massless remnants)
    with np.errstate(divide='ignore'):
        return np.log10(g.cgs.value)


def get_absolute_bol_mag(lum):
    """Computes the absolute bolometric magnitude following
    `IAU Resolution B2 <https://www.iau.org/news/announcements/detail/ann15023/>`_

    Parameters
    ----------
    lum : :class:`~astropy.units.Quantity` [luminosity]
        Luminosity of the star

    Returns
    -------
    M_bol : :class:`~numpy.ndarray`
        Absolute bolometric magnitude
    """
    zero_point_lum = 3.0128e28 * u.watt
    return -2.5 * np.log10(lum / zero_point_lum)


def get_apparent_mag(M_abs, distance):
    """Convert absolute magnitude to apparent magnitude

    Parameters
    ----------
    M_abs : :class:`~numpy.ndarray`
        Absolute magnitude
    distance : :class:`~astropy.units.Quantity` [length]
        Distance

    Returns
    -------
    m_app : :class:`~numpy.ndarray`
        Apparent magnitude
    """
    finite_distance = np.isfinite(distance)
    m_app = np.repeat(np.inf, len(distance))
    m_app[finite_distance] = M_abs[finite_distance] + 5 * np.log10(distance[finite_distance] / (10 * u.pc))
    return m_app


def get_absolute_mag(m_app, distance):
    """Convert apparent magnitude to absolute magnitude

    Parameters
    ----------
    M_abs : :class:`~numpy.ndarray`
        Apparent magnitude
    distance : :class:`~astropy.units.Quantity` [length]
        Distance

    Returns
    -------
    m_app : :class:`~numpy.ndarray`
        Absolute magnitude
    """
    M_abs = m_app - 5 * np.log10(distance / (10 * u.pc))
    return M_abs


def add_mags(*mags, remove_nans=True):
    """Add any number of magnitudes

    Parameters
    ----------
    *mags : `list` or `np.array` or `float` or `int`
        A series of magnitudes. If arrays are given then all must have the same length. If a mixture of single
        values and arrays are given then the single values will be added to each array element
    remove_nans : `bool`, optional
        Whether to remove NaNs from the total (if not the total will be NaN), by default True

    Returns
    -------
    total_mag : :class:`~numpy.ndarray`
        Total magnitude

    Raises
    ------
    ValueError
        If any magnitude is an invalid type
    AssertionError
        If any magnitude array has a different length from another
    """
    total_mag = 0

    # convert lists to numpy arrays
    mags = list(mags)
    for i in range(len(mags)):
        if isinstance(mags[i], list):
            mags[i] = np.array(mags[i])
        if isinstance(mags[i], int):
            mags[i] = float(mags[i])

    # check for dodgy input
    if isinstance(mags[0], (list, np.ndarray)):
        length = len(mags[0])
        for mag in mags[1:]:
            assert len(mag) == length, ("All magnitude arrays must have the same length - one array is of "
                                        f"length {length} but another is {len(mag)}")

    # go through each supplied magnitude
    for mag in mags:
        if not isinstance(mag, (np.ndarray, float, int)):
            raise ValueError(("All `mag`s must either be a list, numpy array, float or an int. Unfortunately "
                              f"for us both, what you have given me is of type `{type(mag).__name__}`..."))

        # compute the default additional magnitude
        additional = 10**(-mag * 0.4)

        # if you want to remove any NaNs then do so
        if remove_nans:
            if isinstance(mag, np.ndarray):
                additional[np.isnan(mag)] = 0.0
            elif np.isnan(mag):
                additional = 0.0
        total_mag += additional

    # hide divide by zero errors (since inf magnitude is correct when all mags are NaN)
    with np.errstate(divide='ignore'):
        return -2.5 * np.log10(total_mag)


def _pan_starrs_footprint(coords):
    """Pan-STARRS 1 footprint used by the Bayestar maps (dec >= -30 deg)"""
    return coords.icrs.dec >= -30 * u.deg


@dataclass(frozen=True)
class DustMapSpec:
    """Specification of a 3D dust map from :mod:`dustmaps` that can be used by :func:`get_extinction`

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
"""Available 3D dust maps, keyed by the names accepted by :func:`get_extinction`"""

DEFAULT_DUSTMAPS = ("bayestar2019",)
"""Dust maps used by default in :func:`get_extinction` and :func:`get_photometry`"""


def list_dustmaps():
    """Print the dust maps that can be used by :func:`get_extinction` and their coverage"""
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


def _load_dustmap(name, init_kwargs):
    """Instantiate the query class for a dust map, hiding its print statements"""
    spec = DUSTMAPS[name]
    if spec.module.split(".")[0] == "dustmaps":
        assert check_dependencies("dustmaps")
    query_class = getattr(importlib.import_module(spec.module), spec.query_class)
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            return query_class(**init_kwargs)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Failed to load the '{name}' dust map. If you haven't downloaded it yet, "
                           f"you can do so with `{spec.fetch_hint}`") from e


def _query_dustmaps(dustmaps, lon, lat, dist):
    """Get E(B-V) for each position, querying each map in turn for positions that are still missing

    Parameters
    ----------
    dustmaps : `list` of `tuple`
        Each element is (name, query object, query kwargs), in order of priority
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


def get_extinction(coords, dustmaps=DEFAULT_DUSTMAPS, dustmap_kwargs=None, R_V=3.3, reflect=True):
    """Calculates the visual extinction values for a set of coordinates using 3D dust maps

    Reddening due to dust is calculated using one or more 3D dust maps from :mod:`dustmaps` (see
    :func:`list_dustmaps` for options). Maps are queried in the order given, with each subsequent map only
    used for positions not covered by earlier maps. For example, ``dustmaps=["bayestar2019", "decaps"]`` uses
    Bayestar wherever possible and fills in the southern Galactic plane with DECaPS.

    The reddening from each map is converted to E(B-V) and then to a visual extinction, assuming a
    total-to-selective extinction ratio of ``R_V`` (default of 3.3, as used by
    `Green+2019 <https://iopscience.iop.org/article/10.3847/1538-4357/ab5362#apjab5362s2>`_).

    .. warning::
        If ``reflect=True``, any positions not covered by any of the maps (outside of each map's footprint,
        e.g. declinations < -30 degrees for Bayestar, or anywhere that a map returns NaN) will be reflected
        around the galactic plane (trying each of [(-l, b), (l, -b), (-l, -b)] in turn) and the dust at the
        first reflected location with coverage will be used instead

    Each map has its own random draws for the probabilistic query modes, which use the global
    :mod:`numpy` random state, so use :func:`numpy.random.seed` for reproducible results.

    Parameters
    ----------
    coords : :class:`~astropy.coordinates.SkyCoord`
        The coordinates at which you wish to calculate extinction values. Must include distances.
    dustmaps : `str` or `list` of `str`
        Name(s) of the dust map(s) to use, in order of priority. See :func:`list_dustmaps` for options.
    dustmap_kwargs : `dict`
        Keyword arguments for each dust map, overriding the defaults in :data:`DUSTMAPS`. Keys are dust map
        names and values are dicts with (optional) keys ``"init"`` (passed when loading the map) and
        ``"query"`` (passed when querying the map). For example, if you only downloaded the mean DECaPS map
        then use ``{"decaps": {"init": {"mean_only": True}, "query": {"mode": "mean"}}}``.
    R_V : `float`
        Total-to-selective extinction ratio used to convert E(B-V) to A_V
    reflect : `bool`
        Whether to reflect positions without coverage about the galactic plane (see warning above)

    Returns
    -------
    Av : :class:`~numpy.ndarray`
        Visual extinction values for each set of coordinates (NaN where no map has coverage)
    """
    dustmaps = _parse_dustmaps(dustmaps)
    dustmap_kwargs = {} if dustmap_kwargs is None else dustmap_kwargs
    for name, kwargs in dustmap_kwargs.items():
        if name not in dustmaps:
            raise ValueError(f"`dustmap_kwargs` given for '{name}', which is not in `dustmaps` ({dustmaps})")
        if not set(kwargs).issubset({"init", "query"}):
            raise ValueError(f"`dustmap_kwargs['{name}']` can only have the keys 'init' and 'query', "
                             f"got {list(kwargs)}")

    # convert to galactic coordinates, ensuring we have distances for the 3D maps
    galactic = coords.galactic
    if not galactic.distance.unit.is_equivalent(u.kpc):
        raise ValueError("`coords` must include distances to use 3D dust maps")
    shape = galactic.shape
    lon = np.atleast_1d(galactic.l.deg).astype(float).ravel()
    lat = np.atleast_1d(galactic.b.deg).astype(float).ravel()
    dist = np.atleast_1d(galactic.distance.to(u.kpc).value).astype(float).ravel()

    # only query positions that are well-defined (e.g. not missing orbits)
    ebv = np.full(len(lon), np.nan)
    finite = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(dist)
    if finite.any():
        # load each dust map (once)
        loaded = []
        for name in dustmaps:
            spec = DUSTMAPS[name]
            user_kwargs = dustmap_kwargs.get(name, {})
            query = _load_dustmap(name, {**spec.init_kwargs, **user_kwargs.get("init", {})})
            loaded.append((name, query, {**spec.query_kwargs, **user_kwargs.get("query", {})}))

        ebv[finite] = _query_dustmaps(loaded, lon[finite], lat[finite], dist[finite])

        # try reflecting anything that isn't covered by the maps
        if reflect:
            for ref_l, ref_b in [(-1, 1), (1, -1), (-1, -1)]:
                todo = finite & np.isnan(ebv)
                if not todo.any():
                    break
                ebv[todo] = _query_dustmaps(loaded, (ref_l * lon[todo]) % 360, ref_b * lat[todo], dist[todo])

    # convert this to a visual extinction
    Av = R_V * ebv
    return Av.reshape(shape)


def get_photometry(filters, population=None, final_bpp=None, final_pos=None, distances=None,
                   ignore_extinction=False, assume_mw_galactocentric=False,
                   main_filter=None, silence_bounds_warning=False,
                   dustmaps=DEFAULT_DUSTMAPS, extinction_kwargs=None):
    """Computes photometry subject to dust extinction using the MIST boloemtric correction grid

    Parameters
    ----------
    filters : `list` of `str`
        Which filters to compute photometry for (e.g. ['Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3']).
        Use :func:`~cogsworth.obs.mist.list_filters` to see the available filters
    population : :class:`~cogsworth.pop.Population`
        The population for which to compute photometry (either supply this or a final_bpp and final_pos)
    final_bpp : :class:`~pandas.DataFrame`
        A dataset of COSMIC binaries at present day - must include these columns: ["sep", "metallicity"] and
        for each star it must have the columns ["teff", "lum", "mass", "rad", "kstar"]
    final_pos : :class:`~astropy.quantity.Quantity`
        Final positions of each system in the galactocentric frame.
        The first `len(self)` entries of each are for bound binaries or primaries, then the final
        `self.disrupted.sum()` entries are for disrupted secondaries. Any missing orbits (where orbit=None
        will be set to `np.inf` for ease of masking.
    ignore_extinction : `bool`
        Whether to ignore extinction
    assume_mw_galactocentric : `bool`
        Whether to assume the population positions/distances are in the Milky Way galactocentric frame
    main_filter : `str`
        The main filter to use for calculating which star is observed as the brighter one
    silence_bounds_warning : `bool`
        Whether to silence the out-of-bounds warning from the MIST bolometric correction grid
    dustmaps : `str` or `list` of `str`
        Name(s) of the 3D dust map(s) to use for extinction, in order of priority. See
        :func:`list_dustmaps` for options and :func:`get_extinction` for details.
    extinction_kwargs : `dict`
        Additional keyword arguments to pass to :func:`get_extinction` (e.g. ``dustmap_kwargs``, ``R_V``,
        ``reflect``)


    Returns
    -------
    photometry : :class:`~pandas.DataFrame`
        Photometry and extinction information for supplied COSMIC binaries in desired `filters`, indexed in
        the same way as `final_bpp` (i.e. by ``bin_num``). For each
        filter, ``{filter}_app`` and ``{filter}_abs`` give the combined apparent and absolute magnitudes of
        bound binaries and mergers (NaN for disrupted binaries), whilst ``{filter}_app_1``/``{filter}_app_2``
        (and similarly for ``abs``) always give the magnitudes of the primary and secondary star individually.
        Stars without any flux (neutron stars, black holes and massless remnants) have infinite magnitudes.
    """
    # check that the input is valid
    if population is None and (final_bpp is None or final_pos is None):
        raise ValueError("Either a population or final_bpp and final_pos must be supplied")
    if distances is None and not assume_mw_galactocentric:
        raise ValueError("Must supply either distances or have `assume_mw_galactocentric=True`")
    if not ignore_extinction and not assume_mw_galactocentric:
        raise ValueError("Cannot calculate extinction due to dust without `assume_mw_galactocentric=True`")
    if isinstance(filters, str):
        filters = [filters]
    check_filters(filters)
    main_filter = filters[0] if main_filter is None else main_filter
    if main_filter not in filters:
        raise ValueError(f"main_filter '{main_filter}' must be one of the supplied filters {filters}")

    if population is not None:
        final_bpp = population.final_bpp
        final_pos = population.final_pos
        disrupted = population.disrupted
    else:
        disrupted = final_bpp["sep"].values < 0.0

    if assume_mw_galactocentric:
        final_coords = SkyCoord(x=final_pos[:, 0], y=final_pos[:, 1], z=final_pos[:, 2],
                                frame="galactocentric", unit=u.kpc, representation_type="cartesian")

    # set up empty photometry table (indexed by bin_num in the same way as final_bpp)
    photometry = pd.DataFrame(index=final_bpp.index)

    if not ignore_extinction:
        # get extinction for every star at once (so each dust map is only loaded once)
        extinction_kwargs = {} if extinction_kwargs is None else extinction_kwargs
        Av = get_extinction(final_coords, dustmaps=dustmaps, **extinction_kwargs)

        # first len(final_bpp) are bound binaries and primaries of disrupted binaries
        photometry['Av_1'] = Av[:len(final_bpp)]

        # secondaries of disrupted binaries follow (match primary otherwise)
        photometry['Av_2'] = photometry['Av_1']
        photometry.loc[disrupted, "Av_2"] = Av[len(final_bpp):]

        # ensure extinction remains in MIST grid range (<= 6) and is not NaN
        photometry.loc[photometry.Av_1 > 6, ['Av_1']] = 6
        photometry.loc[photometry.Av_2 > 6, ['Av_2']] = 6
        photometry = photometry.fillna(6)
    else:
        photometry['Av_1'] = np.zeros(len(final_bpp))
        photometry['Av_2'] = np.zeros(len(final_bpp))

    # get Fe/H using e.g. Bertelli+1994 Eq. 10 (assuming all stars have the solar abundance pattern)
    Z_sun = 0.0142
    FeH = np.log10(final_bpp["metallicity"].values / Z_sun)

    # set up MIST bolometric correction grid
    bc_grid = MISTBolometricCorrectionGrid(bands=filters)
    bc = {
        "app": [None, None],
        "abs": [None, None]
    }

    if distances is None:
        distances = final_coords.icrs.distance
    primary_distances = distances[:len(final_bpp)]
    # secondary distances are the same whilst bound but different when disrupted
    secondary_distances = copy(primary_distances)
    secondary_distances[disrupted] = distances[len(final_bpp):]

    # for each star in the (possibly disrupted/merged) binary
    for ind, dist in zip([1, 2], [primary_distances, secondary_distances]):
        # calculate the surface gravity if necessary
        if f"log_g_{ind}" not in final_bpp:
            final_bpp.insert(len(final_bpp.columns), f"log_g_{ind}",
                             get_log_g(mass=final_bpp[f"mass_{ind}"].values * u.Msun,
                                       radius=final_bpp[f"rad_{ind}"].values * u.Rsun))

        # get the bolometric corrections from MIST isochrones
        bc["app"][ind - 1] = bc_grid.interp(teff=final_bpp[f"teff_{ind}"].values,
                                            logg=final_bpp[f"log_g_{ind}"].values,
                                            feh=FeH, av=photometry[f"Av_{ind}"], bands=filters,
                                            silence_bounds_warning=silence_bounds_warning).values
        bc["abs"][ind - 1] = bc_grid.interp(teff=final_bpp[f"teff_{ind}"].values,
                                            logg=final_bpp[f"log_g_{ind}"].values,
                                            feh=FeH, av=np.zeros(len(final_bpp)), bands=filters,
                                            silence_bounds_warning=silence_bounds_warning).values

        # calculate the absolute bolometric magnitude and set any BH or massless remnants to invisible
        photometry[f"M_abs_{ind}"] = get_absolute_bol_mag(lum=final_bpp[f"lum_{ind}"].values * u.Lsun)
        photometry.loc[np.isin(final_bpp[f"kstar_{ind}"].values, [13, 14, 15]), f"M_abs_{ind}"] = np.inf

        # convert the absolute magnitude to an apparent magnitude
        photometry[f"m_app_{ind}"] = get_apparent_mag(M_abs=photometry[f"M_abs_{ind}"].values,
                                                      distance=dist)

    # go through each filter
    for i, filter in enumerate(filters):
        for prefix, mag_type in [("m", "app"), ("M", "abs")]:
            # apply the bolometric corrections to the apparent magnitude of each star
            filter_mags = [photometry[f"{prefix}_{mag_type}_{ind}"].values - bc[mag_type][ind - 1][:, i]
                           for ind in [1, 2]]

            # total the magnitudes (removing any NaNs), which is only meaningful for bound systems
            total_filter_mag = add_mags(*filter_mags, remove_nans=True)
            total_filter_mag[disrupted] = np.nan
            photometry[f"{filter}_{mag_type}"] = total_filter_mag

            # always also store the magnitudes of each individual star
            for ind in [1, 2]:
                photometry[f"{filter}_{mag_type}_{ind}"] = filter_mags[ind - 1]

            # for the G filter in particular, see which temperature/log-g is getting measured
            if filter == main_filter and mag_type == "app":
                # by default assume the primary is dominant
                photometry["teff_obs"] = final_bpp["teff_1"].values
                photometry["log_g_obs"] = final_bpp["log_g_1"].values

                # overwrite any values where the secondary is brighter
                two_is_brighter = (filter_mags[1] < filter_mags[0]) | (np.isnan(filter_mags[0])
                                                                       & ~np.isnan(filter_mags[1]))
                photometry["secondary_brighter"] = two_is_brighter
                photometry.loc[two_is_brighter, "teff_obs"] = final_bpp["teff_2"].values[two_is_brighter]
                photometry.loc[two_is_brighter, "log_g_obs"] = final_bpp["log_g_2"].values[two_is_brighter]

    return photometry
