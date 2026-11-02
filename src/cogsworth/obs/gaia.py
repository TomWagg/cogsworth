import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import astropy.units as u

__all__ = ["get_gaiamock_sources", "run_gaiamock", "decode_cascade"]

# columns of a sources table, named to match the arguments of gaiamock.run_full_astrometric_cascade
CASCADE_ARGS = ["ra", "dec", "parallax", "pmra", "pmdec", "m1", "m2", "period", "Tp", "ecc",
                "omega", "inc_deg", "w", "phot_g_mean_mag", "f"]

# number of elements in each result row returned by gaiamock.run_full_astrometric_cascade
N_CASCADE_RESULTS = 23

# signal-to-noise thresholds of gaiamock's looser DR4 cuts on orbital solutions
DR4_A0_SNR, DR4_PLX_SNR = 15.0, 5.0

DATA_RELEASES = ["dr3", "dr4", "dr5"]

# C functions from gaiamock, loaded once per process
_C_FUNCS = None


def get_gaiamock_sources(population, G_filter="Gaia_G_EDR3", G_range=(13.0, 21.0),
                         wide_separation=0.5 * u.arcsec, remove_wide_binaries=False, seed=None):
    """Select bound binaries from a population and convert them to gaiamock inputs

    Each bound binary is treated as a single unresolved source, with ``m1`` set to the brighter star in
    ``G_filter`` and the flux ratio ``f`` set by the G magnitudes of the two stars (so ``f = 0`` for dark
    companions). The orientation of each orbit (inclination, angles and time of periastron) is drawn at
    random, since this is not tracked by cogsworth.

    gaiamock doesn't account for binaries wide enough that Gaia resolves them into two sources. A warning is
    raised if any binaries with two luminous stars have an angular semi-major axis larger than
    ``wide_separation``, unless ``remove_wide_binaries=True``, in which case these binaries are removed.

    .. warning::

        This function assumes the final positions and velocities are in the MW galactocentric frame

    Parameters
    ----------
    population : :class:`~cogsworth.pop.Population`
        The population from which to select sources. Its observables must already have been calculated
        with :meth:`~cogsworth.pop.Population.get_observables` and include ``G_filter``
    G_filter : `str`, optional
        Name of the Gaia G-band filter in the population's observables, by default "Gaia_G_EDR3"
    G_range : `tuple` of `float`, optional
        Range of combined apparent G magnitudes to keep. gaiamock's noise model is only valid for G > 13 and
        Gaia doesn't produce astrometry fainter than G = 21, by default (13.0, 21.0)
    wide_separation : :class:`~astropy.units.Quantity` [angle], optional
        Angular semi-major axis above which binaries with two luminous stars are considered wide enough to be
        resolved by Gaia, by default 0.5 arcsec
    remove_wide_binaries : `bool`, optional
        Whether to remove wide binaries rather than warn about them, by default False
    seed : `int` or :class:`~numpy.random.SeedSequence`, optional
        Seed for drawing orbital orientations, by default None

    Returns
    -------
    sources : :class:`~pandas.DataFrame`
        Table of sources indexed by ``bin_num`` with columns matching the arguments of
        :func:`gaiamock.run_full_astrometric_cascade` (``m1`` is the brighter star and ``m2`` the fainter
        star, both in solar masses), plus the extinction in the G-band, ``A_G``, and the angular semi-major
        axis in arcseconds, ``a_arcsec``

    Raises
    ------
    ValueError
        If the population's observables don't include ``G_filter``
    """
    if f"{G_filter}_app" not in population.observables:
        raise ValueError(f"Population observables don't include '{G_filter}', run `get_observables` with "
                         "this filter first")

    final_bpp = population.final_bpp
    observables = population.observables

    # select bound binaries (excluding mergers) within the magnitude range
    G_app = observables[f"{G_filter}_app"].values
    in_range = (G_app >= G_range[0]) & (G_app <= G_range[1])
    mask = ~population.disrupted & (final_bpp["sep"].values > 0) & in_range
    n = mask.sum()
    if n == 0:
        return pd.DataFrame(columns=CASCADE_ARGS + ["A_G", "a_arcsec"], index=final_bpp.index[:0],
                            dtype=float)

    # bound systems come first in the final positions so we can safely mask the first len(population)
    icrs = population.get_final_mw_skycoord()[:len(population)][mask].icrs
    distance = icrs.distance.to(u.pc).value

    bpp = final_bpp[mask]
    G_app = G_app[mask]
    G_abs = observables[f"{G_filter}_abs"].values[mask]

    # treat stars with NaN magnitudes as dark (as in the combined magnitude), then make m1 the brighter star
    G_1, G_2 = (np.nan_to_num(observables[f"{G_filter}_app_{i}"].values[mask], nan=np.inf) for i in [1, 2])
    one_brighter = G_1 <= G_2
    G_bright, G_faint = np.where(one_brighter, G_1, G_2), np.where(one_brighter, G_2, G_1)
    m1 = np.where(one_brighter, bpp["mass_1"], bpp["mass_2"])
    m2 = np.where(one_brighter, bpp["mass_2"], bpp["mass_1"])

    # semi-major axis in AU over distance in pc gives the angular size in arcsec
    a_arcsec = (bpp["sep"].values * u.Rsun).to(u.AU).value / distance

    # draw random orientations (isotropic inclinations)
    rng = np.random.default_rng(seed)
    period = bpp["porb"].values
    sources = pd.DataFrame({
        "ra": icrs.ra.to(u.deg).value,
        "dec": icrs.dec.to(u.deg).value,
        "parallax": 1e3 / distance,
        "pmra": icrs.pm_ra_cosdec.to(u.mas / u.yr).value,
        "pmdec": icrs.pm_dec.to(u.mas / u.yr).value,
        "m1": m1,
        "m2": m2,
        "period": period,
        "Tp": rng.uniform(0, 1, n) * period,
        "ecc": bpp["ecc"].values,
        "omega": rng.uniform(0, 2 * np.pi, n),
        "inc_deg": np.degrees(np.arccos(rng.uniform(-1, 1, n))),
        "w": rng.uniform(0, 2 * np.pi, n),
        "phot_g_mean_mag": G_app,
        "f": 10**(-0.4 * (G_faint - G_bright)),
        "A_G": G_app - G_abs - 5 * np.log10(distance / 10),
        "a_arcsec": a_arcsec,
    }, index=bpp.index)

    # binaries with a dark companion can't be resolved, no matter how wide
    wide = (sources["a_arcsec"] > wide_separation.to(u.arcsec).value) & (sources["f"] > 0)
    if remove_wide_binaries:
        sources = sources[~wide]
    elif wide.any():
        warnings.warn(f"{wide.sum()} binaries have two luminous stars with an angular semi-major axis larger "
                      f"than {wide_separation}. These may be resolved by Gaia, but gaiamock treats them as "
                      "single unresolved sources. Set `remove_wide_binaries=True` to remove wide binaries.")

    return sources


def _load_c_funcs():
    global _C_FUNCS
    import gaiamock
    _C_FUNCS = gaiamock.read_in_C_functions()


def _run_cascade(kwargs, seed, data_release):
    """Run the gaiamock astrometric cascade for a single source"""
    import gaiamock
    if _C_FUNCS is None:
        _load_c_funcs()

    # gaiamock uses the global numpy random state
    np.random.seed(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return list(gaiamock.run_full_astrometric_cascade(**kwargs, data_release=data_release,
                                                          c_funcs=_C_FUNCS))


def run_gaiamock(sources, data_release="dr4", seed=None, processes=1):
    """Simulate Gaia epoch astrometry and fit the astrometric cascade for each source with gaiamock

    Parameters
    ----------
    sources : :class:`~pandas.DataFrame`
        Table of sources, as produced by :func:`get_gaiamock_sources`
    data_release : `str`, optional
        Which Gaia data release to simulate, one of "dr3", "dr4" or "dr5", by default "dr4"
    seed : `int` or :class:`~numpy.random.SeedSequence`, optional
        Seed for the measurement noise. Each source gets its own seed drawn from this one, so results don't
        depend on ``processes``, by default None
    processes : `int`, optional
        Number of processes to use, by default 1

    Returns
    -------
    results : :class:`~pandas.DataFrame`
        Decoded results of the cascade (see :func:`decode_cascade`) with the same index as ``sources``
    """
    if data_release not in DATA_RELEASES:
        raise ValueError(f"`data_release` must be one of {DATA_RELEASES}, not '{data_release}'")

    records = sources[CASCADE_ARGS].to_dict("records")
    seeds = np.random.default_rng(seed).integers(2**32, size=len(records)).tolist()
    args = [(kwargs, s, data_release) for kwargs, s in zip(records, seeds)]

    if processes > 1 and len(args) > 1:
        with Pool(processes, initializer=_load_c_funcs) as pool:
            results = pool.starmap(_run_cascade, args)
    else:
        # restore the global random state afterwards so we don't clobber the user's
        state = np.random.get_state()
        try:
            results = [_run_cascade(*a) for a in args]
        finally:
            np.random.set_state(state)

    return decode_cascade(results, index=sources.index)


def decode_cascade(results, index=None):
    """Convert raw results from :func:`gaiamock.run_full_astrometric_cascade` to a table

    Parameters
    ----------
    results : `list` of `list`
        Result rows from the cascade, each of length 23
    index : `array_like`, optional
        Index for the table, by default None

    Returns
    -------
    decoded : :class:`~pandas.DataFrame`
        Table of results with columns:

        - ``solution_type``: one of "unobservable", "single_star", "accel7", "accel9" or "orbital"
        - ``ruwe``: renormalised unit weight error of the accepted solution
        - ``accel_significance``: significance of the acceleration solution (NaN unless accel7/accel9)
        - ``f2``: goodness of fit of the accepted solution (NaN for unobservable and single star solutions)
        - ``fit_*``: fitted orbital parameters and their uncertainties (NaN unless orbital)
        - ``n_vis_periods``, ``n_obs``: number of visibility periods and observations (NaN unless orbital)
        - ``published_dr3``: whether an orbital solution passes the cuts applied to the DR3 catalogue
        - ``published_dr4_loose``: whether an orbital solution passes gaiamock's looser DR4 cuts
    """
    r = np.asarray(results, dtype=float).reshape(-1, N_CASCADE_RESULTS)
    sol = r[:, 0]
    single, accel9, accel7 = sol == -1, sol == -9, sol == -7
    orbital = sol > 0

    def orbital_only(j):
        return np.where(orbital, r[:, j], np.nan)

    decoded = pd.DataFrame({
        "solution_type": np.select([sol == 0, single, accel9, accel7],
                                   ["unobservable", "single_star", "accel9", "accel7"], "orbital"),
        "ruwe": np.select([single, accel9, accel7, orbital], [r[:, 1], r[:, 12], r[:, 8], r[:, 22]], np.nan),
        "accel_significance": np.where(accel9 | accel7, r[:, 1], np.nan),
        "f2": np.select([orbital, accel9, accel7], [r[:, 21], r[:, 13], r[:, 9]], np.nan),
        "fit_parallax": orbital_only(0),
        "fit_parallax_err": orbital_only(1),
        "fit_period": orbital_only(10),
        "fit_period_err": orbital_only(11),
        "fit_ecc": orbital_only(14),
        "fit_ecc_err": orbital_only(15),
        "fit_inc_deg": orbital_only(16),
        "fit_a0": orbital_only(17),
        "fit_a0_err": orbital_only(18),
        "n_vis_periods": orbital_only(19),
        "n_obs": orbital_only(20),
    }, index=index)

    with np.errstate(divide="ignore", invalid="ignore"):
        a0_snr = decoded["fit_a0"] / decoded["fit_a0_err"]
        plx_snr = decoded["fit_parallax"] / decoded["fit_parallax_err"]
        p = decoded["fit_period"]
        decoded["published_dr3"] = (orbital & (a0_snr > 5) & (plx_snr > 2e4 / p) & (a0_snr > 158 / np.sqrt(p))
                                    & (decoded["fit_ecc_err"] < 0.079 * np.log(p) - 0.244))
        decoded["published_dr4_loose"] = orbital & (a0_snr > DR4_A0_SNR) & (plx_snr > DR4_PLX_SNR)

    return decoded
