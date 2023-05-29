import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from dustmaps.bayestar import BayestarQuery

# HACK around the isochrone import to ignore warnings about Holoview and Multinest
import logging
logging.getLogger("isochrones").setLevel("ERROR")
from isochrones.mist.bc import MISTBolometricCorrectionGrid
logging.getLogger("isochrones").setLevel("WARNING")

__all__ = ["get_log_g", "get_absolute_bol_mag", "get_apparent_mag", "get_absolute_mag", "add_mags",
           "get_extinction", "get_photometry"]


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


def get_extinction(coords):     # pragma: no cover
    """Calculates the visual extinction values for a set of coordinates

    Reddening due to dust is calculated using the Bayestar dustmap. Then the conversion from this to a visual
    extension is done assuming a total-to-selective extinction ratio of 3.3, as is used by
    `Green+2019 <https://iopscience.iop.org/article/10.3847/1538-4357/ab5362#apjab5362s2>`_

    .. warning::
        The dustmap used only covers declinations > -30 degrees, any supplied coordinates below this will be
        reflected around the galactic plane (any of [(-l, b), (l, -b), (-l, -b)]) and the dust at these
        locations will be used instead

    Parameters
    ----------
    coords : :class:`~astropy.coordinates.SkyCoord`
        The coordinates at which you wish to calculate extinction values

    Returns
    -------
    Av : :class:`~numpy.ndarray`
        Visual extinction values for each set of coordinates
    """

    # following section performs reflections for coordinates below -30 deg declination
    # convert to galactic coordinates
    galactic = coords.galactic

    # try all possible reflections about the galactic plane
    for ref_l, ref_b in [(-1, 1), (1, -1), (-1, -1)]:
        # check which things are too low for the dustmap
        too_low = galactic.icrs.dec < (-30 * u.deg)

        # if everything is fine now then stop
        if not too_low.any():
            break

        # apply the reflection to the subset that are too low
        reflected = galactic[too_low]
        reflected.data.lon[()] *= ref_l
        reflected.data.lat[()] *= ref_b
        reflected.cache.clear()

        # check which are fixed now, and reverse the reflection if not
        fixed = reflected.icrs.dec > (-30 * u.deg)
        reflected.data.lon[~fixed] *= ref_l
        reflected.data.lat[~fixed] *= ref_b

        # set the data back in the main coord object
        galactic.data.lon[too_low] = reflected.data.lon
        galactic.data.lat[too_low] = reflected.data.lat

        # clear the cache to ensure consistency
        galactic.cache.clear()

    bayestar = BayestarQuery(max_samples=2, version='bayestar2019')

    # calculate the reddening due to dust
    ebv = bayestar(galactic, mode='random_sample')

    # convert this to a visual extinction
    Av = 3.3 * ebv
    return Av


def get_photometry(final_bpp, final_coords, filters, ignore_extinction=False):
    """Computes photometry subject to dust extinction using the MIST boloemtric correction grid

    Parameters
    ----------
    final_bpp : :class:`~pandas.DataFrame`
        A dataset of COSMIC binaries at present day - must include these columns: ["sep", "metallicity"] and
        for each star it must have the columns ["teff", "lum", "mass", "rad", "kstar"]
    final_coords : `tuple` of :class:`~astropy.coordinates.SkyCoord`
        Final positions and velocities of the binaries at present day. First entry is for binaries or the
        primary in a disrupted system, second entry is for secondaries in a disrupted system.
    filters : `list` of `str`
        Which filters to compute photometry for (e.g. ['J', 'H', 'K', 'G', 'BP', 'RP'])
    ignore_extinction : `bool`
        Whether to ignore extinction

    Returns
    -------
    photometry : :class:`~pandas.DataFrame`
        Photometry and extinction information for supplied COSMIC binaries in desired `filters`
    """
    # set up empty photometry table
    photometry = pd.DataFrame()
    disrupted = final_bpp["sep"].values < 0.0

    if not ignore_extinction:       # pragma: no cover
        # get extinction for bound binaries and primary of disrupted binaries
        photometry['Av_1'] = get_extinction(final_coords[0])

        # get extinction for secondaries of disrupted binaries (leave as np.inf otherwise)
        photometry['Av_2'] = np.repeat(np.inf, len(final_coords[1]))
        photometry.loc[disrupted, "Av_2"] = get_extinction(final_coords[1][disrupted])

        # ensure extinction remains in MIST grid range (<= 6) and is not NaN
        photometry.loc[photometry.Av_1 > 6, ['Av_1']] = 6
        photometry.loc[photometry.Av_2 > 6, ['Av_2']] = 6
        photometry = photometry.fillna(6)
    else:
        photometry['Av_1'] = np.zeros(len(final_coords[0]))
        photometry['Av_2'] = np.zeros(len(final_coords[0]))

    # get Fe/H using e.g. Bertelli+1994 Eq. 10 (assuming all stars have the solar abundance pattern)
    Z_sun = 0.0142
    FeH = np.log10(final_bpp["metallicity"].values / Z_sun)

    # set up MIST bolometric correction grid
    bc_grid = MISTBolometricCorrectionGrid(bands=filters)
    bc = {
        "app": [None, None],
        "abs": [None, None]
    }

    # for each star in the (possibly disrupted/merged) binary
    for ind in [1, 2]:
        # calculate the surface gravity if necessary
        if f"log_g_{ind}" not in final_bpp:
            final_bpp.insert(len(final_bpp.columns), f"log_g_{ind}",
                             get_log_g(mass=final_bpp[f"mass_{ind}"].values * u.Msun,
                                       radius=final_bpp[f"rad_{ind}"].values * u.Rsun))

        # get the bolometric corrections from MIST isochrones
        bc["app"][ind - 1] = bc_grid.interp([final_bpp[f"teff_{ind}"].values,
                                             final_bpp[f"log_g_{ind}"].values,
                                             FeH, photometry[f"Av_{ind}"]], filters)
        bc["abs"][ind - 1] = bc_grid.interp([final_bpp[f"teff_{ind}"].values,
                                             final_bpp[f"log_g_{ind}"].values,
                                             FeH, np.zeros(len(final_bpp))], filters)

        # calculate the absolute bolometric magnitude and set any BH or massless remnants to invisible
        photometry[f"M_abs_{ind}"] = get_absolute_bol_mag(lum=final_bpp[f"lum_{ind}"].values * u.Lsun)
        photometry.loc[np.isin(final_bpp[f"kstar_{ind}"].values, [13, 14, 15]), f"M_abs_{ind}"] = np.inf

        # work out the distance (if the system is bound always use the first `final_coords` SkyCoord)
        distance = np.repeat(np.inf, len(final_bpp)) * u.kpc
        distance[disrupted] = final_coords[ind - 1][disrupted].icrs.distance
        distance[~disrupted] = final_coords[0][~disrupted].icrs.distance

        # convert the absolute magnitude to an apparent magnitude
        photometry[f"m_app_{ind}"] = get_apparent_mag(M_abs=photometry[f"M_abs_{ind}"].values,
                                                      distance=distance)

    # go through each filter
    for i, filter in enumerate(filters):
        for prefix, mag_type in [("m", "app"), ("M", "abs")]:
            # apply the bolometric corrections to the apparent magnitude of each star
            filter_mags = [photometry[f"{prefix}_{mag_type}_{ind}"].values - bc[mag_type][ind - 1][:, i]
                           for ind in [1, 2]]

            # total the magnitudes (removing any NaNs)
            total_filter_mag = add_mags(*filter_mags, remove_nans=True)

            # default to assuming all systems are bound - in this case total magnitude is listed
            # in primary filter mag, and secondary is non-existent
            photometry[f"{filter}_{mag_type}_1"] = total_filter_mag
            photometry[f"{filter}_{mag_type}_2"] = np.repeat(np.inf, len(photometry))

            # for disrupted systems, change filter apparent magnitudes to the values for each individual star
            for ind in [1, 2]:
                photometry.loc[disrupted, f"{filter}_{mag_type}_{ind}"] = filter_mags[ind - 1][disrupted]

            # for the G filter in particular, see which temperature/log-g is getting measured
            if filter == "G" and mag_type == "app":
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
