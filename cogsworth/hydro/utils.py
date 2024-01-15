import numpy as np
import astropy.units as u
import astropy.constants as const

import pynbody

__all__ = ["prepare_snapshot", "dispersion_from_virial_parameter"]


def prepare_snapshot(path, halo_params={"mode": "hyb"}):
    """Prepare a snapshot for use in cogsworth

    Convert a snapshot to physical units, centre it on the main halo, and orient it face-on and side-on.

    Parameters
    ----------
    path : `string`
        Path to the snapshot file or directory (as used in :func:`pynbody.snapshot.load`)
    halo_params : `dict`, optional
        Keyword parameters to be passed to :func:`pynbody.analysis.halo.center`, by default {"mode": "hyb"}

    Returns
    -------
    snap : :class:`pynbody.snapshot.SimSnap`
        The prepared snapshot
    """
    # load in the snapshot and use physical units
    snap = pynbody.load(path)
    snap.physical_units()

    # try to use a halo catalogue to centre the snapshot
    try:
        h = snap.halos()
        pynbody.analysis.halo.center(h[1], **halo_params)
    # otherwise, use pynbody's built-in halo finder to centre the snapshot
    except RuntimeError:
        pynbody.analysis.halo.center(snap, **halo_params)

    # orient the snapshot face-on and side-on
    side_on = pynbody.analysis.angmom.sideon(snap, cen=(0, 0, 0))
    face_on = pynbody.analysis.angmom.faceon(snap, cen=(0, 0, 0))

    return snap


def dispersion_from_virial_parameter(alpha_vir, R, M):
    """Calculate the velocity dispersion from the virial parameter

    (Using the virial parameter definition from
    `Bertoldi & McKee (1992) <https://ui.adsabs.harvard.edu/abs/1992ApJ...395..140B/abstract>`_ Eq. 2.8a)

    Parameters
    ----------
    alpha_vir : `float`
        Virial parameter
    R : :class:`~astropy.units.Quantity` [length]
        Cluster radius
    M : :class:`~astropy.units.Quantity` [mass]
        Cluster mass

    Returns
    -------
    v_disp : :class:`~astropy.units.Quantity` [speed]
        Velocity dispersion
    """
    return np.sqrt(alpha_vir * const.G * M / (5 * R)).to(u.km / u.s)
