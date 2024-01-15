import numpy as np
import astropy.units as u
import astropy.constants as const


def prepare_snapshot():
    raise NotImplementedError


def dispersion_from_virial_parameter(alpha_vir, R, M):
    """Calculate the velocity dispersion from the virial parameter

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
