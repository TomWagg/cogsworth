import numpy as np
import astropy.units as u
import astropy.constants as const


def dispersion_from_virial_parameter(alpha_vir, R, M):
    """Calculate the velocity dispersion from the virial parameter

    Parameters
    ----------
    alpha_vir : `float`
        Virial parameter
    R : :class:`astropy.units.Quantity` [length]
        Cluster radius
    M : :class:`astropy.units.Quantity` [mass]
        Cluster mass

    Returns
    -------
    v_disp : :class:`astropy.units.Quantity` [speed]
        Velocity dispersion
    """
    return np.sqrt(alpha_vir * const.G * M / (5 * R)).to(u.km / u.s)


def quick_lookback_time(z, h=0.71, Omega_M=0.27):
    """Quickly calculate lookback time for a given redshift and cosmology

    Parameters
    ----------
    z : :class:`numpy.ndarray` or `float`
        Redshift
    h : `float`, optional
        Hubble parameter, by default 0.71
    Omega_M : `float`, optional
        $\Omega_M$, by default 0.27

    Returns
    -------
    t : :class:`numpy.ndarray` or `float`
        Lookback time
    """
    # exact solution for a flat universe
    a = 1. / (1. + z)
    x = Omega_M / (1. - Omega_M) / (a * a * a)
    t = (2. / (3. * np.sqrt(1. - Omega_M))) * np.log(np.sqrt(x) / (-1. + np.sqrt(1. + x)))
    t *= 13.777 * (0.71 / h)
    return t
