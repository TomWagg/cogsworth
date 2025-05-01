import numpy as np
import astropy.units as u
import astropy.constants as const
import warnings
import logging
import os.path

from ..tests.optional_deps import check_dependencies

__all__ = ["prepare_snapshot", "dispersion_from_virial_parameter"]


def prepare_snapshot(path, halo_params={"mode": "ssc"}):
    """Prepare a snapshot for use in cogsworth

    Convert a snapshot to physical units, centre it on the main halo, and orient it face-on and side-on.

    Parameters
    ----------
    path : `string`
        Path to the snapshot file or directory (as used in :func:`pynbody.snapshot.load`)
    halo_params : `dict`, optional
        Keyword parameters to be passed to :func:`pynbody.analysis.halo.center`, by default {"mode": "ssc"}

    Returns
    -------
    snap : :class:`pynbody.snapshot.SimSnap`
        The prepared snapshot
    """
    assert check_dependencies("pynbody")
    import pynbody

    # load in the snapshot and use physical units
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*units.*")
        try:
            snap = pynbody.load(path)
            snap.physical_units()
        except RuntimeWarning:          # pragma: no cover
            warnings.filterwarnings("ignore", message=".*units.*")
            BOLD, RESET = "\033[1m", "\033[0m"
            logging.getLogger("cogsworth").warning(f"{BOLD}cogsworth warning:{RESET} Looks like you're loading a snapshot that doesn't specify its units, I'm going to infer them but make sure the outputted units looks right!")
            snap = pynbody.load(path)
            snap.physical_units()

    # try to use a halo catalogue to centre the snapshot
    try:
        h = snap.halos()
        pynbody.analysis.halo.center(h[1], **halo_params)
    # otherwise, use pynbody's built-in halo finder to centre the snapshot
    except RuntimeError:
        BOLD, RESET = "\033[1m", "\033[0m"
        logging.getLogger("cogsworth").warning(f"{BOLD}cogsworth warning:{RESET} I couldn't find a halo catalogue, so I'll use pynbody's built-in halo finder to centre the snapshot")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*units.*")

            try:
                pynbody.analysis.halo.center(snap, **halo_params)
            except ValueError as e:             # pragma: no cover
                if str(e).startswith("Insufficient particles"):
                    logging.getLogger("cogsworth").error((f"{BOLD}cogsworth error:{RESET} pynbody returned this error: {e}. But this often happens when your pynbody installation is not configured correctly to work with FIRE snapshots."))
                    answer = input("Would you like me to try to fix this for you? (This will overwrite the .pynbodyrc in your home directory if it already exists) [Y/n] ")
                    if answer.lower() == "y":
                        home_dir = os.path.expanduser("~")
                        home_answer = input(f"Is your home directory {home_dir}? [Y/n] ")
                        if home_answer.lower() == "n":
                            home_dir = input("Please enter your home directory: ")
                        with open(os.path.join(home_dir, ".pynbodyrc"), "w") as f:
                            f.writelines([
                                "[gadget-units]\n",
                                "pos: kpc a h^-1\n",
                                "\n",
                                "[gadgethdf-type-mapping]\n",
                                "dm: PartType1\n",
                            ])
                        logging.getLogger("cogsworth").info(f"{BOLD}cogsworth info:{RESET} I've written a .pynbodyrc file to your home directory, try running this again.")
                        return
                else:
                    raise e

    # orient the snapshot face-on and side-on
    side_on = pynbody.analysis.angmom.sideon(snap, already_centered=True)
    face_on = pynbody.analysis.angmom.faceon(snap, already_centered=True)

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
