from gala.units import galactic
import gala.potential as gp
import logging

__all__ = ["get_snapshot_potential"]


def get_snapshot_potential(snap, components=[{"label": "star", "attr": "s", "r_s": 3},
                                             {"label": "dark matter", "attr": "dm", "r_s": 10},
                                             {"label": "gas", "attr": "g", "r_s": 3}],
                           out_path=None, nmax=15, lmax=5, verbose=False):
    r"""Compute the potential of a snapshot of a hydrodynamical zoom-in simulation

    Parameters
    ----------
    snap : :class:`pynbody.snapshot.SimSnap`
        The snapshot
    components : `list`, optional
        List of components to add to the potential, each component specifies the label to use, attribute in
        the pynbody snap and scale radius to use, by default includes stars, dark matter and gas with scale
        radii of 3, 10 and 3 kpc respectively
    out_path : `str`, optional
        Path to save the potential (should be a .yml file), by default None
    nmax : `int`, optional
        Maximum value of $n$ for the radial expansion, by default 15
    lmax : `int`, optional
        Maximum value of $\ell$ for the spherical harmonics, by default 5
    verbose : `bool`, optional
        Whether to report on progress, by default False

    Returns
    -------
    pot : :class:`gala.potential.potential.CompositePotential`
        Potential of the snapshot
    """
    # start a composite potential
    pot = gp.CompositePotential()

    # compute the potential for each component
    for comp in components:
        if verbose:
            logging.getLogger("cogsworth").info(f"Computing potential for {comp['label']}")
        subsnap = getattr(snap, comp["attr"])

        # compute the coefficients for the SCF potential
        Snlm, Tnlm = gp.scf.compute_coeffs_discrete(xyz=subsnap["pos"],
                                                    mass=subsnap["mass"] / subsnap["mass"].sum(),
                                                    nmax=nmax, lmax=lmax, r_s=comp["r_s"], skip_m=True)

        # add the SCF potential to the composite potential
        pot[comp["label"]] = gp.scf.SCFPotential(m=subsnap["mass"].sum().tolist(), r_s=comp["r_s"],
                                                 Snlm=Snlm, Tnlm=Tnlm, units=galactic)

    # save the potential to a file if requested
    if out_path is not None:
        pot.save(out_path)

    return pot
