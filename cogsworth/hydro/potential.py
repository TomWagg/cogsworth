import astropy.units as u
from gala.units import galactic
import gala.potential as gp
from .snap import FIRESnapshot


def get_snapshot_potential(components=None, snap_dir=None, snap_num=None, snap_params={}, out_path=None,
                           nmax=15, lmax=5, verbose=False):
    if components is None and (snap_dir is None or snap_num is None):
        raise ValueError("Must provide either `components` or `snap_dir` and `snap_num`")
    elif components is None:
        if verbose:
            print("Loading snapshots from files")
        components = [FIRESnapshot(snap_dir=snap_dir, snap_num=snap_num, particle_type=pt, **snap_params)
                      for pt in ["star", "dark matter", "gas"]]

    pot = gp.CompositePotential()
    for label, snap, r_s in zip(["star", "dark matter", "gas"], components, [3, 10, 3]):
        if verbose:
            print(f"Computing potential for {label}")
        Snlm, Tnlm = gp.scf.compute_coeffs_discrete(xyz=snap.X_s.T.to(u.kpc).value,
                                                    mass=snap.m.to(u.Msun).value / snap.m.sum().value,
                                                    nmax=nmax, lmax=lmax, r_s=r_s, skip_m=True)

        pot[label] = gp.scf.SCFPotential(m=snap.m.sum().value, r_s=r_s, Snlm=Snlm, Tnlm=Tnlm, units=galactic)

    if out_path is not None:
        pot.save(out_path)

    return pot
