import numpy as np
import h5py as h5
import astropy.units as u
import os.path

from cogsworth.tests.optional_deps import check_dependencies
from cogsworth.citations import CITATIONS
from cogsworth.pop import Population

from centre import find_centre
from readsnap import read_snapshot


__all__ = ["FIRESnapshot"]


FIRE_ptypes = ["gas", "dark matter", "dark matter low res", "dark matter low res", "star"]


class FIRESnapshot():
    def __init__(self, snap_dir, snap_num, max_r=30 * u.kpc,
                 min_t_form=13.6 * u.Gyr, particle_type="star", cosmological=True, centre_params={}):
        """Access data from a specific FIRE snapshot.

        Allows you mask data based on radius and age, as well as account for offsets from centre in position
        and velocity.

        Parameters
        ----------
        snap_dir : `str`, optional
            Directory in which to find the snapshot
        snap_num : `int`, optional
            ID of the snapshot
        max_r : :class:`~astropy.units.Quantity` [length], optional
            Maximum distance from galactic centre of particle to include, by default 30*u.kpc
        min_t_form : :class:`~astropy.units.Quantity` [time], optional
            Minimum formation time that a particle can have, by default 13.6*u.Gyr
        particle_type : `str` or `int`, optional
            Which type of particle to consider either one of {"gas", "dark matter", "dark matter low res", 
            "star"} or {0, 1, 2, 4}, by default "star"
        cosmological : `bool`, optional
            Whether the simulation was run in cosmological mode, by default True
        centre_params : `dict`, optional
            Parameters to pass to :meth:`FIRESnapshot.set_centre` when calculating the centre of the galaxy
        """
        # convert particle type to expected int
        if isinstance(particle_type, str):
            if particle_type not in FIRE_ptypes:
                raise ValueError(f"'{particle_type}' is not a valid particle type (choose one of {FIRE_ptypes})")
            particle_type = FIRE_ptypes.index(particle_type)
        self.particle_type = particle_type
        self.snap_dir = snap_dir
        self.snap_num = snap_num
        self.cosmological = cosmological

        # grab the snapshot data
        snap, header = self.get_snap(cosmological=self.cosmological)

        self.set_centre(**centre_params)

        self.h = header["hubble"]
        self.Omega_M = header["Omega0"]
        self.snap_time = _quick_lookback_time(1 / header["time"] - 1, h=self.h, Omega_M=self.Omega_M) * u.Gyr
        self.max_r = max_r
        self.min_t_form = min_t_form

        # get the positions, velocities, masses and ages in right units
        self.p_all = (snap["p"] - self.stellar_centre) * u.kpc / self.h
        self.v_all = (snap["v"] - self.v_cm) * u.km / u.s
        self.m_all = snap["m"] * 1e10 * u.Msun / self.h
        self.Z_all = snap["Z"][:, 0]
        self.ids_all = snap["id"]
        self.t_form_all = _quick_lookback_time(1 / snap["age"] - 1, h=self.h, Omega_M=self.Omega_M) * u.Gyr\
            if particle_type == 4 else None

        self._p = None
        self._v = None
        self._m = None
        self._Z = None
        self._ids = None
        self._t_form = None

        self._X_s = None
        self._V_s = None

    def get_snap(self, ptype=None, h0=False, cosmological=None, header_only=False):
        """Read in a snapshot from a FIRE simulation

        This code is heavily based on a function written by Matt Orr, thanks Matt!

        Parameters
        ----------
        ptype : `int`, optional
            Particle type to use, by default self.particle_type
        h0 : `bool`, optional
            _description_, by default False
        cosmological : `bool`, optional
            Whether the simulation was run in cosmological mode, by default self.cosmological
        header_only : `bool`, optional
            Whether to just return the header information, by default False

        Returns
        -------
        params : `dict`
            The various parameters associated with the snapshot and `self.particle_type` across every file
            from this snapshot (only returned when `header_only` is False)
        header_params : `dict`
            The header parameters for this snapshot

        Raises
        ------
        FileNotFoundError
            No snapshot file could be found
        ValueError
            No particles of type `self.particle_type` in snapshot
        """
        return read_snapshot(self.snap_dir, self.snap_num,
                             ptype=self.particle_type if ptype is None else ptype,
                             h0=h0, cosmological=self.cosmological if cosmological is None else cosmological,
                             header_only=header_only)

    def set_centre(self, centres_dir=None, theta=0.0, phi=0.0, project_ang_mom=True, force_recalculate=False, verbose=False):
        """Set the centre of the galaxy, the centre of mass velocity and normal vector to galactic plane

        If this already exists in a file then it will be read in, otherwise it will be calculated and saved

        Parameters
        ----------
        centres_dir : `str`, optional
            Directory of the files listing the centres of the galaxy, by default subdirectory "centres" in
            `snap_dir`
        theta : `float`, optional
            Angle to use if not projecting, by default 0.0
        phi : , optional
            Angle to use if not projecting, by default 0.0
        project_ang_mom : `bool`, optional
            Whether to project to the plane perpendicular to the total angular momentum, by default True
        force_recalculate : `bool`, optional
            Whether to recalculate even if there's a file with a centre already found, by default False
        verbose : `bool`, optional
            Whether to print out more information, by default False
        """
        centres_dir = os.path.join(self.snap_dir, "centres") if centres_dir is None else centres_dir
        centre_file = os.path.join(centres_dir, f"snap_{self.snap_num}_cents.hdf5")
        if (os.path.exists(centre_file) and not force_recalculate):
            if verbose:
                print("Using previously calculated centre and normal vector from {centre_file}")
            with h5.File(centre_file, 'r') as cent:
                self.n = cent.attrs["NormalVector"]
                self.stellar_centre = cent.attrs["StellarCenter"]
                self.v_cm = cent.attrs["StellarCMVel"]
        else:
            self.stellar_centre, self.v_cm, self.n, _ = find_centre(self.snap_dir, self.snap_num,
                                                                    out_path=centres_dir,
                                                                    theta=theta, phi=phi,
                                                                    project_ang_mom=project_ang_mom)

    def apply_mask(self, max_r=30 * u.kpc, min_t_form=13.6 * u.Gyr):
        """Apply a radius/age cut to the snapshot

        Parameters
        ----------
        max_r : :class:`~astropy.units.Quantity` [length], optional
            Maximum distance from galactic centre of particle to include, by default 30*u.kpc
        min_t_form : :class:`~astropy.units.Quantity` [time], optional
            Minimum formation time that a particle can have, by default 13.6*u.Gyr
        """
        # mask based on max distance to galaxy centre
        r_mask = np.linalg.norm(self.p_all, axis=1) < max_r

        # only stars have formation times
        if self.particle_type == 4:
            # mask based on stellar formation time
            age_mask = self.t_form_all > min_t_form
            total_mask = r_mask & age_mask
        else:
            total_mask = r_mask
        self._p = self.p_all[total_mask]
        self._v = self.v_all[total_mask]
        self._m = self.m_all[total_mask]
        self._Z = self.Z_all[total_mask]
        self._ids = self.ids_all[total_mask]
        self._X_s = None
        self._V_s = None

        if self.particle_type == 4:
            self._t_form = self.t_form_all[total_mask]

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} | {len(self.p)} {FIRE_ptypes[self.particle_type]} particles | "
                f"R < {self.max_r} & t_form > {self.min_t_form}>")

    def __len__(self) -> int:
        return len(self.p)

    @property
    def p(self):
        if self._p is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._p

    @property
    def v(self):
        if self._v is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._v

    @property
    def m(self):
        if self._m is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._m

    @property
    def Z(self):
        if self._Z is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._Z

    @property
    def ids(self):
        if self._ids is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._ids

    @property
    def t_form(self):
        if self._t_form is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._t_form

    @property
    def X_s(self):
        if self._X_s is None:
            self._X_s = np.matmul(self.p.to(u.kpc).value, self.n.T).T * u.kpc
        return self._X_s

    @property
    def V_s(self):
        if self._V_s is None:
            self._V_s = np.matmul(self.v.to(u.km / u.s).value, self.n.T).T * u.km / u.s
        return self._V_s

    @property
    def x(self):
        return self.X_s[0]

    @property
    def y(self):
        return self.X_s[1]

    @property
    def z(self):
        return self.X_s[2]

    @property
    def r(self):
        return np.linalg.norm(self.X_s, axis=0)

    @property
    def v_x(self):
        return self.V_s[0]

    @property
    def v_y(self):
        return self.V_s[1]

    @property
    def v_z(self):
        return self.V_s[2]


def _quick_lookback_time(z, h=0.71, Omega_M=0.27):
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
