import numpy as np
import h5py as h5
import astropy.units as u
import os.path

import readsnap
from meo_python_lib import quick_lookback_time

FIRE_ptypes = ["gas", "dark matter", "dark matter low res", "dark matter low res", "star"]


class FIRESnapshot():
    def __init__(self, snap_num=600,
                 snap_dir="/mnt/home/chayward/firesims/fire2/public_release/core/m11h_res7100/output",
                 centres_dir="/mnt/home/twagg/ceph/FIREmaps/m11h_res7100/centers/",
                 which_centre="main", max_r=30 * u.kpc, min_t_form=13.6 * u.Gyr, particle_type="star"):
        """Access data from a specific FIRE snapshot.

        Allows you mask data based on radius and age, as well as account for offsets from centre in position
        and velocity.

        Parameters
        ----------
        snap_num : `int`, optional
            ID of the snapshot, by default 600
        snap_dir : `str`, optional
            Directory in which to find the snapshot,
            by default "/mnt/home/chayward/firesims/fire2/public_release/core/m11h_res7100/output"
        centres_dir : `str`, optional
            Directory of the files listing the centres of the galaxy,
            by default "/mnt/home/twagg/ceph/FIREmaps/m11h_res7100/centers/"
        which_centre : `str`, optional
            Which centre to use, either "main" or "stellar", by default "main"
        max_r : :class:`~astropy.units.Quantity` [length], optional
            Maximum distance from galactic centre of particle to include, by default 30*u.kpc
        min_t_form : :class:`~astropy.units.Quantity` [time], optional
            Minimum formation time that a particle can have, by default 13.6*u.Gyr
        particle_type : `str` or `int`, optional
            Which type of particle to consider either one of {"gas", "dark matter", "dark matter low res", 
            "star"} or {0, 1, 2, 4}, by default "star"
        """
        # convert particle type to expected int
        if isinstance(particle_type, str):
            if particle_type not in FIRE_ptypes:
                raise ValueError(f"'{particle_type}' is not a valid particle type (choose one of {FIRE_ptypes})")
            particle_type = FIRE_ptypes.index(particle_type)
        self.particle_type = particle_type

        # grab the snapshot data
        snap = readsnap.readsnap(sdir=snap_dir, snum=snap_num, ptype=particle_type, cosmological=1)
        header = readsnap.readsnap(sdir=snap_dir, snum=snap_num, ptype=particle_type, header_only=1, cosmological=1)

        # get info about the centre of the galaxy
        with h5.File(os.path.join(centres_dir, f"snap_{snap_num}_cents.hdf5")) as cent:
            self.n = cent.attrs["NormalVector"]
            self.centre = cent.attrs["Center"]
            self.stellar_centre = cent.attrs["Center"]
            self.v_cm = cent.attrs["StellarCMVel"]

        self.h = header["hubble"]
        self.Omega_M = header["Omega0"]
        self.max_r = max_r
        self.min_t_form = min_t_form

        # get the positions, velocities, masses and ages in right units
        self.p_all = (snap["p"] - (self.centre if which_centre == "main" else self.stellar_centre)) * u.kpc / self.h
        self.v_all = (snap["v"] - self.v_cm) * u.km / u.s
        self.m_all = snap["m"] * 1e10 * u.Msun / self.h
        self.t_form_all = quick_lookback_time(1 / snap["age"] - 1, h=self.h, Omega_M=self.Omega_M) * u.Gyr\
            if particle_type == 4 else None
        
        self._p = None
        self._v = None
        self._m = None
        self._t_form = None

        self._X_s = None
        self._V_s = None

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
        self._X_s = None
        self._V_s = None

        if self.particle_type == 4:
            self._t_form = self.t_form_all[total_mask]

    def __repr__(self) -> str:
        return (f"<{self.__class__.__name__} | {len(self.p)} {FIRE_ptypes[self.particle_type]} particles | "
                f"R < {self.max_r} & t_form > {self.min_t_form}>")

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