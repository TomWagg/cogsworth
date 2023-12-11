import numpy as np
import h5py as h5
import astropy.units as u
import os.path

from cogsworth.tests.optional_deps import check_dependencies
from cogsworth.citations import CITATIONS
from cogsworth.pop import Population

from .centre import find_centre
from .readsnap import read_snapshot
from .utils import quick_lookback_time, dispersion_from_virial_parameter


__all__ = ["FIRESnapshot", "FIREPopulation"]


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
        self.snap_time = quick_lookback_time(1 / header["time"] - 1, h=self.h, Omega_M=self.Omega_M) * u.Gyr
        self.max_r = max_r
        self.min_t_form = min_t_form

        # get the positions, velocities, masses and ages in right units
        self.p_all = (snap["p"] - self.stellar_centre) * u.kpc / self.h
        self.v_all = (snap["v"] - self.v_cm) * u.km / u.s
        self.m_all = snap["m"] * 1e10 * u.Msun / self.h
        self.Z_all = snap["Z"][:, 0]
        self.ids_all = snap["id"]
        self.t_form_all = quick_lookback_time(1 / snap["age"] - 1, h=self.h, Omega_M=self.Omega_M) * u.Gyr\
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
        """FIRE Positions"""
        if self._p is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._p

    @property
    def v(self):
        """FIRE Velocities"""
        if self._v is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._v

    @property
    def m(self):
        """Masses"""
        if self._m is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._m

    @property
    def Z(self):
        """Metallicities"""
        if self._Z is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._Z

    @property
    def ids(self):
        """IDs"""
        if self._ids is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._ids

    @property
    def t_form(self):
        """Formation times"""
        if self._t_form is None:
            self.apply_mask(max_r=self.max_r, min_t_form=self.min_t_form)
        return self._t_form

    @property
    def X_s(self):
        """Galactocentric positions"""
        if self._X_s is None:
            self._X_s = np.matmul(self.p.to(u.kpc).value, self.n.T).T * u.kpc
        return self._X_s

    @property
    def V_s(self):
        """Galactocentric velocities"""
        if self._V_s is None:
            self._V_s = np.matmul(self.v.to(u.km / u.s).value, self.n.T).T * u.km / u.s
        return self._V_s

    @property
    def x(self):
        """Galactocentric x positions"""
        return self.X_s[0]

    @property
    def y(self):
        """Galactocentric y positions"""
        return self.X_s[1]

    @property
    def z(self):
        """Galactocentric z positions"""
        return self.X_s[2]

    @property
    def r(self):
        """Galactocentric radius"""
        return np.linalg.norm(self.X_s, axis=0)

    @property
    def v_x(self):
        """Galactocentric x velocities"""
        return self.V_s[0]

    @property
    def v_y(self):
        """Galactocentric y velocities"""
        return self.V_s[1]

    @property
    def v_z(self):
        """Galactocentric x velocities"""
        return self.V_s[2]


class FIREPopulation(Population):
    def __init__(self, star_particles=None, particle_size=1 * u.pc, particle_boundedness=1.0,
                 sampling_params={"sampling_target": "total_mass",
                                  "trim_extra_samples": True,
                                  "keep_singles": True,}, **kwargs):
        self.star_particles = star_particles
        self.particle_size = particle_size
        self.particle_boundedness = particle_boundedness
        if "n_binaries" not in kwargs:
            kwargs["n_binaries"] = None
        super().__init__(sampling_params=sampling_params, **kwargs)

    def __getitem__(self, ind):
        if self._initC is not None and "particle_id" not in self._initC.columns:
            self._initC["particle_id"] = self._initial_binaries["particle_id"]
        new_pop = super().__getitem__(ind)
        new_pop.star_particles = self.star_particles
        new_pop.particle_size = self.particle_size
        new_pop.particle_boundedness = self.particle_boundedness
        return new_pop

    def sample_initial_binaries(self):
        assert self.star_particles is not None,\
            "`self.star_particles` is None, must provide star particles to sample from"
        initial_binaries_list = [None for _ in range(len(self.star_particles))]
        self._mass_singles, self._mass_binaries, self._n_singles_req, self._n_bin_req = 0.0, 0.0, 0, 0

        i = 0
        for id, particle in self.star_particles.iterrows():
            samples = InitialBinaryTable.sampler('independent', np.linspace(0, 15, 16), np.linspace(0, 15, 16),
                                                binfrac_model=self.BSE_settings["binfrac"],
                                                SF_start=self.max_ev_time.to(u.Myr).value - particle["t_form"] * 1000,
                                                SF_duration=0.0, met=particle["Z"],
                                                total_mass=particle["mass"],
                                                size=particle["mass"] * 0.8,
                                                **self.sampling_params)
        
            # apply the mass cutoff and record particle ID
            samples[0].reset_index(inplace=True)
            samples[0].drop(samples[0][samples[0]["mass_1"] < self.m1_cutoff].index, inplace=True)
            samples[0]["particle_id"] = np.repeat(id, len(samples[0]))

            # save samples
            initial_binaries_list[i] = samples[0]
            self._mass_singles += samples[1]
            self._mass_binaries += samples[2]
            self._n_singles_req += samples[3]
            self._n_bin_req += samples[4]

            i += 1

        self._initial_binaries = pd.concat(initial_binaries_list, ignore_index=True)

        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

        # same for this class
        self.n_binaries = len(self._initial_binaries)
        self.n_binaries_match = len(self._initial_binaries)

        # ensure metallicities remain in a range valid for COSMIC - original value still in initial_galaxy.Z
        self._initial_binaries.loc[self._initial_binaries["metallicity"] < 1e-4, "metallicity"] = 1e-4
        self._initial_binaries.loc[self._initial_binaries["metallicity"] > 0.03, "metallicity"] = 0.03

        self.sample_initial_galaxy()

    def sample_initial_galaxy(self):
        particles = self.star_particles.loc[self._initial_binaries["particle_id"]]

        x, y, z = np.random.normal([particles["x"].values,
                                    particles["y"].values,
                                    particles["z"].values],
                                    self.particle_size.to(u.kpc).value / np.sqrt(3),
                                    size=(3, self.n_binaries_match)) * u.kpc

        self._initial_galaxy = cogsworth.galaxy.Galaxy(self.n_binaries_match, immediately_sample=False)
        self._initial_galaxy._x = x
        self._initial_galaxy._y = y
        self._initial_galaxy._z = z
        self._initial_galaxy._tau = self._initial_binaries["tphysf"].values * u.Myr
        self._initial_galaxy._Z = self._initial_binaries["metallicity"].values
        self._initial_galaxy._which_comp = np.repeat("FIRE", len(self.initial_galaxy._tau))

        v_R = (particles["x"] * particles["v_x"] + particles["y"] * particles["v_y"])\
            / (particles["x"]**2 + particles["y"]**2)**0.5
        v_T = (particles["x"] * particles["v_y"] - particles["y"] * particles["v_x"])\
            / (particles["x"]**2 + particles["y"]**2)**0.5
        v_z = particles["v_z"]

        vel_units = u.km / u.s
        dispersion = dispersion_from_virial_parameter(self.particle_boundedness,
                                                      self.particle_size,
                                                      particles["mass"].values * u.Msun)
        v_R, v_T, v_z = np.random.normal([v_R.values, v_T.values, v_z.values], dispersion / np.sqrt(3),
                                         size=(3, self.n_binaries_match)) * vel_units

        self._initial_galaxy._v_R = v_R
        self._initial_galaxy._v_T = v_T
        self._initial_galaxy._v_z = v_z
