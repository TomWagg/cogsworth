import numpy as np
import astropy.units as u
import pandas as pd

from cosmic.sample.initialbinarytable import InitialBinaryTable

from ..pop import Population
from ..galaxy import Galaxy

from .utils import dispersion_from_virial_parameter


__all__ = ["HydroPopulation"]


class HydroPopulation(Population):
    def __init__(self, star_particles, particle_size=1 * u.pc, virial_parameter=1.0, subset=None,
                 sampling_params={"sampling_target": "total_mass", "trim_extra_samples": True}, **kwargs):
        """A population of stars sampled from a hydrodynamical zoom-in snapshot

        Each star particle in the snapshot is converted to a (binary) stellar population, sampled with COSMIC,
        with initial positions and kinematics determined based on the virial parameter and particle size.

        Parameters
        ----------
        star_particles : :class:`~pandas.DataFrame`
            A table of star particles from a snapshot that contains the mass, metallicity, formation time,
            position and velocity of each particle, as returned by
            :func:`~cogsworth.hydro.rewind.rewind_to_formation`
        particle_size : :class:`~astropy.units.Quantity`, optional
            Size of gaussian for cluster for each star particle, by default 1*u.pc
        virial_parameter : `float`, optional
            Virial parameter for each cluster, for setting velocity dispersions, by default 1.0
        subset : `int` or `list` of `int`, optional
            If set, only use a subset of the star particles, by default use all of them. If an integer is
            passed, a random subset of that size is used, if a list of integers is passed, only those IDs are
            used.
        sampling_params : `dict`, optional
            As in :class:~cogsworth.pop.Population` but changing the defaults to
            {"sampling_target": "total_mass", "trim_extra_samples": True} (i.e. to sample to the total mass of
            the star particle)
        **kwargs : various
            Parameters to pass to :class:`~cogsworth.pop.Population`
        """
        self.star_particles = star_particles
        self.particle_size = particle_size
        self.virial_parameter = virial_parameter
        self._subset_inds = self.star_particles.index.values
        if subset is not None and isinstance(subset, int):
            self._subset_inds = np.random.choice(self._subset_inds, size=subset, replace=False)
        elif subset is not None:
            self._subset_inds = subset
        if "n_binaries" not in kwargs:
            kwargs["n_binaries"] = None
        super().__init__(sampling_params=sampling_params, **kwargs)
        self.__citations__.append("FIRE")

    def __getitem__(self, ind):
        if self._initC is not None and "particle_id" not in self._initC.columns:
            self._initC["particle_id"] = self._initial_binaries["particle_id"]
        new_pop = super().__getitem__(ind)
        new_pop.star_particles = self.star_particles
        new_pop.particle_size = self.particle_size
        new_pop.virial_parameter = self.virial_parameter
        return new_pop

    def get_citations(self, filename=None):
        super().get_citations(filename)
        print("\nNOTE: This population was sampled from a FIRE snapshot...but I don't know which one so I "
              "can't automate this citation, please cite the relevant FIRE paper(s) (e.g. see here: "
              "http://flathub.flatironinstitute.org/fire)")

    def sample_initial_binaries(self):
        """Sample initial binaries from the star particles in the snapshot"""
        initial_binaries_list = [None for _ in range(len(self.star_particles))]
        self._mass_singles, self._mass_binaries, self._n_singles_req, self._n_bin_req = 0.0, 0.0, 0, 0

        for i in self._subset_inds:
            particle = self.star_particles.loc[i]
            samples = InitialBinaryTable.sampler('independent', self.final_kstar1, self.final_kstar2,
                                                 binfrac_model=self.BSE_settings["binfrac"],
                                                 SF_start=(self.max_ev_time - particle["t_form"] * u.Gyr).to(u.Myr).value,
                                                 SF_duration=0.0, met=particle["Z"],
                                                 total_mass=particle["mass"],
                                                 size=particle["mass"] * 0.8,
                                                 **self.sampling_params)

            # apply the mass cutoff and record particle ID
            samples[0].reset_index(inplace=True)
            samples[0].drop(samples[0][samples[0]["mass_1"] < self.m1_cutoff].index, inplace=True)
            samples[0]["particle_id"] = np.repeat(i, len(samples[0]))

            # save samples
            initial_binaries_list[i] = samples[0]
            self._mass_singles += samples[1]
            self._mass_binaries += samples[2]
            self._n_singles_req += samples[3]
            self._n_bin_req += samples[4]

        self._initial_binaries = pd.concat(initial_binaries_list, ignore_index=True)

        # just in case, probably unnecessary
        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

        # these are the same for this class
        self.n_binaries = len(self._initial_binaries)
        self.n_binaries_match = len(self._initial_binaries)

        # ensure metallicities remain in a range valid for COSMIC
        self._initial_binaries.loc[self._initial_binaries["metallicity"] < 1e-4, "metallicity"] = 1e-4
        self._initial_binaries.loc[self._initial_binaries["metallicity"] > 0.03, "metallicity"] = 0.03

        self.sample_initial_galaxy()

    def sample_initial_galaxy(self):
        inds = np.searchsorted(self._subset_inds, self._initial_binaries["particle_id"].values)
        particles = self.star_particles.loc[self._subset_inds[inds]]
        x, y, z = particles["x"].values * u.kpc, particles["y"].values * u.kpc, particles["z"].values * u.kpc
        v_x = particles["v_x"].values * u.km / u.s
        v_y = particles["v_y"].values * u.km / u.s
        v_z = particles["v_z"].values * u.km / u.s
        pos = np.random.normal([x.to(u.kpc).value, y.to(u.kpc).value, z.to(u.kpc).value],
                               self.particle_size.to(u.kpc).value / np.sqrt(3),
                               size=(3, self.n_binaries_match)) * u.kpc

        self._initial_galaxy = Galaxy(self.n_binaries_match, immediately_sample=False)
        self._initial_galaxy._x = pos[0]
        self._initial_galaxy._y = pos[1]
        self._initial_galaxy._z = pos[2]
        self._initial_galaxy._tau = self._initial_binaries["tphysf"].values * u.Myr
        self._initial_galaxy._Z = self._initial_binaries["metallicity"].values
        self._initial_galaxy._which_comp = np.repeat("FIRE", len(self.initial_galaxy._tau))

        v_R = (x * v_x + y * v_y) / (x**2 + y**2)**0.5
        v_T = (x * v_y - y * v_x) / (x**2 + y**2)**0.5

        vel_units = u.km / u.s
        dispersion = dispersion_from_virial_parameter(self.virial_parameter,
                                                      self.particle_size,
                                                      particles["mass"].values * u.Msun)
        v_R, v_T, v_z = np.random.normal([v_R.to(vel_units).value,
                                          v_T.to(vel_units).value,
                                          v_z.to(vel_units).value],
                                         dispersion.to(vel_units).value / np.sqrt(3),
                                         size=(3, self.n_binaries_match)) * vel_units

        self._initial_galaxy._v_R = v_R
        self._initial_galaxy._v_T = v_T
        self._initial_galaxy._v_z = v_z
