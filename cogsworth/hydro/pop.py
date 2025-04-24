import numpy as np
import astropy.units as u
import pandas as pd

from cosmic.sample.initialbinarytable import InitialBinaryTable

from ..pop import Population
from ..sfh import StarFormationHistory

from .utils import dispersion_from_virial_parameter
import warnings


__all__ = ["HydroPopulation"]


class HydroPopulation(Population):
    def __init__(self, star_particles, cluster_radius=3 * u.pc, cluster_mass=1e4 * u.Msun,
                 virial_parameter=1.0, subset=None, sampling_params={}, snapshot_type=None, **kwargs):
        """A population of stars sampled from a hydrodynamical zoom-in snapshot

        Each star particle in the snapshot is converted to a (binary) stellar population, sampled with COSMIC,
        with initial positions and kinematics determined based on the virial parameter and particle size.

        Parameters
        ----------
        star_particles : :class:`~pandas.DataFrame`
            A table of star particles from a snapshot that contains the mass, metallicity, formation time,
            position and velocity of each particle, as returned by
            :func:`~cogsworth.hydro.rewind.rewind_to_formation`
        cluster_radius : :class:`~astropy.units.Quantity`, optional
            Size of gaussian for cluster for each star particle, by default 3 * u.pc
        cluster_mass : :class:`~astropy.units.Quantity`, optional
            Mass of cluster for each star particle for calculating velocity dispersion,
            by default 1e4 * u.Msun
        virial_parameter : `float`, optional
            Virial parameter for each cluster, for setting velocity dispersions, by default 1.0
        subset : `int` or `list` of `int`, optional
            If set, only use a subset of the star particles, by default use all of them. If an integer is
            passed, a random subset of that size is used, if a list of integers is passed, only those IDs are
            used.
        sampling_params : `dict`, optional
            As in :class:`~cogsworth.pop.Population` BUT changing the defaults to
            {"sampling_target": "total_mass", "trim_extra_samples": True} (i.e. to sample to the total mass of
            the star particle)
        snapshot_type : `str`, optional
            What sort of snapshot this to infer what citations to use, by default None (i.e. don't infer).
            Currently supported: "FIRE"
        **kwargs : various
            Parameters to pass to :class:`~cogsworth.pop.Population`
        """
        self.star_particles = star_particles
        self.cluster_radius = cluster_radius
        self.cluster_mass = cluster_mass
        self.virial_parameter = virial_parameter
        self._subset_inds = self.star_particles.index.values
        if subset is not None and isinstance(subset, int):
            self._subset_inds = np.random.choice(self._subset_inds, size=subset, replace=False)
        elif subset is not None:
            self._subset_inds = subset
        if "n_binaries" not in kwargs:
            kwargs["n_binaries"] = None

        base_sampling_params = {"sampling_target": "total_mass", "trim_extra_samples": True}
        base_sampling_params.update(sampling_params)
        super().__init__(sampling_params=base_sampling_params, **kwargs)

        if snapshot_type is not None:
            if snapshot_type.lower() == "fire":
                self.__citations__.append("FIRE")
        self.__citations__.append("pynbody")

    def __repr__(self):
        if self._orbits is None:
            return (f"<{self.__class__.__name__} - {len(self._subset_inds)} star particles - "
                    f"galactic_potential={self.galactic_potential.__class__.__name__}, "
                    f"SFH={self.sfh_model.__name__}>")
        else:
            return (f"<{self.__class__.__name__} - {len(self._subset_inds)} star particles - "
                    f"{self.n_binaries_match} evolved systems - "
                    f"galactic_potential={self.galactic_potential.__class__.__name__}, "
                    f"SFH={self.sfh_model.__name__}>")

    def __getitem__(self, ind):
        # convert any Pandas Series to numpy arrays
        ind = ind.values if isinstance(ind, pd.Series) else ind

        # ensure indexing with the right type
        ALLOWED_TYPES = (int, slice, list, np.ndarray, tuple)
        if not isinstance(ind, ALLOWED_TYPES):          # pragma: no cover
            raise ValueError((f"Can only index using one of {[at.__name__ for at in ALLOWED_TYPES]}, "
                              f"you supplied a '{type(ind).__name__}'"))

        # check validity of indices for array-like types
        if isinstance(ind, (list, tuple, np.ndarray)):
            # check every element is a boolean (if so, convert to bin_nums after asserting length sensible)
            if all(isinstance(x, (bool, np.bool_)) for x in ind):           # pragma: no cover
                assert len(ind) == len(self.bin_nums), "Boolean mask must be same length as the population"
                ind = self.bin_nums[ind]
            # otherwise ensure all elements are integers
            else:
                assert all(isinstance(x, (int, np.integer)) for x in ind), \
                    "Can only index using integers or a boolean mask"
                if len(np.unique(ind)) < len(ind):
                    warnings.warn(("You have supplied duplicate indices, this will invalidate the "
                                   "normalisation of the Population (e.g. mass_binaries will be wrong)"))

        # set up the bin_nums we are selecting
        bin_nums = ind

        # turn ints into arrays and convert slices to exact bin_nums
        if isinstance(ind, int):
            bin_nums = [ind]
        elif isinstance(ind, slice):
            bin_nums = self.bin_nums[ind]
        bin_nums = np.asarray(bin_nums)

        # check that the bin_nums are all valid
        check_nums = np.isin(bin_nums, self.bin_nums)
        if not check_nums.all():            # pragma: no cover
            raise ValueError(("The index that you supplied includes a `bin_num` that does not exist. "
                              f"The first bin_num I couldn't find was {bin_nums[~check_nums][0]}"))

        # start a new population with the same parameters
        new_pop = self.__class__(star_particles=self.star_particles, processes=self.processes,
                                 m1_cutoff=self.m1_cutoff, final_kstar1=self.final_kstar1,
                                 final_kstar2=self.final_kstar2, sfh_model=self.sfh_model,
                                 sfh_params=self.sfh_params, galactic_potential=self.galactic_potential,
                                 v_dispersion=self.v_dispersion, max_ev_time=self.max_ev_time,
                                 timestep_size=self.timestep_size, BSE_settings=self.BSE_settings,
                                 sampling_params=self.sampling_params,
                                 store_entire_orbits=self.store_entire_orbits,
                                 virial_parameter=self.virial_parameter, cluster_radius=self.cluster_radius)

        new_pop.n_binaries = len(bin_nums)
        new_pop.n_binaries_match = len(bin_nums)

        # proxy for checking whether sampling has been done
        if self._mass_binaries is not None:
            new_pop._mass_binaries = self._mass_binaries
            new_pop._mass_singles = self._mass_singles
            new_pop._n_singles_req = self._n_singles_req
            new_pop._n_bin_req = self._n_bin_req

        bin_num_to_ind = {num: i for i, num in enumerate(self.bin_nums)}
        sort_idx = np.argsort(list(bin_num_to_ind.keys()))
        idx = np.searchsorted(list(bin_num_to_ind.keys()), bin_nums, sorter=sort_idx)
        inds = np.asarray(list(bin_num_to_ind.values()))[sort_idx][idx]

        disrupted_bin_num_to_ind = {num: i for i, num in enumerate(self.bin_nums[self.disrupted])}
        sort_idx = np.argsort(list(disrupted_bin_num_to_ind.keys()))
        idx = np.searchsorted(list(disrupted_bin_num_to_ind.keys()),
                              bin_nums[np.isin(bin_nums, self.bin_nums[self.disrupted])],
                              sorter=sort_idx)
        inds_with_disruptions = np.asarray(list(disrupted_bin_num_to_ind.values()))[sort_idx][idx] + len(self)
        all_inds = np.concatenate((inds, inds_with_disruptions)).astype(int)

        if self._initial_galaxy is not None:
            new_pop._initial_galaxy = self._initial_galaxy[inds]

        # checking whether stellar evolution has been done
        if self._bpp is not None:
            # copy over subsets of data when they aren't None
            new_pop._bpp = self._bpp.loc[bin_nums]
            if self._bcm is not None:                   # pragma: no cover
                new_pop._bcm = self._bcm.loc[bin_nums]
            if self._initC is not None:
                new_pop._initC = self._initC.loc[bin_nums]
            if self._kick_info is not None:
                new_pop._kick_info = self._kick_info.loc[bin_nums]
            if self._final_bpp is not None:
                new_pop._final_bpp = self._final_bpp.loc[bin_nums]
            if self._disrupted is not None:
                new_pop._disrupted = self._disrupted[inds]
            if self._classes is not None:               # pragma: no cover
                new_pop._classes = self._classes.iloc[inds]
            if self._observables is not None:           # pragma: no cover
                new_pop._observables = self._observables.iloc[inds]

            # same thing but for arrays with appended disrupted secondaries
            if self._orbits is not None:
                new_pop._orbits = self.orbits[all_inds]
            if self._final_pos is not None:
                new_pop._final_pos = self._final_pos[all_inds]
            if self._final_vel is not None:
                new_pop._final_vel = self._final_vel[all_inds]
        return new_pop

    def get_citations(self, filename=None):
        super().get_citations(filename)
        print("\nNOTE: This population was sampled from a hydrodynamical zoom-in snapshot...but I don't know "
              "which one so I can't automate this citation, please cite the relevant paper(s) (e.g. for FIRE "
              "snapshots see here: http://flathub.flatironinstitute.org/fire)")

    def sample_initial_binaries(self):
        """Sample initial binaries from the star particles in the snapshot"""
        initial_binaries_list = [None for _ in range(len(self.star_particles))]
        self._mass_singles, self._mass_binaries, self._n_singles_req, self._n_bin_req = 0.0, 0.0, 0, 0

        for ibl_ind, i in enumerate(self._subset_inds):
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
            initial_binaries_list[ibl_ind] = samples[0]
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
                               self.cluster_radius.to(u.kpc).value / np.sqrt(3),
                               size=(3, self.n_binaries_match)) * u.kpc

        v_R = (x * v_x + y * v_y) / (x**2 + y**2)**0.5
        v_T = (x * v_y - y * v_x) / (x**2 + y**2)**0.5

        vel_units = u.km / u.s
        dispersion = dispersion_from_virial_parameter(self.virial_parameter,
                                                      self.cluster_radius, self.cluster_mass)
        v_R, v_T, v_z = np.random.normal([v_R.to(vel_units).value,
                                          v_T.to(vel_units).value,
                                          v_z.to(vel_units).value],
                                         dispersion.to(vel_units).value / np.sqrt(3),
                                         size=(3, self.n_binaries_match)) * vel_units

        self._initial_galaxy = StarFormationHistory(self.n_binaries_match, immediately_sample=False)
        self._initial_galaxy._tau = self._initial_binaries["tphysf"].values * u.Myr
        self._initial_galaxy._Z = self._initial_binaries["metallicity"].values
        self._initial_galaxy._which_comp = np.repeat("FIRE", len(self.initial_galaxy._tau))

        self._initial_galaxy._x = pos[0]
        self._initial_galaxy._y = pos[1]
        self._initial_galaxy._z = pos[2]
        self._initial_galaxy.v_R = v_R
        self._initial_galaxy.v_T = v_T
        self._initial_galaxy.v_z = v_z

    def perform_stellar_evolution(self, **kwargs):
        """Perform stellar evolution on systems sampled from the star particles
        and track their parent particles"""
        super().perform_stellar_evolution(**kwargs)
        self._initC["particle_id"] = self._initial_binaries["particle_id"]
