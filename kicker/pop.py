import time
import os
from copy import copy
from multiprocessing import Pool
import warnings
import numpy as np
import astropy.units as u
import astropy.coordinates as coords
import h5py as h5
import pandas as pd
from tqdm import tqdm
import healpy as hp
import matplotlib.pyplot as plt

from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
from gaiaunlimited.utils import get_healpix_centers

from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler.independent import Sample
from cosmic.evolve import Evolve
import gala.potential as gp
import gala.dynamics as gd

from kicker import galaxy
from kicker.kicks import integrate_orbit_with_events
from kicker.events import identify_events
from kicker.classify import determine_final_classes
from kicker.observables import get_photometry

__all__ = ["Population", "load"]


class Population():
    """Class for creating and evolving populations of binaries throughout the Milky Way

    Parameters
    ----------
    n_binaries : `int`
        How many binaries to sample for the population
    processes : `int`, optional
        How many processes to run if you want multithreading, by default 8
    m1_cutoff : `float`, optional
        The minimum allowed primary mass, by default 0
    final_kstar1 : `list`, optional
        Desired final types for primary star, by default list(range(16))
    final_kstar2 : `list`, optional
        Desired final types for secondary star, by default list(range(16))
    galaxy_model : :class:`~kicker.galaxy.Galaxy`, optional
        A Galaxy class to use for sampling the initial galaxy parameters, by default
        :class:`~kicker.galaxy.Frankel2018`
    galactic_potential : :class:`~gala.potential.potential.PotentialBase`, optional
        Galactic potential to use for evolving the orbits of binaries, by default
        :class:`~gala.potential.potential.MilkyWayPotential`
    v_dispersion : :class:`~astropy.units.Quantity` [velocity], optional
        Velocity dispersion to apply relative to the local circular velocity, by default 5*u.km/u.s
    max_ev_time : :class:`~astropy.units.Quantity` [time], optional
        Maximum evolution time for both COSMIC and Gala, by default 12.0*u.Gyr
    timestep_size : :class:`~astropy.units.Quantity` [time], optional
        Size of timesteps to use in galactic evolution, by default 1*u.Myr
    BSE_settings : `dict`, optional
        Any BSE settings to pass to COSMIC
    store_entire_orbits : `bool`, optional
        Whether to store the entire orbit for each binary, by default True. If not then only the final
        PhaseSpacePosition will be stored. This cuts down on both memory usage and disk space used if you
        save the Population.

    Attributes
    ----------
    mass_singles : `float`
        Total mass in single stars needed to generate population
    mass_binaries : `float`
        Total mass in binaries needed to generate population
    n_singles_req : `int`
        Number of single stars needed to generate population
    n_bin_req : `int`
        Number of binaries needed to generate population
    bpp : :class:`~pandas.DataFrame`
        Evolutionary history of each binary
    bcm : :class:`~pandas.DataFrame`
        Final state of each binary
    initC : :class:`~pandas.DataFrame`
        Initial conditions for each binary
    kick_info : :class:`~pandas.DataFrame`
        Information about the kicks that occur for each binary
    orbits : `list` of :class:`~gala.dynamics.Orbit`
        The orbits of each binary within the galaxy from its birth until :attr:`max_ev_time` with timesteps of
        :attr:`timestep_size`. Note that disrupted binaries will have two entries (for both stars).
    classes : `list`
        The classes associated with each produced binary (see :meth:`~kicker.classify.list_classes` for a
        list of available classes and their meanings)
    final_coords : `tuple` of :class:`~astropy.coordinates.SkyCoord`
        A SkyCoord object of the final positions of each binary in the galactocentric frame.
        For bound binaries only the first SkyCoord is populated, for disrupted binaries each SkyCoord
        corresponds to the individual components. Any missing orbits (where orbit=None or there is no
        secondary component) will be set to `np.inf` for ease of masking.
    final_bpp : :class:`~pandas.DataFrame`
        The final state of each binary (taken from the final entry in :attr:`bpp`)
    disrupted : :class:`~numpy.ndarray` of `bool`
        A mask on the binaries of whether they were disrupted
    observables : :class:`~pandas.DataFrame`
        Observables associated with the final binaries. See `get_observables` for more details on the columns
    bin_nums : :class:`~np.ndarray`
        An array containing the unique COSMIC `bin_nums` of each binary in the population - these can be
        used an indices for the population
    """
    def __init__(self, n_binaries, processes=8, m1_cutoff=0, final_kstar1=list(range(16)),
                 final_kstar2=list(range(16)), galaxy_model=galaxy.Frankel2018,
                 galactic_potential=gp.MilkyWayPotential(), v_dispersion=5 * u.km / u.s,
                 max_ev_time=12.0*u.Gyr, timestep_size=1 * u.Myr, BSE_settings={}, store_entire_orbits=True):

        # require a sensible number of binaries
        if n_binaries <= 0:
            raise ValueError("You need to input a *nonnegative* number of binaries")

        self.n_binaries = n_binaries
        self.n_binaries_match = n_binaries
        self.processes = processes
        self.m1_cutoff = m1_cutoff
        self.final_kstar1 = final_kstar1
        self.final_kstar2 = final_kstar2
        self.galaxy_model = galaxy_model
        self.galactic_potential = galactic_potential
        self.v_dispersion = v_dispersion
        self.max_ev_time = max_ev_time
        self.timestep_size = timestep_size
        self.pool = None
        self.store_entire_orbits = store_entire_orbits

        self._initial_binaries = None
        self._mass_singles = None
        self._mass_binaries = None
        self._n_singles_req = None
        self._n_bin_req = None
        self._bpp = None
        self._bcm = None
        self._initC = None
        self._kick_info = None
        self._orbits = None
        self._classes = None
        self._final_coords = None
        self._final_bpp = None
        self._disrupted = None
        self._escaped = None
        self._observables = None
        self._bin_nums = None

        self.BSE_settings = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0,
                             'pts1': 0.001, 'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5,
                             'ck': 1000, 'bwind': 0.0, 'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1,
                             'acc2': 1.5, 'grflag': 1, 'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0,
                             'ifflag': 0, 'bconst': 3000, 'sigma': 265.0, 'gamma': -2.0, 'pisn': 45.0,
                             'natal_kick_array': [[-100.0, -100.0, -100.0, -100.0, 0.0],
                                                  [-100.0, -100.0, -100.0, -100.0, 0.0]], 'bhsigmafrac': 1.0,
                             'polar_kick_angle': 90, 'qcrit_array': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             'cekickflag': 2, 'cehestarflag': 0, 'cemergeflag': 0, 'ecsn': 2.25,
                             'ecsn_mlow': 1.6, 'aic': 1, 'ussn': 0, 'sigmadiv': -20.0, 'qcflag': 5,
                             'eddlimflag': 0, 'fprimc_array': [2.0/21.0, 2.0/21.0, 2.0/21.0, 2.0/21.0,
                                                               2.0/21.0, 2.0/21.0, 2.0/21.0, 2.0/21.0,
                                                               2.0/21.0, 2.0/21.0, 2.0/21.0, 2.0/21.0,
                                                               2.0/21.0, 2.0/21.0, 2.0/21.0, 2.0/21.0],
                             'bhspinflag': 0, 'bhspinmag': 0.0, 'rejuv_fac': 1.0, 'rejuvflag': 0, 'htpmb': 1,
                             'ST_cr': 1, 'ST_tide': 1, 'bdecayfac': 1, 'rembar_massloss': 0.5, 'kickflag': 0,
                             'zsun': 0.014, 'bhms_coll_flag': 0, 'don_lim': -1, 'acc_lim': -1, 'binfrac': 0.5}
        self.BSE_settings.update(BSE_settings)

    def __repr__(self):
        if self._orbits is None:
            return (f"<{self.__class__.__name__} - {self.n_binaries} systems - "
                    f"galactic_potential={self.galactic_potential.__class__.__name__}, "
                    f"SFH={self.galaxy_model.__name__}>")
        else:
            return (f"<{self.__class__.__name__} - {self.n_binaries_match} evolved systems - "
                    f"galactic_potential={self.galactic_potential.__class__.__name__}, "
                    f"galaxy_model={self.galaxy_model.__name__}>")

    def __len__(self):
        return self.n_binaries_match

    def __getitem__(self, ind):
        # ensure indexing with the right type
        if not isinstance(ind, (int, slice, list, np.ndarray, tuple)):
            raise ValueError(("Can only index using an `int`, `list`, `ndarray` or `slice`, you supplied a "
                              f"`{type(ind).__name__}`"))

        # create a list of bin nums from ind
        bin_nums = ind
        if isinstance(ind, int):
            bin_nums = [ind]
        if isinstance(ind, slice):
            bin_nums = list(range(ind.stop)[ind])
        bin_nums = np.asarray(bin_nums)

        # check that the bin_nums are all valid
        possible_bin_nums = self.final_bpp["bin_num"]
        check_nums = np.isin(bin_nums, possible_bin_nums)
        if not check_nums.all():
            raise ValueError(("The index that you supplied includes a `bin_num` that does not exist. "
                              f"The first bin_num I couldn't find was {bin_nums[~check_nums][0]}"))

        # start a new population with the same parameters
        new_pop = self.__class__(n_binaries=len(bin_nums), processes=self.processes,
                                 m1_cutoff=self.m1_cutoff, final_kstar1=self.final_kstar1,
                                 final_kstar2=self.final_kstar2, galaxy_model=self.galaxy_model,
                                 galactic_potential=self.galactic_potential, v_dispersion=self.v_dispersion,
                                 max_ev_time=self.max_ev_time, timestep_size=self.timestep_size,
                                 BSE_settings=self.BSE_settings, store_entire_orbits=self.store_entire_orbits)

        # proxy for checking whether sampling has been done
        if self._mass_binaries is not None:
            new_pop._mass_binaries = self._mass_binaries
            new_pop._mass_singles = self._mass_singles
            new_pop._n_singles_req = self._n_singles_req
            new_pop._n_bin_req = self._n_bin_req

        if self._initial_galaxy is not None:
            # since we are indexing on bin nums, need to convert that to actual indices
            ind_range = np.arange(len(possible_bin_nums))
            new_pop._initial_galaxy = self._initial_galaxy[ind_range[possible_bin_nums.isin(bin_nums)]]

        # checking whether stellar evolution has been done
        if self._bpp is not None:
            # copy over subsets of the stellar evolution tables when they aren't None
            new_pop._bpp = self._bpp[self._bpp["bin_num"].isin(bin_nums)]
            if self._bcm is not None:
                new_pop._bcm = self._bcm[self._bcm["bin_num"].isin(bin_nums)]
            if self._initC is not None:
                new_pop._initC = self._initC[self._initC["bin_num"].isin(bin_nums)]
            if self._kick_info is not None:
                new_pop._kick_info = self._kick_info[self._kick_info["bin_num"].isin(bin_nums)]

            # same sort of thing for later parameters
            mask = self.final_bpp["bin_num"].isin(bin_nums).values
            new_pop._final_bpp = self.final_bpp[mask]

            if self._orbits is not None:
                new_pop._orbits = self.orbits[mask]
            if self._disrupted is not None:
                new_pop._disrupted = self._disrupted[mask]
            if self._classes is not None:
                new_pop._classes = self._classes[mask]
            if self._final_coords is not None:
                new_pop._final_coords = [self._final_coords[i][mask] for i in range(2)]
            if self._disrupted is not None:
                new_pop._disrupted = self._disrupted[mask]
            if self._observables is not None:
                new_pop._observables = self._observables[mask]

        return new_pop

    @property
    def bin_nums(self):
        if self._bin_nums is None:
            if self._bpp is not None:
                self._bin_nums = self.final_bpp["bin_num"].unique()
            else:
                raise ValueError("You need to evolve binaries to get a list of `bin_nums`!")
        return self._bin_nums

    @property
    def initial_galaxy(self):
        if self._initial_galaxy is None:
            self.sample_initial_binaries()
        return self._initial_galaxy

    @property
    def mass_singles(self):
        if self._mass_singles is None:
            self.sample_initial_binaries()
        return self._mass_singles

    @property
    def mass_binaries(self):
        if self._mass_binaries is None:
            self.sample_initial_binaries()
        return self._mass_binaries

    @property
    def n_singles_req(self):
        if self._n_singles_req is None:
            self.sample_initial_binaries()
        return self._n_singles_req

    @property
    def n_bin_req(self):
        if self._n_bin_req is None:
            self.sample_initial_binaries()
        return self._n_bin_req

    @property
    def bpp(self):
        if self._bpp is None:
            self.perform_stellar_evolution()
        return self._bpp

    @property
    def bcm(self):
        if self._bcm is None:
            self.perform_stellar_evolution()
        return self._bcm

    @property
    def initC(self):
        if self._initC is None:
            self.perform_stellar_evolution()
        return self._initC

    @property
    def kick_info(self):
        if self._kick_info is None:
            self.perform_stellar_evolution()
        return self._kick_info

    @property
    def orbits(self):
        if self._orbits is None:
            self.perform_galactic_evolution()
        return self._orbits

    @property
    def classes(self):
        if self._classes is None:
            self._classes = determine_final_classes(population=self)
        return self._classes

    @property
    def final_coords(self):
        if self._final_coords is None:
            self._final_coords = self.get_final_coords()
        return self._final_coords

    @property
    def final_bpp(self):
        if self._final_bpp is None:
            self._final_bpp = self.bpp.drop_duplicates(subset="bin_num", keep="last")
            self._final_bpp.insert(len(self._final_bpp.columns), "metallicity",
                                   self.initC["metallicity"].values)
        return self._final_bpp

    @property
    def disrupted(self):
        if self._disrupted is None:
            # check for disruptions in THREE different ways because COSMIC isn't always consistent (:
            self._disrupted = (self.final_bpp["bin_num"].isin(self.kick_info[self.kick_info["disrupted"] == 1.0]["bin_num"].unique())
                               & (self.final_bpp["sep"] < 0.0)
                               & self.final_bpp["bin_num"].isin(self.bpp[self.bpp["evol_type"] == 11.0]["bin_num"])).values
        return self._disrupted

    @property
    def escaped(self):
        if self._escaped is None:
            self._escaped = [np.repeat(False, len(self)), np.repeat(False, len(self))]

            # do it for all systems and then also for the secondaries of disrupted systems
            for ind, mask in enumerate([np.repeat(True, len(self)), self.disrupted]):
                # get the current velocity
                v_curr = np.sum(self.final_coords[ind][mask].velocity.d_xyz**2, axis=0)**(0.5)

                # get the escape velocity at the current position based on galactic potential
                pos = np.asarray([self.final_coords[ind][mask].galactocentric.x.to(u.kpc),
                                  self.final_coords[ind][mask].galactocentric.y.to(u.kpc),
                                  self.final_coords[ind][mask].galactocentric.z.to(u.kpc)]) * u.kpc

                # 0.5 * m * v_esc**2 = m * (-Phi)
                v_esc = np.sqrt(-2 * self.galactic_potential(pos))
                self._escaped[ind][mask] = v_curr >= v_esc
        return self._escaped

    @property
    def observables(self):
        if self._observables is None:
            self._observables = self.get_observables()
        return self._observables

    def create_population(self, with_timing=True):
        """Create an entirely evolved population of binaries.

        This will sample the initial binaries and initial galaxy and then perform both the :py:mod:`cosmic`
        and :py:mod:`gala` evolution.

        Parameters
        ----------
        with_timing : `bool`, optional
            Whether to print messages about the timing, by default True
        """
        if with_timing:
            start = time.time()
            print(f"Run for {self.n_binaries} binaries")

        self.sample_initial_binaries()
        if with_timing:
            print(f"Ended up with {self.n_binaries_match} binaries with m1 > {self.m1_cutoff} solar masses")
            print(f"[{time.time() - start:1.0e}s] Sample initial binaries")
            lap = time.time()

        self.pool = Pool(self.processes) if self.processes > 1 else None
        self.perform_stellar_evolution()
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Evolve binaries (run COSMIC)")
            lap = time.time()

        self.perform_galactic_evolution(progress_bar=with_timing)
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")

        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        if with_timing:
            print(f"Overall: {time.time() - start:1.1f}s")

    def sample_initial_galaxy(self):
        """Sample the initial galactic times, positions and velocities"""
        # initialise the initial galaxy class with correct number of binaries
        self._initial_galaxy = self.galaxy_model(size=self.n_binaries_match)

        # work out the initial velocities of each binary
        vel_units = u.km / u.s

        # calculate the Galactic circular velocity at the initial positions
        v_circ = self.galactic_potential.circular_velocity(q=[self._initial_galaxy.positions.x,
                                                              self._initial_galaxy.positions.y,
                                                              self._initial_galaxy.positions.z]).to(vel_units)

        # add some velocity dispersion
        v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
                                         self.v_dispersion.to(vel_units) / np.sqrt(3),
                                         size=(3, self.n_binaries_match))
        v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units
        self._initial_galaxy._v_R = v_R
        self._initial_galaxy._v_T = v_T
        self._initial_galaxy._v_z = v_z

    def sample_initial_binaries(self):
        """Sample the initial binary parameters for the population"""
        self._bin_nums = None

        # overwrite the binary fraction is the user just wants single stars
        binfrac = self.BSE_settings["binfrac"] if self.BSE_settings["binfrac"] != 0.0 else 1.0
        self._initial_binaries, self._mass_singles, self._mass_binaries, self._n_singles_req,\
            self._n_bin_req = InitialBinaryTable.sampler('independent',
                                                         self.final_kstar1, self.final_kstar2,
                                                         binfrac_model=binfrac,
                                                         primary_model='kroupa01', ecc_model='sana12',
                                                         porb_model='sana12', qmin=-1,
                                                         SF_start=self.max_ev_time.to(u.Myr).value,
                                                         SF_duration=0.0, met=0.02, size=self.n_binaries)

        # check if the user just wants single stars instead of binaries
        if self.BSE_settings["binfrac"] == 0.0:
            mass_1, mass_tot = Sample().sample_primary(primary_model="kroupa01",
                                                       size=len(self._initial_binaries))
            self._initial_binaries["mass_1"] = mass_1
            self._initial_binaries["mass0_1"] = mass_1
            self._initial_binaries["kstar_1"] = np.where(mass_1 > 0.7, 1, 0)
            self._initial_binaries["kstar_2"] = np.zeros(len(self._initial_binaries))
            self._initial_binaries["mass_2"] = np.zeros(len(self._initial_binaries))
            self._initial_binaries["mass0_2"] = np.zeros(len(self._initial_binaries))
            self._initial_binaries["porb"] = np.zeros(len(self._initial_binaries))
            self._initial_binaries["sep"] = np.zeros(len(self._initial_binaries))
            self._initial_binaries["ecc"] = np.zeros(len(self._initial_binaries))

            self._mass_singles = mass_tot
            self._mass_binaries = 0.0
            self._n_singles_req = self.n_binaries
            self._n_bin_req = 0

        # apply the mass cutoff
        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

        # count how many binaries actually match the criteria (may be larger than `n_binaries` due to sampler)
        self.n_binaries_match = len(self._initial_binaries)

        # check that any binaries remain
        if self.n_binaries_match == 0:
            raise ValueError(("Your choice of `m1_cutoff` resulted in all samples being thrown out. Consider"
                              " a larger sample size or a less stringent mass cut"))

        self.sample_initial_galaxy()

        # update the metallicity and birth times of the binaries to match the galaxy
        self._initial_binaries["metallicity"] = self._initial_galaxy.Z
        self._initial_binaries["tphysf"] = self._initial_galaxy.tau.to(u.Myr).value

        # ensure metallicities remain in a range valid for COSMIC - original value still in initial_galaxy.Z
        self._initial_binaries.loc[self._initial_binaries["metallicity"] < 1e-4, "metallicity"] = 1e-4
        self._initial_binaries.loc[self._initial_binaries["metallicity"] > 0.03, "metallicity"] = 0.03

    def perform_stellar_evolution(self):
        """Perform the (binary) stellar evolution of the sampled binaries"""
        # delete any cached variables
        self._final_bpp = None
        self._observables = None
        self._bin_nums = None
        self._disrupted = None
        self._escaped = None

        # if no initial binaries have been sampled then we need to create some
        if self._initial_binaries is None and self._initC is None:
            print("Warning: Initial binaries not yet sampled, performing sampling now.")
            self.sample_initial_binaries()

        # if initC exists then we can use that instead of initial binaries
        elif self._initial_binaries is None:
            self._initial_binaries = self._initC

        no_pool_existed = self.pool is None and self.processes > 1
        if no_pool_existed:
            self.pool = Pool(self.processes)

        # catch any warnings about overwrites
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*initial binary table is being overwritten.*")
            warnings.filterwarnings("ignore", message=".*to a different value than assumed in the mlwind.*")

            # perform the evolution!
            self._bpp, self._bcm, self._initC,\
                self._kick_info = Evolve.evolve(initialbinarytable=self._initial_binaries,
                                                BSEDict=self.BSE_settings, pool=self.pool)

        if no_pool_existed:
            self.pool.close()
            self.pool.join()
            self.pool = None

        # check if there are any NaNs in the final bpp table rows or the kick_info
        nans = np.isnan(self.final_bpp["sep"])
        kick_info_nans = np.isnan(self._kick_info["delta_vsysx_1"])

        # if we detect NaNs
        if nans.any() or kick_info_nans.any():      # pragma: no cover
            # make sure the user knows bad things have happened
            print("WARNING! PANIC! THE SKY THE FALLING!")
            print("------------------------------------")
            print("(NaNs detected)")

            # store the bad things for later
            nan_bin_nums = np.unique(np.concatenate((self.final_bpp[nans]["bin_num"].values,
                                                     self._kick_info[kick_info_nans]["bin_num"].values)))
            self._bpp[self._bpp["bin_num"].isin(nan_bin_nums)].to_hdf("nans.h5", key="bpp")
            self._initC[self._initC["bin_num"].isin(nan_bin_nums)].to_hdf("nans.h5", key="initC")
            self._kick_info[self._kick_info["bin_num"].isin(nan_bin_nums)].to_hdf("nans.h5", key="kick_info")

            # update the population to delete any bad binaries
            n_nan = len(nan_bin_nums)
            self.n_binaries_match -= n_nan
            self._bpp = self._bpp[~self._bpp["bin_num"].isin(nan_bin_nums)]
            self._bcm = self._bcm[~self._bcm["bin_num"].isin(nan_bin_nums)]
            self._kick_info = self._kick_info[~self._kick_info["bin_num"].isin(nan_bin_nums)]
            self._initC = self._initC[~self._initC["bin_num"].isin(nan_bin_nums)]

            not_nan = ~self.final_bpp["bin_num"].isin(nan_bin_nums)
            self._initial_galaxy._tau = self._initial_galaxy._tau[not_nan]
            self._initial_galaxy._Z = self._initial_galaxy._Z[not_nan]
            self._initial_galaxy._positions = self._initial_galaxy._positions[not_nan]
            self._initial_galaxy._v_R = self._initial_galaxy._v_R[not_nan]
            self._initial_galaxy._v_T = self._initial_galaxy._v_T[not_nan]
            self._initial_galaxy._v_z = self._initial_galaxy._v_z[not_nan]
            self._initial_galaxy._which_comp = self._initial_galaxy._which_comp[not_nan]
            self._initial_galaxy._size -= n_nan

            # reset final bpp
            self._final_bpp = None

            print(f"WARNING: {n_nan} bad binaries removed from tables - but normalisation may be off")
            print("I've added the offending binaries to the `nan.h5` file, do with them what you will")

    def perform_galactic_evolution(self, quiet=False, progress_bar=True):
        """Use :py:mod:`gala` to perform the orbital integration for each evolved binary

        Parameters
        ----------
        quiet : `bool`, optional
            Whether to silence any warnings about failing orbits, by default False
        """
        # delete any cached variables
        self._final_coords = None
        self._observables = None

        # turn the drawn coordinates into an astropy representation
        rep = self.initial_galaxy.positions.represent_as("cylindrical")

        # create differentials based on the velocities (dimensionless angles allows radians conversion)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dif = coords.CylindricalDifferential(self.initial_galaxy._v_R,
                                                 (self.initial_galaxy._v_T
                                                  / rep.rho).to(u.rad / u.Gyr),
                                                 self.initial_galaxy._v_z)

        # combine the representation and differentials into a Gala PhaseSpacePosition
        w0s = gd.PhaseSpacePosition(rep.with_differentials(dif))

        # identify the pertinent events in the evolution
        events = identify_events(full_bpp=self.bpp, full_kick_info=self.kick_info)

        # if we want to use multiprocessing
        if self.pool is not None or self.processes > 1:
            # track whether a pool already existed
            pool_existed = self.pool is not None

            # if not, create one
            if not pool_existed:
                self.pool = Pool(self.processes)

            # setup arguments and evolve the orbits from birth until present day
            args = [(w0s[i], self.max_ev_time - self.initial_galaxy.tau[i], self.max_ev_time,
                     copy(self.timestep_size), self.galactic_potential,
                     events[i], self.store_entire_orbits, quiet) for i in range(self.n_binaries_match)]

            if progress_bar:
                orbits = self.pool.starmap(integrate_orbit_with_events,
                                           tqdm(args, total=self.n_binaries_match))
            else:
                orbits = self.pool.starmap(integrate_orbit_with_events, args)

            # if a pool didn't exist before then close the one just created
            if not pool_existed:
                self.pool.close()
                self.pool.join()
                self.pool = None
        else:
            # otherwise just use a for loop to evolve the orbits from birth until present day
            orbits = []
            for i in range(self.n_binaries_match):
                orbits.append(integrate_orbit_with_events(w0=w0s[i], potential=self.galactic_potential,
                                                          t1=self.max_ev_time - self.initial_galaxy.tau[i],
                                                          t2=self.max_ev_time, dt=copy(self.timestep_size),
                                                          events=events[i], quiet=quiet,
                                                          store_all=self.store_entire_orbits))

        self._orbits = np.array(orbits, dtype="object")

    def get_final_coords(self):
        """Get the final coordinates of each binary (or each component in disrupted binaries)

        Returns
        -------
        final_coords : `tuple` of :class:`~astropy.coordinates.SkyCoord`
            A SkyCoord object of the final positions of each binary in the galactocentric frame.
            For bound binaries only the first SkyCoord is populated, for disrupted binaries each SkyCoord
            corresponds to the individual components. Any missing orbits (where orbit=None or there is no
            secondary component) will be set to `np.inf` for ease of masking.
        """
        # pool all of the orbits into a single numpy array
        final_kinematics = np.ones((len(self.orbits), 2, 6)) * np.inf
        for i, orbit in enumerate(self.orbits):
            # check if the orbit is missing
            if orbit is None:
                print("Warning: Detected `None` orbit, entering coordinates as `np.inf`")

            # check if it has been disrupted
            elif isinstance(orbit, list):
                final_kinematics[i, 0, :3] = orbit[0][-1].pos.xyz.to(u.kpc).value
                final_kinematics[i, 1, :3] = orbit[1][-1].pos.xyz.to(u.kpc).value
                final_kinematics[i, 0, 3:] = orbit[0][-1].vel.d_xyz.to(u.km / u.s)
                final_kinematics[i, 1, 3:] = orbit[1][-1].vel.d_xyz.to(u.km / u.s)

            # otherwise just save the system in the primary
            else:
                final_kinematics[i, 0, :3] = orbit[-1].pos.xyz.to(u.kpc).value
                final_kinematics[i, 0, 3:] = orbit[-1].vel.d_xyz.to(u.km / u.s)

        # turn the array into two SkyCoords
        final_coords = [coords.SkyCoord(x=final_kinematics[:, i, 0] * u.kpc,
                                        y=final_kinematics[:, i, 1] * u.kpc,
                                        z=final_kinematics[:, i, 2] * u.kpc,
                                        v_x=final_kinematics[:, i, 3] * u.km / u.s,
                                        v_y=final_kinematics[:, i, 4] * u.km / u.s,
                                        v_z=final_kinematics[:, i, 5] * u.km / u.s,
                                        frame="galactocentric") for i in [0, 1]]
        return final_coords[0], final_coords[1]

    def get_observables(self, filters=['J', 'H', 'K', 'G', 'BP', 'RP'], ignore_extinction=False):
        """Get observables associated with the binaries at present day.

        These include: extinction due to dust, absolute and apparent bolometric magnitudes for each star,
        apparent magnitudes in each filter and observed temperature and surface gravity for each binary.

        For bound binaries and stellar mergers, only the column `{filter}_app_1` is relevant. For
        disrupted binaries, `{filter}_app_1` is for the primary star and `{filter}_app_2` is for
        the secondary star.

        Parameters
        ----------
        filters : `list`, optional
            Which filters to compute observables for, by default ['J', 'H', 'K', 'G', 'BP', 'RP']
        ignore_extinction : `bool`
            Whether to ignore extinction
        """
        return get_photometry(self.final_bpp, self.final_coords, filters, ignore_extinction=ignore_extinction)

    def get_gaia_observed_bin_nums(self):
        """Get a list of ``bin_nums`` of systems that are bright enough to be observed by Gaia.

        This is calculated based on the Gaia selection function provided by :mod:`gaiaunlimited`. This
        function returns a **random sample** of systems where systems are included in the observed subset
        with a probability given by Gaia's completeness at their location.

        E.g. if Gaia's completeness is 0 for a source of a given magnitude and location then it will never be
        included. Similarly, if the completeness is 1 then it will always be included. However, if the
        completeness is 0.5 then it will only be included in the list of ``bin_nums`` half of the time.

        Returns
        -------
        primary_observed : :class:`~numpy.ndarray`
            A list of binary numbers (that can be used in tables like :attr:`final_bpp`) for which the
            bound binary/disrupted primary would be observed
        secondary_observed : :class:`~numpy.ndarray`
            A list of binary numbers (that can be used in tables like :attr:`final_bpp`) for which the
            disrupted secondary would be observed
        """
        # get coordinates of the centres of the healpix pixels in a nside=2**7
        coords_of_centers = get_healpix_centers(7)

        # get the Gaia selection function for this healpix order
        dr3sf = DR3SelectionFunctionTCG()

        # work out the index of each pixel for every binary
        pix_inds = self.get_healpix_inds(nside=128)

        # loop over first (all bound binaries & primaries from disrupted binaries)
        # and then (secondaries from disrupted binaries)
        observed = []
        all_bin_nums = self.final_bpp["bin_num"].values
        for pix, g_mags, bin_nums in zip(pix_inds, [self.observables["G_app_1"].values,
                                                    self.observables["G_app_2"][self.disrupted].values],
                                         [all_bin_nums, all_bin_nums[self.disrupted]]):
            # get the coordinates of the corresponding pixels
            comp_coords = coords_of_centers[pix]

            # ensure any NaNs in the magnitudes are just set to super faint
            g_mags = np.nan_to_num(g_mags, nan=1000)
            g_mags = g_mags.value if hasattr(g_mags, 'unit') else g_mags

            # by default, assume Gaia has 0 completeness for each source
            completeness = np.zeros(len(g_mags))

            # only bother getting completeness for things brighter than G=22 (everything fainter is 0)
            bright_enough = g_mags < 22.0
            observed_bin_nums = np.array([])
            if bright_enough.any():
                completeness[bright_enough] = dr3sf.query(comp_coords[bright_enough], g_mags[bright_enough])

                # draw a random sample from the systems based on Gaia's completeness at each coordinate
                observed_bin_nums = bin_nums[np.random.uniform(size=len(completeness)) < completeness]

            observed.append(observed_bin_nums)

        primary_observed, secondary_observed = observed
        return primary_observed, secondary_observed

    def get_healpix_inds(self, nside=128):
        """Get the indices of the healpix pixels that each binary is in

        Parameters
        ----------
        nside : `int`, optional
            Healpix nside parameter, by default 128

        Returns
        -------
        pix : :class:`~numpy.ndarray`
            The indices for the bound binaries/primaries of disrupted binaries
            (corresponds to ``self.final_coords[0]``)
        disrupted_pix : :class:`~numpy.ndarray`
            The indices for the secondaries of disrupted binaries
            (corresponds to ``self.final_coords[1][self.disrupted]``)
        """
        # get the coordinates in right format
        colatitudes = [np.pi/2 - self.final_coords[i].icrs.dec.to(u.rad).value for i in [0, 1]]
        longitudes = [self.final_coords[i].icrs.ra.to(u.rad).value for i in [0, 1]]

        # find the pixels for each bound binary/primary and for each disrupted secondary
        pix = hp.ang2pix(nside, nest=True, theta=colatitudes[0], phi=longitudes[0])
        disrupted_pix = hp.ang2pix(nside, nest=True,
                                   theta=colatitudes[1][self.disrupted], phi=longitudes[1][self.disrupted])
        return pix, disrupted_pix

    def plot_map(self, nside=128, coord="C",
                 cmap="magma", norm="log", unit=None, show=True, **mollview_kwargs):
        r"""Plot a healpix map of the final positions of all binaries in population

        Parameters
        ----------
        nside : `int`, optional
            Healpix nside parameter, by default 128
        coord : `str`, optional
            Which coordinates to plot. One of ["C[elestial]", "G[alactic", "E[quatorial]"], by default "C"
        cmap : `str`, optional
            A `matplotlib colormap <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
            to use for the plot, by default "magma"
        norm : `str`, optional
            How to normalise the total number of binaries (anything linear or log), by default "log"
        unit : `str`, optional
            A label to use for the unit of the colour bar, by default ":math:`\log_{10}(N_{\rm binaries})`"
            if ``norm==log`` and ":math:`N_{\rm binaries}`" if ``norm==linear``
        show : `bool`, optional
            Whether to immediately show the plot, by default True
        **mollview_kwargs
            Any additional arguments that you want to pass to healpy's
            `mollview <https://healpy.readthedocs.io/en/latest/generated/healpy.visufunc.mollview.html>`_
        """
        pix, disrupted_pix = self.get_healpix_inds(nside=nside)

        # initialise an empty map
        m = np.zeros(hp.nside2npix(nside))

        # count the unique pixel values and how many sources are in each
        inds, counts = np.unique(np.concatenate([pix, disrupted_pix]), return_counts=True)

        # apply a log if desired
        if norm == "log":
            counts = np.log10(counts)

        # fill in the map
        m[inds] = counts

        # if the user wants a different coordinate then use the list format to convert
        if coord.lower() not in ["c", "celestial"]:
            coord = ["C", coord]
        # otherwise just pass plain old celestial
        else:
            coord = "C"

        # if a unit isn't provided then we can write one down automatically
        if unit is None:
            unit = r"$\log(N_{\rm binaries})$" if norm == "log" else r"$N_{\rm binaries}$"

        # create a mollview plot
        hp.mollview(m, nest=True, cmap=cmap, coord=coord, unit=unit, **mollview_kwargs)

        # show it if the user wants
        if show:
            plt.show()

    def save(self, file_name, overwrite=False):
        """Save a Population to disk

        This will produce 4 files:
            - An HDF5 file containing most of the data
            - A .npy file containing the orbits
            - A .txt file detailing the Galactic potential used
            - A .txt file detailing the initial galaxy model used

        Parameters
        ----------
        file_name : `str`
            A file name to use. Either no file extension or ".h5".
        overwrite : `bool`, optional
            Whether to overwrite any existing files, by default False

        Raises
        ------
        FileExistsError
            If `overwrite=False` and files already exist
        """
        if file_name[-3:] != ".h5":
            file_name += ".h5"
        if os.path.isfile(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise FileExistsError((f"{file_name} already exists. Set `overwrite=True` to overwrite "
                                       "the file."))
        self.bpp.to_hdf(file_name, key="bpp")
        self.bcm.to_hdf(file_name, key="bcm")
        self.initC.to_hdf(file_name, key="initC")
        self.kick_info.to_hdf(file_name, key="kick_info")

        self.galactic_potential.save(file_name.replace('.h5', '-potential.txt'))
        self.initial_galaxy.save(file_name, key="initial_galaxy")
        np.save(file_name.replace(".h5", "-orbits.npy"), np.array(self.orbits, dtype="object"))

        with h5.File(file_name, "a") as file:
            numeric_params = np.array([self.n_binaries, self.n_binaries_match, self.processes, self.m1_cutoff,
                                       self.v_dispersion.to(u.km / u.s).value,
                                       self.max_ev_time.to(u.Gyr).value, self.timestep_size.to(u.Myr).value,
                                       self.mass_singles, self.mass_binaries, self.n_singles_req,
                                       self.n_bin_req])
            num_par = file.create_dataset("numeric_params", data=numeric_params)
            num_par.attrs["store_entire_orbits"] = self.store_entire_orbits

            k_stars = np.array([self.final_kstar1, self.final_kstar2])
            file.create_dataset("k_stars", data=k_stars)

            # save BSE settings
            d = file.create_dataset("BSE_settings", data=[])
            for key in self.BSE_settings:
                d.attrs[key] = self.BSE_settings[key]


def load(file_name):
    """Load a Population from a series of files

    Parameters
    ----------
    file_name : `str`
        Base name of the files to use. Should either have no file extension or ".h5"

    Returns
    -------
    pop : `Population`
        The loaded Population
    """
    if file_name[-3:] != ".h5":
        file_name += ".h5"

    BSE_settings = {}
    with h5.File(file_name, "r") as file:
        numeric_params = file["numeric_params"][...]
        k_stars = file["k_stars"][...]

        store_entire_orbits = file["numeric_params"].attrs["store_entire_orbits"]

        # load in BSE settings
        for key in file["BSE_settings"].attrs:
            BSE_settings[key] = file["BSE_settings"].attrs[key]

    initial_galaxy = galaxy.load(file_name, key="initial_galaxy")
    galactic_potential = gp.potential.load(file_name.replace('.h5', '-potential.txt'))

    p = Population(n_binaries=int(numeric_params[0]), processes=int(numeric_params[2]),
                   m1_cutoff=numeric_params[3], final_kstar1=k_stars[0], final_kstar2=k_stars[1],
                   galaxy_model=initial_galaxy.__class__, galactic_potential=galactic_potential,
                   v_dispersion=numeric_params[4] * u.km / u.s, max_ev_time=numeric_params[5] * u.Gyr,
                   timestep_size=numeric_params[6] * u.Myr, BSE_settings=BSE_settings,
                   store_entire_orbits=store_entire_orbits)

    p.n_binaries_match = int(numeric_params[1])
    p._mass_singles = numeric_params[7]
    p._mass_binaries = numeric_params[8]
    p._n_singles_req = numeric_params[9]
    p._n_bin_req = numeric_params[10]

    p._initial_galaxy = initial_galaxy

    p._bpp = pd.read_hdf(file_name, key="bpp")
    p._bcm = pd.read_hdf(file_name, key="bcm")
    p._initC = pd.read_hdf(file_name, key="initC")
    p._kick_info = pd.read_hdf(file_name, key="kick_info")

    p._orbits = np.load(file_name.replace(".h5", "-orbits.npy"), allow_pickle=True)

    return p


class EvolvedPopulation(Population):
    def __init__(self, n_binaries, mass_singles=None, mass_binaries=None, n_singles_req=None, n_bin_req=None,
                 bpp=None, bcm=None, initC=None, kick_info=None, **pop_kwargs):
        super().__init__(n_binaries=n_binaries, **pop_kwargs)

        self._mass_singles = mass_singles
        self._mass_binaries = mass_binaries
        self._n_singles_req = n_singles_req
        self._n_bin_req = n_bin_req
        self._bpp = bpp
        self._bcm = bcm
        self._initC = initC
        self._kick_info = kick_info

    def sample_initial_binaries(self):
        raise NotImplementedError("`EvolvedPopulation` cannot sample new binaries, use `Population` instead")

    def perform_stellar_evolution(self):
        raise NotImplementedError("`EvolvedPopulation` cannot do stellar evolution, use `Population` instead")

    def create_population(self, with_timing=True):
        """Create an entirely evolved population of binaries with sampling or stellar evolution

        This will sample the initial galaxy and then perform the :py:mod:`gala` evolution.

        Parameters
        ----------
        with_timing : `bool`, optional
            Whether to print messages about the timing, by default True
        """
        if with_timing:
            start = time.time()
            print(f"Run for {self.n_binaries} binaries")

        self.sample_initial_galaxy()
        if with_timing:
            print(f"[{time.time() - start:1.0e}s] Sample initial galaxy")
            lap = time.time()

        self.pool = Pool(self.processes) if self.processes > 1 else None
        self.perform_galactic_evolution(progress_bar=with_timing)
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")

        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        if with_timing:
            print(f"Overall: {time.time() - start:1.1f}s")
