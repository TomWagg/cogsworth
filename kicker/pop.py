import time
import os
from multiprocessing import Pool
import numpy as np
import astropy.units as u
import astropy.coordinates as coords
import h5py as h5

from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
import gala.potential as gp
import gala.dynamics as gd

from kicker.galaxy import Frankel2018
from kicker.kicks import integrate_orbit_with_events
from kicker.events import identify_events
from kicker.classify import determine_final_classes


class Population():
    """Class for creating and evolving populations of binaries throughout the Milky Way

    Parameters
    ----------
    n_binaries : `int`
        How many binaries to sample for the population
    processes : `int`, optional
        How many processes to run if you want multithreading, by default 8
    m1_cutoff : `float`, optional
        The minimum allowed primary mass, by default 7
    final_kstar1 : `list`, optional
        Desired final types for primary star, by default list(range(14))
    final_kstar2 : `list`, optional
        Desired final types for secondary star, by default list(range(14))
    galaxy_model : `kicker.galaxy.Galaxy`, optional
        A Galaxy class to use for sampling the initial galaxy parameters, by default kicker.galaxy.Frankel2018
    galactic_potential : `gala.potential.PotentialBase`, optional
        Galactic potential to use for evolving the orbits of binaries, by default gp.MilkyWayPotential()
    v_dispersion : `float`, optional
        Velocity dispersion to apply relative to the local circular velocity, by default 5*u.km/u.s
    max_ev_time : `float`, optional
        Maximum evolution time for both COSMIC and Gala, by default 12.0*u.Gyr
    timestep_size : `float`, optional
        Size of timesteps to use in galactic evolution, by default 1*u.Myr

    Attributes
    ----------
    initial_binaries : `pandas.DataFrame`
        Table of inital binaries that have been sampled
    mass_singles : `float`
        Total mass in single stars needed to generate population
    mass_binaries : `float`
        Total mass in binaries needed to generate population
    n_singles_req : `int`
        Number of single stars needed to generate population
    n_bin_req : `int`
        Number of binaries needed to generate population
    bpp : `pandas.DataFrame`
        Evolutionary history of each binary
    bcm : `pandas.DataFrame`
        Final state of each binary
    initC : `pandas.DataFrame`
        Initial conditions for each binary
    kick_info : `pandas.DataFrame`
        Information about the kicks that occur for each binary
    orbits : `list of gala.dynamics.Orbit`
        The orbits of each binary within the galaxy from its birth until `self.max_ev_time` with timesteps of
        `self.timestep_size`. Note that disrupted binaries will have two entries (for both stars).
    classes : `list`
        The classes associated with each produced binary (see classify.list_classes for a list of available
        classes and their meanings)
    """
    def __init__(self, n_binaries, processes=8, m1_cutoff=7, final_kstar1=list(range(14)),
                 final_kstar2=list(range(14)), galaxy_model=Frankel2018,
                 galactic_potential=gp.MilkyWayPotential(), v_dispersion=5 * u.km / u.s,
                 max_ev_time=12.0*u.Gyr, timestep_size=1 * u.Myr):
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

        # TODO: give users access to changing these settings
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
                             'zsun': 0.014, 'bhms_coll_flag': 0, 'don_lim': -1, 'acc_lim': -1}

    @property
    def initial_binaries(self):
        if self._initial_binaries is None:
            self.sample_initial_binaries()
        return self._initial_binaries

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

    def create_population(self, with_timing=True):
        """Create an entirely evolved population of binaries.

        This will sample the initial binaries and initial galaxy and then 
        perform both the COSMIC and Gala evolution

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
            print(f"Ended up with {self.n_binaries_match} binaries with masses > {self.m1_cutoff} solar masses")
            print(f"[{time.time() - start:1.0e}s] Sample initial binaries")
            lap = time.time()

        self.pool = Pool(self.processes) if self.processes else None
        self.perform_stellar_evolution()
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Evolve binaries (run COSMIC)")
            lap = time.time()

        self.perform_galactic_evolution()
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")

        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        if with_timing:
            print(f"Overall: {time.time() - start:1.1f}s")

    def sample_initial_binaries(self):
        """Sample the initial binary parameters for the population"""
        self._initial_binaries, self._mass_singles, self._mass_binaries, self._n_singles_req,\
            self._n_bin_req = InitialBinaryTable.sampler('independent', self.final_kstar1, self.final_kstar2,
                                                         binfrac_model=0.5, primary_model='kroupa01',
                                                         ecc_model='sana12', porb_model='sana12',
                                                         qmin=-1, SF_start=self.max_ev_time.to(u.Myr).value,
                                                         SF_duration=0.0, met=0.02, size=self.n_binaries)

        # apply the mass cutoff
        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

        # count how many binaries actually match the criteria (may be larger than `n_binaries` due to sampler)
        self.n_binaries_match = len(self._initial_binaries)

        # initialise the initial galaxy class with correct number of binaries
        self.initial_galaxy = self.galaxy_model(size=self.n_binaries_match)

        # work out the initial velocities of each binary
        vel_units = u.km / u.s

        # calculate the Galactic circular velocity at the initial positions
        v_circ = self.galactic_potential.circular_velocity(q=[self.initial_galaxy.x,
                                                              self.initial_galaxy.y,
                                                              self.initial_galaxy.z]).to(vel_units)

        # add some velocity dispersion
        v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
                                         self.v_dispersion.to(vel_units) / np.sqrt(3),
                                         size=(3, self.n_binaries_match))
        v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units
        self.initial_galaxy.v_R = v_R
        self.initial_galaxy.v_T = v_T
        self.initial_galaxy.v_z = v_z

        # update the metallicity and birth times of the binaries to match the galaxy
        self._initial_binaries["metallicity"] = self.initial_galaxy.Z
        self._initial_binaries["tphysf"] = (self.max_ev_time - self.initial_galaxy.tau).to(u.Myr).value

    def perform_stellar_evolution(self):
        """Perform the (binary) stellar evolution of the sampled binaries"""
        if self._initial_binaries is None:
            print("Warning: Initial binaries not yet sampled, performing sampling now.")
            self.sample_initial_binaries()

        no_pool_existed = self.pool is None and self.processes > 1
        if no_pool_existed:
            self.pool = Pool(self.processes)

        self._bpp, self._bcm, self._initC,\
            self._kick_info = Evolve.evolve(initialbinarytable=self._initial_binaries,
                                            BSEDict=self.BSE_settings, pool=self.pool)

        if no_pool_existed:
            self.pool.close()
            self.pool.join()

    def perform_galactic_evolution(self):
        # turn the drawn coordinates into an astropy representation
        rep = coords.CylindricalRepresentation(self.initial_galaxy.rho,
                                               self.initial_galaxy.phi,
                                               self.initial_galaxy.z)

        # create differentials based on the velocities (dimensionless angles allows radians conversion)
        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            dif = coords.CylindricalDifferential(self.initial_galaxy.v_R,
                                                 (self.initial_galaxy.v_T
                                                  / self.initial_galaxy.rho).to(u.rad / u.Gyr),
                                                 self.initial_galaxy.v_z)

        # combine the representation and differentials into a Gala PhaseSpacePosition
        w0s = gd.PhaseSpacePosition(rep.with_differentials(dif))

        # identify the pertinent events in the evolution
        events = identify_events(lookback_times=self.initial_galaxy.tau, full_bpp=self.bpp,
                                 full_kick_info=self.kick_info)

        # if we want to use multiprocessing
        if self.pool is not None or self.processes > 1:
            # track whether a pool already existed
            pool_existed = self.pool is not None

            # if not, create one
            if not pool_existed:
                self.pool = Pool(self.processes)

            # setup arguments and evolve the orbits from birth until present day
            args = [(w0s[i], self.galactic_potential, events[i],
                     self.initial_galaxy.tau[i], self.max_ev_time,
                     self.timestep_size) for i in range(self.n_binaries_match)]
            orbits = self.pool.starmap(integrate_orbit_with_events, args)

            # if a pool didn't exist before then close the one just created
            if not pool_existed:
                self.pool.close()
                self.pool.join()
        else:
            # otherwise just use a for loop to evolve the orbits from birth until present day
            orbits = []
            for i in range(self.n_binaries_match):
                orbits.append(integrate_orbit_with_events(w0=w0s[i], potential=self.galactic_potential,
                                                          events=events[i], t1=self.initial_galaxy.tau[i],
                                                          t2=self.max_ev_time, dt=self.timestep_size))

        self._orbits = orbits

    def save(self, file_name, overwrite=False):
        if file_name[-3:] != ".h5":
            file_name += ".h5"
        if os.path.isfile(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise FileExistsError((f"{file_name} already exists. Set `overwrite=True` to overwrite "
                                       "the file."))
        self.initial_binaries.to_hdf(file_name, key="initial_binaries")
        self.bpp.to_hdf(file_name, key="bpp")
        self.bcm.to_hdf(file_name, key="bcm")
        self.initC.to_hdf(file_name, key="initC")
        self.kick_info.to_hdf(file_name, key="kick_info")

        self.galactic_potential.save(f"{file_name.replace('.h5', '-potential.txt')}")
        self.initial_galaxy.save(file_name, key="initial_galaxy")

        with h5.File(file_name, "a") as file:
            numeric_params = np.array([self.n_binaries, self.n_binaries_match, self.processes, self.m1_cutoff,
                                       self.v_dispersion.to(u.km / u.s).value,
                                       self.max_ev_time.to(u.Gyr).value, self.timestep_size.to(u.Myr).value,
                                       self.mass_singles, self.mass_binaries, self.n_singles_req,
                                       self.n_bin_req])
            file.create_dataset("numeric_params", data=numeric_params)

            k_stars = np.array([self.final_kstar1, self.final_kstar2])
            file.create_dataset("k_stars", data=k_stars)

        # still need orbits