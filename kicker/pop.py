from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve

import kicker.kicks as kicks
import time
from multiprocessing import Pool


class Population():
    def __init__(self, n_binaries, processes=8, m1_cutoff=7, final_kstar1=list(range(14)),
                 final_kstar2=list(range(14))):
        self.n_binaries = n_binaries
        self.processes = processes
        self.m1_cutoff = m1_cutoff
        self.final_kstar1 = final_kstar1
        self.final_kstar2 = final_kstar2

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

    def create_population(self, with_timing=True):
        if with_timing:
            start = time.time()
            print(f"Run for {self.n_binaries} binaries")

        self.sample_initial_binaries()
        if with_timing:
            print(f"Ended up with {len(self._initial_binaries)} binaries with masses > {self.m1_cutoff} solar masses")
            print(f"[{time.time() - start:1.0e}s] Sample binaries")
            lap = time.time()

        self.pool = Pool(self.processes) if self.processes else None
        self.perform_stellar_evolution()
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Evolve binaries (run COSMIC)")
            lap = time.time()

        self._orbits = kicks.evolve_binaries_in_galaxy(self._bpp, self._kick_info, pool=self.pool)
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")

        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None

        if with_timing:
            print(f"Overall: {time.time() - start:1.1f}s")

    def sample_initial_binaries(self):
        self._initial_binaries, self._mass_singles, self._mass_binaries, self._n_singles_req,\
            self._n_bin_req = InitialBinaryTable.sampler('independent', self.final_kstar1, self.final_kstar2,
                                                         binfrac_model=0.5, primary_model='kroupa01',
                                                         ecc_model='sana12', porb_model='sana12',
                                                         qmin=-1, SF_start=13700.0, SF_duration=0.0,
                                                         met=0.02, size=self.n_binaries)
        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

    def sample_initial_galaxy():
        pass

    def perform_stellar_evolution(self):
        if self._initial_binaries is None:
            print("Warning: Initial binaries not yet sampled, performing sampling now.")
            self.sample_initial_binaries()
        self._bpp, self._bcm, self._initC,\
            self._kick_info = Evolve.evolve(initialbinarytable=self._initial_binaries,
                                            BSEDict=self.BSE_settings, pool=self.pool)

    def perform_galactic_evolution():
        pass
