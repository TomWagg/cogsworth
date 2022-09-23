from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.sample.sampler import independent
from cosmic.evolve import Evolve

import cosmic_gala
import time
from multiprocessing import Pool


def run_sim(n_binaries, BSESettings={}, nproc=8, with_timing=True, m1_cutoff=7):
    if with_timing:
        start = time.time()

    final_kstar1 = list(range(14))
    final_kstar2 = list(range(14))

    if with_timing:
        print(f"Run for {n_binaries} binaries")

    BSEDict = {'xi': 1.0, 'bhflag': 1, 'neta': 0.5, 'windflag': 3, 'wdflag': 1, 'alpha1': 1.0, 'pts1': 0.001,
               'pts3': 0.02, 'pts2': 0.01, 'epsnov': 0.001, 'hewind': 0.5, 'ck': 1000, 'bwind': 0.0,
               'lambdaf': 0.0, 'mxns': 3.0, 'beta': -1.0, 'tflag': 1, 'acc2': 1.5, 'grflag': 1,
               'remnantflag': 4, 'ceflag': 0, 'eddfac': 1.0, 'ifflag': 0, 'bconst': 3000, 'sigma': 265.0,
               'gamma': -2.0, 'pisn': 45.0,
               'natal_kick_array': [[-100.0,-100.0,-100.0,-100.0,0.0], [-100.0,-100.0,-100.0,-100.0,0.0]],
               'bhsigmafrac': 1.0, 'polar_kick_angle': 90,
               'qcrit_array': [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
               'cekickflag': 2, 'cehestarflag': 0, 'cemergeflag': 0, 'ecsn': 2.25, 'ecsn_mlow': 1.6,
               'aic': 1, 'ussn': 0, 'sigmadiv':-20.0, 'qcflag': 5, 'eddlimflag': 0,
               'fprimc_array': [2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,
                                2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0,2.0/21.0],
               'bhspinflag': 0, 'bhspinmag': 0.0, 'rejuv_fac': 1.0, 'rejuvflag': 0, 'htpmb': 1,
               'ST_cr': 1, 'ST_tide': 1, 'bdecayfac': 1, 'rembar_massloss': 0.5, 'kickflag': 0,
               'zsun': 0.014, 'bhms_coll_flag': 0, 'don_lim': -1, 'acc_lim': -1}

    BSEDict.update(BSESettings)

    InitialBinaries, mass_singles, mass_binaries,\
        n_singles, n_binaries = InitialBinaryTable.sampler('independent', final_kstar1, final_kstar2,
                                                           binfrac_model=0.5, primary_model='kroupa01',
                                                           ecc_model='sana12', porb_model='sana12',
                                                           qmin=-1, SF_start=13700.0, SF_duration=0.0,
                                                           met=0.02, size=n_binaries)

    InitialBinaries = InitialBinaries[InitialBinaries["mass_1"] >= m1_cutoff]

    print(f"Ended up with {len(InitialBinaries)} binaries with masses above {m1_cutoff} solar masses")

    if with_timing:
        print(f"[{time.time() - start:1.0e}s] Sample binaries")
        lap = time.time()

    with Pool(nproc) as pool:
        bpp, bcm, initC, kick_info  = Evolve.evolve(initialbinarytable=InitialBinaries, BSEDict=BSEDict,
                                                    pool=pool)

        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Evolve binaries (run COSMIC)")
            lap = time.time()

        orbits = cosmic_gala.evolve_binaries_in_galaxy(bpp, kick_info, pool=pool)

    if with_timing:
        print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")
        print(f"Overall: {time.time() - start:1.1f}s")

    return bpp, kick_info, orbits


def main():
    run_sim(10000)


if __name__ == "__main__":
    main()
