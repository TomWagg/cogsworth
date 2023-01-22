from multiprocessing import Pool
from os import path
import numpy as np
import astropy.units as u
import pandas as pd

import gala.integrate as gi
import legwork
import kicker


def past_pool_func(pot, w0, t1, t2, dt, store_all, Integrator):
    return pot.integrate_orbit(w0=w0, t1=t1, t2=t2, dt=dt, store_all=store_all,
                               Integrator=Integrator).pos.xyz.ravel().to(u.kpc).value


def future_pool_func(pot, w0, bin_num, t_merge, dt, final_bpp, esc):
    if not esc:
        start_time = final_bpp.tphys[bin_num] * u.Myr
        final = pot.integrate_orbit(w0, t1=start_time, t2=start_time + t_merge, dt=dt,
                                    store_all=False, Integrator=gi.DOPRI853Integrator)
        return final.pos.xyz.ravel().to(u.kpc).value
    else:
        return [0, 0, 0]


def get_distance_sample(big_data_path="/epyc/ssd/users/tomwagg/pops/dco_mergers/",
                        small_data_path="../data/",
                        label="test",
                        t_merge_max=1 * u.Tyr):
    # evolve 10 million binaries with a primary mass cutoff
    p = kicker.pop.Population(n_binaries=10_000_000, m1_cutoff=7, processes=32, store_entire_orbits=False)
    p.create_population()

    # grab the objects that end as DCOs
    dcos = p[p.classes[p.classes["dco"]].index.values]

    # work out when the future ones will merge
    t_merge = legwork.evol.get_t_merge_ecc(ecc_i=dcos.final_bpp["ecc"].values,
                                           a_i=dcos.final_bpp["sep"].values * u.Rsun,
                                           m_1=dcos.final_bpp["mass_1"].values * u.Msun,
                                           m_2=dcos.final_bpp["mass_1"].values * u.Msun).to(u.Myr)

    # only keep the ones that merge within cutoff
    dcos = dcos[dcos.bin_nums[t_merge < t_merge_max]]
    t_merge = t_merge[t_merge < t_merge_max]

    # grab the objects that merged as DCOs
    merged = p.bpp[p.bpp["sep"] == 0].index.unique()
    previous_dco = p.bpp[((p.bpp["kstar_1"] == 14) & (p.bpp["kstar_2"] == 14))
                         | ((p.bpp["kstar_1"] == 13) & (p.bpp["kstar_2"] == 14))
                         | ((p.bpp["kstar_1"] == 14) & (p.bpp["kstar_2"] == 13))
                         | ((p.bpp["kstar_1"] == 13) & (p.bpp["kstar_2"] == 13))].index.unique()
    mergers = p[np.intersect1d(merged, previous_dco)]

    merger_pop = p[np.concatenate((dcos.bin_nums, mergers.bin_nums))]
    merger_pop.save(path.join(big_data_path, f"merger-distance-pop-{label}"), overwrite=True)

    # free up some memory
    del p, merger_pop

    # PAST MERGERS
    # ------------

    # find current time for binaries and merger times (final time at which they were DCOs)
    present_times = mergers.final_bpp["tphys"].values * u.Myr
    merger_times = mergers.bpp[((mergers.bpp["kstar_1"] == 14) & (mergers.bpp["kstar_2"] == 14))
                               | ((mergers.bpp["kstar_1"] == 13) & (mergers.bpp["kstar_2"] == 14))
                               | ((mergers.bpp["kstar_1"] == 14) & (mergers.bpp["kstar_2"] == 13))
                               | ((mergers.bpp["kstar_1"] == 13) & (mergers.bpp["kstar_2"] == 13))]
    merger_times = merger_times.drop_duplicates(subset="bin_num", keep="last")["tphys"].values * u.Myr

    # throw all of that into a list of arguments for the pool
    args = [(mergers.galactic_potential, mergers.orbits[i][-1], present_times[i], merger_times[i],
            -1 * u.Myr, False, gi.DOPRI853Integrator) for i in range(len(mergers))]

    # integrate the orbits backwards to the merger time and save the locations of the merger
    mergers.pool = Pool(mergers.processes)
    past_mergers = mergers.pool.starmap(past_pool_func, args) * u.kpc
    mergers.pool.close()
    mergers.pool.join()
    mergers.pool = None
    past_bin_nums = mergers.bin_nums

    del mergers

    # FUTURE MERGERS
    # ------------

    # track the subpopulation that escape the galaxy
    esc = dcos.escaped[0]
    esc_dcos = dcos[dcos.bin_nums[esc]]

    # integrate orbits forwards if they merge soon enough and haven't escaped
    args = [(dcos.galactic_potential, dcos.orbits[i][-1], bin_num, t_merge[i], 1 * u.Myr, dcos.final_bpp,
            esc[i]) for i, bin_num in enumerate(dcos.bin_nums)]

    dcos.pool = Pool(dcos.processes)
    future_mergers = dcos.pool.starmap(future_pool_func, args) * u.kpc
    dcos.pool.close()
    dcos.pool.join()
    dcos.pool = None

    # if the DCO has escaped then we don't need integration and can just do it simple
    movement = (esc_dcos.final_coords[0].velocity * t_merge[esc]).d_xyz.to(u.kpc)
    future_mergers[esc] = (np.asarray([esc_dcos.final_coords[0].x.to(u.kpc),
                                       esc_dcos.final_coords[0].y.to(u.kpc),
                                       esc_dcos.final_coords[0].z.to(u.kpc)]) * u.kpc + movement).T

    future_bin_nums = dcos.bin_nums

    bin_nums = np.concatenate((past_bin_nums, future_bin_nums))
    t_merge = np.concatenate((merger_times, t_merge)).to(u.Myr).value
    x = np.concatenate((past_mergers[:, 0], future_mergers[:, 0])).to(u.kpc).value
    y = np.concatenate((past_mergers[:, 1], future_mergers[:, 1])).to(u.kpc).value
    z = np.concatenate((past_mergers[:, 2], future_mergers[:, 2])).to(u.kpc).value

    data = {"bin_num": bin_nums, "t_merge": t_merge, "x": x, "y": y, "z": z}
    df = pd.DataFrame(data=data)
    df.to_hdf(path.join(small_data_path, f"merger-loc-times-{label}.h5"), key="df")


for i in range(10):
    get_distance_sample(label=i)
