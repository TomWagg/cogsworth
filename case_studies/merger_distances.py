import astropy.units as u
from multiprocessing import Pool

import gala.integrate as gi

import numpy as np
import legwork
import kicker

print("START")

p = kicker.pop.Population(n_binaries=100_000_000, m1_cutoff=7, processes=32, store_entire_orbits=False)
p.create_population()

dcos = p[p.classes[p.classes["dco"]].index.values]
dcos.save("/epyc/ssd/users/tomwagg/pop/data/distances-future", overwrite=True)

t_merge = legwork.evol.get_t_merge_ecc(ecc_i=dcos.final_bpp["ecc"].values,
                                       a_i=dcos.final_bpp["sep"].values * u.Rsun,
                                       m_1=dcos.final_bpp["mass_1"].values * u.Msun,
                                       m_2=dcos.final_bpp["mass_1"].values * u.Msun)

esc = dcos.escaped[0]
esc_dcos = dcos[dcos.bin_nums[esc]]
gal_dcos = dcos[dcos.bin_nums[~esc]]

args = [(dcos.galactic_potential, dcos.orbits[i][-1],
         bin_num, t_merge[i], 1 * u.Myr, dcos.final_bpp, esc[i]) for i, bin_num in enumerate(dcos.bin_nums)]

def future_pool_func(pot, w0, bin_num, t_merge, dt, final_bpp, esc):
    if not esc and t_merge < 1 * u.Tyr:
        start_time = final_bpp.tphys[bin_num] * u.Myr
        final = pot.integrate_orbit(w0, t1=start_time, t2=start_time + t_merge, dt=dt,
                                    store_all=False, Integrator=gi.DOPRI853Integrator)
        return final.pos.xyz.ravel().to(u.kpc).value
    else:
        return [0, 0, 0]

dcos.pool = Pool(dcos.processes)
future_mergers = dcos.pool.starmap(future_pool_func, args) * u.kpc
dcos.pool.close()
dcos.pool.join()
dcos.pool = None

movement = (esc_dcos.final_coords[0].velocity * t_merge[esc]).d_xyz.to(u.kpc)
future_mergers[esc] = (np.asarray([esc_dcos.final_coords[0].x.to(u.kpc),
                                   esc_dcos.final_coords[0].y.to(u.kpc),
                                   esc_dcos.final_coords[0].z.to(u.kpc)]) * u.kpc + movement).T

future_merger_distances = np.sum(future_mergers[t_merge < 1 * u.Tyr]**2, axis=1)**(0.5)
np.save("../data/future_merger_distances.npy", future_merger_distances.to(u.kpc).value)

print(future_merger_distances)

merged = p.bpp[p.bpp["sep"] == 0].index.unique()
previous_dco = p.bpp[((p.bpp["kstar_1"] == 14) & (p.bpp["kstar_2"] == 14))
                  | ((p.bpp["kstar_1"] == 13) & (p.bpp["kstar_2"] == 14))
                  | ((p.bpp["kstar_1"] == 14) & (p.bpp["kstar_2"] == 13))
                  | ((p.bpp["kstar_1"] == 13) & (p.bpp["kstar_2"] == 13))].index.unique()
mergers = p[np.intersect1d(merged, previous_dco)]
mergers.save("/epyc/ssd/users/tomwagg/pop/data/distances-past", overwrite=True)

del p

# mergers = kicker.pop.load("/epyc/ssd/users/tomwagg/pop/data/distances-past")

present_times = mergers.final_bpp["tphys"].values * u.Myr
merger_times = mergers.bpp[((mergers.bpp["kstar_1"] == 14) & (mergers.bpp["kstar_2"] == 14))
                           | ((mergers.bpp["kstar_1"] == 13) & (mergers.bpp["kstar_2"] == 14))
                           | ((mergers.bpp["kstar_1"] == 14) & (mergers.bpp["kstar_2"] == 13))
                           | ((mergers.bpp["kstar_1"] == 13) & (mergers.bpp["kstar_2"] == 13))].drop_duplicates(subset="bin_num", keep="last")["tphys"].values * u.Myr

args = [(mergers.galactic_potential, mergers.orbits[i][-1], present_times[i], merger_times[i],
         -1 * u.Myr, False, gi.DOPRI853Integrator) for i in range(len(mergers))]

def pool_func(pot, w0, t1, t2, dt, store_all, Integrator):
    return pot.integrate_orbit(w0=w0, t1=t1, t2=t2, dt=dt, store_all=store_all,
                               Integrator=Integrator).pos.xyz.ravel().to(u.kpc).value

mergers.pool = Pool(mergers.processes)
past_mergers = mergers.pool.starmap(pool_func, args) * u.kpc
mergers.pool.close()
mergers.pool.join()
mergers.pool = None

past_merger_distances = np.sum(past_mergers**2, axis=1)**(0.5)
np.save("../data/past_merger_distances.npy", past_merger_distances.to(u.kpc).value)

print(past_merger_distances)
