import cogsworth
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import gala.dynamics as gd


plt.rc('font', family='serif')
plt.rcParams['text.usetex'] = False
fs = 24

# update various fontsizes to match
params = {'figure.figsize': (12, 8),
          'legend.fontsize': fs,
          'axes.labelsize': fs,
          'xtick.labelsize': 0.9 * fs,
          'ytick.labelsize': 0.9 * fs,
          'axes.linewidth': 1.1,
          'xtick.major.size': 7,
          'xtick.minor.size': 4,
          'ytick.major.size': 7,
          'ytick.minor.size': 4}
plt.rcParams.update(params)


def sn_distance_histograms(p, bins=np.geomspace(1e0, 1e4, 500), fig=None, axes=None, show=True):
    if fig is None or axes is None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 14))

    bin_centres = np.array([(bins[i] + bins[i + 1]) / 2. for i in range(len(bins) - 1)])
    widths = np.insert(bin_centres[1:] - bin_centres[:-1], -1, bin_centres[1] - bin_centres[0])

    ax = axes[0]
    ax.hist([p.primary_sn_distances.to(u.pc).value[p.sn_singles],
             p.primary_sn_distances.to(u.pc).value[p.sn_1],
             p.secondary_sn_distances.to(u.pc).value[p.sn_2],
             np.concatenate((p.primary_sn_distances.to(u.pc).value[p.sn_1_merger], p.secondary_sn_distances.to(u.pc).value[p.sn_2_merger]))],
            bins=bins, label=["Single", "Primary", "Secondary", "Merger Product"], stacked=True,
            color=["grey", plt.cm.viridis(0.4), plt.cm.viridis(0.7), "gold"]);
    ax.legend(fontsize=0.7*fs)
    ax.set_ylabel(ylabel="Number of SN")

    ax = axes[1]
    phist, bins = np.histogram(p.primary_sn_distances.to(u.pc).value, bins=bins)
    shist, bins = np.histogram(p.secondary_sn_distances.to(u.pc).value, bins=bins)
    hist = phist + shist
    ax.bar(bin_centres, 1 - np.cumsum(hist) / np.sum(hist), width=widths, color="#c78ee6")
    ax.set_yscale("log")
    ax.set_ylabel(r"Fraction of SNe > $d$")

    for ax in axes:
        ax.set(xscale="log", xlabel="SN distance from parent cluster [pc]")
        ax.grid(linewidth=0.5, color="lightgrey")

    if show:
        plt.show()
    return fig, axes


def subpop_masks(p):
    sn_rows = p.bpp[((p.bpp["evol_type"] == 15) | (p.bpp["evol_type"] == 16)) & ~p.bpp["bin_num"].isin(p.avoid_these)]
    sn_initC = p.initC.loc[sn_rows["bin_num"]]
    single_bin_nums = sn_initC[sn_initC["kstar_2"] == 15].index.values
    
    primary_sn_rows = sn_rows[sn_rows["evol_type"] == 15]
    secondary_sn_rows = sn_rows[sn_rows["evol_type"] == 16]

    p.sn_singles = primary_sn_rows["bin_num"].isin(single_bin_nums)
    
    sn_1_sep_zero = primary_sn_rows["bin_num"].isin(primary_sn_rows["bin_num"][primary_sn_rows["sep"] == 0.0])
    p.sn_1_merger = ~p.sn_singles & sn_1_sep_zero
    p.sn_1 = ~p.sn_singles & ~sn_1_sep_zero
    
    sn_2_sep_zero = secondary_sn_rows["bin_num"].isin(secondary_sn_rows["bin_num"][secondary_sn_rows["sep"] == 0.0])
    p.sn_2_merger = sn_2_sep_zero
    p.sn_2 = ~sn_2_sep_zero
    
    print(p.sn_singles.sum(), p.sn_1.sum(), p.sn_2.sum(), p.sn_1_merger.sum() + p.sn_2_merger.sum())
    
    return p.sn_singles, p.sn_1, p.sn_1_merger, p.sn_2, p.sn_2_merger


star_particles = pd.read_hdf("FIRE_star_particles.h5")
try:
    particle_orbits = np.load("particle_orbits.npy", allow_pickle=True)
except:
    particle_orbits = [None for _ in range(len(star_particles))]
    for i in range(len(star_particles)):
        if i % 1000 == 0:
            print(i)
        w0 = gd.PhaseSpacePosition(pos=star_particles.iloc[i][["x", "y", "z"]].values * u.kpc,
                                vel=star_particles.iloc[i][["v_x", "v_y", "v_z"]].values * u.km / u.s)
        min_dt = (p.max_ev_time - star_particles.iloc[i]["t_form"] * u.Gyr).to(u.Myr).value
        if min_dt <= 0.01:
            continue
        particle_orbits[i] = p.galactic_potential.integrate_orbit(w0,
                                                                  t1=star_particles.iloc[i]["t_form"] * u.Gyr,
                                                                  t2=p.max_ev_time,
                                                                  dt=min(min_dt - 0.01, 1) * u.Myr)
    np.save("particle_orbits.npy", particle_orbits)

    
file_names = ["beta_0.25", "beta_0.5", "beta_0.75", "bhflag_3", "ecsn_kick_-265"]
    
for file_name in [file_names[4]]:
    p = cogsworth.pop.load(f"/mnt/home/twagg/ceph/pops/supernovae/{file_name}")
    
    uni, counts = np.unique(p.bpp[p.bpp["evol_type"] == 15]["bin_num"], return_counts=True)
    x = uni[counts > 1]
    uni, counts = np.unique(p.bpp[p.bpp["evol_type"] == 16]["bin_num"], return_counts=True)
    y = uni[counts > 1]
    p.avoid_these = np.concatenate((x, y))
    
    p._orbits_file = f"/mnt/home/twagg/ceph/pops/supernovae/{file_name}.h5"
    
    bad_binaries = np.isin(p.bpp["bin_num"].values, p.avoid_these)
    sn_rows = [p.bpp[(p.bpp["evol_type"] == 15) & ~bad_binaries], p.bpp[(p.bpp["evol_type"] == 16) & ~bad_binaries]]
    kicked_nums = [sn_rows[i]["bin_num"].values for i in range(2)]
    kicked_mask = [np.isin(p.bin_nums, kicked_nums[0]), np.isin(p.bin_nums, kicked_nums[1])]
    
    sn_distances = [np.zeros(len(kicked_nums[0])) * u.kpc, np.zeros(len(kicked_nums[1])) * u.kpc]
    sn_locations = [np.zeros((len(kicked_nums[0]), 3)) * u.kpc, np.zeros((len(kicked_nums[1]), 3)) * u.kpc]
    for i in [0, 1]:
        child_orbits = p.primary_orbits[kicked_mask[i]] if i == 0 else p.secondary_orbits[kicked_mask[i]]
        parent_orbits = particle_orbits[p.initC.loc[kicked_nums[i]]["particle_id"].values]
        for j in range(len(kicked_nums[i])):
            parent_orbit = parent_orbits[j]
            child_orbit = child_orbits[j]
            sn_time = sn_rows[i]["tphys"].iloc[j]

            parent_pos = parent_orbit.pos[(parent_orbit.t - parent_orbit.t[0]).to(u.Myr).value < sn_time][-1]
            child_pos = child_orbit.pos[(child_orbit.t - child_orbit.t[0]).to(u.Myr).value < sn_time][-1]

            sn_distances[i][j] = sum((parent_pos - child_pos).xyz**2)**(0.5)
            sn_locations[i][j] = child_pos.xyz
            
    np.savez(f"/mnt/home/twagg/ceph/pops/supernovae/sn_positions-{file_name}",
             sn_distances[0].to(u.kpc).value, sn_distances[1].to(u.kpc).value,
             sn_locations[0].to(u.kpc).value, sn_locations[1].to(u.kpc).value)

    subpop_masks(p)
    p.primary_sn_distances, p.secondary_sn_distances = sn_distances[0], sn_distances[1]
    
    fig, axes = sn_distance_histograms(p, show=False)
    plt.savefig(f"hists/{file_name}.pdf", format="pdf", dpi=300, bbox_inches="tight")