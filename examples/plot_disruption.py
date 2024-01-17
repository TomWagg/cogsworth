"""
Disruption Orbit Animation
==========================

Create animations of an orbit of a disrupted binary in various projections.
"""

import cogsworth
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import gala.dynamics as gd

p = cogsworth.pop.Population(100, processes=6, final_kstar1=[13, 14], timestep_size=0.2 * u.Myr)
p.create_population()

g_cen_dist = np.sum(p.final_pos**2, axis=1)**(0.5)
good_ones = (g_cen_dist[:len(p)][p.disrupted] < 30 * u.kpc) & (g_cen_dist[len(p):] < 30 * u.kpc)

bin_nums = p.bin_nums[p.disrupted][good_ones]
potential_bpp = p.bpp.loc[bin_nums]
print(len(potential_bpp[potential_bpp["evol_type"] == 16]))
bin_num = potential_bpp[potential_bpp["evol_type"] == 16].iloc[0]["bin_num"]

bpp_rows = p.bpp.loc[bin_num]
split_time = bpp_rows[(bpp_rows["evol_type"].isin([15, 16])) & (bpp_rows["sep"] > 0.0)].iloc[-1]["tphys"] * u.Myr

primary_orbit = p.orbits[:len(p)][p.final_bpp["bin_num"] == bin_num][0]
secondary_orbit = p.orbits[len(p):][p.bin_nums[p.disrupted] == bin_num][0]

primary_orbit = primary_orbit[primary_orbit.t < (primary_orbit.t[0] + split_time * 5)]
secondary_orbit = secondary_orbit[secondary_orbit.t < (secondary_orbit.t[0] + split_time * 5)]

times = primary_orbit.t
times -= times[0]

combined_orbit = gd.Orbit(pos=np.array([primary_orbit.pos.xyz.T, secondary_orbit.pos.xyz.T]).T,
                          vel=np.array([primary_orbit.vel.d_xyz.T, secondary_orbit.vel.d_xyz.T]).T,
                          t=times)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

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
plt.style.use("dark_background")

for ax in axes:
    ax.set_facecolor("black")
    for side in ['bottom', 'top', 'left', 'right']:
        ax.spines[side].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')

fig.set_facecolor("black")

# faster = 2 / split_time.to(u.Myr).value

segment_style = [{"color": "C0"}, {"color": "C1"}]

fig, anim = combined_orbit.animate(stride=5, segment_nsteps=50, underplot_full_orbit=True, axes=axes,
                                   FuncAnimation_kwargs={"interval": 50},
                                   segment_style=segment_style, marker_style={"color": "C1"})

plt.show()

# from matplotlib import animation
# writergif = animation.PillowWriter(fps=1000 / faster)
# anim.save("test.gif", writer=writergif)
