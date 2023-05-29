"""
Disruption Orbit
================

Create animations and plots of an orbit of a disrupted binary

TODO
"""

import cogsworth
import matplotlib.pyplot as plt
import astropy.units as u
import numpy as np
import gala.dynamics as gd

p = cogsworth.pop.Population(100, processes=6, final_kstar1=[13, 14], timestep_size=0.2 * u.Myr)
p.create_population()

good_ones = (p.disrupted) & (p.final_coords[0].icrs.distance < 30 * u.kpc)\
    & (p.final_coords[1].icrs.distance < 30 * u.kpc)

bin_nums = p.final_bpp[good_ones]["bin_num"].values
print(len(p.bpp[(p.bpp["evol_type"] == 16) & p.bpp["bin_num"].isin(bin_nums)]))
bin_num = p.bpp[(p.bpp["evol_type"] == 16) & p.bpp["bin_num"].isin(bin_nums)].iloc[0]["bin_num"]

bpp_rows = p.bpp.loc[bin_num]
split_time = bpp_rows[(bpp_rows["evol_type"].isin([15, 16])) & (bpp_rows["sep"] > 0.0)].iloc[-1]["tphys"] * u.Myr

primary_orbit = p.orbits[p.final_bpp["bin_num"] == bin_num][0][0]
secondary_orbit = p.orbits[p.final_bpp["bin_num"] == bin_num][0][1]

primary_orbit = primary_orbit[primary_orbit.t < (primary_orbit.t[0] + split_time * 10)]
secondary_orbit = secondary_orbit[secondary_orbit.t < (secondary_orbit.t[0] + split_time * 10)]

times = primary_orbit.t
times -= times[0]

combined_orbit = gd.Orbit(pos=np.array([primary_orbit.pos.xyz.T, secondary_orbit.pos.xyz.T]).T,
                          vel=np.array([primary_orbit.vel.d_xyz.T, secondary_orbit.vel.d_xyz.T]).T,
                          t=times)

fig, axes = plt.subplots(1, 3, figsize=(21, 7))

from matplotlib.colors import LogNorm

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

x_max = max(abs(combined_orbit.pos.x.min()), combined_orbit.pos.x.max()) * 1.1
y_max = max(abs(combined_orbit.pos.y.min()), combined_orbit.pos.y.max()) * 1.1
z_max = max(abs(combined_orbit.pos.z.min()), combined_orbit.pos.z.max()) * 1.1

x = np.linspace(-x_max, x_max, 100)
y = np.linspace(-y_max, y_max, 100)
z = np.linspace(-z_max, z_max, 100)

# p.galactic_potential.plot_contours(grid=(x, y, 1), ax=axes[0], cmap="Purples")
# p.galactic_potential.plot_contours(grid=(x, 1, z), ax=axes[1], cmap="Purples")
# p.galactic_potential.plot_contours(grid=(1, y, z), ax=axes[2], cmap="Purples")

# axes[0].set_xlim(-x_max, x_max)
# axes[0].set_ylim(-y_max, y_max)
# axes[1].set_xlim(-x_max, x_max)
# axes[2].set_ylim(-z_max, z_max)
# axes[1].set_xlim(-y_max, y_max)
# axes[2].set_ylim(-z_max, z_max)

for ax in axes:
    ax.set_facecolor("black")
    for side in ['bottom', 'top', 'left', 'right']:
        ax.spines[side].set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.yaxis.label.set_color('white')
    ax.xaxis.label.set_color('white')

fig.set_facecolor("black")

print(split_time)

faster = 2 / split_time.to(u.Myr).value

segment_style = [{"color": "C0"}, {"color": "C1"}]


fig, anim = combined_orbit.animate(stride=1, segment_nsteps=50, underplot_full_orbit=True, axes=axes,
                                   FuncAnimation_kwargs={"interval": faster},
                                   segment_style=segment_style, marker_style={"color": "C1"})

plt.show()

# from matplotlib import animation
# writergif = animation.PillowWriter(fps=1000 / faster)
# anim.save("test.gif", writer=writergif)
