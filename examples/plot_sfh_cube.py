"""
Custom star formation history demo
==================================

A demo of a rather ridiculous custom star formation history in action.

``cogsworth`` allows you to define custom star formation histories (SFHs) for your
:class:`~cogsworth.pop.Population` and are extremely flexible. Here we take that flexibility to the extreme by
creating a custom model that produces a galaxy in the shape of a cube with everything formed 42 Myr ago.

You can see in the plot below that the initial positions of the stars are confined to a cube, whilst the final
positions are rotated due to the galaxy's rotation. The colourbar shows the initial azimuthal angle of the
stars to highlight this.

Check out the `SFH tutorial <../tutorials/pop_settings/initial_galaxy.ipynb>`_ for more details.
"""

import cogsworth
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u

# sphinx_gallery_start_ignore
# plt.style.use("dark_background")
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
          'ytick.minor.size': 4,
          'savefig.dpi': 300}
plt.rcParams.update(params)
# sphinx_gallery_end_ignore

class TheCube(cogsworth.sfh.StarFormationHistory):
    def __init__(self, size, components=None, 
                 component_masses=None, immediately_sample=True):
        super().__init__(size=size, components=components,
                         component_masses=component_masses,
                         immediately_sample=immediately_sample)

    def sample(self):
        self.draw_lookback_times()
        self.draw_positions()
        self.get_metallicity()
        self._which_comp = np.repeat("CUBE", self.size)

    def draw_lookback_times(self):
        self._tau = np.repeat(42, self.size) * u.Myr

    def draw_positions(self):
        self._x = np.random.uniform(-30, 30, self.size) * u.kpc
        self._y = np.random.uniform(-30, 30, self.size) * u.kpc
        self._z = np.random.uniform(-30, 30, self.size) * u.kpc

    def get_metallicity(self):
        nonsense = np.abs(self._x * self._y * self._z)
        self._Z = (nonsense - nonsense.min()) / nonsense.max()


p_cube = cogsworth.pop.Population(1000, sfh_model=TheCube, processes=1, BSE_settings={"binfrac": 1.0})
p_cube.create_population()

# remove disrupted systems (they go to random locations)
p_cube = p_cube[~p_cube.disrupted]

fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
fig.subplots_adjust(wspace=0.1)

# plot initial positions on the left and final positions on the right
scatter = axes[0].scatter(p_cube.initial_galaxy.x, p_cube.initial_galaxy.y,
                          c=p_cube.initial_galaxy.phi.value, cmap="twilight")
scatter = axes[1].scatter(p_cube.final_pos[:, 0], p_cube.final_pos[:, 1],
                          c=p_cube.initial_galaxy.phi.value, cmap="twilight")

# create a colourbar with custom ticks and labels
cbar = fig.colorbar(scatter, label=r"$\phi_{\rm init} \, [rad]$", ax=axes, pad=0.0)
cbar.set_ticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
               labels=[r'$-\pi$', r'$-\pi / 2$', "0", r'$\pi / 2$', r'$\pi$'])

for ax in axes:
    ax.set(xlabel=r"Galactocentric $x \, [\rm kpc]$",
           xlim=(-40, 40), ylim=(-40, 40))
axes[0].set_ylabel(r"Galactocentric $y \, [\rm kpc]$")

axes[0].annotate("Initial positions", xy=(0.5, 0.95), xycoords="axes fraction",
                 ha="center", va="top", fontsize=0.7*fs)
axes[1].annotate("Final positions", xy=(0.42, 0.96), xycoords="axes fraction",
                 ha="center", va="top", fontsize=0.7*fs, rotation=12)

axes[1].annotate("TheCube...and it's ROTATING!", xy=(0, 1.05), xycoords="axes fraction",
                 ha="center", va="center", fontsize=fs)

plt.show()
