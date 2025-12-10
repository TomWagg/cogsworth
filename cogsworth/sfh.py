import sys
import yaml
import h5py as h5
import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.interpolate import interp1d
from scipy.integrate import quad, cumulative_trapezoid
from scipy.special import lambertw
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from astropy.coordinates import SkyCoord
import logging

from types import FunctionType

# for action-based potentials
import gala.potential as gp
from gala.units import galactic
from gala.potential.potential.io import to_dict as potential_to_dict, from_dict as potential_from_dict

from cogsworth.tests.optional_deps import check_dependencies

from cogsworth.citations import CITATIONS


__all__ = ["StarFormationHistory", "DistributionFunctionBasedSFH", "Wagg2022",
           "BurstUniformDisc", "ConstantUniformDisc", "ConstantPlummerSphere",
           "SandersBinney2015", "SpheroidalDwarf", "CarinaDwarf", "load", "concat"]


class StarFormationHistory():
    """Class for a generic galactic star formation history model from which to sample

    This class sets out an outline for sampling from a star formation history model but a subclass will be
    needed for things to function properly.

    All attributes listed below are a given value for the sampled points in the galaxy. If one hasn't been
    sampled/calculated when accessed then it will be automatically sampled/calculated. If sampling, ALL values
    will be sampled.

    Parameters
    ----------
    size : `int`
        Number of points to sample from the model
    components : `list`, optional
        List of component names, by default None
    component_masses : `list`, optional
        List of masses associated with each component (must be the same length as `components`),
        by default None
    immediately_sample : `bool`, optional
        Whether to immediately sample the points, by default True

    """
    def __init__(self, size, components=None, component_masses=None,
                 immediately_sample=True, **kwargs):
        self._components = components
        self._component_masses = component_masses
        self._size = size
        self._tau = None
        self._Z = None
        self._x = None
        self._y = None
        self._z = None
        self._which_comp = None

        # check for any extra parameters that have been passed
        # this may occur when loading from a file and the user was writing a custom class
        if len(kwargs) > 0:
            for key in kwargs:
                setattr(self, key, kwargs[key])

        self.__citations__ = ["cogsworth"]

        if immediately_sample:
            self.sample()

    def __len__(self):
        return self.size

    def __repr__(self):
        return f"<{self.__class__.__name__}, size={len(self)}>"

    def __add__(self, other):
        return concat(self, other)

    def __getitem__(self, ind):
        # ensure indexing with the right type
        if not isinstance(ind, (int, slice, list, np.ndarray, tuple)):
            raise ValueError(("Can only index using an `int`, `list`, `ndarray` or `slice`, you supplied a "
                              f"`{type(ind).__name__}`"))

        # work out any extra kwargs we might need to set
        kwargs = self.__dict__
        actual_kwargs = {}
        saved_attributes = {}
        array_attributes = ["_tau", "_Z", "_x", "_y", "_z", "_which_comp", "v_R", "v_T", "v_z", "v_x", "v_y"]
        for key in list(kwargs.keys()):
            # only keep attributes that have no underscores and aren't velocity components
            if key[0] != "_" and key not in array_attributes:
                actual_kwargs[key] = kwargs[key]
            # also keep any component specific attributes
            elif key.startswith("_component"):
                actual_kwargs[key[1:]] = kwargs[key]
            # save other attributes for later
            elif key not in array_attributes and key != "_size":
                saved_attributes[key] = kwargs[key]

        # pre-mask tau to get the length easily
        tau = np.atleast_1d(self.tau[ind])

        new_sfh = self.__class__(size=len(tau), immediately_sample=False, **actual_kwargs)

        new_sfh._tau = tau
        new_sfh._Z = np.atleast_1d(self._Z[ind])
        new_sfh._x = np.atleast_1d(self._x[ind])
        new_sfh._y = np.atleast_1d(self._y[ind])
        new_sfh._z = np.atleast_1d(self._z[ind])
        new_sfh._which_comp = np.atleast_1d(self._which_comp[ind])

        # if we have any of the velocity components then we need to slice them too
        vel_comps = ["v_R", "v_T", "v_z", "v_x", "v_y"]
        for vel in vel_comps:
            if hasattr(self, vel):
                setattr(new_sfh, vel, np.atleast_1d(getattr(self, vel)[ind]))

        for attr in saved_attributes:
            setattr(new_sfh, attr, saved_attributes[attr])

        return new_sfh

    @property
    def size(self):
        """The number of points in the star formation history model"""
        return self._size

    @size.setter
    def size(self, value):
        if not isinstance(value, int):
            raise ValueError("Size must be an integer")
        if value <= 0:
            raise ValueError("Size must be greater than 0")
        self._size = value

    @property
    def components(self):
        """A list of the components in the star formation history model

        Returns
        -------
        components : ``list`` of ``str``
            The list of the components in the star formation history model
        """
        return self._components

    @property
    def component_masses(self):
        """The masses of the components in the star formation history model

        Returns
        -------
        component_masses : ``list`` of ``float``, [Msun]
            The masses of the components in the star formation history model
        """
        return self._component_masses

    @property
    def tau(self):
        """The lookback times of the sampled points

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            The lookback times of the sampled points
        """
        if self._tau is None:
            self.sample()
        return self._tau

    @property
    def Z(self):
        """The metallicities of the sampled points

        Returns
        -------
        Z : :class:`~astropy.units.Quantity` [dimensionless]
            The metallicities of the sampled points (absolute metallicity **not** solar metallicity)
        """
        if self._Z is None:
            self.sample()
        return self._Z

    @property
    def x(self):
        """The galactocentric x positions of the sampled points

        Returns
        -------
        x : :class:`~astropy.units.Quantity` [length]
            The galactocentric x positions of the sampled points
        """
        if self._x is None:
            self.sample()
        return self._x

    @property
    def y(self):
        """The galactocentric y positions of the sampled points

        Returns
        -------
        y : :class:`~astropy.units.Quantity` [length]
            The galactocentric y positions of the sampled points
        """
        if self._y is None:
            self.sample()
        return self._y

    @property
    def z(self):
        """The galactocentric z positions of the sampled points

        Returns
        -------
        z : :class:`~astropy.units.Quantity` [length]
            The galactocentric z positions of the sampled points
        """
        if self._z is None:
            self.sample()
        return self._z

    @property
    def rho(self):
        """The galactocentric cylindrical radius of the sampled points

        A shortcut for the radius in the x-y plane, :math:`\\sqrt{x^2 + y^2}`

        Returns
        -------
        rho : :class:`~astropy.units.Quantity` [length]
            The galactocentric cylindrical radius of the sampled points
        """
        return (self.x**2 + self.y**2)**(0.5)

    @property
    def phi(self):
        """The galactocentric azimuthal angle of the sampled points

        A shortcut for :math:`\\arctan(y / x)`

        Returns
        -------
        phi : :class:`~astropy.units.Quantity` [angle]
            The galactocentric azimuthal angle of the sampled points
        """
        return np.arctan2(self.y, self.x)

    @property
    def positions(self):
        """The galactocentric positions of the sampled points

        Returns
        -------
        positions : :class:`~astropy.units.Quantity` [length], shape=(3, :attr:`~size`)
            The galactocentric positions of the sampled points
        """
        return [self.x.to(u.kpc).value, self.y.to(u.kpc).value, self.z.to(u.kpc).value] * u.kpc

    @property
    def which_comp(self):
        """The component each point belongs to

        Returns
        -------
        which_comp : ``numpy.ndarray`` of ``str``
            The component each point belongs to
        """
        if self._which_comp is None:
            self.sample()
        return self._which_comp

    def get_citations(self, filename=None):
        """Print the citations for the packages/papers used in the star formation history"""
        if not hasattr(self, "__citations__") or len(self.__citations__) == 0:
            print("No citations need for this star formation history model")
            return

        # ask users for a filename to save the bibtex to
        if filename is None:
            filename = input("Filename for generating a bibtex file (leave blank to just print to terminal): ")
        filename = filename + ".bib" if not filename.endswith(".bib") and filename != "" else filename

        # construct citation string
        cite_tags = []
        bibtex = []
        for section in CITATIONS:
            for citation in self.__citations__:
                if citation in CITATIONS[section]:
                    if citation != "cogsworth":
                        cite_tags.extend(CITATIONS[section][citation]["tags"])
                    bibtex.append(CITATIONS[section][citation]["bibtex"])
        cite_str = ",".join(cite_tags)
        bibtex_str = "\n\n".join(bibtex)

        # print the acknowledgement
        BOLD, RESET, GREEN = "\033[1m", "\033[0m", "\033[0;32m"
        print(f"{BOLD}{GREEN}You can paste this acknowledgement into the relevant section of your manuscript"
              + RESET)
        print(r"This research made use of \texttt{cogsworth} \citep{"
              + ",".join(CITATIONS["general"]["cogsworth"]["tags"])
              + r"} and a model for galactic star formation based on the following papers \citep{"
              + cite_str + "}.\n")

        # either print bibtex to terminal or save to file
        if filename != "":
            print(f"{BOLD}{GREEN}The associated bibtex can be found in {filename} - happy writing!{RESET}")
            with open(filename, "w") as f:
                f.write(bibtex_str)
        else:
            print(f"{BOLD}{GREEN}And paste this bibtex into your .bib file - happy writing!{RESET}")
            print(bibtex_str)

    def sample(self):
        """Sample from the distributions for each component, combine and save in class attributes"""
        if self.size is None:
            raise ValueError("`self.size` has not been set")

        if self._component_masses is None or self._components is None:
            raise ValueError("`self.components` or `self.component_masses` has not been set")

        # work out the weight to assign to each component
        total_mass = np.sum(self._component_masses)
        mass_fractions = np.divide(self._component_masses, total_mass)

        # convert these weights to a number of binaries
        sizes = np.zeros(len(mass_fractions)).astype(int)
        for i in range(len(self._components) - 1):
            sizes[i] = np.round(mass_fractions[i] * self._size)
        sizes[-1] = self._size - np.sum(sizes)

        # create an array of which component each point belongs to
        self._which_comp = np.concatenate([[com] * sizes[i] for i, com in enumerate(self._components)])

        self._tau = np.zeros(self._size) * u.Gyr
        rho = np.zeros(self._size) * u.kpc
        z = np.zeros(self._size) * u.kpc

        # go through each component and get lookback time, radius and height
        for i, com in enumerate(self._components):
            com_mask = self._which_comp == com
            self._tau[com_mask] = self.draw_lookback_times(sizes[i], component=com)
            rho[com_mask] = self.draw_radii(sizes[i], component=com)
            z[com_mask] = self.draw_heights(sizes[i], component=com)

        # shuffle the samples so components are well mixed (mostly for plotting)
        random_order = np.random.permutation(self._size)
        self._tau = self._tau[random_order]
        rho = rho[random_order]
        z = z[random_order]
        self._which_comp = self._which_comp[random_order]

        # draw a random azimuthal angle
        phi = self.draw_phi()

        self._x = rho * np.sin(phi)
        self._y = rho * np.cos(phi)
        self._z = z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

        return self._tau, self.positions, self.Z

    def draw_lookback_times(self, size=None, component=None):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def draw_radii(self, size=None, component=None):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def draw_heights(self, size=None, component=None):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def draw_phi(self, size=None, component=None):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def get_metallicity(self):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def plot(self, coordinates="cartesian", component=None, colour_by=None, show=True, cbar_norm=LogNorm(),
             cbar_label=r"Metallicity, $Z$", cmap="plasma", xlim=None, ylim=None, zlim=None,
             fig=None, axes=None, **kwargs):
        if fig is None or axes is None:
            fig, axes = plt.subplots(2, 1, figsize=(10 * 1.2475, 14),
                                     gridspec_kw={'height_ratios': [4, 14]},
                                     sharex=True)
        if colour_by is None:
            colour_by = self.Z.value

        if coordinates == "cylindrical":
            x = self.rho
            y1 = self.z
            y2 = self.phi
        elif coordinates == "cartesian":
            x = self.x
            y1 = self.z
            y2 = self.y
            axes[1].set_aspect("equal")
        else:
            raise ValueError("Invalid coordinates specified")

        if component is not None:
            mask = self._which_comp == component
            x = x[mask]
            y1 = y1[mask]
            y2 = y2[mask]
            colour_by = colour_by[mask]

        fig.subplots_adjust(hspace=0)

        s = kwargs.pop("s", 0.1)

        axes[0].scatter(x, y1, c=colour_by, s=s, cmap=cmap, norm=cbar_norm, **kwargs)

        axes[0].set_xlabel(r"$x$ [kpc]", labelpad=15)
        axes[0].xaxis.tick_top()
        axes[0].xaxis.set_label_position("top")

        axes[0].set_ylabel(r"$z$ [kpc]")

        scatt = axes[1].scatter(x.value, y2.value, c=colour_by, s=s, cmap=cmap, norm=cbar_norm, **kwargs)
        cbar = fig.colorbar(scatt, ax=axes, pad=0.0)
        cbar.set_label(cbar_label)

        axes[1].set_xlabel(r"$x$ [kpc]")
        axes[1].set_ylabel(r"$y$ [kpc]")

        if xlim is not None:
            axes[0].set_xlim(xlim)
            axes[1].set_xlim(xlim)
        if ylim is not None:
            axes[1].set_ylim(ylim)
        if zlim is not None:
            axes[0].set_ylim(zlim)

        if show:
            plt.show()

        return fig, axes

    def save(self, file_name, key="sfh"):
        """Save the entire class to storage.

        Data will be stored in an hdf5 file using `file_name`.

        Parameters
        ----------
        file_name : `str`
            A name to use for the hdf5 file in which samples will be stored. If this doesn't end in ".h5" then
            ".h5" will be appended.
        key : `str`, optional
            Key to use for the hdf5 file, by default "sfh"
        """
        # append file extension if necessary
        if file_name[-3:] != ".h5":
            file_name += ".h5"

        # store data in a dataframe and save this to file
        data = {
            "tau": self.tau.to(u.Gyr),
            "Z": self.Z,
            "x": self.x.to(u.kpc),
            "y": self.y.to(u.kpc),
            "z": self.z.to(u.kpc),
            "which_comp": self.which_comp
        }

        # additionally store velocity components if they exist
        for attr in ["v_R", "v_T", "v_z"]:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr).to(u.km / u.s).value

        df = pd.DataFrame(data=data)
        df.to_hdf(file_name, key=key)

        # convert parameters into something storable
        params = simplify_params(self.__dict__.copy())

        # check whether the class is part of the default module, get parent recursively if not
        module = sys.modules[__name__]
        class_name = self.__class__.__name__
        class_obj = self
        while not hasattr(module, class_name):
            class_obj = class_obj.__class__.__bases__[0]
            class_name = class_obj.__name__

        # warn the user if we saved a different class name
        if class_name != self.__class__.__name__:
            logging.getLogger("cogsworth").warning(("cogsworth warning: StarFormationHistory class being "
                                                    f"saved as `{class_name}` instead of "
                                                    f"`{self.__class__.__name__}`. Data will be copied but "
                                                    "new sampling will draw from the "
                                                    f"functions in `{class_name}` rather than the "
                                                    "custom class you used."))
        params["class_name"] = class_name

        # dump it all into the file attrs using yaml
        with h5.File(file_name, "a") as file:
            file[key].attrs["params"] = yaml.dump(params, default_flow_style=None)

            # if there's a potential associated with the SFH then save it too
            if hasattr(self, "potential"):
                pot_dict = potential_to_dict(self.potential)
                file[key].attrs["potential"] = yaml.dump(pot_dict, default_flow_style=None)


class BurstUniformDisc(StarFormationHistory):
    """An extremely simple star formation history, with all stars formed at ``t_burst`` in a uniform disc with
    height ``z_max`` and radius ``R_max`` disc, all with metallicity ``Z``.

    Parameters
    ----------

    size : `int`
        Number of points to sample from the model
    t_burst : :class:`~astropy.units.Quantity` [time]
        Lookback time at which all stars are formed
    z_max : :class:`~astropy.units.Quantity` [length]
        Maximum height of the disc
    R_max : :class:`~astropy.units.Quantity` [length]
        Maximum radius of the disc
    Z : `float`, optional
        Metallicity of the disc, by default 0.02
    """
    def __init__(self, size, t_burst=12 * u.Gyr, z_max=2 * u.kpc, R_max=15 * u.kpc, Z_all=0.02, **kwargs):
        self.t_burst = t_burst
        self.z_max = z_max
        self.R_max = R_max
        self.Z_all = Z_all
        super().__init__(size=size, components=kwargs.pop("components", ["disc"]),
                         component_masses=kwargs.pop("component_masses", [1]), **kwargs)

    def draw_lookback_times(self, size=None, component=None):
        return np.repeat(self.t_burst.value, size) * self.t_burst.unit

    def draw_radii(self, size=None, component=None):
        return np.random.uniform(0, self.R_max.value**2, size)**(0.5) * self.R_max.unit

    def draw_heights(self, size=None, component=None):
        return np.random.uniform(-self.z_max.value, self.z_max.value, size) * self.z_max.unit

    def draw_phi(self, size=None):
        # if no size is given then use the class value
        size = self._size if size is None else size
        return np.random.uniform(0, 2 * np.pi, size) * u.rad

    def get_metallicity(self):
        return np.repeat(self.Z_all, self.size) * u.dimensionless_unscaled


class ConstantUniformDisc(BurstUniformDisc):
    """A simple star formation history, with all stars formed at a constant rate between ``t_burst`` 
    and the present day in a uniform disc with height ``z_max`` and radius ``R_max`` disc, all with
    metallicity ``Z``.

    Based on :class:`BurstUniformDisc`.
    """
    def draw_lookback_times(self, size=None, component=None):
        return np.random.uniform(0, self.t_burst.value, size) * self.t_burst.unit


class ConstantPlummerSphere(StarFormationHistory):
    """A simple star formation history, with all stars formed at a constant rate between ``tau_min``
    and ``tau_max`` in a Plummer sphere potential, all with metallicity ``Z``.

    This star formation history sampled BOTH positions and velocities self-consistently in a Plummer
    potential.

    Parameters
    ----------
    size : `int`
        Number of points to sample from the model
    tau_min : :class:`~astropy.units.Quantity` [time]
        Minimum lookback time
    tau_max : :class:`~astropy.units.Quantity` [time]
        Maximum lookback time
    Z_all : `float`
        Metallicity of the sphere
    M : :class:`~astropy.units.Quantity` [mass]
        Total mass of the Plummer sphere
    a : :class:`~astropy.units.Quantity` [length]
        Plummer scale radius
    r_trunc : :class:`~astropy.units.Quantity` [length], optional
        Truncation radius for the Plummer sphere, by default None (i.e. no truncation). If set, stars
        will only be sampled within this radius. For some guidance on setting this value, note that you will
        lose 1 - (r_trunc**3 / (r_trunc**2 + a**2)**1.5) of the mass of the Plummer sphere, where `a` is the
        Plummer scale radius. So setting r_trunc = 5 a will lose ~6% of the mass, r_trunc = 10 a will
        lose ~0.5% of the mass.
    """
    def __init__(self, size, tau_min, tau_max, Z_all, M, a, r_trunc=None, **kwargs):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.Z_all = Z_all
        self.a = a
        self.M = M
        self.r_trunc = r_trunc
        super().__init__(size=size, components=kwargs.pop('components', ["sphere"]),
                         component_masses=kwargs.pop('component_masses', [1]),
                         **kwargs)

    def draw_lookback_times(self):
        """Draw lookback times uniformly between tau_min and tau_max"""
        return np.random.uniform(self.tau_min.to(u.Gyr).value,
                                 self.tau_max.to(u.Gyr).value, self.size) * u.Gyr

    def get_metallicity(self):
        """Fix all metallicities to Z_all"""
        return np.repeat(self.Z_all, self.size) * u.dimensionless_unscaled

    def sample(self):
        # sample times
        self._tau = self.draw_lookback_times()

        # sample positions in a Plummer sphere
        u_max = 1.0 if self.r_trunc is None else self.r_trunc**3 / (self.r_trunc**2 + self.a**2)**1.5
        u_rand = np.random.uniform(0, u_max, self.size)
        r = self.a * (u_rand**(-2/3) - 1.0)**(-0.5)

        # uniformly sample isotropic directions
        cos_theta = np.random.uniform(-1, 1, self.size)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2 * np.pi, self.size)

        # set positions, components and metallicities
        self._x = r * sin_theta * np.cos(phi)
        self._y = r * sin_theta * np.sin(phi)
        self._z = r * cos_theta
        self._which_comp = np.repeat("sphere", self.size)
        self._Z = self.get_metallicity()

        # radii in Plummer units: r' = r / a
        r_dimless = (r / self.a).decompose().value

        # potential and escape speed in Plummer units (G=M=a=1)
        phi_dimless = -1.0 / np.sqrt(1.0 + r_dimless**2)
        vesc_dimless = np.sqrt(-2.0 * phi_dimless)

        # we want q = v / v_esc in [0, 1] with PDF ∝ q^2 (1 - q^2)^(7/2)
        # precompute maximum of g(q) = q^2 (1 - q^2)^(7/2), which occurs at q^2 = 2/9
        g = lambda q: q**2 * (1.0 - q**2)**3.5
        g_max = g(np.sqrt(2.0 / 9.0))

        # perform vectorised rejection sampling
        q = np.empty(self.size)
        remaining = np.ones(self.size, dtype=bool)

        # keep sampling until we have all values
        while np.any(remaining):
            n_rem = remaining.sum()

            # pick random q values in the valid random of 0, 1
            q_try = np.random.uniform(0.0, 1.0, n_rem)

            # pick random y values in the bounding rectangle of 0, g_max
            y = np.random.uniform(0.0, g_max, n_rem)

            accept = y < g(q_try)

            # get inds of remaining points we just sampled
            idx_rem = np.nonzero(remaining)[0]

            # for those that were accepted, save the q value and mark as done
            q[idx_rem[accept]] = q_try[accept]
            remaining[idx_rem[accept]] = False

        # speed in Plummer units: v = q * v_esc
        v_dimless = q * vesc_dimless

        # convert to physical speed
        v_phys = v_dimless * np.sqrt(const.G * self.M / self.a).to(u.km / u.s)

        # random isotropic velocity directions
        cos_theta_v = np.random.uniform(-1.0, 1.0, self.size)
        sin_theta_v = np.sqrt(1.0 - cos_theta_v**2)
        phi_v = np.random.uniform(0.0, 2.0 * np.pi, self.size)

        # save and convert velocities
        self.v_x = v_phys * sin_theta_v * np.cos(phi_v)
        self.v_y = v_phys * sin_theta_v * np.sin(phi_v)
        self.v_z = v_phys * cos_theta_v

        self.v_T = np.sqrt(((-self.x * self.v_y + self.y * self.v_x)**2) / (self.x**2 + self.y**2))
        self.v_R = (self.x * self.v_x + self.y * self.v_y) / np.sqrt(self.x**2 + self.y**2)

        return self._tau, self.positions, self.Z


class Wagg2022(StarFormationHistory):
    """A semi-empirical model defined in
    `Wagg+2022 <https://ui.adsabs.harvard.edu/abs/2021arXiv211113704W/abstract>`_
    (see Figure 1 and Section 2.2.1 for a detailed explanation.), heavily based on
    `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_.

    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    Parameters
    ----------
    galaxy_age : :class:`~astropy.units.Quantity` [time], optional
        Maximum lookback time, by default 12*u.Gyr
    tsfr : :class:`~astropy.units.Quantity` [time], optional
        Star formation timescale, by default 6.8*u.Gyr
    alpha : `float`, optional
        Disc inside-out growth parameter, by default 0.3
    Fm : `int`, optional
        Metallicity at centre of disc at tm, by default -1
    gradient : :class:`~astropy.units.Quantity` [1/length], optional
        Metallicity gradient, by default -0.075/u.kpc
    Rnow : :class:`~astropy.units.Quantity` [length], optional
        Radius at which present day metallicity is solar, by default 8.7*u.kpc
    gamma : `float`, optional
        Time dependence of chemical enrichment, by default 0.3
    zsun : `float`, optional
        Solar metallicity, by default 0.0142
    """
    def __init__(self, size, components=["low_alpha_disc", "high_alpha_disc", "bulge"],
                 component_masses=[2.585e10, 2.585e10, 0.91e10],
                 tsfr=6.8 * u.Gyr, alpha=0.3, Fm=-1, gradient=-0.075 / u.kpc, Rnow=8.7 * u.kpc,
                 gamma=0.3, zsun=0.0142, galaxy_age=12 * u.Gyr, **kwargs):
        self.tsfr = tsfr
        self.alpha = alpha
        self.Fm = Fm
        self.gradient = gradient
        self.Rnow = Rnow
        self.gamma = gamma
        self.zsun = zsun
        self.galaxy_age = galaxy_age
        super().__init__(size=size, components=components, component_masses=component_masses, **kwargs)
        self.__citations__.extend(["Wagg+2022", "Frankel+2018", "Bovy+2016", "Bovy+2019", "McMillan+2011"])

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        """Inverse CDF sampling of lookback times. low_alpha and high_alpha discs uses
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 4,
        separated and normalised at 8 Gyr. The bulge matches the distribution in Fig. 7 of
        `Bovy+19 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.4740B/abstract>`_ but accounts
        for sample's bias.

        Parameters
        ----------
        size : `int`
            How many times to draw
        component : `str`
            Which component of the Milky Way

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        # if no size is given then use the class value
        size = self._size if size is None else size
        if component == "low_alpha_disc":
            U = np.random.rand(size)
            norm = 1 / quad(lambda x: np.exp(-(self.galaxy_age.value - x) / self.tsfr.value), 0, 8)[0]
            tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr.value) + 1)
        elif component == "high_alpha_disc":
            U = np.random.rand(size)
            norm = 1 / quad(lambda x: np.exp(-(self.galaxy_age.value - x) / self.tsfr.value), 8, 12)[0]
            tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr.value)
                                     + np.exp(8 * u.Gyr / self.tsfr))
        elif component == "bulge":
            tau = beta.rvs(a=2, b=3, loc=6, scale=6, size=size) * u.Gyr
        return tau

    def draw_radii(self, size=None, component="low_alpha_disc"):
        """Inverse CDF sampling of galactocentric radii using
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 5.
        The scale length is calculated using Eq. 6 the low_alpha disc and is fixed at 1/0.43 and 1.5 kpc
        respectively for the high_alpha disc and bulge.

        Parameters
        ----------
        size : `int`
            How many radii to draw
        component : `str`
            Which component of the Milky Way

        Returns
        -------
        rho : :class:`~astropy.units.Quantity` [length]
            Random Galactocentric radius
        """
        # if no size is given then use the class value
        size = self._size if size is None else size

        if component == "low_alpha_disc":
            R_0 = 4 * u.kpc * (1 - self.alpha * (self._tau[self._which_comp == component] / (8 * u.Gyr)))
        elif component == "high_alpha_disc":
            R_0 = 1 / 0.43 * u.kpc
        else:
            R_0 = 1.5 * u.kpc

        U = np.random.rand(size)
        rho = - R_0 * (lambertw((U - 1) / np.exp(1), k=-1).real + 1)
        return rho

    def draw_heights(self, size=None, component="low_alpha_disc"):
        """Inverse CDF sampling of heights using
        `McMillan 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.414.2446M/abstract>`_ Eq. 3
        and various scale lengths.

        Parameters
        ----------
        size : `int`
            How many heights to draw
        component : `str`
            Which component of the Milky Way

        Returns
        -------
        z : :class:`~astropy.units.Quantity` [length]
            Random heights
        """
        # if no size is given then use the class value
        size = self._size if size is None else size

        if component == "low_alpha_disc":
            z_d = 0.3 * u.kpc
        elif component == "high_alpha_disc":
            z_d = 0.95 * u.kpc
        else:
            z_d = 0.2 * u.kpc
        U = np.random.rand(size)
        z = np.random.choice([-1, 1], size) * z_d * np.log(1 - U)
        return z

    def draw_phi(self, size=None):
        """Draw random azimuthal angles

        Parameters
        ----------
        size : `int`, optional
            How many angles to draw, by default self._size

        Returns
        -------
        phi : :class:`~astropy.units.Quantity` [angle]
            Azimuthal angles
        """
        # if no size is given then use the class value
        size = self._size if size is None else size
        return np.random.uniform(0, 2 * np.pi, size) * u.rad

    def get_metallicity(self):
        """Convert radius and time to metallicity using
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 7 and
        `Bertelli+1994 <https://ui.adsabs.harvard.edu/abs/1994A%26AS..106..275B/abstract>`_ Eq. 9 but
        assuming all stars have the solar abundance pattern (so no factor of 0.977)

        Returns
        -------
        Z : :class:`~astropy.units.Quantity` [dimensionless]
            Metallicities corresponding to radii and times
        """
        rho = (self.x**2 + self.y**2)**(0.5)
        FeH = self.Fm + self.gradient * rho - (self.Fm + self.gradient * self.Rnow)\
            * (1 - (self._tau / self.galaxy_age))**self.gamma
        return np.power(10, FeH + np.log10(self.zsun))


class DistributionFunctionBasedSFH(StarFormationHistory):
    """A star formation history based on a distribution function.
    This is an abstract base class and should not be instantiated directly.

    Parameters
    ----------
    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    potential : :class:`~agama.Potential` or :class:`Potential <gala.potential.potential.PotentialBase>`
        The gravitational potential in which to sample the distribution function
    df : `function` or `dict` or ``list`` of either
        Either a function that represents the distribution function, taking J as an argument,
        or the keyword arguments to pass to the distribution function(s) using
        :class:`agama.DistributionFunction`. If a `dict` is given then the same
        distribution function will be used for all components. If a ``list`` of `dict` is
        given then each component will use the corresponding distribution function.
    """
    def __init__(self, size, potential, df, **kwargs):
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        self.potential = potential
        self._agama_pot = potential if isinstance(potential, agama.Potential) else potential.as_interop("agama")

        if isinstance(df, dict):
            self._df = agama.DistributionFunction(potential=self.agama_pot, **df)
        elif isinstance(df, FunctionType):
            self._df = df
        elif isinstance(df, list):
            self._df = [agama.DistributionFunction(potential=self.agama_pot, **df_kw)
                        if isinstance(df_kw, dict) else df_kw for df_kw in df]

        super().__init__(size=size, **kwargs)

    @property
    def agama_pot(self):
        return self._agama_pot
    
    @property
    def df(self):
        return self._df
    
    def sample(self):
        """Sample from the distributions for each component, combine and save in class attributes"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        self.draw_lookback_times()

        self._x = np.zeros(self._size) * u.kpc
        self._y = np.zeros(self._size) * u.kpc
        self._z = np.zeros(self._size) * u.kpc

        self.v_x = np.zeros(self._size) * u.km / u.s
        self.v_y = np.zeros(self._size) * u.km / u.s
        self.v_z = np.zeros(self._size) * u.km / u.s

        self.v_R = np.zeros(self._size) * u.km / u.s
        self.v_T = np.zeros(self._size) * u.km / u.s

        if self._which_comp is None and self.components is None:
            self._which_comp = np.array(["main"] * self._size)

        for i, com in enumerate(self.components if self.components is not None else ["main"]):
            com_mask = self._which_comp == com

            df = self.df[i] if isinstance(self.df, list) else self.df
            xv, _ = agama.GalaxyModel(self.agama_pot, df).sample(com_mask.sum())

            # convert units for velocity
            xv[:, 3:] *= (u.kpc / u.Myr).to(u.km / u.s)

            # save the positions/velocities
            self._x[com_mask] = xv[:, 0] * u.kpc
            self._y[com_mask] = xv[:, 1] * u.kpc
            self._z[com_mask] = xv[:, 2] * u.kpc
            self.v_x[com_mask] = xv[:, 3] * u.km / u.s
            self.v_y[com_mask] = xv[:, 4] * u.km / u.s
            self.v_z[com_mask] = xv[:, 5] * u.km / u.s

        # work out the velocities by rotating using SkyCoord
        full_coord = SkyCoord(x=self._x, y=self._y, z=self._z, v_x=self.v_x, v_y=self.v_y, v_z=self.v_z,
                              frame="galactocentric").represent_as("cylindrical")

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            self.v_R = full_coord.differentials['s'].d_rho
            self.v_T = (full_coord.differentials['s'].d_phi * full_coord.rho).to(u.km / u.s)
            self.v_z = full_coord.differentials['s'].d_z

        # compute the metallicity given the other values
        self.get_metallicity()


class SandersBinney2015(DistributionFunctionBasedSFH):
    """Star formation history model based on a Quasi-Isothermal Disc distribution function from
    `Sanders & Binney 2015 <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3479S/abstract>`_.

    This class doesn't account for the extended distribution function described in SB15, instead following
    the quasi-isothermal DF described in Section 2.2 of that paper. We follow their prescription for the
    time evolution of the velocity dispersions and the metallicity distribution, but do not include radial
    migration.

    Parameters are inherited from :class:`DistributionFunctionBasedSFH` and :class:`StarFormationHistory`
    but additionally with the following:

    Parameters
    ----------
    time_bins : `int`, optional
        Number of time bins to use when computing different radial and vertical velocity dispersions, which
        accounts for how these parameters evolve with time. More bins means a more accurate representation
        of the SFH but takes longer to compute. By default 5.
    verbose : `bool`, optional
        Whether to print out information about the setup and sampling of the model, by default False
    """
    def __init__(self, size, time_bins=5, verbose=False,
                 tau_m=12 * u.Gyr, tau_S=0.43 * u.Gyr, tau_T=10 * u.Gyr,
                 tau_F=8 * u.Gyr, tau_1=0.11 * u.Gyr,
                 **kwargs):
        self._size = size
        self.time_bins = time_bins
        self.tau_m = tau_m
        self.tau_S = tau_S
        self.tau_T = tau_T
        self.tau_F = tau_F
        self.tau_1 = tau_1
        self.verbose = verbose
        self._inv_cdf = None
        self._guiding_radius_interp = None
        self._omega_interp = None
        self._kappa_interp = None
        self._nu_interp = None

        # ensure we don't pass components twice
        for var in ["components", "component_masses"]:          # pragma: no cover
            if var in kwargs:
                kwargs.pop(var)

        # save whether the user wants to immediately sample (we're going to tell super not to either way)
        immediately_sample = kwargs.pop("immediately_sample", True)

        super().__init__(size=size, components=["thin_disc", "thick_disc"],
                         df=None, immediately_sample=False, **kwargs)
        self.__citations__.append("Sanders&Binney2015")

        if immediately_sample:
            self.sample()

    def _precompute_interpolations(self):
        interp_needed = (self._inv_cdf is None or self._guiding_radius_interp is None
                         or self._omega_interp is None or self._kappa_interp is None
                         or self._nu_interp is None)
        if not interp_needed:       # pragma: no cover
            return
        if self.verbose:
            print("Pre-computing lookback time, guiding radius and frequency interpolations")

        # interpolate the inverse CDF for lookback time distribution
        # pdf taken from Sanders & Binney 2015 Eq. 10
        tau_range = np.linspace(0, self.tau_m * (1 - 1e-10), 100000)
        tau_pdf = np.exp(tau_range / self.tau_F - self.tau_S / (self.tau_m - tau_range))
        tau_cdf = cumulative_trapezoid(tau_pdf, tau_range, initial=0)
        self._inv_cdf = interp1d(tau_cdf / tau_cdf[-1], tau_range, bounds_error=True)
        self.galaxy_age = self.tau_m

        # pre-compute frequencies at a range of guiding radii
        R_g_range = np.linspace(1e-2, 100, 10000) * u.kpc
        J_phi = R_g_range * self.potential.circular_velocity(q=[R_g_range, 0 * R_g_range, 0 * R_g_range])
        self._guiding_radius_interp = interp1d(J_phi.to(u.kpc**2 / u.Myr).value, R_g_range.value,
                                               bounds_error=False, fill_value="extrapolate")

        omega = self._get_omega(R_g_range)
        kappa = self._get_kappa(R_g_range, omega)
        nu = self._get_nu(R_g_range)
        self._omega_interp = interp1d(R_g_range.value, omega.value)
        self._kappa_interp = interp1d(R_g_range.value, kappa.value)
        self._nu_interp = interp1d(R_g_range.value, nu.value)

    def _get_omega(self, R_g):
        """Get the circular frequency at a given guiding radius

        Parameters
        ----------
        R_g : float
            Guiding radius in same units as potential

        .. math::
            \\Omega(R_g) = \\frac{v_c(R_g)}{R_g}
        """
        R_g = np.atleast_1d(R_g)
        return (self.potential.circular_velocity(q=[R_g, 0 * R_g, 0 * R_g]) / R_g).to(1 / u.Myr)

    def _get_kappa(self, R_g, omega):
        """Get the radial epicyclic frequency at a given guiding radius

        Parameters
        ----------
        R_g : float
            Guiding radius in same units as potential
        omega : float
            Circular frequency at the guiding radius in 1/time

        .. math::
            \\kappa(R_g) = \\sqrt{4 \\Omega^2 + R_g \\frac{{\\rm d}\\Omega^2}{{\\rm d}R}}
        """
        omega = np.atleast_1d(omega)
        R_g = np.atleast_1d(R_g)
        d_omega_2_dR = np.gradient(omega**2, R_g)
        return np.sqrt(4 * omega**2 + R_g * d_omega_2_dR).to(1 / u.Myr)

    def _get_nu(self, R_g):
        """Get the vertical epicyclic frequency at a given guiding radius

        Parameters
        ----------
        R_g : float
            Guiding radius in same units as potential

        .. math::
            \\nu(R_g) = \\sqrt{\\frac{\\partial^2 \\Phi}{\\partial z^2}}
        """
        R_g = np.atleast_1d(R_g)
        return (self.potential.hessian(q=[R_g, 0 * R_g, 0 * R_g])[2, 2]**0.5).to(1 / u.Myr)

    def _get_sigma_i(self, i, R_g, tau, component):
        """Get the radial or vertical velocity dispersion at a given guiding radius and lookback time

        Follows `Sanders & Binney 2015 <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3479S/abstract>`_
        Eq. 4 and 10.

        Parameters
        ----------
        i : `str`
            Either "R" or "z" for radial or vertical velocity dispersion
        R_g : float
            Guiding radius in kpc
        tau : :class:`~astropy.units.Quantity` [time]
            Lookback time
        component : `str`
            Either "thin_disc" or "thick_disc"

        Returns
        -------
        sigma_i : float
            Velocity dispersion in the specified direction in km/s
        """
        assert component in ["thin_disc", "thick_disc"], "Component must be 'thin_disc' or 'thick_disc'"
        assert i in ["R", "z"], "i must be 'R' or 'z'"
        sigma_R0 = (48.3 if component == "thin_disc" else 50.5)
        sigma_z0 = (30.7 if component == "thin_disc" else 51.3)
        R_sigma = (7.8 if component == "thin_disc" else 6.2)

        beta = (0.33 if i == 'R' else 0.4) if component == "thin_disc" else 0
        return (sigma_R0 if i == "R" else sigma_z0) * np.exp((8 - R_g) / R_sigma) * ((tau + self.tau_1) / (self.tau_T + self.tau_1))**beta

    def _generate_df(self, J, component, tau):
        """Generate a distribution function for a given component and lookback time

        Follows `Sanders & Binney 2015 <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3479S/abstract>`_
        Eq. 3.1.

        Parameters
        ----------
        J : `array-like`, shape (N, 3)
            Actions in (J_r, J_phi, J_z) in units of kpc^2 / Myr
        component : `str`
            Either "thin_disc" or "thick_disc"
        tau : :class:`~astropy.units.Quantity` [time]
            Lookback time

        Returns
        -------
        df_val : `array-like`, shape (N,)
            Value of the distribution function at the given actions
        """
        assert component in ["thin_disc", "thick_disc"], "Component must be 'thin_disc' or 'thick_disc'"

        J_r, J_z, J_phi = J.T

        # only compute the DF where the prior interpolations are valid
        df_val = np.full_like(J_r, np.nan)
        valid = (J_phi >= 1e-5) & (J_phi <= 100)

        R_d = 3.45 if component == "thin_disc" else 2.31
        L_0 = 0.01

        # get guiding radii
        R_g = np.zeros_like(J_r)
        R_g[valid] = self._guiding_radius_interp(J_phi[valid])
        valid &= (R_g >= 1e-2) & (R_g <= 100)
        R_g = R_g[valid]

        # get frequencies at guiding radii based on potential
        omega = self._omega_interp(R_g)
        kappa = self._kappa_interp(R_g)
        nu = self._nu_interp(R_g)

        # time dependent velocity dispersions
        kms_to_kpcMyr = (u.km / u.s).to(u.kpc / u.Myr)
        sigma_R = self._get_sigma_i("R", R_g, tau, component) * kms_to_kpcMyr
        sigma_z = self._get_sigma_i("z", R_g, tau, component) * kms_to_kpcMyr

        # construct DF
        prefactor = 1 / (8 * np.pi**3) * (1 + np.tanh(J_phi[valid] / L_0))          # no units
        exp_terms = [
            omega / (R_d**2 * kappa**2) * np.exp(-R_g / R_d),                       # units of Myr/kpc^2
            (kappa / sigma_R**2) * np.exp(-kappa * J_r[valid] / sigma_R**2),        # units of Myr/kpc^2
            (nu / sigma_z**2) * np.exp(-nu * J_z[valid] / sigma_z**2)               # units of Myr/kpc^2
        ]
        df_val[valid] = prefactor * np.prod(exp_terms, axis=0)                      # units of Myr^3/kpc^6
        return df_val

    def draw_lookback_times(self):
        """Draw lookback times for all stars using inverse CDF sampling

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        U = np.random.rand(self._size)
        self._tau = self._inv_cdf(U) * u.Gyr
        return self._tau

    def get_metallicity(self):
        """Calculate the metallicity based on the radius and lookback time 
        BUT use the prescription from Frankel+2018, the SB15 one is outdated.
        """
        Fm, gradient, Rnow, gamma = -1, -0.075 / u.kpc, 8.7 * u.kpc, 0.3
        FeH = Fm + gradient * self.rho - (Fm + gradient * Rnow) * (1 - (self.tau / self.galaxy_age))**gamma
        self._Z = np.power(10, FeH + np.log10(0.0142))
        return self._Z

    def sample(self):
        """Sample from the distributions for each component, combine and save in class attributes"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        self._precompute_interpolations()

        if self.verbose:
            print("Initiating sampling procedure")

        self.draw_lookback_times()

        is_thin_disc = self.tau < self.tau_T
        sizes = [np.sum(is_thin_disc), np.sum(~is_thin_disc)]

        self._which_comp = np.where(self.tau < self.tau_T, "thin_disc", "thick_disc")

        self._x = np.zeros(self._size) * u.kpc
        self._y = np.zeros(self._size) * u.kpc
        self._z = np.zeros(self._size) * u.kpc

        self.v_x = np.zeros(self._size) * u.km / u.s
        self.v_y = np.zeros(self._size) * u.km / u.s
        self.v_z = np.zeros(self._size) * u.km / u.s

        self.v_R = np.zeros(self._size) * u.km / u.s
        self.v_T = np.zeros(self._size) * u.km / u.s

        for size, com in zip(sizes, self.components):
            if size == 0:          # pragma: no cover
                continue
            com_mask = self._which_comp == com

            if com == "thin_disc":
                time_bin_edges = np.linspace(0, self.tau_T.to(u.Gyr).value, self.time_bins + 1) * u.Gyr
            else:
                time_bin_edges = np.array([self.tau_T.to(u.Gyr).value, self.tau_m.to(u.Gyr).value]) * u.Gyr

            if self.verbose:
                print(f"  Sampling {size} stars from the {com}")

            # loop over each bin of time and sample from the corresponding DF
            for t0, t1 in zip(time_bin_edges[:-1], time_bin_edges[1:]):
                in_bin = com_mask & (self.tau >= t0) & (self.tau < t1)
                n_in_bin = np.sum(in_bin)
                if n_in_bin == 0:           # pragma: no cover
                    continue

                if self.verbose:
                    print(f"    Sampling {n_in_bin} stars with lookback times between {t0:.2f} and {t1:.2f}")

                df = agama.DistributionFunction(lambda J: self._generate_df(J=J, component=com,
                                                                            tau=(t0 + t1) / 2))
                xv, _ = agama.GalaxyModel(self._agama_pot, df).sample(n_in_bin)

                # convert units for velocity
                xv[:, 3:] *= (u.kpc / u.Myr).to(u.km / u.s)

                # save the positions/velocities
                self._x[in_bin] = xv[:, 0] * u.kpc
                self._y[in_bin] = xv[:, 1] * u.kpc
                self._z[in_bin] = xv[:, 2] * u.kpc
                self.v_x[in_bin] = xv[:, 3] * u.km / u.s
                self.v_y[in_bin] = xv[:, 4] * u.km / u.s
                self.v_z[in_bin] = xv[:, 5] * u.km / u.s

        # work out the velocities by rotating using SkyCoord
        full_coord = SkyCoord(x=self._x, y=self._y, z=self._z, v_x=self.v_x, v_y=self.v_y, v_z=self.v_z,
                              frame="galactocentric").represent_as("cylindrical")

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            self.v_R = full_coord.differentials['s'].d_rho
            self.v_T = (full_coord.differentials['s'].d_phi * full_coord.rho).to(u.km / u.s)
            self.v_z = full_coord.differentials['s'].d_z

        # compute the metallicity given the other values
        self.get_metallicity()


class SpheroidalDwarf(DistributionFunctionBasedSFH):
    """An action-based model for dwarf spheroidal galaxies and globular clusters
    `Pascale+2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2423P/abstract>`_.

    Parameters are the same as :class:`DistributionFunctionBasedSFH` and
    :class:`StarFormationHistory` but additionally with the following:

    Parameters
    ----------
    J_0_star : `float`
        The action scale that naturally defines the length- and velocity-scale
    alpha : `float`
        A non-negative, dimensionless parameter that mainly regulates the model's density profile
    eta : `float`
        A non-negative, dimensionless parameter that mainly controls the radial or tangential bias of the
        model velocity distribution; models sharing the parameters $(\\alpha, \\eta)$ are homologous.
    fixed_Z : `float`
        Fixed metallicity for all stars in the dwarf galaxy
    tau_min : :class:`~astropy.units.Quantity` [time]
        Minimum lookback time for star formation, by default 10 Gyr
    galaxy_age : :class:`~astropy.units.Quantity` [time]
        Maximum lookback time for star formation, by default 12 Gyr
    mass : `float`, optional
        Total mass of the galactic potential. If not given, a potential must be provided. If given, this will
        be used to create a NFW potential with scale radius 1 kpc and concentration 1. By default None.
    """
    def __init__(self, size, J_0_star, alpha, eta, fixed_Z, tau_min, galaxy_age, mass=None, **kwargs):
        # set mass and, potentially (...hehe), the potential
        self.mass = mass
        if "potential" not in kwargs and self.mass is None:
            raise ValueError("You must provide either a potential or a mass for the SpheroidalDwarf model")
        elif "potential" not in kwargs and self.mass is not None:
            kwargs["potential"] = gp.NFWPotential(m=mass, r_s=1.0, units=galactic)

        kwargs["df"] = lambda J: self._generate_df(J)
        self.J_0_star = J_0_star
        self.alpha = alpha
        self.eta = eta
        self.fixed_Z = fixed_Z
        self.tau_min = tau_min
        self.galaxy_age = galaxy_age

        self._agama_pot = None
        self._df = None

        # ensure we don't pass components twice
        for var in ["components", "component_masses"]:          # pragma: no cover
            if var in kwargs:
                kwargs.pop(var)

        super().__init__(size=size, components=None, component_masses=None, **kwargs)
        self.__citations__.append("Pascale+2019")

    def draw_lookback_times(self, size=None):
        """Uniform sampling of lookback times between tau_min and tau_max

        Parameters
        ----------
        size : `int`
            How many times to draw
        component : `str`
            Which component of the Milky Way

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        # if no size is given then use the class value
        size = self._size if size is None else size
        self._tau = np.random.uniform(self.tau_min.to(u.Gyr).value,
                                      self.galaxy_age.to(u.Gyr).value, size) * u.Gyr
        return self._tau

    def get_metallicity(self):
        """Convert radius and time to metallicity using
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 7 and
        `Bertelli+1994 <https://ui.adsabs.harvard.edu/abs/1994A%26AS..106..275B/abstract>`_ Eq. 9 but
        assuming all stars have the solar abundance pattern (so no factor of 0.977)

        Returns
        -------
        Z : :class:`~astropy.units.Quantity` [dimensionless]
            Metallicities corresponding to radii and times
        """
        self._Z = np.repeat(self.fixed_Z, self._size)
        return self._Z

    def _generate_df(self, J):
        """Get the distribution function for a dwarf galaxy disk
        
        This assumes spherical symmetry and follows Eq. 7 (instead of the more general Eq. 5) of
        `Pascale+2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2423P/abstract>`_.
        """
        J0_no_units = (self.J_0_star).decompose(galactic).value
        Jr, Jz, Jphi = J.T
        kJ = Jr + self.eta * (np.abs(Jphi) + Jz)
        return np.exp(-(kJ / J0_no_units)**self.alpha)


class CarinaDwarf(SpheroidalDwarf):
    """A model for the Carina dwarf spheroidal galaxy based on
    `Pascale+2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2423P/abstract>`_.

    Parameters are the same as :class:`SpheroidalDwarf` but with the following defaults:

    Parameters
    ----------
    size : `int`
        How many stars to simulate
    """
    def __init__(self, size, **kwargs):
        super().__init__(size=size, mass=8.69e8 * u.Msun, J_0_star=0.677 * u.kpc*u.km/u.s,
                         alpha=0.946, eta=0.5, **kwargs)


def load(file_name, key="sfh"):
    """Load an entire class from storage.

    Data should be stored in an hdf5 file using `file_name`.

    Parameters
    ----------
    file_name : `str`
        A name of the .h5 file in which samples are stored and .txt file in which parameters are stored
    key : `str`, optional
        Key to use for the hdf5 file, by default "sfh"
    """
    # append file extension if necessary
    if file_name[-3:] != ".h5":
        file_name += ".h5"

    # assume no potential unless we find one
    pot = None

    # load the parameters back in using yaml
    with h5.File(file_name, "r") as file:
        if key not in file.keys():
            raise ValueError((f"Can't find a saved SFH in {file_name} under the key {key}."))
        params = yaml.load(file[key].attrs["params"], Loader=yaml.Loader)

        # load associated potential if it exists
        if "potential" in file[key].attrs:
            potential_data = file[key].attrs["potential"]
            pot = potential_from_dict(yaml.load(potential_data, Loader=yaml.Loader))

    # get the current module, get a class using the name, delete it from parameters that will be passed
    module = sys.modules[__name__]

    sfh_class = getattr(module, params["class_name"])
    del params["class_name"]

    # ensure no samples are taken
    params["immediately_sample"] = False

    # complicate the parameters to add units back in
    complicated_params = complicate_params(params)

    # add associated potential if it exists
    if pot is not None:
        complicated_params["potential"] = pot

    # create a new sfh using the parameters
    loaded_sfh = sfh_class(**complicated_params)

    # read in the data and save it into the class
    df = pd.read_hdf(file_name, key=key)
    loaded_sfh._tau = df["tau"].values * u.Gyr
    loaded_sfh._Z = df["Z"].values * u.dimensionless_unscaled
    loaded_sfh._which_comp = df["which_comp"].values
    loaded_sfh._x = df["x"].values * u.kpc
    loaded_sfh._y = df["y"].values * u.kpc
    loaded_sfh._z = df["z"].values * u.kpc

    # additionally read in velocity components if they exist
    for attr in ["v_R", "v_T", "v_z"]:
        if attr in df:
            setattr(loaded_sfh, attr, df[attr].values * u.km / u.s)

    # return the newly created class
    return loaded_sfh


def concat(*sfhs):
    """Concatenate multiple StarFormationHistory objects together.

    Parameters
    ----------
    *sfhs : `StarFormationHistory`
        Any number of StarFormationHistory objects to concatenate

    Returns
    -------
    `StarFormationHistory`
        A new StarFormationHistory object that is the concatenation of all the input objects
    """
    # check that all the objects are of the same type
    sfhs = list(sfhs)
    assert all([isinstance(sfh, StarFormationHistory) for sfh in sfhs])
    if len(sfhs) == 1:
        return sfhs[0]
    elif len(sfhs) == 0:
        raise ValueError("No objects to concatenate")

    # create a new object with the same parameters as the first
    new_sfh = sfhs[0][:]

    # concatenate the velocity components if they exist
    for attr in ["_tau", "_Z", "_which_comp", "_x", "_y", "_z", "v_R", "v_T", "v_z"]:
        if hasattr(sfhs[0], attr):
            setattr(new_sfh, attr, np.concatenate([getattr(sfh, attr) for sfh in sfhs]))

    new_sfh._size = len(new_sfh._tau)

    return new_sfh


def simplify_params(params, dont_save=["_tau", "_Z", "_x", "_y", "_z", "_which_comp", "v_R", "v_T", "v_z",
                                       "v_x", "v_y", "_df", "_agama_pot", "potential", "__citations__",
                                       "sfh_params", "_guiding_radius_interp", "_omega_interp",
                                       "_kappa_interp", "_nu_interp", "_inv_cdf"]):
    # delete any keys that we don't want to save
    delete_keys = [key for key in params.keys() if key in dont_save]
    for key in delete_keys:
        del params[key]

    # convert any arrays to lists and split up units from values
    params_copy = params.copy()
    for key, item in params_copy.items():
        if hasattr(item, 'unit'):
            params[key] = item.value
            params[key+'_unit'] = str(item.unit)

        if hasattr(params[key], 'tolist'):
            params[key] = params[key].tolist()

    return params


def complicate_params(params):
    # combine units with their values
    params_copy = params.copy()
    for key in params_copy.keys():
        if "_unit" in key:
            continue
        if key + "_unit" in params:
            params[key] *= u.Unit(params[key + '_unit'])
            del params[key + '_unit']

        if key[0] == "_":
            params[key[1:]] = params[key]
            del params[key]
    return params
