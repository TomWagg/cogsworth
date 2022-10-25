import sys
import yaml
import numpy as np
import astropy.units as u
from scipy.integrate import quad
from scipy.special import lambertw
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from astropy.coordinates import SkyCoord

__all__ = ["Galaxy", "Frankel2018", "load"]


class Galaxy():
    """Class for a generic galaxy model from which to sample

    This class sets out an outline for sampling from a galaxy model but a subclass will be needed for things
    to function properly.

    All attributes listed below are a given value for the sampled points in the galaxy. If ine hasn't been
    sampled/calculated when accessed then it will be automatically sampled/calculated. If sampling, ALL values
    will be sampled.

    Parameters
    ----------
    size : `int`
        Number of points to sample from the galaxy model
    components : `list`, optional
        List of component names, by default None
    component_masses : `list`, optional
        List of masses associated with each component (must be the same length as `components`),
        by default None
    immediately_sample : `bool`, optional
        Whether to immediately sample the points from the galaxy, by default True


    Attributes
    ----------
    tau : :class:`~astropy.units.Quantity` [time]
        Lookback time
    Z : :class:`~astropy.units.Quantity` [dimensionless]
        Metallicity
    positions : :class:`~astropy.coordinates.SkyCoord`
        Initial positions in Galactocentric frame

    """
    def __init__(self, size, components=None, component_masses=None,
                 immediately_sample=True):
        self._components = components
        self._component_masses = component_masses
        self._size = size
        self._tau = None
        self._Z = None
        self._positions = None
        self._which_comp = None

        if immediately_sample:
            self.sample()

    def __repr__(self):
        return f"<{self.__class__.__name__}, size={self.size}>"

    def __getitem__(self, ind):
        # ensure indexing with the right type
        if not isinstance(ind, (int, slice, list, np.ndarray, tuple)):
            raise ValueError(("Can only index using an `int`, `list`, `ndarray` or `slice`, you supplied a "
                              f"`{type(ind).__name__}`"))

        # work out any extra kwargs we might need to set
        kwargs = self.__dict__
        actual_kwargs = {}
        for key in list(kwargs.keys()):
            if key[0] != "_":
                actual_kwargs[key] = kwargs[key]

        # pre-mask tau to get the length easily
        tau = self.tau[ind]

        new_gal = self.__class__(size=len(tau),
                                 components=self.components, component_masses=self.component_masses,
                                 immediately_sample=False, **actual_kwargs)

        new_gal._tau = tau
        new_gal._Z = self._Z[ind]
        new_gal._positions = self._positions[ind]
        new_gal._which_comp = self._which_comp[ind]

        return new_gal

    @property
    def size(self):
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
        return self._components

    @property
    def component_masses(self):
        return self._component_masses

    @property
    def tau(self):
        if self._tau is None:
            self.sample()
        return self._tau

    @property
    def Z(self):
        if self._Z is None:
            self.sample()
        return self._Z

    @property
    def positions(self):
        if self._positions is None:
            self.sample()
        return self._positions

    @property
    def which_comp(self):
        if self._which_comp is None:
            self.sample()
        return self._which_comp

    def sample(self):
        """Sample from the Galaxy distributions for each component, combine and save in class attributes"""
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

        self._positions = SkyCoord(x=rho * np.sin(phi), y=rho * np.cos(phi), z=z,
                                   frame="galactocentric", representation_type="cartesian")

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

        return self._tau, self._positions, self.Z

    def draw_lookback_times(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def draw_radii(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def draw_heights(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def draw_phi(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def get_metallicity(self):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def plot(self, coordinates="cartesian", component=None, colour_by=None, show=True, cbar_norm=LogNorm(),
             cbar_label=r"Metallicity, $Z$", cmap="plasma", xlim=None, ylim=None, zlim=None, **kwargs):
        fig, axes = plt.subplots(2, 1, figsize=(10 * 1.2475, 14), gridspec_kw={'height_ratios': [4, 14]},
                                 sharex=True)
        if colour_by is None:
            colour_by = self.Z.value

        if coordinates == "cylindrical":
            x = self.positions.represent_as("cylindrical").rho
            y1 = self.positions.represent_as("cylindrical").z
            y2 = self.positions.represent_as("cylindrical").phi
        elif coordinates == "cartesian":
            x = self.positions.x
            y1 = self.positions.z
            y2 = self.positions.y
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

        axes[0].scatter(x, y1, c=colour_by, s=0.1, cmap=cmap, norm=cbar_norm, **kwargs)

        axes[0].set_xlabel(r"$x$ [kpc]", labelpad=15)
        axes[0].xaxis.tick_top()
        axes[0].xaxis.set_label_position("top")

        axes[0].set_ylabel(r"$z$ [kpc]")

        scatt = axes[1].scatter(x.value, y2.value, c=colour_by, s=0.1, cmap=cmap, norm=cbar_norm, **kwargs)
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

    def save(self, file_name, key="galaxy"):
        """Save the entire class to storage.

        Data will be stored in an hdf5 file using `file_name` and a small txt file will be created using the
        same file name.

        Parameters
        ----------
        file_name : `str`
            A name to use for the hdf5 file in which samples will be stored. If this doesn't end in ".h5" then
            ".h5" will be appended.
        key : `str`, optional
            Key to use for the hdf5 file, by default "galaxy"
        """
        # append file extension if necessary
        if file_name[-3:] != ".h5":
            file_name += ".h5"

        # store data in a dataframe and save this to file
        data = {
            "tau": self.tau.to(u.Gyr),
            "Z": self.Z,
            "x": self.positions.x.to(u.kpc),
            "y": self.positions.y.to(u.kpc),
            "z": self.positions.z.to(u.kpc),
            "which_comp": self.which_comp
        }
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
            print((f"Warning: Galaxy class being saved as `{class_name}` instead of "
                   f"`{self.__class__.__name__}`. Data will be copied but new sampling will draw from the "
                   f"functions in `{class_name}` rather than the custom class you used."))
        params["class_name"] = class_name

        # dump it all into a file using yaml
        with open(file_name.replace(".h5", "-galaxy-params.txt"), "w") as file:
            yaml.dump(params, file, default_flow_style=False)


class Frankel2018(Galaxy):
    """A semi-empirical galaxy model based on
    `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_. This model was used in
    detail in `Wagg+2022 <https://ui.adsabs.harvard.edu/abs/2021arXiv211113704W/abstract>`_ -
    see Figure 1 and Section 2.2.1 for a detailed explanation.

    Parameters are the same as :class:`Galaxy` but additionally with the following:

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
        rho = (self.positions.x**2 + self.positions.y**2)**(0.5)
        FeH = self.Fm + self.gradient * rho - (self.Fm + self.gradient * self.Rnow)\
            * (1 - (self._tau / self.galaxy_age))**self.gamma
        return np.power(10, FeH + np.log10(self.zsun))


def load(file_name, key="galaxy"):
    """Load an entire class from storage.

    Data should be stored in an hdf5 file using `file_name` and a small txt file with the
    same file name.

    Parameters
    ----------
    file_name : `str`
        A name of the .h5 file in which samples are stored and .txt file in which parameters are stored
    key : `str`, optional
        Key to use for the hdf5 file, by default "galaxy"
    """
    # append file extension if necessary
    if file_name[-3:] != ".h5":
        file_name += ".h5"

    # load the parameters back in using yaml
    with open(file_name.replace(".h5", "-galaxy-params.txt"), "r") as file:
        params = yaml.load(file.read(), Loader=yaml.Loader)

    # get the current module, get a class using the name, delete it from parameters that will be passed
    module = sys.modules[__name__]

    galaxy_class = getattr(module, params["class_name"])
    del params["class_name"]

    # ensure no samples are taken
    params["immediately_sample"] = False

    # create a new galaxy using the parameters
    galaxy = galaxy_class(**complicate_params(params))

    # read in the data and save it into the class
    df = pd.read_hdf(file_name, key=key)
    galaxy._tau = df["tau"].values * u.Gyr
    galaxy._Z = df["Z"].values * u.dimensionless_unscaled
    galaxy._which_comp = df["which_comp"].values

    galaxy._positions = SkyCoord(x=df["x"].values * u.kpc, y=df["y"].values * u.kpc, z=df["z"].values * u.kpc,
                                 frame="galactocentric", representation_type="cartesian")

    # return the newly created class
    return galaxy


def simplify_params(params, dont_save=["_tau", "_Z", "_positions", "_which_comp", "_v_R", "_v_T", "_v_z"]):
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
