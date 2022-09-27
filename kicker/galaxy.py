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


class Galaxy():
    """Class for a generic galaxy model from which to sample

    This class sets out an outline for sampling from a galaxy model but a subclass will be needed for things
    to function properly.

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
    R_sun : `float`, optional
        Distance of the Sun from the Galactic centre, used for calculating distances, by default 8.2*u.kpc


    Attributes
    ----------
    Each attribute listed below is a given value for the sampled points in the galaxy. If it hasn't been
    sampled/calculated when accessed then it will be automatically sampled/calculated. If sampling, ALL values
    will be sampled.

    tau : `np.array`
        Lookback time
    Z : `np.array`
        Metallicity
    z : `np.array`
        Galactocentric height
    rho : `np.array`
        Galactocentric radius
    phi : `np.array`
        Galactocentric azimuthal angle relative to Sun
    Dg : `np.array`
        Distance from Galactic centre
    D : `np.array`
        Distance from Sun
    x : `np.array`
        Galactocentric distance along x-axis (parallel with vector pointing at Sun)
    y : `np.array`
        Galactocentric distance along y-axis (perpendicular with vector pointing at Sun)

    """
    def __init__(self, size=None, components=None, component_masses=None,
                 immediately_sample=True, R_sun=8.2 * u.kpc):
        self._components = components
        self._component_masses = component_masses
        self._size = size
        self._R_sun = R_sun
        self._tau = None
        self._Z = None
        self._z = None
        self._rho = None
        self._phi = None
        self._Dg = None
        self._D = None
        self._x = None
        self._y = None

        if immediately_sample:
            self.sample()

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if value <= 0:
            raise ValueError("Size must be greater than 0")
        if not isinstance(value, int):
            raise ValueError("Size must be an integer")

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
    def rho(self):
        if self._rho is None:
            self.sample()
        return self._rho

    @property
    def z(self):
        if self._z is None:
            self.sample()
        return self._z

    @property
    def phi(self):
        if self._phi is None:
            self.sample()
        return self._phi

    @property
    def Z(self):
        if self._Z is None:
            self.sample()
        return self._Z

    @property
    def D(self):
        if self._D is None:
            self._D = np.sqrt(self._z**2 + self._rho**2 + self._R_sun**2
                              - 2 * self._R_sun * self._rho * np.cos(self._phi))
        return self._D

    @property
    def Dg(self):
        if self._Dg is None:
            self._Dg = (self._rho**2 + self._z**2)**(0.5)
        return self._Dg

    @property
    def x(self):
        if self._x is None:
            self._x = self._rho * np.cos(self._phi)
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._y = self._rho * np.sin(self._phi)
        return self._y

    @property
    def which_comp(self):
        if self._which_comp is None:
            self.sample()
        return self._which_comp

    def sample(self):
        """Sample from the Galaxy distributions for each component, combine and save in class attributes"""
        if self.size is None:
            raise ValueError("`self.size` has not been set")

        # reset calculated values
        self._D = None
        self._rho = None

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
        self._rho = np.zeros(self._size) * u.kpc
        self._z = np.zeros(self._size) * u.kpc

        # go through each component and get lookback time, radius and height
        for i, com in enumerate(self._components):
            com_mask = self._which_comp == com
            self._tau[com_mask] = self.draw_lookback_times(sizes[i], component=com)
            self._rho[com_mask] = self.draw_radii(sizes[i], component=com)
            self._z[com_mask] = self.draw_heights(sizes[i], component=com)

        # shuffle the samples so components are well mixed (mostly for plotting)
        random_order = np.random.permutation(self._size)
        self._tau = self._tau[random_order]
        self._rho = self._rho[random_order]
        self._z = self._z[random_order]
        self._which_comp = self._which_comp[random_order]

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

        # draw a random azimuthal angle
        self._phi = self.draw_phi()
        return self._tau, (self._rho, self.z, self.phi), self.Z

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
             cbar_label=r"Metallicity, $Z$", cmap="plasma", **kwargs):
        fig, axes = plt.subplots(2, 1, figsize=(10 * 1.2475, 14), gridspec_kw={'height_ratios': [4, 14]},
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
            "z": self.z.to(u.kpc),
            "rho": self.rho.to(u.kpc),
            "phi": self.phi.to(u.rad),
            "Dg": self.Dg.to(u.kpc),
            "D": self.D.to(u.kpc),
            "x": self.x.to(u.kpc),
            "y": self.y.to(u.kpc),
            "which_comp": self.which_comp
        }
        df = pd.DataFrame(data=data)
        df.to_hdf(file_name, key=key)

        # convert parameters into something storable and append the name of the class
        params = simplify_params(self.__dict__.copy())
        params["class_name"] = self.__class__.__name__

        # dump it all into a file using yaml
        with open(file_name.replace(".h5", "-galaxy-params.txt"), "w") as file:
            yaml.dump(params, file, default_flow_style=False)


class Frankel2018(Galaxy):
    """A semi-empirical galaxy model based on Frankel+2018. This model was used in detail in Wagg+2022 -
    see Figure 1 and Section 2.2.1 for a detailed explanation.

    Attributes
    ----------
    As `Galaxy` but additionally the following:

    galaxy_age : `float`, optional
        Maximum lookback time, by default 12*u.Gyr
    tsfr : `float`, optional
        Star formation timescale, by default 6.8*u.Gyr
    alpha : `float`, optional
        Disc inside-out growth parameter, by default 0.3
    Fm : `int`, optional
        Metallicity at centre of disc at tm, by default -1
    gradient : `float`, optional
        Metallicity gradient, by default -0.075/u.kpc
    Rnow : `float`, optional
        Radius at which present day metallicity is solar, by default 8.7*u.kpc
    gamma : `float`, optional
        Time dependence of chemical enrichment, by default 0.3
    zsun : `float`, optional
        Solar metallicity, by default 0.0142
    """
    def __init__(self, components=["low_alpha_disc", "high_alpha_disc", "bulge"],
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
        super().__init__(components=components, component_masses=component_masses, **kwargs)

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        """Inverse CDF sampling of lookback times. low_alpha and high_alpha discs uses Frankel+2018 Eq.4,
        separated and normalised at 8 Gyr. The bulge matches the distribution in Fig.7 of Bovy+19 but accounts
        for sample's bias.

        Parameters
        ----------
        size : `int`
            How many times to draw
        component: `str`
            Which component of the Milky Way

        Returns
        -------
        tau: `float/array`
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
        """Inverse CDF sampling of galactocentric radii using Frankel+2018 Eq.5. The scale length is
        calculated using Eq. 6 the low_alpha disc and is fixed at 1/0.43 and 1.5 kpc respectively for the
        high_alpha disc and bulge.

        Parameters
        ----------
        size : `int`
            How many radii to draw
        component: `str`
            Which component of the Milky Way

        Returns
        -------
        rho: `float/array`
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
        """Inverse CDF sampling of heights using McMillan 2011 Eq. 3 and various scale lengths.

        Parameters
        ----------
        size : `int`
            How many heights to draw
        component: `str`
            Which component of the Milky Way

        Returns
        -------
        z: `float/array`
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
        phi : `np.array`
            Azimuthal angles
        """
        # if no size is given then use the class value
        size = self._size if size is None else size
        return np.random.uniform(0, 2 * np.pi, size) * u.rad

    def get_metallicity(self):
        """Convert radius and time to metallicity using Frankel+2018 Eq.7 and Bertelli+1994 Eq.9

        Returns
        -------
        Z: `float/array`
            Metallicities corresponding to radii and times
        """
        FeH = self.Fm + self.gradient * self._rho - (self.Fm + self.gradient * self.Rnow)\
            * (1 - (self._tau / self.galaxy_age))**self.gamma
        return np.power(10, 0.977 * FeH + np.log10(self.zsun))


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
    galaxy._z = df["z"].values * u.kpc
    galaxy._rho = df["rho"].values * u.kpc
    galaxy._phi = df["phi"].values * u.rad
    galaxy._Dg = df["Dg"].values * u.kpc
    galaxy._D = df["D"].values * u.kpc
    galaxy._x = df["x"].values * u.kpc
    galaxy._y = df["y"].values * u.kpc
    galaxy._which_comp = df["which_comp"].values

    # return the newly created class
    return galaxy


def simplify_params(params, dont_save=["_tau", "_Z", "_z", "_rho", "_phi", "_Dg",
                                       "_D", "_x", "_y", "_which_comp", "v_R", "v_T", "v_z"]):
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
