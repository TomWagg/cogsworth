import sys
import yaml
import h5py as h5
import numpy as np
import astropy.units as u
from scipy.integrate import quad
from scipy.special import lambertw
from scipy.stats import beta
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
from astropy.coordinates import SkyCoord

# for action-based potentials
import gala.potential as gp
from gala.units import galactic

from cogsworth.tests.optional_deps import check_dependencies

from cogsworth.citations import CITATIONS


__all__ = ["StarFormationHistory", "Wagg2022", "QuasiIsothermalDisk", "load"]


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


    Attributes
    ----------
    tau : :class:`~astropy.units.Quantity` [time]
        Lookback time
    Z : :class:`~astropy.units.Quantity` [dimensionless]
        Metallicity
    x : :class:`~astropy.units.Quantity` [length]
        Galactocentric x position
    y : :class:`~astropy.units.Quantity` [length]
        Galactocentric y position
    z : :class:`~astropy.units.Quantity` [length]
        Galactocentric z position

    """
    def __init__(self, size, components=None, component_masses=None,
                 immediately_sample=True):
        self._components = components
        self._component_masses = component_masses
        self._size = size
        self._tau = None
        self._Z = None
        self._x = None
        self._y = None
        self._z = None
        self._which_comp = None

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
        for key in list(kwargs.keys()):
            if key[0] != "_" and key not in ["v_R", "v_T", "v_z"]:
                actual_kwargs[key] = kwargs[key]
            elif key.startswith("_component"):
                actual_kwargs[key[1:]] = kwargs[key]

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
        if hasattr(self, "v_R"):
            new_sfh.v_R = np.atleast_1d(self.v_R[ind])
        if hasattr(self, "v_T"):
            new_sfh.v_T = np.atleast_1d(self.v_T[ind])
        if hasattr(self, "v_z"):
            new_sfh.v_z = np.atleast_1d(self.v_z[ind])

        return new_sfh

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
    def x(self):
        if self._x is None:
            self.sample()
        return self._x

    @property
    def y(self):
        if self._y is None:
            self.sample()
        return self._y

    @property
    def z(self):
        if self._z is None:
            self.sample()
        return self._z

    @property
    def rho(self):
        return (self.x**2 + self.y**2)**(0.5)

    @property
    def phi(self):
        return np.arctan2(self.y, self.x)

    @property
    def positions(self):
        return [self.x.to(u.kpc).value, self.y.to(u.kpc).value, self.z.to(u.kpc).value] * u.kpc

    @property
    def which_comp(self):
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
             cbar_label=r"Metallicity, $Z$", cmap="plasma", xlim=None, ylim=None, zlim=None, **kwargs):
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

        if xlim is not None:
            axes[0].set_xlim(xlim)
            axes[1].set_xlim(xlim)
        if ylim is not None:
            axes[1].set_ylim(ylim)
        if zlim is not None:
            axes[0].set_ylim(zlim)

        if show:
            plt.show()

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
            print((f"Warning: StarFormationHistory class being saved as `{class_name}` instead of "
                   f"`{self.__class__.__name__}`. Data will be copied but new sampling will draw from the "
                   f"functions in `{class_name}` rather than the custom class you used."))
        params["class_name"] = class_name

        # dump it all into the file attrs using yaml
        with h5.File(file_name, "a") as file:
            file[key].attrs["params"] = yaml.dump(params, default_flow_style=None)


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


class QuasiIsothermalDisk(StarFormationHistory):      # pragma: no cover
    """A quasi-isothermal distribution function with parameters from
    `Sanders & Binney 2015 <https://ui.adsabs.harvard.edu/abs/2015MNRAS.449.3479S/abstract>`_.

    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    Parameters
    ----------
    galaxy_age : :class:`~astropy.units.Quantity` [time], optional
        Maximum lookback time, by default 12*u.Gyr
    tsfr : :class:`~astropy.units.Quantity` [time], optional
        Star formation timescale, by default 6.8*u.Gyr
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
    def __init__(self, size, tsfr=6.8 * u.Gyr, Fm=-1, gradient=-0.075 / u.kpc, Rnow=8.7 * u.kpc,
                 gamma=0.3, zsun=0.0142, galaxy_age=12 * u.Gyr, **kwargs):
        self.tsfr = tsfr
        self.Fm = Fm
        self.gradient = gradient
        self.Rnow = Rnow
        self.gamma = gamma
        self.zsun = zsun
        self.galaxy_age = galaxy_age

        self._agama_pot = None
        self._df = None

        # ensure we don't pass components twice
        for var in ["components", "component_masses"]:
            if var in kwargs:
                kwargs.pop(var)

        # TODO: Perhaps create a "action-based-potential" subclass, there's a lot repeated here

        super().__init__(size=size, components=None, component_masses=None, **kwargs)

    @property
    def agama_pot(self):
        if self._agama_pot is None:
            self.get_DF()
        return self._agama_pot

    @property
    def df(self):
        if self._df is None:
            self.get_DF()
        return self._df

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

    def get_DF(self):
        """Get the distribution function for a quasi-isothermal disk based on the Gala MW potential"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        # create Gala potential
        gala_pot = gp.MilkyWayPotential2022()

        # convert gala potential components into agama components (TODO: check if this is in Gala yet)
        agama_components = []
        for p in gala_pot["disk"].get_three_potentials().values():
            agama_components.append(
                dict(
                    type="miyamotonagai",
                    mass=p.parameters["m"].value,
                    scaleradius=p.parameters["a"].value,
                    scaleheight=p.parameters["b"].value,
                )
            )
        for k in ["bulge", "nucleus"]:
            p = gala_pot[k]
            agama_components.append(
                dict(
                    type="dehnen",
                    mass=p.parameters["m"].value,
                    scaleradius=p.parameters["c"].value,
                    gamma=1.0,
                )
            )
        p = gala_pot["halo"]
        agama_components.append(
            dict(
                type="nfw", mass=p.parameters["m"].value, scaleradius=p.parameters["r_s"].value
            )
        )

        # use the components to create an agama potential
        self._agama_pot = agama.Potential(*agama_components)

        # additionally work out the distribution function
        self._df = agama.DistributionFunction(
            type='QuasiIsothermal',
            Rdisk=3.45,
            Rsigmar=7.8,
            Rsigmaz=7.8,
            sigmar0=(48.3*u.km/u.s).decompose(galactic).value,
            sigmaz0=(30.7*u.km/u.s).decompose(galactic).value,
            potential=self._agama_pot,
            Sigma0=1.
        )
        return self._df, self._agama_pot

    def sample(self):
        """Sample from the Galaxy distribution and save in class attributes"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        # create an array of which component each point belongs to
        self._which_comp = np.repeat("low_alpha_disc", self.size)
        self._tau = self.draw_lookback_times(size=self.size, component="low_alpha_disc")

        # get cartesian coordinates from agama
        xv, _ = agama.GalaxyModel(self.agama_pot, self.df).sample(self.size)

        # convert units for velocity
        xv[:, 3:] *= (u.kpc / u.Myr).to(u.km / u.s)

        # save the positions
        self._x = xv[:, 0] * u.kpc
        self._y = xv[:, 1] * u.kpc
        self._z = xv[:, 2] * u.kpc

        # work out the velocities by rotating using SkyCoord
        full_coord = SkyCoord(x=self._x, y=self._y, z=self._z,
                              v_x=xv[:, 3] * u.km / u.s, v_y=xv[:, 4] * u.km / u.s, v_z=xv[:, 5] * u.km / u.s,
                              frame="galactocentric").represent_as("cylindrical")

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            self.v_R = full_coord.differentials['s'].d_rho
            self.v_T = (full_coord.differentials['s'].d_phi * full_coord.rho).to(u.km / u.s)
            self.v_z = full_coord.differentials['s'].d_z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

        return self._tau, self.positions, self.Z


class SpheroidalDwarf(StarFormationHistory):      # pragma: no cover
    """An action-based model for dwarf spheroidal galaxies and globular clusters
    `Pascale+2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.488.2423P/abstract>`_.

    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    Parameters
    ----------
    mass : `float`
        Total mass of the galactic potential
    J_0_star : `float`
        The action scale that naturally defines the length- and velocity-scale
    alpha : `float`
        A non-negative, dimensionless parameter that mainly regulates the models density profile
    eta : `float`
        A non-negative, dimensionless parameter that mainly controls the radial or tangential bias of the
        model velocity distribution; models sharing the parameters $(\\alpha, \\eta)$ are homologous.
    galaxy_age : :class:`~astropy.units.Quantity` [time], optional
        Maximum lookback time, by default 12*u.Gyr
    tsfr : :class:`~astropy.units.Quantity` [time], optional
        Star formation timescale, by default 6.8*u.Gyr
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
    def __init__(self, size, mass, J_0_star, alpha, eta, tsfr=6.8 * u.Gyr, Fm=-1, gradient=-0.075 / u.kpc,
                 Rnow=8.7 * u.kpc, gamma=0.3, zsun=0.0142, galaxy_age=12 * u.Gyr, **kwargs):
        self.mass = mass
        self.J_0_star = J_0_star
        self.alpha = alpha
        self.eta = eta
        self.tsfr = tsfr
        self.Fm = Fm
        self.gradient = gradient
        self.Rnow = Rnow
        self.gamma = gamma
        self.zsun = zsun
        self.galaxy_age = galaxy_age

        self._agama_pot = None
        self._df = None

        # ensure we don't pass components twice
        for var in ["components", "component_masses"]:
            if var in kwargs:
                kwargs.pop(var)

        super().__init__(size=size, components=None, component_masses=None, **kwargs)

    @property
    def agama_pot(self):
        if self._agama_pot is None:
            self.get_DF()
        return self._agama_pot

    @property
    def df(self):
        if self._df is None:
            self.get_DF()
        return self._df

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

    def get_DF(self):
        """Get the distribution function for a dwarf galaxy disk based on an NFW profile"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        gala_pot = gp.NFWPotential(m=self.mass, r_s=1.0, units=galactic)
        self._agama_pot = agama.Potential(
            type="nfw",
            mass=gala_pot.parameters["m"].decompose(galactic).value,
            scaleradius=gala_pot.parameters["r_s"].decompose(galactic).value,
        )

        J0_no_units = (self.J_0_star).decompose(galactic).value

        def dwarf_df(J):
            Jr, Jz, Jphi = J.T
            kJ = Jr + self.eta * (np.abs(Jphi) + Jz)
            return np.exp(-(kJ / J0_no_units)**self.alpha)
        self._df = dwarf_df
        return self._df, self._agama_pot

    def sample(self):
        """Sample from the Galaxy distribution and save in class attributes"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        # create an array of which component each point belongs to
        self._which_comp = np.repeat("low_alpha_disc", self.size)
        self._tau = self.draw_lookback_times(size=self.size, component="low_alpha_disc")

        # get cartesian coordinates from agama
        xv, _ = agama.GalaxyModel(self.agama_pot, self.df).sample(self.size)

        # convert units for velocity
        xv[:, 3:] *= (u.kpc / u.Myr).to(u.km / u.s)

        # save the positions
        self._x = xv[:, 0] * u.kpc
        self._y = xv[:, 1] * u.kpc
        self._z = xv[:, 2] * u.kpc

        # work out the velocities by rotating using SkyCoord
        full_coord = SkyCoord(x=self._x, y=self._y, z=self._z,
                              v_x=xv[:, 3] * u.km / u.s, v_y=xv[:, 4] * u.km / u.s, v_z=xv[:, 5] * u.km / u.s,
                              frame="galactocentric").represent_as("cylindrical")

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            self.v_R = full_coord.differentials['s'].d_rho
            self.v_T = (full_coord.differentials['s'].d_phi * full_coord.rho).to(u.km / u.s)
            self.v_z = full_coord.differentials['s'].d_z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

        return self._tau, self.positions, self.Z


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

    # load the parameters back in using yaml
    with h5.File(file_name, "r") as file:
        if key not in file.keys():
            raise ValueError((f"Can't find a saved SFH in {file_name} under the key {key}."))
        params = yaml.load(file[key].attrs["params"], Loader=yaml.Loader)

    # get the current module, get a class using the name, delete it from parameters that will be passed
    module = sys.modules[__name__]

    sfh_class = getattr(module, params["class_name"])
    del params["class_name"]

    # ensure no samples are taken
    params["immediately_sample"] = False

    # create a new sfh using the parameters
    loaded_sfh = sfh_class(**complicate_params(params))

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
                                       "_df", "_agama_pot", "__citations__"]):
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
