import numpy as np
import astropy.units as u
from scipy.integrate import quad
from scipy.special import lambertw
from scipy.stats import beta


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
    R : `np.array`
        Galactocentric radius
    theta : `np.array`
        Galactocentric azimuthal angle relative to Sun
    rho : `np.array`
        Distance from Galactic centre
    D : `np.array`
        Distance from Sun
    x : `np.array`
        Galactocentric distance along x-axis (parallel with vector pointing at Sun)
    y : `np.array`
        Galactocentric distance along y-axis (perpendicular with vector pointing at Sun)

    """
    def __init__(self, size, components=None, component_masses=None,
                 immediately_sample=True, R_sun=8.2 * u.kpc):
        self._components = components
        self._component_masses = component_masses
        self._size = size
        self._R_sun = R_sun
        self._tau = None
        self._Z = None
        self._z = None
        self._R = None
        self._theta = None
        self._rho = None
        self._D = None
        self._x = None
        self._y = None

        if immediately_sample:
            self.sample()

    @property
    def size(self):
        return self._size

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
    def R(self):
        if self._R is None:
            self.sample()
        return self._R

    @property
    def z(self):
        if self._z is None:
            self.sample()
        return self._z

    @property
    def theta(self):
        if self._theta is None:
            self.sample()
        return self._theta

    @property
    def Z(self):
        if self._Z is None:
            self.sample()
        return self._Z

    @property
    def D(self):
        if self._D is None:
            self._D = np.sqrt(self._z**2 + self._R**2 + self._R_sun**2
                              - 2 * self._R_sun * self._R * np.cos(self._theta))
        return self._D

    @property
    def rho(self):
        if self._rho is None:
            self._rho = (self._R**2 + self._z**2)**(0.5)
        return self._rho

    @property
    def x(self):
        if self._x is None:
            self._x = self._R * np.cos(self._theta)
        return self._x

    @property
    def y(self):
        if self._y is None:
            self._y = self._R * np.sin(self._theta)
        return self._y

    def sample(self):
        """Sample from the Galaxy distributions for each component, combine and save in class attributes"""
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
        self._R = np.zeros(self._size) * u.kpc
        self._z = np.zeros(self._size) * u.kpc

        # go through each component and get lookback time, radius and height
        for i, com in enumerate(self._components):
            com_mask = self._which_comp == com
            self._tau[com_mask] = self.draw_lookback_times(sizes[i], component=com)
            self._R[com_mask] = self.draw_radii(sizes[i], component=com)
            self._z[com_mask] = self.draw_heights(sizes[i], component=com)

        # shuffle the samples so components are well mixed (mostly for plotting)
        random_order = np.random.permutation(self._size)
        self._tau = self._tau[random_order]
        self._R = self._R[random_order]
        self._z = self._z[random_order]
        self._which_comp = self._which_comp[random_order]

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

        # draw a random azimuthal angle
        self._theta = self.draw_theta()
        return self._tau, (self._R, self.z, self.theta), self.Z

    def draw_lookback_times(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def draw_radii(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def draw_heights(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def draw_theta(self, size=None, component=None):
        raise NotImplementedError("This Galaxy model has not implemented this method")

    def get_metallicity(self):
        raise NotImplementedError("This Galaxy model has not implemented this method")


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
        R: `float/array`
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
        R = - R_0 * (lambertw((U - 1) / np.exp(1), k=-1).real + 1)
        return R

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

    def draw_theta(self, size=None):
        """Draw random azimuthal angles

        Parameters
        ----------
        size : `int`, optional
            How many angles to draw, by default self._size

        Returns
        -------
        theta : `np.array`
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
        FeH = self.Fm + self.gradient * self._R - (self.Fm + self.gradient * self.Rnow)\
            * (1 - (self._tau / self.galaxy_age))**self.gamma
        return np.power(10, 0.977 * FeH + np.log10(self.zsun))
