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
import pandas as pd
from astropy.coordinates import SkyCoord
import logging

from types import FunctionType

# for action-based potentials
import gala.potential as gp
from gala.units import galactic
from gala.potential.potential.io import to_dict as potential_to_dict, from_dict as potential_from_dict

from cogsworth.utils import check_dependencies
from cogsworth.plot import plot_sfh
from cogsworth.citations import CITATIONS


__all__ = ["StarFormationHistory", "CompositeStarFormationHistory",
           "DistributionFunctionBasedSFH", "Wagg2022",
           "Frankel2018SFH", "LowAlphaDiscWagg2022", "HighAlphaDiscWagg2022", "BulgeWagg2022",
           "MilkyWayBarSormani2022",
           "BurstUniformDisc", "ConstantUniformDisc", "ConstantPlummerSphere",
           "SandersBinney2015", "SpheroidalDwarf", "CarinaDwarf", "load", "concat"]


def _exponential_disc(size, scale_height):
    """Inverse CDF sampling of heights using
    `McMillan 2011 <https://ui.adsabs.harvard.edu/abs/2011MNRAS.414.2446M/abstract>`_ Eq. 3
    and various scale lengths.

    Parameters
    ----------
    size : `int`
        How many heights to draw
    scale_height: :class:`~astropy.units.Quantity` [length]
        Scale height for the exponential disc

    Returns
    -------
    z : :class:`~astropy.units.Quantity` [length]
        Random heights
    """
    return np.random.choice([-1, 1], size) * scale_height * np.log(1 - np.random.rand(size))

def _frankel2018_metallicity_relation(sfh):
    """Convert radius and time to metallicity using
    `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 7 and
    `Bertelli+1994 <https://ui.adsabs.harvard.edu/abs/1994A%26AS..106..275B/abstract>`_ Eq. 9 but
    assuming all stars have the solar abundance pattern (so no factor of 0.977)

    Parameters
    ----------
    sfh : :class:`~cogsworth.sfh.StarFormationHistory`
        The star formation history for which to calculate the metallicity relation
    
    Returns
    -------
    Z : :class:`~astropy.units.Quantity` [dimensionless]
        Metallicities corresponding to radii and times
    """
    FeH = (
        sfh.Fm + sfh.gradient * sfh.rho - (sfh.Fm + sfh.gradient * sfh.Rnow) *
        (1 - (sfh._tau / sfh.galaxy_age))**sfh.gamma
    )
    return np.power(10, FeH + np.log10(sfh.zsun))

class StarFormationHistory():
    """Class for a generic galactic star formation history model from which to sample

    This class sets out an outline for sampling from a star formation history model but a subclass will be
    needed for things to function properly.

    All attributes listed below are a given value for the sampled points in the galaxy. If one hasn't been
    sampled/calculated when accessed then it will be automatically sampled/calculated. If sampling, ALL values
    will be sampled.
    """
    def __init__(self, **kwargs):
        self._tau = None
        self._Z = None
        
        self._x = None
        self._y = None
        self._z = None

        self._v_R = None
        self._v_T = None
        self._v_z = None

        self._v_x = None
        self._v_y = None

        self._composite_weight = 1.0

        # check for any extra parameters that have been passed
        # this may occur when loading from a file and the user was writing a custom class
        if len(kwargs) > 0:
            for key in kwargs:
                setattr(self, key, kwargs[key])

        self.__citations__ = ["cogsworth"]

    def __len__(self):
        if self._tau is not None:
            return len(self._tau)
        else:
            return 0

    def __repr__(self):
        if self._tau is not None:
            return f"<{self.__class__.__name__}, size={len(self)}>"
        else:
            return f"<{self.__class__.__name__}, [not yet sampled]>"

    def __add__(self, other):
        if isinstance(other, StarFormationHistory):
            return CompositeStarFormationHistory(
                components=[self, other],
                component_ratios=[self._composite_weight, other._composite_weight]
            )
        elif isinstance(other, CompositeStarFormationHistory):
            return CompositeStarFormationHistory(
                components=[self] + other.components,
                component_ratios=np.concatenate([[self._composite_weight], other.component_ratios])
            )
        else:
            return NotImplemented
    
    def __mul__(self, other):
        if not isinstance(other, (int, float)):
            return NotImplemented
        new_sfh = self.copy()
        new_sfh._composite_weight *= other
        return new_sfh
    
    def __rmul__(self, other):
        return self.__mul__(other)

    def __getitem__(self, ind):
        # ensure indexing with the right type
        if not isinstance(ind, (int, slice, list, np.ndarray, tuple)):
            raise ValueError(("Can only index using an `int`, `list`, `ndarray` or `slice`, you supplied a "
                              f"`{type(ind).__name__}`"))
        
        # if no sampling has occurred, then indexing is trivial and we just return a copy of the class
        if self._tau is None:
            return self.__class__(**self.__dict__)

        # work out any extra kwargs we might need to set
        kwargs = self.__dict__
        actual_kwargs = {}
        saved_attributes = {}
        array_attributes = ["_tau", "_Z", "_x", "_y", "_z", "_v_R", "_v_T", "_v_z", "_v_x", "_v_y"]
        for key in list(kwargs.keys()):
            # only keep attributes that have no underscores and aren't velocity components
            if key[0] != "_" and key not in array_attributes:
                actual_kwargs[key] = kwargs[key]
            # save other attributes for later
            elif key not in array_attributes and key != "_size":
                saved_attributes[key] = kwargs[key]

        # pre-mask tau to get the length easily
        tau = np.atleast_1d(self.tau[ind])

        new_sfh = self.__class__(**actual_kwargs)

        new_sfh._tau = tau
        new_sfh._Z = np.atleast_1d(self._Z[ind])
        new_sfh._x = np.atleast_1d(self._x[ind])
        new_sfh._y = np.atleast_1d(self._y[ind])
        new_sfh._z = np.atleast_1d(self._z[ind])

        # if we have any of the velocity components then we need to slice them too
        vel_comps = ["_v_R", "_v_T", "_v_z", "_v_x", "_v_y"]
        for vel in vel_comps:
            if hasattr(self, vel) and getattr(self, vel) is not None:
                setattr(new_sfh, vel, np.atleast_1d(getattr(self, vel)[ind]))

        for attr in saved_attributes:
            setattr(new_sfh, attr, saved_attributes[attr])

        return new_sfh
    
    def copy(self):
        """Return a copy of this star formation history"""
        return self[:]

    @property
    def tau(self):
        """The lookback times of the sampled points

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            The lookback times of the sampled points
        """
        if self._tau is None:
            raise ValueError("Star formation history has not yet been sampled")
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
            raise ValueError("Star formation history has not yet been sampled")
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
            raise ValueError("Star formation history has not yet been sampled")
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
            raise ValueError("Star formation history has not yet been sampled")
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
            raise ValueError("Star formation history has not yet been sampled")
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
    def v_R(self):
        r"""The galactocentric radial velocity of the sampled points"""
        if self._v_R is None:
            raise ValueError("This star formation history model does not have radial velocities sampled.")
        return self._v_R
    
    @property
    def v_T(self):
        r"""The galactocentric tangential velocity of the sampled points"""
        if self._v_T is None:
            raise ValueError("This star formation history model does not have tangential velocities sampled.")
        return self._v_T
    
    @property
    def v_z(self):
        r"""The galactocentric vertical velocity of the sampled points"""
        if self._v_z is None:
            raise ValueError("This star formation history model does not have vertical velocities sampled.")
        return self._v_z
    
    @property
    def v_phi(self):
        r"""The galactocentric azimuthal velocity of the sampled points

        A shortcut for :math:`(v_T / \rho)`
        """
        return self.v_T / self.rho
    
    @property
    def v_x(self):
        r"""The galactocentric x velocity of the sampled points
        
        A shortcut for :math:`v_R \cos(\phi) - v_T \sin(\phi)`
        """
        if self._v_x is not None:
            return self._v_x
        return self.v_R * np.cos(self.phi) - self.v_T * np.sin(self.phi)
    
    @property
    def v_y(self):
        r"""The galactocentric y velocity of the sampled points
        
        A shortcut for :math:`v_R \sin(\phi) + v_T \cos(\phi)`
        """
        if self._v_y is not None:
            return self._v_y
        return self.v_R * np.sin(self.phi) + self.v_T * np.cos(self.phi)
    
    @property
    def velocities(self):
        return [self.v_x.to(u.km / u.s).value,
                self.v_y.to(u.km / u.s).value,
                self.v_z.to(u.km / u.s).value] * (u.km / u.s)

    @staticmethod
    def sfh_citation_statement(citations, filename=None):
        """Print the citations for the packages/papers used in the star formation history

        Parameters
        ----------
        citations : `list` of `str`
            The list of citations to include in the statement, these should be keys in the `CITATIONS` dict
        filename : `str`, optional
            Filename for generating a bibtex file (leave blank to just print to terminal), by default None
        """
        # ask users for a filename to save the bibtex to
        if filename is None:            # pragma: no cover
            filename = input("Filename for generating a bibtex file (leave blank to just print to terminal): ")

        # construct citation string
        cite_tags = []
        bibtex = []
        for section in CITATIONS:
            for citation in citations:
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

    def get_citations(self, filename=None):
        """Print the citations for the packages/papers used in the star formation history"""
        if not hasattr(self, "__citations__") or len(self.__citations__) == 0:          # pragma: no cover
            print("No citations needed for this star formation history model")
            return
        
        self.sfh_citation_statement(self.__citations__, filename=filename)

    def sample(self, size):
        """Sample from the distributions for each component, combine and save in class attributes"""
        # erase any existing samples
        for attr in ["_tau", "_Z", "_x", "_y", "_z", "_v_R", "_v_T", "_v_z", "_v_x", "_v_y"]:
            setattr(self, attr, None)

        self._tau = np.zeros(size) * u.Gyr
        rho = np.zeros(size) * u.kpc
        z = np.zeros(size) * u.kpc

        # get lookback time, radius and height
        self._tau = self.draw_lookback_times(size)
        rho = self.draw_radii(size)
        z = self.draw_heights(size)

        # draw a random azimuthal angle
        phi = self.draw_phi(size)

        # set cartesian values
        self._x = rho * np.sin(phi)
        self._y = rho * np.cos(phi)
        self._z = z

        # compute the metallicity given the other values
        self._Z = self.get_metallicity()

    def draw_lookback_times(self, size):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def draw_radii(self, size):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def draw_heights(self, size):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def draw_phi(self, size):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def get_metallicity(self):
        raise NotImplementedError("This StarFormationHistory model has not implemented this method")

    def plot(self, **kwargs):
        """Plot the star formation history using the default plotting function

        See :func:`~cogsworth.plotting.plot_sfh` for more details and options.
        """
        return plot_sfh(self, **kwargs)

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
            "z": self.z.to(u.kpc)
        }

        # additionally store velocity components if they exist
        for attr in ["_v_R", "_v_T", "_v_z", "_v_x", "_v_y"]:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                data[attr] = getattr(self, attr).to(u.km / u.s)

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



class CompositeStarFormationHistory():
    """A star formation history that is a combination of multiple other star formation histories.

    This class allows you to combine multiple star formation histories together, for example to create a
    composite star formation history for the Milky Way with a disc and a bulge. You can also use this to
    combine multiple instances of the same star formation history, for example to create a composite star
    formation history for the Milky Way with a thin and thick disc.

    Parameters
    ----------
    components : `list` of :class:`~StarFormationHistory`
        The star formation histories to combine. These will be sampled in proportion to their size.
    """
    def __init__(self, components, component_ratios, **kwargs):
        self.components = components

        # normalise component ratios to sum to 1
        component_ratios = np.array(component_ratios)
        component_ratios /= component_ratios.sum()

        self.component_ratios = component_ratios
        self.__citations__ = list(set(
            citation for component in self.components for citation in component.__citations__
        ))

        self._sort_order = None

    @classmethod
    def from_file(cls, file_name, key="sfh"):
        """Load a composite star formation history from file

        Parameters
        ----------
        file_name : `str`
            The name of the hdf5 file from which to load the star formation history. If this doesn't end in ".h5"
            then ".h5" will be appended.
        key : `str`, optional
            The key under which the data is stored in the hdf5 file, by default "sfh"

        Returns
        -------
        sfh : :class:`~CompositeStarFormationHistory`
            The loaded composite star formation history
        """
        n_components = 0
        component_ratios = []
        with h5.File(file_name, "r") as file:
            while f"{key}_{n_components}" in file:
                n_components += 1
                component_ratios.append(file[f"{key}_{n_components - 1}"].attrs.get("component_ratio", 1.0))

        return cls(
            components=[load(file_name, key=f"{key}_{i}") for i in range(n_components)],
            component_ratios=component_ratios
        )


    def __getitem__(self, ind):
        if not isinstance(ind, (int, slice, list, np.ndarray, tuple)):
            raise ValueError(("Can only index using an `int`, `list`, `ndarray` or `slice`, you supplied a "
                              f"`{type(ind).__name__}`"))
        
        # first we turn `ind` into an array of indices if it isn't already
        if isinstance(ind, int):
            ind = np.array([ind])
        elif isinstance(ind, slice):
            ind = np.arange(len(self))[ind]
        elif isinstance(ind, (list, tuple)):
            ind = np.array(ind)

        # if it's a boolean array, we convert it to indices
        if ind.dtype == bool:
            ind = np.where(ind)[0]

        # if this object already has a sort order (from a prior indexing), translate logical indices
        # to internal component indices so subsequent indexing composes correctly
        if self._sort_order is not None:
            ind = self._sort_order[ind]

        # get unique sorted indices and the inverse mapping such that unique_ind[inverse] == ind
        # this handles both unsorted indices and duplicates in one step
        unique_ind, inverse = np.unique(ind, return_inverse=True)

        # now we need to figure out which component each index is in, some components may have nothing with
        # an index and will be left as length 0 components
        component_ind_ranges = np.cumsum([0] + [len(component) for component in self.components])

        new_components = []
        for i in range(len(self.components)):
            component_mask = (unique_ind >= component_ind_ranges[i]) & (unique_ind < component_ind_ranges[i + 1])
            component_inds = unique_ind[component_mask] - component_ind_ranges[i]
            new_components.append(self.components[i][component_inds])

        new_csfh = CompositeStarFormationHistory(
            components=new_components,
            component_ratios=self.component_ratios
        )

        # store the inverse mapping so __getattr__ can reorder/duplicate values to match the requested ind
        if not np.array_equal(inverse, np.arange(len(ind))):
            new_csfh._sort_order = inverse

        return new_csfh

    def __getattr__(self, name):
        """When we try to access an attribute, if it's one that needs combining from the components"""
        COMBINE_ATTRS = ["tau", "Z", "x", "y", "z", "phi", "rho",
                         "v_x", "v_y", "v_z", "v_R", "v_T", "v_phi"]
        if name in COMBINE_ATTRS or (name[0] == "_" and name[1:] in COMBINE_ATTRS):
            component_vals = [getattr(component, name) for component in self.components]
            if all(val is not None for val in component_vals):
                result = np.concatenate(component_vals)
                if self._sort_order is not None:
                    return result[self._sort_order]
                return result
            else:
                return None
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        
    def __setattr__(self, name, value):
        """When we try to set an attribute, if it's one that needs combining from the components then we set it on the
        relevant component instead"""
        COMBINE_ATTRS = ["_tau", "_Z", "_x", "_y", "_z", "_v_x", "_v_y", "_v_z", "_v_R", "_v_T"]
        if name in COMBINE_ATTRS:
            if not isinstance(value, (np.ndarray, list, int, float)):
                raise ValueError(f"Can only set attribute `{name}` using an `int`, `float`, `list` or `ndarray`, you supplied a `{type(value).__name__}`")
            if isinstance(value, list):
                value = np.array(value)

            # if there's a sort order, undo it so values are in internal component order
            if self._sort_order is not None:
                _, first_occurrence = np.unique(self._sort_order, return_index=True)
                value = value[first_occurrence]

            # update the attribute in each individual component
            n_internal = sum(len(c) for c in self.components)
            all_indices = np.arange(n_internal)
            component_ind_ranges = np.cumsum([0] + [len(component) for component in self.components])

            for i in range(len(self.components)):
                component_mask = (all_indices >= component_ind_ranges[i]) & (all_indices < component_ind_ranges[i + 1])
                component_vals = value[component_mask]
                setattr(self.components[i], name, component_vals)
        else:
            super().__setattr__(name, value)
        
    def __len__(self):
        # check if the first component has any ._tau samples yet
        if self.components[0]._tau is None:
            return 0
        
        # otherwise should be safe to count total length of tau
        return len(self.tau)
    
    def __repr__(self):
        component_string = ', '.join([f"{component.__class__.__name__} ({ratio:.0%})"
                                      for component, ratio in zip(self.components, self.component_ratios)])
        length_str = f"size={len(self)}" if len(self) > 0 else "[not yet sampled]"
        return (
            f"<{self.__class__.__name__}, {length_str}, {len(self.components)} "
            f"components | {component_string}>"
        )
    
    def __add__(self, other):
        if isinstance(other, StarFormationHistory):
            return CompositeStarFormationHistory(
                components=self.components + [other],
                component_ratios=np.concatenate([self.component_ratios, [other._composite_weight]])
            )
        elif isinstance(other, CompositeStarFormationHistory):
            return CompositeStarFormationHistory(
                components=self.components + other.components,
                component_ratios=np.concatenate([self.component_ratios, other.component_ratios])
            )
        else:
            return NotImplemented
        
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
    def velocities(self):
        """The galactocentric velocities of the sampled points

        Returns
        -------
        velocities : :class:`~astropy.units.Quantity` [velocity], shape=(3, :attr:`~size`)
            The galactocentric velocities of the sampled points
        """
        return [self.v_x.to(u.km / u.s).value,
                self.v_y.to(u.km / u.s).value,
                self.v_z.to(u.km / u.s).value] * (u.km / u.s)

    def sample(self, size):
        """Sample from the distributions for each component, combine and save in class attributes"""

        # convert the component ratios to a number of binaries
        sizes = np.zeros(len(self.component_ratios)).astype(int)
        for i in range(len(self.components) - 1):
            sizes[i] = np.round(self.component_ratios[i] * size)
        sizes[-1] = size - np.sum(sizes)

        for i, component in enumerate(self.components):
            component.sample(sizes[i])

    def plot(self, **kwargs):
        """Plot the star formation history using the default plotting function

        See :func:`~cogsworth.plotting.plot_sfh` for more details and options.
        """
        return plot_sfh(self, **kwargs)

    def save(self, file_name, key="sfh"):
        """Save the entire class to storage.

        Data will be stored in an hdf5 file using `file_name`.

        Parameters
        ----------
        file_name : `str`
        key : `str`, optional
            The key under which to store the data in the hdf5 file, by default "sfh"
        """
        # append file extension if necessary
        if file_name[-3:] != ".h5":
            file_name += ".h5"

        for i, component in enumerate(self.components):
            component.save(file_name, key=f"{key}_{i}")

        with h5.File(file_name, "a") as file:
            for i, cr in enumerate(self.component_ratios):
                file[f"{key}_{i}"].attrs["component_ratio"] = cr

    def get_citations(self, filename=None):
        """Print the citations for the packages/papers used in the star formation history"""
        if not hasattr(self, "__citations__") or len(self.__citations__) == 0:      # pragma: no cover
            print("No citations needed for this star formation history model")
            return

        StarFormationHistory.sfh_citation_statement(self.__citations__, filename=filename)

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
    def __init__(self, t_burst=12 * u.Gyr, z_max=2 * u.kpc, R_max=15 * u.kpc, Z_all=0.02, **kwargs):
        self.t_burst = t_burst
        self.z_max = z_max
        self.R_max = R_max
        self.Z_all = Z_all
        super().__init__(**kwargs)

    def draw_lookback_times(self, size):
        return np.repeat(self.t_burst.value, size) * self.t_burst.unit

    def draw_radii(self, size):
        return np.random.uniform(0, self.R_max.value**2, size)**(0.5) * self.R_max.unit

    def draw_heights(self, size):
        return np.random.uniform(-self.z_max.value, self.z_max.value, size) * self.z_max.unit

    def draw_phi(self, size):
        # if no size is given then use the class value
        size = self._size if size is None else size
        return np.random.uniform(0, 2 * np.pi, size) * u.rad

    def get_metallicity(self):
        return np.repeat(self.Z_all, len(self)) * u.dimensionless_unscaled


class ConstantUniformDisc(BurstUniformDisc):
    """A simple star formation history, with all stars formed at a constant rate between ``t_burst`` 
    and the present day in a uniform disc with height ``z_max`` and radius ``R_max`` disc, all with
    metallicity ``Z``.

    Based on :class:`BurstUniformDisc`.
    """
    def draw_lookback_times(self, size):
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
    def __init__(self, tau_min, tau_max, Z_all, M, a, r_trunc=None, **kwargs):
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.Z_all = Z_all
        self.a = a
        self.M = M
        self.r_trunc = r_trunc
        super().__init__(**kwargs)

    def draw_lookback_times(self, size):
        """Draw lookback times uniformly between tau_min and tau_max"""
        return np.random.uniform(self.tau_min.to(u.Gyr).value, self.tau_max.to(u.Gyr).value, size) * u.Gyr

    def get_metallicity(self, size):
        """Fix all metallicities to Z_all"""
        return np.repeat(self.Z_all, size) * u.dimensionless_unscaled

    def sample(self, size):
        # sample times
        self._tau = self.draw_lookback_times(size)

        # sample positions in a Plummer sphere
        u_max = 1.0 if self.r_trunc is None else self.r_trunc**3 / (self.r_trunc**2 + self.a**2)**1.5
        u_rand = np.random.uniform(0, u_max, size)
        r = self.a * (u_rand**(-2/3) - 1.0)**(-0.5)

        # uniformly sample isotropic directions
        cos_theta = np.random.uniform(-1, 1, size)
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = np.random.uniform(0, 2 * np.pi, size)

        # set positions, components and metallicities
        self._x = r * sin_theta * np.cos(phi)
        self._y = r * sin_theta * np.sin(phi)
        self._z = r * cos_theta
        self._Z = self.get_metallicity(size)

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
        q = np.empty(size)
        remaining = np.ones(size, dtype=bool)

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
        cos_theta_v = np.random.uniform(-1.0, 1.0, size)
        sin_theta_v = np.sqrt(1.0 - cos_theta_v**2)
        phi_v = np.random.uniform(0.0, 2.0 * np.pi, size)

        # save and convert velocities
        self._v_x = v_phys * sin_theta_v * np.cos(phi_v)
        self._v_y = v_phys * sin_theta_v * np.sin(phi_v)
        self._v_z = v_phys * cos_theta_v

        self._v_T = np.sqrt(((-self.x * self.v_y + self.y * self.v_x)**2) / (self.x**2 + self.y**2))
        self._v_R = (self.x * self.v_x + self.y * self.v_y) / np.sqrt(self.x**2 + self.y**2)


class Frankel2018SFH(StarFormationHistory):
    """A star formation history for a component of the Milky Way, based on
    `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_.

    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    Parameters
    ----------
    scale_height : :class:`~astropy.units.Quantity` [length], optional
        Scale height of the disc, by default 0.3*u.kpc
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
    galaxy_age : :class:`~astropy.units.Quantity` [time], optional
        Maximum lookback time, by default 12*u.Gyr
    """
    def __init__(self, scale_length=None, scale_height=None,
                 tsfr=6.8 * u.Gyr, alpha=0.3, Fm=-1, gradient=-0.075 / u.kpc, Rnow=8.7 * u.kpc,
                 gamma=0.3, zsun=0.0142, galaxy_age=12 * u.Gyr, **kwargs):
        self.scale_length = scale_length
        self.scale_height = scale_height
        self.tsfr = tsfr
        self.alpha = alpha
        self.Fm = Fm
        self.gradient = gradient
        self.Rnow = Rnow
        self.gamma = gamma
        self.zsun = zsun
        self.galaxy_age = galaxy_age
        super().__init__(**kwargs)
        self.__citations__.extend(["Frankel+2018", "McMillan+2011"])

    def draw_radii(self, size):
        """Inverse CDF sampling of galactocentric radii using
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 5.
        The scale length is calculated using Eq. 6.

        Parameters
        ----------
        size : `int`
            How many radii to draw

        Returns
        -------
        rho : :class:`~astropy.units.Quantity` [length]
            Random Galactocentric radius
        """
        return -self.scale_length * (lambertw((np.random.rand(size) - 1) / np.exp(1), k=-1).real + 1)

    def draw_heights(self, size):
        """Draw heights from an exponential distribution with scale height given by the class attribute.

        Parameters
        ----------
        size : `int`
            How many heights to draw

        Returns
        -------
        z : :class:`~astropy.units.Quantity` [length]
            Random heights above the plane
        """
        return _exponential_disc(size, self.scale_height)

    def draw_phi(self, size):
        return np.random.uniform(0, 2 * np.pi, size) * u.rad

    def get_metallicity(self):
        return _frankel2018_metallicity_relation(self)

class LowAlphaDiscWagg2022(Frankel2018SFH):
    """A star formation history for the low-alpha disc of the Milky Way, used in Wagg+2022

    Parameters are the same as :class:`Frankel2018SFH`
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("scale_height", 0.3 * u.kpc)
        super().__init__(**kwargs)

    def draw_lookback_times(self, size):
        """Inverse CDF sampling of lookback times using
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 4,
        separated and normalised at 8 Gyr.

        Parameters
        ----------
        size : `int`
            How many times to draw

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        U = np.random.rand(size)
        norm = 1 / quad(lambda x: np.exp(-(self.galaxy_age.value - x) / self.tsfr.value), 0, 8)[0]
        return self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr.value) + 1)

    def draw_radii(self, size):
        self.scale_length = 4 * u.kpc * (1 - self.alpha * (self._tau / (8 * u.Gyr)))
        return super().draw_radii(size)
    
class HighAlphaDiscWagg2022(Frankel2018SFH):
    """A star formation history for the high-alpha disc of the Milky Way, used in Wagg+2022

    Parameters are the same as :class:`Frankel2018SFH`
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("scale_height", 0.95 * u.kpc)
        kwargs.setdefault("scale_length", 1/0.43 * u.kpc)
        super().__init__(**kwargs)

    def draw_lookback_times(self, size):
        """Inverse CDF sampling of lookback times using
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ Eq. 4,
        separated and normalised at 8 Gyr.

        Parameters
        ----------
        size : `int`
            How many times to draw

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        U = np.random.rand(size)
        norm = 1 / quad(lambda x: np.exp(-(self.galaxy_age.value - x) / self.tsfr.value), 8, 12)[0]
        return self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr.value)
                                  + np.exp(8 * u.Gyr / self.tsfr))
    

class BulgeWagg2022(Frankel2018SFH):
    """A star formation history for the bulge of the Milky Way used in Wagg+2022

    Parameters are the same as :class:`Frankel2018SFH`
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("scale_height", 0.2 * u.kpc)
        kwargs.setdefault("scale_length", 1.5 * u.kpc)
        super().__init__(**kwargs)
        self.__citations__.extend(["Bovy+2019", "Bovy+2016"])

    def draw_lookback_times(self, size):
        """Inverse CDF sampling of lookback times using a beta distribution, fit to match the distribution 
        in Fig. 7 of `Bovy+19 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.4740B/abstract>`_ but
        accounting for the sample's bias.

        Parameters
        ----------
        size : `int`
            How many times to draw

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        return beta.rvs(a=2, b=3, loc=6, scale=6, size=size) * u.Gyr

class Wagg2022(CompositeStarFormationHistory):
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
    def __init__(self, tsfr=6.8 * u.Gyr, alpha=0.3, Fm=-1, gradient=-0.075 / u.kpc, Rnow=8.7 * u.kpc,
                 gamma=0.3, zsun=0.0142, galaxy_age=12 * u.Gyr, **kwargs):
        self.tsfr = tsfr
        self.alpha = alpha
        self.Fm = Fm
        self.gradient = gradient
        self.Rnow = Rnow
        self.gamma = gamma
        self.zsun = zsun
        self.galaxy_age = galaxy_age

        components = [
            LowAlphaDiscWagg2022(tsfr=tsfr, alpha=alpha, Fm=Fm, gradient=gradient, Rnow=Rnow,
                                 gamma=gamma, zsun=zsun, galaxy_age=galaxy_age),
            HighAlphaDiscWagg2022(tsfr=tsfr, alpha=alpha, Fm=Fm, gradient=gradient, Rnow=Rnow,
                                  gamma=gamma, zsun=zsun, galaxy_age=galaxy_age),
            BulgeWagg2022(tsfr=tsfr, alpha=alpha, Fm=Fm, gradient=gradient, Rnow=Rnow,
                          gamma=gamma, zsun=zsun, galaxy_age=galaxy_age)
        ]
        component_ratios = [2.585e10, 2.585e10, 0.91e10]

        super().__init__(components=components, component_ratios=component_ratios, **kwargs)


class MilkyWayBarSormani2022(StarFormationHistory):
    """A star formation history for the Milky Way bar, based on
    `Sormani+2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.514L...1S/abstract>`_.

    Positions and velocities use the agama Density sampler, and metallicities
    use the `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_
    gradient relation.

    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    Parameters
    ----------
    potential : :class:`~gala.potential.PotentialBase`
        A potential to use for sampling velocities
    kappa : `float`, optional
        Kappa parameter for the axisymmetric Jeans equations, by default 0.7
    present_day_bar_angle : :class:`~astropy.units.Quantity` [angle], optional
        Present-day angle of the bar major axis with respect to the Galactocentric x-axis (Sun-GC line), by default 28*u.deg
    pattern_speed : :class:`~astropy.units.Quantity` [angle/time], optional
        Pattern speed of the bar, by default 40*u.km/u.s/u.kpc
    x_max : :class:`~astropy.units.Quantity` [length], optional
        Maximum x coordinate (bar major axis), by default 20*u.kpc
    y_max : :class:`~astropy.units.Quantity` [length], optional
        Maximum y coordinate (bar intermediate axis), by default 20*u.kpc
    z_max : :class:`~astropy.units.Quantity` [length], optional
        Maximum z coordinate (bar minor axis), by default 20*u.kpc
    Fm : `float`, optional
        Metallicity at the Galactic centre at lookback time ``galaxy_age``, by default -1
    gradient : :class:`~astropy.units.Quantity` [1/length], optional
        Radial metallicity gradient, by default -0.075/u.kpc
    Rnow : :class:`~astropy.units.Quantity` [length], optional
        Radius at which the present-day metallicity is solar, by default 8.7*u.kpc
    gamma : `float`, optional
        Time exponent for chemical enrichment, by default 0.3
    zsun : `float`, optional
        Solar metallicity, by default 0.0142
    galaxy_age : :class:`~astropy.units.Quantity` [time], optional
        Maximum lookback time, by default 12*u.Gyr
    """
    def __init__(self, potential, kappa=0.7,
                 present_day_bar_angle=28 * u.deg, pattern_speed=40 * u.km / u.s / u.kpc,
                 x_max=20 * u.kpc, y_max=20 * u.kpc, z_max=20 * u.kpc,
                 Fm=-1, gradient=-0.075 / u.kpc, Rnow=8.7 * u.kpc,
                 gamma=0.3, zsun=0.0142, galaxy_age=8 * u.Gyr, **kwargs):
        self.present_day_bar_angle = present_day_bar_angle
        self.pattern_speed = pattern_speed
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max
        self.Fm = Fm
        self.gradient = gradient
        self.Rnow = Rnow
        self.gamma = gamma
        self.zsun = zsun
        self.galaxy_age = galaxy_age
        self.potential = potential
        self.kappa = kappa
        super().__init__(**kwargs)
        self.__citations__.extend(["Sormani+2022", "Frankel+2018", "Sanders&Binney2015"])

        tau_range = np.linspace(0, self.galaxy_age.to(u.Gyr).value, 100000)
        valid = (tau_range >= 0) & (tau_range < 12)
        tau_pdf = np.zeros_like(tau_range)
        tau_pdf[valid] = np.exp(tau_range[valid] / 8 - 0.43 / (12 - tau_range[valid]))
        tau_cdf = cumulative_trapezoid(tau_pdf, tau_range, initial=0)
        self._inv_cdf = interp1d(tau_cdf / tau_cdf[-1], tau_range, bounds_error=True)

    def draw_lookback_times(self, size):
        """Use inverse CDF sampling to draw lookback times from SB15 model, only between 0 and ``galaxy_age``

        Parameters
        ----------
        size : `int`
            Number of lookback times to draw

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        return self._inv_cdf(np.random.rand(size)) * u.Gyr

    def get_metallicity(self):
        """Compute metallicities using the
        `Frankel+2018 <https://ui.adsabs.harvard.edu/abs/2018ApJ...865...96F/abstract>`_ relation.

        Returns
        -------
        Z : :class:`~astropy.units.Quantity` [dimensionless]
            Metallicities
        """
        return _frankel2018_metallicity_relation(self)

    def _bar_density(self, x, y, z):
        """Total bar density (sum of all three Sormani+2022 components) in 10^10 M_sun kpc^-3"""
        valid = ((np.abs(x) < self.x_max.to(u.kpc).value)
                 & (np.abs(y) < self.y_max.to(u.kpc).value)
                 & (np.abs(z) < self.z_max.to(u.kpc).value))
        density = np.zeros_like(x)
        density[valid] = (self._bar_comp_1_density(x[valid], y[valid], z[valid])
                          + self._bar_comp_2_density(x[valid], y[valid], z[valid])
                          + self._bar_comp_3_density(x[valid], y[valid], z[valid]))
        return density

    def sample(self, size):
        """Sample from the bar density using Agama's Density sampler

        Parameters
        ----------
        size : `int`
            Number of samples to return
        """
        for attr in ["_tau", "_Z", "_x", "_y", "_z", "_v_R", "_v_T", "_v_z", "_v_x", "_v_y"]:
            setattr(self, attr, None)

        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        # convert potential to agama potential if necessary
        if not isinstance(self.potential, agama.Potential):
            self.potential = self.potential.as_interop('agama')

        def density_func(x):
            return self._bar_density(x[:, 0], x[:, 1], x[:, 2])

        # use agama.Density
        density = agama.Density(density_func, symmetry='t')
        xv, _ = density.sample(size, potential=self.potential, kappa=self.kappa)

        # draw lookback times first — needed to determine the bar angle at each star's birth
        self._tau = self.draw_lookback_times(size)

        # rotate bar-frame positions into the Galactocentric frame.
        # assuming constant pattern speed, the bar major axis was at angle
        #   phi_birth = phi_0 - omega_p * tau
        # at each star's birth (lookback time tau), where phi_0 is the present-day bar angle
        # measured from the Galactocentric x-axis (Sun-GC direction).
        phi_0 = self.present_day_bar_angle.to(u.rad).value
        omega_p = self.pattern_speed.to(u.Gyr**-1) / (2 * np.pi) # convert pattern speed to radians per Gyr
        phi_birth = phi_0 - (omega_p * self._tau).to(u.dimensionless_unscaled).value

        x_bar = xv[:, 0]
        y_bar = xv[:, 1]
        self._x = (x_bar * np.cos(phi_birth) - y_bar * np.sin(phi_birth)) * u.kpc
        self._y = (x_bar * np.sin(phi_birth) + y_bar * np.cos(phi_birth)) * u.kpc
        self._z = xv[:, 2] * u.kpc

        # also rotate velocities
        v_x = xv[:, 3] * u.kpc / u.Myr
        v_y = xv[:, 4] * u.kpc / u.Myr
        self._v_x = (v_x * np.cos(phi_birth) - v_y * np.sin(phi_birth))
        self._v_y = (v_x * np.sin(phi_birth) + v_y * np.cos(phi_birth))
        self._v_z = xv[:, 5] * u.kpc / u.Myr

        # convert to km/s
        self._v_x = self._v_x.to(u.km / u.s)
        self._v_y = self._v_y.to(u.km / u.s)
        self._v_z = self._v_z.to(u.km / u.s)

        self._Z = self.get_metallicity()

    def _bar_comp_1_density(self, x, y, z):
        """Density of the X-shaped/boxy-peanut bulge component from
        `Sormani+2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.514L...1S/abstract>`_ Table 1.

        Parameters
        ----------
        x, y, z : float or array-like
            Galactocentric Cartesian coordinates in kpc, in the bar frame (x-axis along bar major axis)

        Returns
        -------
        rho : float or array-like
            Density in units of 10^10 M_sun / kpc^3
        """
        # Table 1 parameters for component 1 (X-shaped/boxy-peanut bulge)
        p = {
            "rho0":  0.316,   # 10^10 M_sun kpc^-3, central density normalisation
            "x0":    0.490,   # kpc, scale length along bar major axis
            "y0":    0.392,   # kpc, scale length along bar minor axis
            "z0":    0.229,   # kpc, scale length along vertical axis
            "c_par": 1.991,   # exponent for combining in-plane and vertical components
            "c_per": 2.232,   # exponent for combining x and y components
            "m":     0.873,   # exponent in the sech argument
            "n":     1.940,   # exponent for the X-shape arm radii
            "alpha": 0.626,   # X-shape strength
            "c":     1.342,   # X-shape arm tilt (z slope)
            "xc":    0.751,   # kpc, x scale length for X-shape arms
            "yc":    0.469,   # kpc, y scale length for X-shape arms
            "r_cut": 4.370,   # kpc, outer exponential cutoff radius
        }

        # 3D generalised ellipsoidal radius: {[(|x|/x0)^c_per + (|y|/y0)^c_per]^(c_par/c_per) + (|z|/z0)^c_par}^(1/c_par)
        xy_term = (np.abs(x) / p["x0"])**p["c_per"] + (np.abs(y) / p["y0"])**p["c_per"]
        a1 = (xy_term**(p["c_par"] / p["c_per"]) + (np.abs(z) / p["z0"])**p["c_par"])**(1.0 / p["c_par"])

        # X-shape arm radii: a_± = sqrt(((x ± c*z)/xc)^2 + (y/yc)^2)
        a_plus  = np.sqrt(((x + p["c"] * z) / p["xc"])**2 + (y / p["yc"])**2)
        a_minus = np.sqrt(((x - p["c"] * z) / p["xc"])**2 + (y / p["yc"])**2)

        # Spherical radius for outer Gaussian cutoff
        r = np.sqrt(x**2 + y**2 + z**2)

        boxy_profile = 1.0 / np.cosh(a1**p["m"])   # sech(a1^m)
        x_shape = 1.0 + p["alpha"] * (np.exp(-a_plus**p["n"]) + np.exp(-a_minus**p["n"]))
        outer_cut = np.exp(-(r / p["r_cut"])**2)

        return p["rho0"] * boxy_profile * x_shape * outer_cut

    def _bar_elongated_density(self, x, y, z, p):
        """Shared density kernel for the elongated bar components (2 and 3) from
        `Sormani+2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.514L...1S/abstract>`_.

        The functional form is:
        rho = rho0 * exp(-a^n) * sech^2(z/z0) * exp(-(R/R_out)^n_out) * exp(-(R_in/R)^n_in)

        Parameters
        ----------
        x, y, z : float or array-like
            Galactocentric Cartesian coordinates in kpc, in the bar frame
        p : dict
            Parameter dictionary with keys: rho0, x0, y0, z0, c_per, n, R_out, n_out, R_in, n_in
        """
        # In-plane generalised ellipsoidal radius: [(|x|/x0)^c_per + (|y|/y0)^c_per]^(1/c_per)
        a = ((np.abs(x) / p["x0"])**p["c_per"] + (np.abs(y) / p["y0"])**p["c_per"])**(1.0 / p["c_per"])

        # Cylindrical radius; guarded against R=0 for the inner-cutoff term
        R = np.sqrt(x**2 + y**2)
        R_safe = np.maximum(R, 1e-5)

        in_plane  = np.exp(-a**p["n"])
        z_profile = (1.0 / np.cosh(z / p["z0"]))**2                      # sech^2(z/z0)
        outer_cut = np.exp(-(R_safe / p["R_out"])**p["n_out"])
        inner_cut = np.exp(-(p["R_in"] / R_safe)**p["n_in"])

        return p["rho0"] * in_plane * z_profile * outer_cut * inner_cut

    def _bar_comp_2_density(self, x, y, z):
        """Density of the extended bar component from
        `Sormani+2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.514L...1S/abstract>`_ Table 1.

        Parameters
        ----------
        x, y, z : float or array-like
            Galactocentric Cartesian coordinates in kpc, in the bar frame (x-axis along bar major axis)

        Returns
        -------
        rho : float or array-like
            Density in units of 10^10 M_sun / kpc^3
        """
        # Table 1 parameters for component 2 (extended bar)
        p = {
            "rho0":  0.050,    # 10^10 M_sun kpc^-3, central density normalisation
            "x0":    5.364,    # kpc, scale length along bar major axis
            "y0":    0.959,    # kpc, scale length along bar minor axis
            "z0":    0.611,    # kpc, scale height
            "c_per": 0.970,    # exponent for combining x and y components
            "n":     3.051,    # in-plane density exponent
            "R_out": 3.190,    # kpc, outer radial cutoff scale
            "n_out": 16.731,   # outer radial cutoff exponent (large positive → sharp outer edge)
            "R_in":  0.558,    # kpc, inner radial cutoff scale
            "n_in":  3.196,    # inner radial cutoff exponent (central density hole)
        }
        return self._bar_elongated_density(x, y, z, p)

    def _bar_comp_3_density(self, x, y, z):
        """Density of the long bar component from
        `Sormani+2022 <https://ui.adsabs.harvard.edu/abs/2022MNRAS.514L...1S/abstract>`_ Table 1.

        Parameters
        ----------
        x, y, z : float or array-like
            Galactocentric Cartesian coordinates in kpc, in the bar frame (x-axis along bar major axis)

        Returns
        -------
        rho : float or array-like
            Density in units of 10^10 M_sun / kpc^3
        """
        # Table 1 parameters for component 3 (long bar)
        p = {
            "rho0":  1743.049,  # 10^10 M_sun kpc^-3, central density normalisation
            "x0":    0.478,     # kpc, scale length along bar major axis
            "y0":    0.267,     # kpc, scale length along bar minor axis
            "z0":    0.252,     # kpc, scale height
            "c_per": 1.879,     # exponent for combining x and y components
            "n":     0.980,     # in-plane density exponent
            "R_out": 2.204,     # kpc, radial cutoff scale
            "n_out": -27.291,   # radial cutoff exponent (negative → acts as inner cutoff at R_out)
            "R_in":  7.607,     # kpc, radial cutoff scale (outer edge of long bar)
            "n_in":  1.630,     # radial cutoff exponent
        }
        return self._bar_elongated_density(x, y, z, p)


class DistributionFunctionBasedSFH(StarFormationHistory):
    """A star formation history based on a distribution function.
    This is an abstract base class and should not be instantiated directly.

    Parameters
    ----------
    Parameters are the same as :class:`StarFormationHistory` but additionally with the following:

    potential : :class:`~agama.Potential` or :class:`Potential <gala.potential.potential.PotentialBase>`
        The gravitational potential in which to sample the distribution function
    df : `function` or `dict`
        Either a function that represents the distribution function, taking J as an argument,
        or the keyword arguments to pass to the distribution function(s) using
        :class:`agama.DistributionFunction`.
    """
    def __init__(self, potential, df, **kwargs):
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        self.potential = potential
        self._agama_pot = potential if isinstance(potential, agama.Potential) else potential.as_interop("agama")

        if isinstance(df, dict):
            self._df = agama.DistributionFunction(potential=self.agama_pot, **df)
        elif isinstance(df, FunctionType):
            self._df = df
        elif df is not None:            # pragma: no cover
            raise ValueError(("`df` must be either a function or a dict of keyword arguments to pass "
                              "to `agama.DistributionFunction`"))

        super().__init__(**kwargs)

    @property
    def agama_pot(self):
        return self._agama_pot
    
    @property
    def df(self):
        return self._df
    
    def sample(self, size):
        """Sample from the distributions for each component, combine and save in class attributes"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        self.draw_lookback_times(size)

        self._x = np.zeros(size) * u.kpc
        self._y = np.zeros(size) * u.kpc
        self._z = np.zeros(size) * u.kpc

        self._v_x = np.zeros(size) * u.km / u.s
        self._v_y = np.zeros(size) * u.km / u.s
        self._v_z = np.zeros(size) * u.km / u.s

        self._v_R = np.zeros(size) * u.km / u.s
        self._v_T = np.zeros(size) * u.km / u.s

        xv, _ = agama.GalaxyModel(self.agama_pot, self.df).sample(size)

        # convert units for velocity
        xv[:, 3:] *= (u.kpc / u.Myr).to(u.km / u.s)

        # save the positions/velocities
        self._x = xv[:, 0] * u.kpc
        self._y = xv[:, 1] * u.kpc
        self._z = xv[:, 2] * u.kpc
        self._v_x = xv[:, 3] * u.km / u.s
        self._v_y = xv[:, 4] * u.km / u.s
        self._v_z = xv[:, 5] * u.km / u.s

        # work out the velocities by rotating using SkyCoord
        full_coord = SkyCoord(x=self._x, y=self._y, z=self._z, v_x=self._v_x, v_y=self._v_y, v_z=self._v_z,
                              frame="galactocentric").represent_as("cylindrical")

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            self._v_R = full_coord.differentials['s'].d_rho
            self._v_T = (full_coord.differentials['s'].d_phi * full_coord.rho).to(u.km / u.s)
            self._v_z = full_coord.differentials['s'].d_z

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
    def __init__(self, time_bins=5, verbose=False,
                 tau_m=12 * u.Gyr, tau_S=0.43 * u.Gyr, tau_T=10 * u.Gyr,
                 tau_F=8 * u.Gyr, tau_1=0.11 * u.Gyr,
                 **kwargs):
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

        super().__init__(df=None, **kwargs)
        self.__citations__.append("Sanders&Binney2015")

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
        df_val = np.zeros_like(J_r)
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

    def draw_lookback_times(self, size):
        """Draw lookback times for all stars using inverse CDF sampling

        Returns
        -------
        tau : :class:`~astropy.units.Quantity` [time]
            Random lookback times
        """
        U = np.random.rand(size)
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

    def sample(self, size):
        """Sample from the distributions for each component, combine and save in class attributes"""
        assert check_dependencies("agama")
        import agama
        agama.setUnits(**{k: galactic[k] for k in ['length', 'mass', 'time']})

        self._precompute_interpolations()

        if self.verbose:
            print("Initiating sampling procedure")

        self.draw_lookback_times(size)

        is_thin_disc = self.tau < self.tau_T
        sizes = [np.sum(is_thin_disc), np.sum(~is_thin_disc)]

        which_comp = np.where(is_thin_disc, "thin_disc", "thick_disc")

        self._x = np.zeros(size) * u.kpc
        self._y = np.zeros(size) * u.kpc
        self._z = np.zeros(size) * u.kpc

        self._v_x = np.zeros(size) * u.km / u.s
        self._v_y = np.zeros(size) * u.km / u.s
        self._v_z = np.zeros(size) * u.km / u.s

        self._v_R = np.zeros(size) * u.km / u.s
        self._v_T = np.zeros(size) * u.km / u.s

        for com_size, com in zip(sizes, ["thin_disc", "thick_disc"]):
            if com_size == 0:          # pragma: no cover
                continue
            com_mask = which_comp == com

            if com == "thin_disc":
                time_bin_edges = np.linspace(0, self.tau_T.to(u.Gyr).value, self.time_bins + 1) * u.Gyr
            else:
                time_bin_edges = np.array([self.tau_T.to(u.Gyr).value, self.tau_m.to(u.Gyr).value]) * u.Gyr

            if self.verbose:
                print(f"  Sampling {com_size} stars from the {com}")

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
                self._v_x[in_bin] = xv[:, 3] * u.km / u.s
                self._v_y[in_bin] = xv[:, 4] * u.km / u.s
                self._v_z[in_bin] = xv[:, 5] * u.km / u.s

        # work out the velocities by rotating using SkyCoord
        full_coord = SkyCoord(x=self._x, y=self._y, z=self._z, v_x=self._v_x, v_y=self._v_y, v_z=self._v_z,
                              frame="galactocentric").represent_as("cylindrical")

        with u.set_enabled_equivalencies(u.dimensionless_angles()):
            self._v_R = full_coord.differentials['s'].d_rho
            self._v_T = (full_coord.differentials['s'].d_phi * full_coord.rho).to(u.km / u.s)
            self._v_z = full_coord.differentials['s'].d_z

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
    def __init__(self, J_0_star, alpha, eta, fixed_Z, tau_min, galaxy_age, mass=None, **kwargs):
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

        super().__init__(**kwargs)
        self.__citations__.append("Pascale+2019")

    def draw_lookback_times(self, size):
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
        self._tau = np.random.uniform(self.tau_min.to(u.Gyr).value,
                                      self.galaxy_age.to(u.Gyr).value, size) * u.Gyr
        return self._tau

    def get_metallicity(self):
        """Fixed metallicity for all stars in the dwarf galaxy

        Returns
        -------
        Z : :class:`~astropy.units.Quantity` [dimensionless]
            Metallicities
        """
        self._Z = np.repeat(self.fixed_Z, len(self._tau))
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
    """
    def __init__(self, **kwargs):
        super().__init__(mass=8.69e8 * u.Msun, J_0_star=0.677 * u.kpc*u.km/u.s,
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

    # check whether this might actually be a composite star formation history
    with h5.File(file_name, "r") as file:
        if f"{key}_0" in file.keys():
            return CompositeStarFormationHistory.from_file(file_name, key=key)

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
    loaded_sfh._x = df["x"].values * u.kpc
    loaded_sfh._y = df["y"].values * u.kpc
    loaded_sfh._z = df["z"].values * u.kpc

    # additionally read in velocity components if they exist
    for attr in ["_v_R", "_v_T", "_v_z", "_v_x", "_v_y", "_v_z"]:
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
    if len(sfhs) == 1:
        return sfhs[0]
    elif len(sfhs) == 0:
        raise ValueError("No objects to concatenate")

    if all([isinstance(sfh, CompositeStarFormationHistory) for sfh in sfhs]):
        # ensure that each composite has the same number of components
        n_comps = [len(sfh.components) for sfh in sfhs]
        if len(set(n_comps)) != 1:
            raise ValueError(("All CompositeStarFormationHistory objects to concatenate must have the "
                              "same number of components"))
        components = [concat(*[sfhs[i].components[j] for i in range(len(sfhs))]) for j in range(n_comps[0])]

        return CompositeStarFormationHistory(
            components=components,
            component_ratios=sfhs[0].component_ratios       # assume the same ratios as the first
        )

    elif all([isinstance(sfh, StarFormationHistory) for sfh in sfhs]):

        # create a new object with the same parameters as the first
        new_sfh = sfhs[0].copy()

        # concatenate the velocity components if they exist
        for attr in ["_tau", "_Z", "_x", "_y", "_z", "_v_R", "_v_T", "_v_z"]:
            if hasattr(sfhs[0], attr) and not getattr(sfhs[0], attr) is None:
                setattr(new_sfh, attr, np.concatenate([getattr(sfh, attr) for sfh in sfhs]))

        new_sfh._size = len(new_sfh._tau)

        return new_sfh
    else:
        raise ValueError(("All SFHs to concatenate must be of the same type, either all "
                          "CompositeStarFormationHistory or all StarFormationHistory"))


def simplify_params(params, dont_save=["_tau", "_Z", "_x", "_y", "_z", "_which_comp", "_v_R", "_v_T", "_v_z",
                                       "_v_x", "_v_y", "_df", "_agama_pot", "potential", "__citations__",
                                       "_guiding_radius_interp", "_omega_interp",
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
