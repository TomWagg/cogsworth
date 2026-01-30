import time
import os
from copy import copy
from multiprocessing import Pool
import warnings
import numpy as np
import astropy.units as u
import astropy.coordinates as coords
import h5py as h5
import pandas as pd
from tqdm import tqdm
import yaml
import logging
import matplotlib.pyplot as plt

from cosmic.sample.initialbinarytable import InitialBinaryTable
from cosmic.evolve import Evolve
from cosmic.checkstate import set_checkstates
from cosmic.output import save_initC, load_initC
from cosmic.utils import parse_inifile
from cosmic import __version__ as cosmic_version
import gala.potential as gp
import gala.dynamics as gd
import gala.integrate as gi
from gala.potential.potential.io import to_dict as potential_to_dict, from_dict as potential_from_dict
from gala import __version__ as gala_version

from cogsworth import sfh
from cogsworth._version import __version__
from cogsworth.kicks import integrate_orbit_with_events
from cogsworth.events import identify_events
from cogsworth.classify import determine_final_classes
from cogsworth.obs.observables import get_photometry
from cogsworth.tests.optional_deps import check_dependencies
from cogsworth.plot import plot_cartoon_evolution, plot_galactic_orbit
from cogsworth.utils import translate_COSMIC_tables, get_default_BSE_settings

from cogsworth.citations import CITATIONS

__all__ = ["Population", "EvolvedPopulation", "load", "concat"]
_GLOBAL = {}


class Population():
    """Class for creating and evolving populations of binaries throughout the Milky Way

    Parameters
    ----------
    n_binaries : `int`
        How many binaries to sample for the population
    processes : `int`, optional
        How many processes to run if you want multithreading, by default 8
    m1_cutoff : `float`, optional
        The minimum allowed primary mass, by default 0
    final_kstar1 : `list`, optional
        Desired final types for primary star, by default list(range(16))
    final_kstar2 : `list`, optional
        Desired final types for secondary star, by default list(range(16))
    sfh_model : :class:`~cogsworth.sfh.StarFormationHistory`, optional
        A StarFormationHistory class to use for sampling the initial galaxy parameters, by default
        :class:`~cogsworth.sfh.Wagg2022`
    sfh_params : `dict`, optional
        Any additional parameters to pass to your chosen ``SFH model`` when it is initialised
    galactic_potential : :class:`Potential <gala.potential.potential.PotentialBase>`, optional
        Galactic potential to use for evolving the orbits of binaries, by default
        :class:`~gala.potential.potential.MilkyWayPotential`
    v_dispersion : :class:`~astropy.units.Quantity` [velocity], optional
        Velocity dispersion to apply relative to the local circular velocity, by default 5*u.km/u.s
    max_ev_time : :class:`~astropy.units.Quantity` [time], optional
        Maximum evolution time for both COSMIC and Gala, by default 12.0*u.Gyr
    timestep_size : :class:`~astropy.units.Quantity` [time], optional
        Size of timesteps to use in galactic evolution, by default 1*u.Myr
    BSE_settings : `dict`, optional
        Any BSE settings to pass to COSMIC, superseded by `ini_file` if provided
    ini_file : `str`, optional
        Path to an ini file to use for the COSMIC stellar evolution, supersedes `BSE_settings` by default None
    use_default_BSE_settings : `bool`, optional
        Whether to use the default COSMIC BSE settings, by default False. If False then at least one of
        `BSE_settings` or `ini_file` must be provided.
    bcm_timestep_conditions : List of lists, optional
        Any timestep conditions to pass to COSMIC evolution. This will affect the rows that are output in the
        the BCM table, by default only the first and last timesteps are output. For more details check out the
        `relevant COSMIC docs <https://cosmic-popsynth.github.io/COSMIC/examples/index.html#dynamically-set-time-resolution-for-bcm-array>`_
    sampling_params : `dict`, optional
        Any additional parameters to pass to the COSMIC sampling (see
        :meth:`~cosmic.sample.sampler.independent.get_independent_sampler`)
    store_entire_orbits : `bool`, optional
        Whether to store the entire orbit for each binary, by default True. If not then only the final
        PhaseSpacePosition will be stored. This cuts down on both memory usage and disk space used if you
        save the Population (as well as how long it takes to reload the data).
    bpp_columns : `list`, optional
        Which columns COSMIC should store in the `bpp` table. If None, default columns are used.
        See https://cosmic-popsynth.github.io/COSMIC/pages/output_info.html for a list of columns.
    bcm_columns : `list`, optional
        Which columns COSMIC should store in the `bcm` table. If None, default columns are used.
        See https://cosmic-popsynth.github.io/COSMIC/pages/output_info.html for a list of columns.
    error_file_path : `str`, optional
        Path to a folder in which to store any error files generated during evolution, by default "./"
    integrator : :class:`~gala.integrate.Integrator`, optional
        The integrator used by gala for evolving the orbits of binaries in the galactic potential
        (default is :class:`~gala.integrate.DOPRI853Integrator`).
    integrator_kwargs : `dict`, optional
        Any additional keyword arguments to pass to the gala integrator, by default an empty dict
    """
    def __init__(self, n_binaries, processes=8, m1_cutoff=0, final_kstar1=list(range(16)),
                 final_kstar2=list(range(16)), sfh_model=sfh.Wagg2022, sfh_params={},
                 galactic_potential=gp.MilkyWayPotential(version='v2'), v_dispersion=5 * u.km / u.s,
                 max_ev_time=12.0*u.Gyr, timestep_size=1 * u.Myr, BSE_settings={}, ini_file=None,
                 use_default_BSE_settings=False, sampling_params={},
                 bcm_timestep_conditions=[], store_entire_orbits=True,
                 bpp_columns=None, bcm_columns=None, error_file_path="./",
                 integrator=gi.DOPRI853Integrator, integrator_kwargs={}):

        # require a sensible number of binaries if you are not targetting total mass
        if not ("sampling_target" in sampling_params and sampling_params["sampling_target"] == "total_mass"):
            if n_binaries <= 0:
                raise ValueError("You need to input a *nonnegative* number of binaries")

        # warn users that everything will soon explode without settings
        if not use_default_BSE_settings and ini_file is None and BSE_settings == {}:
            logging.getLogger("cogsworth").warning(("cogsworth warning: You have not provided any binary "
                                                    "stellar evolution (BSE) settings. You must provide "
                                                    "either `BSE_settings` or an `ini_file` in order to "
                                                    "perform stellar evolution with COSMIC. You may set "
                                                    "`use_default_BSE_settings=True` to use the default "
                                                    "settings listed in cogsworth, but do so at your own "
                                                    "risk and be sure to understand the choices that you "
                                                    "have made.\nRun `cogsworth.utils.list_BSE_defaults()` "
                                                    "to see these."))

        self.n_binaries = n_binaries
        self.n_binaries_match = n_binaries
        self.processes = processes
        self.m1_cutoff = m1_cutoff
        self.final_kstar1 = final_kstar1
        self.final_kstar2 = final_kstar2
        self.sfh_model = sfh_model
        self.sfh_params = sfh_params
        self.galactic_potential = galactic_potential
        self.v_dispersion = v_dispersion
        self.max_ev_time = max_ev_time
        self.timestep_size = timestep_size
        self.pool = None
        self.pool_config = None
        self.store_entire_orbits = store_entire_orbits
        self.bpp_columns = bpp_columns
        self.bcm_columns = bcm_columns
        self.error_file_path = error_file_path
        self.integrator = integrator
        self.integrator_kwargs = integrator_kwargs

        self._file = None
        self._initial_binaries = None
        self._initial_galaxy = None
        self._mass_singles = None
        self._mass_binaries = None
        self._n_singles_req = None
        self._n_bin_req = None
        self._bpp = None
        self._bcm = None
        self._initC = None
        self._kick_info = None
        self._orbits = None
        self._classes = None
        self._final_pos = None
        self._final_vel = None
        self._final_bpp = None
        self._disrupted = None
        self._escaped = None
        self._observables = None
        self._bin_nums = None

        self.__citations__ = ["cogsworth", "cosmic", "gala"]

        self.BSE_settings = get_default_BSE_settings() if use_default_BSE_settings else {}
        self.BSE_settings.update(BSE_settings if ini_file is None else parse_inifile(ini_file)[0])

        self.sampling_params = {'primary_model': 'kroupa01', 'ecc_model': 'sana12', 'porb_model': 'sana12',
                                'qmin': -1, 'keep_singles': False}
        self.sampling_params.update(sampling_params)
        self.bcm_timestep_conditions = bcm_timestep_conditions

    def __repr__(self):
        if self._orbits is None:
            return (f"<{self.__class__.__name__} - {self.n_binaries} systems - "
                    f"galactic_potential={self.galactic_potential.__class__.__name__}, "
                    f"SFH={self.sfh_model.__name__}>")
        else:
            return (f"<{self.__class__.__name__} - {self.n_binaries_match} evolved systems - "
                    f"galactic_potential={self.galactic_potential.__class__.__name__}, "
                    f"sfh_model={self.sfh_model.__name__}>")

    def __len__(self):
        return self.n_binaries_match

    def __add__(self, other):
        return concat(self, other)

    def __getitem__(self, ind):
        # convert any Pandas Series to numpy arrays
        ind = ind.values if isinstance(ind, pd.Series) else ind

        # if the population is associated with a file, make sure it's entirely loaded before slicing
        if self._file is not None:
            parts = ["initial_binaries", "bpp", "initial_galaxy", "orbits"]
            vars = [self._initial_binaries, self._bpp, self._initial_galaxy, self._orbits]
            masks = {f"has_{p}": False for p in parts}
            with h5.File(self._file, "r") as f:
                for p in parts:
                    masks[f"has_{p}"] = p in f
            missing_parts = [p for i, p in enumerate(parts) if (masks[f"has_{p}"] and vars[i] is None)]

            if len(missing_parts) > 0:
                raise ValueError(("This population was loaded from a file but you haven't loaded all parts "
                                  "yet. You need to do this before indexing it. The missing parts are: "
                                  f"{missing_parts}. You either need to access each of these variables or "
                                  "reload the entire population using all parts."))

        # ensure indexing with the right type
        ALLOWED_TYPES = (int, slice, list, np.ndarray, tuple)
        if not isinstance(ind, ALLOWED_TYPES):
            raise ValueError((f"Can only index using one of {[at.__name__ for at in ALLOWED_TYPES]}, "
                              f"you supplied a '{type(ind).__name__}'"))

        # check validity of indices for array-like types
        if isinstance(ind, (list, tuple, np.ndarray)):
            # check every element is a boolean (if so, convert to bin_nums after asserting length sensible)
            if all(isinstance(x, (bool, np.bool_)) for x in ind):
                assert len(ind) == len(self.bin_nums), "Boolean mask must be same length as the population"
                ind = self.bin_nums[ind]
            # otherwise ensure all elements are integers
            else:
                assert all(isinstance(x, (int, np.integer)) for x in ind), \
                    "Can only index using integers or a boolean mask"
                if len(np.unique(ind)) < len(ind):
                    warnings.warn(("You have supplied duplicate indices, this will invalidate the "
                                   "normalisation of the Population (e.g. mass_binaries will be wrong)"))

        # set up the bin_nums we are selecting
        bin_nums = ind

        # turn ints into arrays and convert slices to exact bin_nums
        if isinstance(ind, int):
            bin_nums = [ind]
        elif isinstance(ind, slice):
            bin_nums = self.bin_nums[ind]
        bin_nums = np.asarray(bin_nums)

        # check that the bin_nums are all valid
        check_nums = np.isin(bin_nums, self.bin_nums)
        if not check_nums.all():
            raise ValueError(("The index that you supplied includes a `bin_num` that does not exist. "
                              f"The first bin_num I couldn't find was {bin_nums[~check_nums][0]}"))

        # start a new population with the same parameters
        new_pop = self.__class__(n_binaries=len(bin_nums), processes=self.processes,
                                 m1_cutoff=self.m1_cutoff, final_kstar1=self.final_kstar1,
                                 final_kstar2=self.final_kstar2, sfh_model=self.sfh_model,
                                 sfh_params=self.sfh_params, galactic_potential=self.galactic_potential,
                                 v_dispersion=self.v_dispersion, max_ev_time=self.max_ev_time,
                                 timestep_size=self.timestep_size, BSE_settings=self.BSE_settings,
                                 sampling_params=self.sampling_params,
                                 bcm_timestep_conditions=self.bcm_timestep_conditions,
                                 store_entire_orbits=self.store_entire_orbits,
                                 bpp_columns=self.bpp_columns,
                                 bcm_columns=self.bcm_columns,
                                 error_file_path=self.error_file_path,
                                 integrator=self.integrator,
                                 integrator_kwargs=self.integrator_kwargs)
        new_pop.n_binaries_match = new_pop.n_binaries

        # proxy for checking whether sampling has been done
        if self._mass_binaries is not None:
            new_pop._mass_binaries = self._mass_binaries
            new_pop._mass_singles = self._mass_singles
            new_pop._n_singles_req = self._n_singles_req
            new_pop._n_bin_req = self._n_bin_req

        bin_num_to_ind = {num: i for i, num in enumerate(self.bin_nums)}
        sort_idx = np.argsort(list(bin_num_to_ind.keys()))
        idx = np.searchsorted(list(bin_num_to_ind.keys()), bin_nums, sorter=sort_idx)
        inds = np.asarray(list(bin_num_to_ind.values()))[sort_idx][idx]

        if self._initial_galaxy is not None:
            new_pop._initial_galaxy = self._initial_galaxy[inds]
        if self._initC is not None:
            new_pop._initC = self._initC.loc[bin_nums]
        if self._initial_binaries is not None:
            new_pop._initial_binaries = self._initial_binaries.loc[bin_nums]

        # checking whether stellar evolution has been done
        if self._bpp is not None:
            # copy over subsets of data when they aren't None
            new_pop._bpp = self._bpp.loc[bin_nums]
            if self._bcm is not None:
                new_pop._bcm = self._bcm.loc[bin_nums]
            if self._kick_info is not None:
                new_pop._kick_info = self._kick_info.loc[bin_nums]
            if self._final_bpp is not None:
                new_pop._final_bpp = self._final_bpp.loc[bin_nums]
            if self._disrupted is not None:
                new_pop._disrupted = self._disrupted[inds]
            if self._classes is not None:
                new_pop._classes = self._classes.iloc[inds]
            if self._observables is not None:
                new_pop._observables = self._observables.iloc[inds]

            if self._orbits is not None or self._final_pos is not None or self._final_vel is not None:
                disrupted_bin_num_to_ind = {num: i for i, num in enumerate(self.bin_nums[self.disrupted])}
                sort_idx = np.argsort(list(disrupted_bin_num_to_ind.keys()))
                idx = np.searchsorted(list(disrupted_bin_num_to_ind.keys()),
                                      bin_nums[np.isin(bin_nums, self.bin_nums[self.disrupted])],
                                      sorter=sort_idx)
                inds_with_disruptions = np.asarray(list(disrupted_bin_num_to_ind.values()))[sort_idx][idx]\
                      + len(self)
                all_inds = np.concatenate((inds, inds_with_disruptions)).astype(int)

            # same thing but for arrays with appended disrupted secondaries
            if self._orbits is not None:
                new_pop._orbits = self.orbits[all_inds]
            if self._final_pos is not None:
                new_pop._final_pos = self._final_pos[all_inds]
            if self._final_vel is not None:
                new_pop._final_vel = self._final_vel[all_inds]
        return new_pop

    def copy(self):
        """Create a copy of the population"""
        return self[:]

    def get_citations(self, filename=None):
        """Print the citations for the packages/papers used in the population"""
        # ask users for a filename to save the bibtex to
        if filename is None:
            filename = input("Filename for generating a bibtex file (leave blank to just print to terminal): ")
        filename = filename + ".bib" if not filename.endswith(".bib") and filename != "" else filename

        sections = {
            "general": "",
            "sfh": r"The \texttt{cogsworth} population used a star formation history model based on the following papers",
            "observables": "Population observables were estimated using dust maps and MIST isochrones",
            "gaia": "Observability of systems with Gaia was predicted using an empirical selection function",
            "legwork": "Calculation of LISA gravitational wave signatures was performed using LEGWORK",
            "FIRE": "The initial conditions for the population were sampled from the FIRE simulations",
            "hydro": "Hydrodynamical simulations used to inform the star formation history and galactic potential were processed using pynbody",
        }

        acknowledgement = r"This research made use of \texttt{cogsworth} and its dependencies"

        # construct citation string
        bibtex = []
        for section in sections:
            cite_tags = []
            for citation in self.__citations__:
                if citation in CITATIONS[section]:
                    cite_tags.extend(CITATIONS[section][citation]["tags"])
                    bibtex.append(CITATIONS[section][citation]["bibtex"])
            if len(cite_tags) > 0:
                cite_str = ",".join(cite_tags)
                acknowledgement += sections[section] + r" \citep{" + cite_str + "}. "
        bibtex_str = "\n\n".join(bibtex)

        # print the acknowledgement
        BOLD, RESET, GREEN = "\033[1m", "\033[0m", "\033[0;32m"
        print("\nYou can paste this acknowledgement into the relevant section of your manuscript:")
        print(f"{BOLD}{GREEN}{acknowledgement}{RESET}")

        # either print bibtex to terminal or save to file
        if filename != "":
            print(f"The associated bibtex can be found in {filename}")
            with open(filename, "w") as f:
                f.write(bibtex_str)
        else:
            print("\nAnd paste this bibtex into your .bib file:")
            print(f"{BOLD}{GREEN}{bibtex_str}{RESET}")
        print("Good luck with the paper writing ◝(ᵔᵕᵔ)◜")

    @property
    def bin_nums(self):
        """An array of the unique identifiers for each binary in a population.

        This is a helper function which pulls from the bin_nums available in :attr:`final_bpp`,
        or :attr:`initC`, or :attr:`initial_binaries` if available. If none of these are available, an
        error is raised.

        Returns
        -------
        bin_nums : :class:`~numpy.ndarray`
            An array of the unique identifiers for each binary in the population.

        Raises
        ------
        ValueError
            If no binaries have been sampled or evolved yet.
        """
        if self._bin_nums is None:
            if self._final_bpp is not None:
                self._bin_nums = self._final_bpp["bin_num"].unique()
            elif self._initC is not None:
                self._bin_nums = self._initC["bin_num"].unique()
            elif self._initial_binaries is not None:
                self._bin_nums = np.unique(self._initial_binaries.index.values)
            else:
                raise ValueError("You need to first sample binaries to get a list of `bin_nums`!")
        return self._bin_nums

    @property
    def initial_galaxy(self):
        """The initial sampled galactic star formation history.

        This contains the initial conditions for the galaxy that was sampled to generate the population. This
        includes the birth times, positions, velocities, and metallicities of the stars in the galaxy.

        Returns
        -------
        initial_galaxy : :class:`~cogsworth.sfh.StarFormationHistory`
            The initial galaxy that was sampled to generate the population.

        Raises
        ------
        ValueError
            If no galaxy has been sampled yet.
        """
        if self._initial_galaxy is None and self._file is not None:
            self._initial_galaxy = sfh.load(self._file, key="initial_galaxy")
            self.sfh_model = self._initial_galaxy.__class__
        elif self._initial_galaxy is None:
            raise ValueError("No galaxy sampled yet, run `sample_initial_galaxy` to generate one.")
        return self._initial_galaxy

    @property
    def mass_singles(self):
        """The total mass in single stars needed to generate the population.

        Returns
        -------
        mass_singles : `float`
            The total mass in single stars needed to generate the population (in solar masses).

        Raises
        ------
        ValueError
            If no population sampled yet.
        """
        if self._mass_singles is None:
            raise ValueError("No population sampled yet, run `sample_initial_binaries` to do so.")
        return self._mass_singles

    @property
    def mass_binaries(self):
        """The total mass in binaries needed to generate the population.

        Returns
        -------
        mass_binaries : `float`
            The total mass in binaries

        Raises
        ------
        ValueError
            If no population sampled yet.
        """
        if self._mass_binaries is None:
            raise ValueError("No population sampled yet, run `sample_initial_binaries` to do so.")
        return self._mass_binaries

    @property
    def n_singles_req(self):
        """The number of single stars needed to generate the population.

        Returns
        -------
        n_singles_req : `int`
            The number of single stars needed to generate the population.

        Raises
        ------
        ValueError
            If no population sampled yet.
        """
        if self._n_singles_req is None:
            raise ValueError("No population sampled yet, run `sample_initial_binaries` to do so.")
        return self._n_singles_req

    @property
    def n_bin_req(self):
        """The number of binary stars needed to generate the population.

        Returns
        -------
        n_bin_req : `int`
            The number of binary stars needed to generate the population.

        Raises
        ------
        ValueError
            If no population sampled yet.
        """
        if self._n_bin_req is None:
            raise ValueError("No population sampled yet, run `sample_initial_binaries` to do so.")
        return self._n_bin_req

    @property
    def initial_binaries(self):
        """The initial binaries that were sampled to generate the population.

        This is only used when evolution has not been performed yet. If evolution has been performed, the
        initial binaries are stored in the :attr:`initC` table instead (since this also includes information
        on the assumed binary physics).

        Returns
        -------
        initial_binaries : :class:`~pandas.DataFrame`
            The initial binaries that were sampled to generate the population.

        Raises
        ------
        ValueError
            If no binaries have been sampled yet.
        """
        # use initC if available
        if self._initial_binaries is None and self._initC is not None:
            return self._initC

        # if not, try to load them from the file
        if self._initial_binaries is None and self._file is not None:       # pragma: no cover
            try:
                self._initial_binaries = pd.read_hdf(self._file, key="initial_binaries")
            except KeyError:
                try:
                    self._initial_binaries = pd.read_hdf(self._file, key="initC")
                except KeyError:
                    raise ValueError(f"No initial binaries found in population file ({self._file})")
        elif self._initial_binaries is None:        # pragma: no cover
            raise ValueError("No binaries sampled yet, run `sample_initial_binaries` to do so.")
        return self._initial_binaries

    @property
    def bpp(self):
        """A table of the evolutionary history of each binary.

        Each row of this table corresponds to an important timestep in the evolution of a binary. Events that
        log a timestep include: stellar type of a star changing, mass transfer, a supernova, a common envelope
        event, a binary merger, or a disruption. The columns of the table are described in detail `in the
        COSMIC documentation <https://cosmic-popsynth.github.io/docs/stable/output_info/index.html#bpp>`__.

        Returns
        -------
        bpp : :class:`~pandas.DataFrame`
            The evolutionary history of each binary.

        Raises
        ------
        ValueError
            If no stellar evolution has been performed yet.
        """
        if self._bpp is None and self._file is not None:
            self._bpp = pd.read_hdf(self._file, key="bpp")
        elif self._bpp is None:
            raise ValueError("No stellar evolution performed yet, run `perform_stellar_evolution` to do so.")
        return self._bpp

    @property
    def bcm(self):
        """A table of the evolutionary history of each binary at dynamically chosen timesteps.

        Each row of this table corresponds to a timestep in the evolution of a binary. Timesteps are set
        based on user-defined ``bcm_timestep_conditions``. The columns of the table are described in
        detail
        `in the COSMIC documentation <https://cosmic-popsynth.github.io/docs/stable/output_info/index.html#bcm>`__.

        Returns
        -------
        bcm : :class:`~pandas.DataFrame`
            The evolutionary history of each binary at dynamically chosen timesteps. Note this will be
            ``None`` if no timestep conditions have been set (in ``bcm_timestep_conditions``).

        Raises
        ------
        Warning
            If no timestep conditions (``bcm_timestep_condition``) have been set for the BCM table.

        ValueError
            If no stellar evolution has been performed yet.
        """
        if self._bcm is None and self._file is not None:
            has_bcm = None
            with h5.File(self._file, "r") as f:
                has_bcm = "bcm" in f
            self._bcm = pd.read_hdf(self._file, key="bcm") if has_bcm else None
        elif self._bcm is None:
            if len(np.ravel(self.bcm_timestep_conditions)) == 0:        # pragma: no cover
                logging.getLogger("cogsworth").warning(("cogsworth warning: You haven't set any timestep "
                                                        "conditions for the BCM table, so it is not "
                                                        "calculated. Set `bcm_timestep_conditions` to get a "
                                                        "BCM table."))
            else:
                raise ValueError("No stellar evolution performed yet, run `perform_stellar_evolution` to do so.")
        return self._bcm

    @property
    def initC(self):
        """A table of initial conditions for each binary.

        This table contains the initial conditions for each binary that was sampled. Each row corresponds to
        a binary and the columns are described in detail in the COSMIC documentation.

        Returns
        -------
        initC : :class:`~pandas.DataFrame`
            The initial conditions for each binary.

        Raises
        ------
        ValueError
            If no stellar evolution has been performed yet.
        """
        if self._initC is None and self._file is not None:
            self._initC = load_initC(self._file, key="initC", settings_key="initC_settings")
        elif self._initC is None:
            raise ValueError("No stellar evolution performed yet, run `perform_stellar_evolution` to do so.")
        return self._initC

    @property 
    def kick_info(self):
        """A table of the kicks that occur for each binary.

        Each row of this table corresponds to a potential supernova kick, such that there are two rows for
        each binary in the population. The columns of the table are described in detail
        `in the COSMIC documentation <https://cosmic-popsynth.github.io/docs/stable/output_info/index.html#kick-info>`__.

        Returns
        -------
        kick_info : :class:`~pandas.DataFrame`
            Information about the kicks that occur for each binary.

        Raises
        ------
        ValueError
            If no stellar evolution has been performed yet.
        """
        if self._kick_info is None and self._file is not None:
            self._kick_info = pd.read_hdf(self._file, key="kick_info")
        if self._kick_info is None:
            raise ValueError("No stellar evolution performed yet, run `perform_stellar_evolution` to do so.")
        return self._kick_info

    @property
    def orbits(self):
        """The orbits of each system within the galaxy from its birth until :attr:`max_ev_time`.

        This list will have length = `len(self) + self.disrupted.sum()`, where the first section are for
        bound binaries and disrupted primaries and the last section are for disrupted secondaries.

        Returns
        -------
        orbits : `list` of :class:`~gala.dynamics.Orbit`, shape (len(self) + self.disrupted.sum(),)
            The orbits of each system within the galaxy from its birth until :attr:`max_ev_time`.

        Raises
        ------
        ValueError
            If no orbits have been calculated yet.
        """
        # if orbits are uncalculated and no file is provided then throw an error
        if self._orbits is None and self._file is None:
            raise ValueError("No orbits calculated yet, run `perform_galactic_evolution` to do so")
        # otherwise if orbits are uncalculated but a file is provided then load the orbits from the file
        elif self._orbits is None:
            # load the entire file into memory
            with h5.File(self._file, "r") as f:
                if "orbits" not in f:
                    raise ValueError(f"No orbits found in population file ({self._file})")

                offsets = f["orbits"]["offsets"][...]
                pos, vel = f["orbits"]["pos"][...] * u.kpc, f["orbits"]["vel"][...] * u.km / u.s
                t = f["orbits"]["t"][...] * u.Myr

            # convert positions, velocities and times into a list of orbits
            self._orbits = np.array([gd.Orbit(pos[:, offsets[i]:offsets[i + 1]],
                                              vel[:, offsets[i]:offsets[i + 1]],
                                              t[offsets[i]:offsets[i + 1]]) for i in range(len(offsets) - 1)])

            # also calculate the final positions and velocities while you're at it
            final_inds = offsets[1:] - 1
            self._final_pos = pos[:, final_inds].T
            self._final_vel = vel[:, final_inds].T
        return self._orbits

    @property
    def primary_orbits(self):
        """The orbits of primary stars

        Returns
        -------
        primary_orbits : :class:`~gala.dynamics.Orbit`, shape (len(self),)
            The orbits of the primary stars in the population
        """
        return self.orbits[:len(self)]

    @property
    def secondary_orbits(self):
        """The orbits of secondary stars

        Returns
        -------
        secondary_orbits : :class:`~gala.dynamics.Orbit`, shape (len(self),)
            The orbits of the secondary stars in the population
        """
        order = np.argsort(np.concatenate((self.bin_nums[~self.disrupted], self.bin_nums[self.disrupted])))
        return np.concatenate((self.primary_orbits[~self.disrupted], self.orbits[len(self):]))[order] 

    @property
    def classes(self):
        """A table of classes that apply to each binary.

        Each row corresponds to a binary and the columns are a boolean of whether they meet the criteria for
        a class. For a full description of classes and their criteria,
        run :func:`~cogsworth.classify.list_classes`.

        Returns
        -------
        classes : :class:`~pandas.DataFrame`
            The classes that apply to each binary.
        """
        if self._classes is None:
            BOLD, RESET = "\033[1m", "\033[0m"
            logging.getLogger("cogswoth").info(f"{BOLD}cogsworth info:{RESET} No classes calculated yet, running now")
            self._classes = determine_final_classes(population=self)
        return self._classes

    @property
    def final_pos(self):
        """The final position of each binary (or star from a disrupted binary) in the galaxy.

        Returns
        -------
        final_pos : :class:`~numpy.ndarray`, shape (len(self) + self.disrupted.sum(), 3)
            The final position of each binary (or star from a disrupted binary) in the galaxy.
        """
        if self._final_pos is None:
            self._final_pos, self._final_vel = self._get_final_coords()
        return self._final_pos

    @property
    def final_vel(self):
        """The final velocity of each binary (or star from a disrupted binary) in the galaxy.

        Returns
        -------
        final_vel : :class:`~numpy.ndarray`, shape (len(self) + self.disrupted.sum(), 3)
            The final velocity of each binary (or star from a disrupted binary) in the galaxy.
        """
        if self._final_vel is None:
            self._final_pos, self._final_vel = self._get_final_coords()
        return self._final_vel

    @property
    def final_bpp(self):
        """The final state of each binary in the population.

        This is simply the final row from the :attr:`bpp` table for each binary.

        Returns
        -------
        final_bpp : :class:`~pandas.DataFrame`
            The final state of each binary in the population.
        """
        if self._final_bpp is None:
            self._final_bpp = self.bpp.drop_duplicates(subset="bin_num", keep="last")
            self._final_bpp.insert(len(self._final_bpp.columns), "metallicity",
                                   self.initC["metallicity"].values)
        return self._final_bpp

    @property
    def disrupted(self):
        """A mask of whether a binary was disrupted during its evolution.

        Returns
        -------
        disrupted : :class:`~numpy.ndarray`, shape (len(self),)
            A mask of whether a binary was disrupted during its evolution.
        """
        if self._disrupted is None:
            # check for disruptions in THREE different ways because COSMIC isn't always consistent (:
            self._disrupted = (self.final_bpp["bin_num"].isin(self.kick_info[self.kick_info["disrupted"] == 1.0]["bin_num"].unique())
                               & (self.final_bpp["sep"] < 0.0)
                               & self.final_bpp["bin_num"].isin(self.bpp[self.bpp["evol_type"] == 11.0]["bin_num"])).values
        return self._disrupted

    @property
    def escaped(self):
        """A mask of whether a binary escaped the galaxy during its evolution.

        This is calculated by comparing the final velocity of the binary to the escape velocity at its
        final position.

        Returns
        -------
        escaped : :class:`~numpy.ndarray`, shape (len(self),)
            A mask of whether a binary escaped the galaxy during its evolution.
        """
        if self._escaped is None:
            self._escaped = np.repeat(False, len(self))

            # get the current velocity
            v_curr = np.sum(self.final_vel**2, axis=1)**(0.5)

            # 0.5 * m * v_esc**2 = m * (-Phi)
            v_esc = np.sqrt(-2 * self.galactic_potential(self.final_pos.T))
            self._escaped = v_curr >= v_esc
        return self._escaped

    @property
    def observables(self):
        """A table of the observable properties of each binary.

        Returns
        -------
        observables : :class:`~pandas.DataFrame`
            The observable properties of each binary. Columns are defined in
            :func:`~cogsworth.obs.observables.get_observables`.

        Raises
        ------
        ValueError
            If no observables have been calculated yet.
        """
        if self._observables is None:
            raise ValueError("Observables not yet calculated, run `get_observables` to do so")
        else:
            return self._observables

    def _close_pool(self):
        """Close the multiprocessing pool if it exists."""
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self.pool_config = None

    def _ensure_pool(self, quiet):
        """Ensure the multiprocessing pool is set up."""
        # if the config has changed and an old pool exists, close the old pool
        config = (self.galactic_potential, self.max_ev_time, self.store_entire_orbits, quiet,
                  self.integrator, self.integrator_kwargs)
        if self.pool is not None and self.pool_config != config:        # pragma: no cover
            self._close_pool()

        # set up a new pool if needed
        if self.processes > 1 and self.pool is None:
            self.pool = Pool(self.processes, initializer=_init_pool, initargs=(*config,))
            self.pool_config = config

    def create_population(self, with_timing=True):
        """Create an entirely evolved population of binaries.

        This will sample the initial binaries and initial galaxy and then perform both the :py:mod:`cosmic`
        and :py:mod:`gala` evolution.

        Parameters
        ----------
        with_timing : `bool`, optional
            Whether to print messages about the timing, by default True
        """
        if with_timing:
            start = time.time()
            print(f"Run for {self.n_binaries} binaries")

        self.sample_initial_binaries()
        if with_timing:
            print(f"Ended up with {self.n_binaries_match} binaries with m1 > {self.m1_cutoff} solar masses")
            print(f"[{time.time() - start:1.0e}s] Sample initial binaries")
            lap = time.time()

        if self.bcm_timestep_conditions != []:
            set_checkstates(self.bcm_timestep_conditions)

        self._ensure_pool(quiet=False)
        self.perform_stellar_evolution()
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Evolve binaries (run COSMIC)")
            lap = time.time()

        self.perform_galactic_evolution(progress_bar=with_timing)
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")

        self._close_pool()

        if with_timing:
            print(f"Overall: {time.time() - start:1.1f}s")

    def sample_initial_galaxy(self):
        """Sample the initial galactic times, positions and velocities"""
        # initialise the initial galaxy class with correct number of binaries
        self._initial_galaxy = self.sfh_model(size=self.n_binaries_match, **self.sfh_params)

        # add relevant citations
        self.__citations__.extend([c for c in self._initial_galaxy.__citations__ if c != "cogsworth"])

        # if velocities are already set then just immediately return
        if all(hasattr(self._initial_galaxy, attr) for attr in ["v_R", "v_T", "v_z"]):   # pragma: no cover
            return

        # work out the initial velocities of each binary
        vel_units = u.km / u.s

        # calculate the Galactic circular velocity at the initial positions
        v_circ = self.galactic_potential.circular_velocity(q=[self._initial_galaxy.x,
                                                              self._initial_galaxy.y,
                                                              self._initial_galaxy.z],
                                                           t=(self.max_ev_time - self._initial_galaxy.tau)).to(vel_units)

        # add some velocity dispersion
        v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
                                         self.v_dispersion.to(vel_units) / np.sqrt(3),
                                         size=(3, self.n_binaries_match))
        v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units
        self._initial_galaxy.v_R = v_R
        self._initial_galaxy.v_T = v_T
        self._initial_galaxy.v_z = v_z

    def sample_initial_binaries(self, initC=None, overwrite_initC_settings=True, reset_sampled_kicks=True):
        """Sample the initial binary parameters for the population.

        Alternatively, copy initial conditions from another population

        Parameters
        ----------
        initC : :class:`~pandas.DataFrame`, optional
            Initial conditions from a different Population, by default None (new sampling performed)
        overwrite_initC_settings : `bool`, optional
            Whether to overwrite initC settings in the case where the new population has a different set of
            `BSE_settings`, by default True
        reset_sampled_kicks : `bool`, optional
            Whether to reset any sampled kicks in the population to ensure new ones are drawn, by default True
        """
        self._bin_nums = None
        self._final_bpp = None
        self._initC = None
        self._initial_binaries = None

        # if an initC table is provided then use that instead of sampling
        if initC is not None:
            self._initial_binaries = copy(initC)

            # if we are allowed to overwrite setting then replace columns
            if overwrite_initC_settings:
                for key in self.BSE_settings:
                    if key in self._initial_binaries:
                        self._initial_binaries[key] = self.BSE_settings[key]

            # reset sampled kicks if desired
            if reset_sampled_kicks:
                cols = ["natal_kick_1", "phi_1", "theta_1", "natal_kick_2", "phi_2", "theta_2"]
                for col in cols:
                    self._initial_binaries[col] = -100.0
        else:
            if "binfrac" not in self.BSE_settings:
                raise ValueError("You must specify a binary fraction (e.g. `binfrac: 0.5`) in "
                                 "`Population.BSE_settings` to sample binaries")
            if self.BSE_settings["binfrac"] == 0.0 and not self.sampling_params["keep_singles"]:
                raise ValueError(("You've chosen a binary fraction of 0.0 but set `keep_singles=False` (in "
                                  "self.sampling_params), so you'll draw 0 samples...I don't think you "
                                  "wanted to do that?"))
            self._initial_binaries, self._mass_singles, self._mass_binaries, self._n_singles_req, \
                self._n_bin_req = InitialBinaryTable.sampler('independent',
                                                             self.final_kstar1, self.final_kstar2,
                                                             binfrac_model=self.BSE_settings["binfrac"],
                                                             SF_start=self.max_ev_time.to(u.Myr).value,
                                                             SF_duration=0.0, met=0.02, size=self.n_binaries,
                                                             **self.sampling_params)

        # apply the mass cutoff
        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

        # reset index to match new `bin_num`s
        self._initial_binaries.reset_index(inplace=True)

        # count how many binaries actually match the criteria (may be larger than `n_binaries` due to sampler)
        self.n_binaries_match = len(self._initial_binaries)

        # check that any binaries remain
        if self.n_binaries_match == 0:
            raise ValueError(("Your choice of `m1_cutoff` resulted in all samples being thrown out. Consider"
                              " a larger sample size or a less stringent mass cut"))

        self.sample_initial_galaxy()

        # update the metallicity and birth times of the binaries to match the galaxy
        self._initial_binaries["metallicity"] = self._initial_galaxy.Z
        self._initial_binaries["tphysf"] = self._initial_galaxy.tau.to(u.Myr).value

        # ensure metallicities remain in a range valid for COSMIC - original value still in initial_galaxy.Z
        self._initial_binaries.loc[self._initial_binaries["metallicity"] < 1e-4, "metallicity"] = 1e-4
        self._initial_binaries.loc[self._initial_binaries["metallicity"] > 0.03, "metallicity"] = 0.03

    def perform_stellar_evolution(self):
        """Perform the (binary) stellar evolution of the sampled binaries"""
        # delete any cached variables
        self._final_bpp = None
        self._observables = None
        self._bin_nums = None
        self._disrupted = None
        self._escaped = None

        if self.bcm_timestep_conditions != []:
            set_checkstates(self.bcm_timestep_conditions)

        # if no initial binaries have been sampled then we need to create some
        if self._initial_binaries is None and self._initC is None:
            logging.getLogger("cogsworth").warning(("cogsworth warning: Initial binaries not yet sampled, "
                                                    "performing sampling now."))
            self.sample_initial_binaries()

        no_pool_existed = self.pool is None and self.processes > 1
        if no_pool_existed:
            self._ensure_pool(quiet=False)

        # catch any warnings about overwrites
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*initial binary table is being overwritten.*")
            warnings.filterwarnings("ignore", message=".*to a different value than assumed in the mlwind.*")

            ibt = self.initial_binaries if self._initC is None else self._initC
            BSEDict = self.BSE_settings
            if "kickflag" in ibt.columns and BSEDict != {}:
                BSEDict = {}
                logging.getLogger("cogsworth").warning("cogsworth warning: You passed settings for BSE (in `Population.BSE_settings`) but "
                                                       "your initial binary table already has settings saved "
                                                       "in its columns. cogsworth will use the settings "
                                                       "found in the table.")

            # perform the evolution!
            self._bpp, bcm, self._initC, \
                self._kick_info = Evolve.evolve(initialbinarytable=ibt,
                                                BSEDict=BSEDict, pool=self.pool,
                                                timestep_conditions=self.bcm_timestep_conditions,
                                                bpp_columns=self.bpp_columns, bcm_columns=self.bcm_columns)

            # only save BCM when it has interesting timesteps
            if self.bcm_timestep_conditions != []:
                self._bcm = bcm

        if no_pool_existed:
            self._close_pool()

        # check if there are any NaNs in the final bpp table rows or the kick_info
        nans = np.isnan(self.final_bpp["sep"])
        kick_info_nans = np.isnan(self._kick_info["delta_vsysx_1"])

        # if we detect NaNs
        if nans.any() or kick_info_nans.any():      # pragma: no cover
            # make sure the user knows bad things have happened
            logging.getLogger("cogsworth").warning("NaNs detected in COSMIC evolution")

            # store the bad things for later
            nan_bin_nums = np.unique(np.concatenate((self.final_bpp[nans]["bin_num"].values,
                                                     self._kick_info[kick_info_nans]["bin_num"].values)))
            self._bpp[self._bpp["bin_num"].isin(nan_bin_nums)].to_hdf("nans.h5", key="bpp")
            self._initC[self._initC["bin_num"].isin(nan_bin_nums)].to_hdf("nans.h5", key="initC")
            self._kick_info[self._kick_info["bin_num"].isin(nan_bin_nums)].to_hdf("nans.h5", key="kick_info")

            # update the population to delete any bad binaries
            n_nan = len(nan_bin_nums)
            self.n_binaries_match -= n_nan
            self._bpp = self._bpp[~self._bpp["bin_num"].isin(nan_bin_nums)]

            if self._bcm is not None:
                self._bcm = self._bcm[~self._bcm["bin_num"].isin(nan_bin_nums)]
            self._kick_info = self._kick_info[~self._kick_info["bin_num"].isin(nan_bin_nums)]
            self._initC = self._initC[~self._initC["bin_num"].isin(nan_bin_nums)]

            not_nan = ~self.final_bpp["bin_num"].isin(nan_bin_nums)
            self._initial_galaxy._tau = self._initial_galaxy._tau[not_nan]
            self._initial_galaxy._Z = self._initial_galaxy._Z[not_nan]
            self._initial_galaxy._x = self._initial_galaxy._x[not_nan]
            self._initial_galaxy._y = self._initial_galaxy._y[not_nan]
            self._initial_galaxy._z = self._initial_galaxy._z[not_nan]

            for attr in ["v_R", "v_T", "v_z", "_which_comp"]:
                if hasattr(self._initial_galaxy, attr):
                    setattr(self._initial_galaxy, attr, getattr(self._initial_galaxy, attr)[not_nan])
            self._initial_galaxy._size -= n_nan

            # reset final bpp
            self._final_bpp = None

            logging.getLogger("cogsworth").warning((f"{n_nan} bad binaries removed from tables - but "
                                                    "normalisation may be off. I've added the offending "
                                                    "binaries to a `nan.h5` file with their initC, bpp, "
                                                    "and kick_info tables"))

    def perform_galactic_evolution(self, quiet=False, progress_bar=True):
        """Use :py:mod:`gala` to perform the orbital integration for each evolved binary

        Parameters
        ----------
        quiet : `bool`, optional
            Whether to silence any warnings about failing orbits, by default False
        """
        # delete any cached variables that are based on orbits
        self._final_pos = None
        self._final_vel = None
        self._observables = None

        if self._initial_galaxy is None:            # pragma: no cover
            logging.getLogger("cogsworth").warning(("cogsworth warning: Initial galaxy not yet sampled, "
                                                    "performing sampling now."))
            self.sample_initial_galaxy()

        if self._initC is None and self._initial_binaries is None:          # pragma: no cover
            logging.getLogger("cogsworth").warning(("cogsworth warning: Initial binaries not yet sampled, "
                                                    "performing sampling now."))
            self.sample_initial_binaries()

        if self._bpp is None:           # pragma: no cover
            logging.getLogger("cogsworth").warning(("cogsworth warning: Stellar evolution not yet performed, "
                                                    "performing evolution now."))
            self.perform_stellar_evolution()

        v_phi = (self.initial_galaxy.v_T / self.initial_galaxy.rho)
        v_X = (self.initial_galaxy.v_R * np.cos(self.initial_galaxy.phi)
               - self.initial_galaxy.rho * np.sin(self.initial_galaxy.phi) * v_phi)
        v_Y = (self.initial_galaxy.v_R * np.sin(self.initial_galaxy.phi)
               + self.initial_galaxy.rho * np.cos(self.initial_galaxy.phi) * v_phi)

        # combine the representation and differentials into a Gala PhaseSpacePosition
        w0s = gd.PhaseSpacePosition(pos=[a.to(u.kpc).value for a in [self.initial_galaxy.x,
                                                                     self.initial_galaxy.y,
                                                                     self.initial_galaxy.z]] * u.kpc,
                                    vel=[a.to(u.km/u.s).value for a in [v_X, v_Y,
                                                                        self.initial_galaxy.v_z]] * u.km/u.s)

        # randomly drawn phase and inclination angles as necessary
        for col in ["phase_sn_1", "phase_sn_2"]:
            if col not in self.initC:
                self.initC[col] = np.random.uniform(0, 2 * np.pi, len(self.initC))
        for col in ["inc_sn_1", "inc_sn_2"]:
            if col not in self.initC:
                self.initC[col] = np.arccos(2 * np.random.rand(len(self.initC)) - 1.0)

        # identify the pertinent events in the evolution
        primary_events, secondary_events = identify_events(p=self)

        # if we want to use multiprocessing
        if self.pool is not None or self.processes > 1:
            # track whether a pool already existed
            pool_existed = self.pool is not None
            self._ensure_pool(quiet=quiet)

            # setup arguments to combine primary and secondaries into a single list
            primary_args = [(w0s[i], self.max_ev_time - self.initial_galaxy.tau[i],
                             copy(self.timestep_size), primary_events[i])
                            for i in range(self.n_binaries_match)]
            secondary_args = [(w0s[i], self.max_ev_time - self.initial_galaxy.tau[i],
                               copy(self.timestep_size), secondary_events[i])
                              for i in range(self.n_binaries_match) if secondary_events[i] is not None]
            args = primary_args + secondary_args

            # evolve the orbits from birth until present day
            if progress_bar:
                orbits = self.pool.starmap(_orbit_worker, tqdm(args, total=self.n_binaries_match))
            else:
                orbits = self.pool.starmap(_orbit_worker, args)

            # if a pool didn't exist before then close the one just created
            if not pool_existed:
                self._close_pool()
        else:
            # otherwise just use a for loop to evolve the orbits from birth until present day
            orbits = []
            for i in range(self.n_binaries_match):
                orbits.append(integrate_orbit_with_events(w0=w0s[i], potential=self.galactic_potential,
                                                          t1=self.max_ev_time - self.initial_galaxy.tau[i],
                                                          t2=self.max_ev_time, dt=copy(self.timestep_size),
                                                          events=primary_events[i], quiet=quiet,
                                                          store_all=self.store_entire_orbits,
                                                          integrator=self.integrator,
                                                          integrator_kwargs=self.integrator_kwargs))
            for i in range(self.n_binaries_match):
                if secondary_events[i] is None:
                    continue
                orbits.append(integrate_orbit_with_events(w0=w0s[i], potential=self.galactic_potential,
                                                          t1=self.max_ev_time - self.initial_galaxy.tau[i],
                                                          t2=self.max_ev_time, dt=copy(self.timestep_size),
                                                          events=secondary_events[i], quiet=quiet,
                                                          store_all=self.store_entire_orbits,
                                                          integrator=self.integrator,
                                                          integrator_kwargs=self.integrator_kwargs))

        # check for bad orbits
        bad_orbits = np.array([orbit is None for orbit in orbits])

        # if there are any bad orbits then warn the user and remove them from the population
        if any(bad_orbits):             # pragma: no cover
            # start a warning message
            bad_bin_nums = np.concatenate((self.bin_nums, self.bin_nums[self.disrupted]))[bad_orbits]
            warning_message = (
                f"cogsworth warning: {bad_orbits.sum()} orbit(s) failed numerical integration, removing them."
                " This can occur due to NaNs in stellar evolution or extreme orbits that Gala cannot handle."
            )

            # if we're going to save them then find a file name
            if self.error_file_path is not None:
                # decide on file name (avoid overwriting existing file)
                file_num = 1
                file_name = os.path.join(self.error_file_path, "bad_orbits.h5")
                while os.path.exists(file_name):
                    file_name = os.path.join(self.error_file_path, f"bad_orbits_{file_num}.h5")
                    file_num += 1

                # save the bad orbits population
                self.initC.loc[bad_bin_nums].to_hdf(file_name, key="initC")
                self.bpp.loc[bad_bin_nums].to_hdf(file_name, key="bpp")
                self.kick_info.loc[bad_bin_nums].to_hdf(file_name, key="kick_info")
                self.initial_galaxy[np.isin(self.bin_nums, bad_bin_nums)].save(file_name, key="sfh")
        
                warning_message += f" Information for these systems was saved to `{file_name}`."
                warning_message += " This includes their initC, bpp, kick_info, and initial galaxy objects."
            logging.getLogger("cogsworth").warning(warning_message)

            # mask them out from the main population
            new_self = self[~np.isin(self.bin_nums, bad_bin_nums)]
            self.__dict__.update(new_self.__dict__)

            # also mask them out from the orbits
            orbits = np.array(orbits, dtype="object")[~bad_orbits]

        self._orbits = np.array(orbits, dtype="object")

    def _get_final_coords(self):
        """Get the final coordinates of each binary (or each component in disrupted binaries)

        Returns
        -------
        final_pos, final_vel : :class:`~astropy.quantity.Quantity`
            Final positions and velocities of each system in the galactocentric frame.
            The first `len(self)` entries of each are for bound binaries or primaries, then the final
            `self.disrupted.sum()` entries are for disrupted secondaries. Any missing orbits (where orbit=None
            will be set to `np.inf` for ease of masking.
        """
        if self._file is not None:
            with h5.File(self._file, "r") as f:
                offsets = f["orbits"]["offsets"][...]
                pos, vel = f["orbits"]["pos"][...] * u.kpc, f["orbits"]["vel"][...] * u.km / u.s

            final_inds = offsets[1:] - 1

            self._final_pos = pos[:, final_inds].T
            self._final_vel = vel[:, final_inds].T
            del pos, vel
        else:
            # pool all of the orbits into a single numpy array
            self._final_pos = np.ones((len(self.orbits), 3)) * np.inf
            self._final_vel = np.ones((len(self.orbits), 3)) * np.inf
            for i, orbit in enumerate(self.orbits):
                # check if the orbit is missing
                if orbit is None:
                    warnings.warn("Warning: Detected `None` orbit, entering coordinates as `np.inf`")
                else:
                    self._final_pos[i] = orbit[-1].pos.xyz.to(u.kpc).value
                    self._final_vel[i] = orbit[-1].vel.d_xyz.to(u.km / u.s).value
            self._final_pos *= u.kpc
            self._final_vel *= u.km / u.s
        return self._final_pos, self._final_vel

    def get_observables(self, **kwargs):
        """Get observables associated with the binaries at present day.

        These include: extinction due to dust, absolute and apparent bolometric magnitudes for each star,
        apparent magnitudes in each filter and observed temperature and surface gravity for each binary.

        For bound binaries and stellar mergers, only the column `{filter}_app_1` is relevant. For
        disrupted binaries, `{filter}_app_1` is for the primary star and `{filter}_app_2` is for
        the secondary star.

        Parameters
        ----------
        **kwargs to pass to :func:`~cogsworth.obs.observables.get_photometry`
        """
        self.__citations__.extend(["MIST", "MESA", "bayestar2019"])
        self._observables = get_photometry(population=self, **kwargs)
        return self._observables

    def get_gaia_observed_bin_nums(self, ra=None, dec=None, gaia_G_filter="Gaia_G_EDR3"):
        """Get a list of ``bin_nums`` of systems that are bright enough to be observed by Gaia.

        This is calculated based on the Gaia selection function provided by :mod:`gaiaunlimited`. This
        function returns a **random sample** of systems where systems are included in the observed subset
        with a probability given by Gaia's completeness at their location.

        E.g. if Gaia's completeness is 0 for a source of a given magnitude and location then it will never be
        included. Similarly, if the completeness is 1 then it will always be included. However, if the
        completeness is 0.5 then it will only be included in the list of ``bin_nums`` half of the time.

        Parameters
        ----------
        ra : :class:`~numpy.ndarray` or `None`
            Right ascension of the binaries. Either supply value or set to "auto" to use the RA from the
            population's observables.
        dec : :class:`~numpy.ndarray` or `None`
            Declination of the binaries. Either supply value or set to "auto" to use the Dec from the
            population's observables.
        gaia_G_filter : str
            The name of the Gaia G-band filter to use for apparent magnitude (e.g. "Gaia_G_EDR3").

        Returns
        -------
        primary_observed : :class:`~numpy.ndarray`
            A list of binary numbers (that can be used in tables like :attr:`final_bpp`) for which the
            bound binary/disrupted primary would be observed
        secondary_observed : :class:`~numpy.ndarray`
            A list of binary numbers (that can be used in tables like :attr:`final_bpp`) for which the
            disrupted secondary would be observed
        """
        assert check_dependencies("gaiaunlimited")
        from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
        from gaiaunlimited.utils import get_healpix_centers

        self.__citations__.append("gaia-selection-function")
        # get coordinates of the centres of the healpix pixels in a nside=2**7
        coords_of_centers = get_healpix_centers(7)

        # get the Gaia selection function for this healpix order
        dr3sf = DR3SelectionFunctionTCG()

        # work out the index of each pixel for every binary
        pix_inds = self.get_healpix_inds(ra=ra, dec=dec, nside=128)

        # loop over first (all bound binaries & primaries from disrupted binaries)
        # and then (secondaries from disrupted binaries)
        observed = []
        for pix, g_mags, bin_nums in zip([pix_inds[:len(self)], pix_inds[len(self):]],
                                         [self.observables[f"{gaia_G_filter}_app_1"].values,
                                          self.observables[f"{gaia_G_filter}_app_2"][self.disrupted].values],
                                         [self.bin_nums, self.bin_nums[self.disrupted]]):
            # get the coordinates of the corresponding pixels
            comp_coords = coords_of_centers[pix]

            # ensure any NaNs in the magnitudes are just set to super faint
            g_mags = np.nan_to_num(g_mags, nan=1000)
            g_mags = g_mags.value if hasattr(g_mags, 'unit') else g_mags

            # by default, assume Gaia has 0 completeness for each source
            completeness = np.zeros(len(g_mags))

            # only bother getting completeness for things brighter than G=22 (everything fainter is 0)
            bright_enough = g_mags < 22.0
            observed_bin_nums = np.array([])
            if bright_enough.any():
                completeness[bright_enough] = dr3sf.query(comp_coords[bright_enough], g_mags[bright_enough])

                # draw a random sample from the systems based on Gaia's completeness at each coordinate
                observed_bin_nums = bin_nums[np.random.uniform(size=len(completeness)) < completeness]

            observed.append(observed_bin_nums)

        primary_observed, secondary_observed = observed
        return primary_observed, secondary_observed

    def get_final_mw_skycoord(self):
        """Get the final positions and velocities as an astropy SkyCoord object

        ..warning::

            This function assumes the final positions and velocities are in the MW galactocentric frame"""
        return coords.SkyCoord(x=self.final_pos[:, 0], y=self.final_pos[:, 1], z=self.final_pos[:, 2],
                               v_x=self.final_vel[:, 0], v_y=self.final_vel[:, 1], v_z=self.final_vel[:, 2],
                               representation_type="cartesian", unit=u.kpc, frame="galactocentric")

    def plot_sky_locations(self, fig=None, ax=None, show=True, show_galactic_plane=True, **kwargs):
        """Plot the final positions of the binaries in the Milky Way galactocentric frame"""
        assert check_dependencies(["healpy", "matplotlib"])
        import matplotlib.pyplot as plt

        # get the final positions and velocities
        final_coords = self.get_final_mw_skycoord().icrs

        # plot the positions
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # plot the galactic plane if desired
        if show_galactic_plane:
            galactic_plane = coords.SkyCoord(l=np.linspace(1e-10, 2 * np.pi, 10000), b=np.zeros(10000),
                                             unit="rad", frame="galactic").transform_to("icrs")
            in_order = np.argsort(galactic_plane.ra.value)
            ax.plot(galactic_plane.ra.value[in_order], galactic_plane.dec.value[in_order],
                    label="Galactic Plane", color="black" if ax.get_facecolor() == (1,1,1,0) else "white",
                    linestyle="dotted")

        ax.scatter(final_coords.ra.value, final_coords.dec.value, **kwargs)
        ax.set(xlabel="Right Ascension [deg]", ylabel="Declination [deg]")
        ax.legend()

        if show:
            plt.show()
        return fig, ax

    def get_healpix_inds(self, ra=None, dec=None, nside=128):
        """Get the indices of the healpix pixels that each binary is in

        Parameters
        ----------
        ra : `float` or `str`
            Either the right ascension of the system in radians or "auto" to automatically calculate assuming
            Milky Way galactocentric coordinates
        dec : `float` or `str`
            Either the declination of the system in radians or "auto" to automatically calculate assuming
            Milky Way galactocentric coordinates
        nside : `int`, optional
            Healpix nside parameter, by default 128

        Returns
        -------
        pix : :class:`~numpy.ndarray`
            The indices for each system
        """
        assert check_dependencies("healpy")
        import healpy as hp

        if ra is None or dec is None:
            raise ValueError("You must provide both `ra` and `dec`, or set them to 'auto'")
        if ra == "auto" or dec == "auto":
            final_coords = self.get_final_mw_skycoord()
            ra = final_coords.icrs.ra.to(u.rad).value
            dec = final_coords.icrs.dec.to(u.rad).value

        # get the coordinates in right format
        colatitudes = np.pi / 2 - dec
        longitudes = ra

        # find the pixels for each bound binary/primary and for each disrupted secondary
        pix = hp.ang2pix(nside, nest=True, theta=colatitudes, phi=longitudes)
        return pix

    def plot_map(self, ra=None, dec=None, nside=128, coord="C",
                 cmap="magma", norm="log", unit=None, show=True, **mollview_kwargs):
        r"""Plot a healpix map of the final positions of all binaries in population

        Parameters
        ----------
        nside : `int`, optional
            Healpix nside parameter, by default 128
        coord : `str`, optional
            Which coordinates to plot. One of ["C[elestial]", "G[alactic", "E[quatorial]"], by default "C"
        cmap : `str`, optional
            A `matplotlib colormap <https://matplotlib.org/stable/gallery/color/colormap_reference.html>`_
            to use for the plot, by default "magma"
        norm : `str`, optional
            How to normalise the total number of binaries (anything linear or log), by default "log"
        unit : `str`, optional
            A label to use for the unit of the colour bar, by default ":math:`\log_{10}(N_{\rm binaries})`"
            if ``norm==log`` and ":math:`N_{\rm binaries}`" if ``norm==linear``
        show : `bool`, optional
            Whether to immediately show the plot, by default True
        **mollview_kwargs
            Any additional arguments that you want to pass to healpy's
            `mollview <https://healpy.readthedocs.io/en/latest/generated/healpy.visufunc.mollview.html>`_
        """
        assert check_dependencies(["healpy", "matplotlib"])
        import healpy as hp
        import matplotlib.pyplot as plt

        pix = self.get_healpix_inds(ra=ra, dec=dec, nside=nside)

        # initialise an empty map
        m = np.zeros(hp.nside2npix(nside))

        # count the unique pixel values and how many sources are in each
        inds, counts = np.unique(pix, return_counts=True)

        # apply a log if desired
        if norm == "log":
            counts = np.log10(counts)

        # fill in the map
        m[inds] = counts

        # if the user wants a different coordinate then use the list format to convert
        if coord.lower() not in ["c", "celestial"]:
            coord = ["C", coord]
        # otherwise just pass plain old celestial
        else:
            coord = "C"

        # if a unit isn't provided then we can write one down automatically
        if unit is None:
            unit = r"$\log(N_{\rm binaries})$" if norm == "log" else r"$N_{\rm binaries}$"

        # create a mollview plot
        hp.mollview(m, nest=True, cmap=cmap, coord=coord, unit=unit, **mollview_kwargs)

        # show it if the user wants
        if show:
            plt.show()

    def translate_tables(self, **kwargs):
        """Translate the COSMIC BSE tables to human readable format

        Parameters
        ----------

        **kwargs : `various`
            Any arguments to pass to :func:`~cogsworth.utils.translate_COSMIC_tables`
        """
        with pd.option_context('mode.chained_assignment', None):
            self._bpp = translate_COSMIC_tables(self._bpp, **kwargs)
            self._final_bpp = translate_COSMIC_tables(self._final_bpp, **kwargs)

            if self._bcm is not None:
                kwargs.update({"evol_type": False})
                self._bcm = translate_COSMIC_tables(self._bcm, **kwargs)

    def plot_cartoon_binary(self, bin_num, **kwargs):
        """Plot a cartoon of the evolution of a single binary

        Parameters
        ----------
        bin_num : `int`
            Which binary to plot
        **kwargs : `various`
            Keyword arguments to pass, see :func:`~cogsworth.plot.plot_cartoon_evolution` for options

        Returns
        -------
        fig, ax : :class:`~matplotlib.pyplot.figure`, :class:`~matplotlib.pyplot.axis`
            Figure and axis of the plot
        """
        return plot_cartoon_evolution(self.bpp, bin_num, **kwargs)

    def plot_orbit(self, bin_num, show_sn=True, show_merger=True, sn_kwargs={}, show=True,
                   show_legend=True, **kwargs):
        """Plot the Galactic orbit of a binary system

        Parameters
        ----------
        bin_num : `int`
            Which binary to plot
        show_sn : `bool`, optional
            Whether to show the position of the SNe, by default True
        show_merger : `bool`, optional
            Whether to show the position of a merger, by default True
        sn_kwargs : `dict`, optional
            Any additional arguments to pass to the SN scatter plot call, by default {}
        show : `bool`, optional
            Whether to immediately show the plot
        show_legend : `bool`, optional
            Whether to show the legend
        **kwargs : `various`
            Any additional arguments to pass to :func:`cogsworth.plot.plot_galactic_orbit`

        Returns
        -------
        fig, axes : :class:`~matplotlib.pyplot.figure`, :class:`~numpy.ndarray`
            Figure and axes of the plot

        Raises
        ------
        ValueError
            If the `bin_num` isn't in the population
        """
        # check that the bin_num is in the population
        if bin_num not in self.bin_nums:
            raise ValueError("bin_num not in population")
        # find the index of the binary and whether it was disrupted
        ind = np.where(self.bin_nums == bin_num)[0][0]
        disrupted = self.disrupted[ind]

        # set the orbits
        primary_orbit = self.primary_orbits[ind]
        secondary_orbit = self.secondary_orbits[ind] if disrupted else None

        fig, axes = kwargs.pop("fig", None), kwargs.pop("axes", None)
        if fig is None or axes is None:
            fig, axes = plt.subplots(1, 3, figsize=(25, 7))
            fig.subplots_adjust(wspace=0.35)

        # create the orbit plot
        fig, axes = plot_galactic_orbit(primary_orbit, secondary_orbit, show=False,
                                        show_legend=False, fig=fig, axes=axes, **kwargs)

        # extra bit if you want to show the SN locations
        if show_sn or show_merger:
            # set the colours
            colours = [None, None, "black"]
            if "primary_kwargs" in kwargs and "color" in kwargs["primary_kwargs"]:      # pragma: no cover
                colours[0] = kwargs["primary_kwargs"]["color"]
            if "secondary_kwargs" in kwargs and "color" in kwargs["secondary_kwargs"]:  # pragma: no cover
                colours[1] = kwargs["secondary_kwargs"]["color"]

            # loop over the primary and secondary supernova + merger
            rows = self.bpp.loc[bin_num]
            for loop in zip([((rows["evol_type"] == 15) | ((rows["evol_type"] == 16) & (rows["sep"] == 0.0))),
                             (rows["evol_type"] == 16) & (rows["sep"] != 0.0),
                             rows["sep"] == 0.0],
                            [primary_orbit, (secondary_orbit if disrupted else primary_orbit), primary_orbit],
                            colours,
                            ["Primary supernova", "Secondary supernova", "Merger"],
                            [show_sn, show_sn, show_merger]):
                mask, orbit, colour, l, show_obj = loop
                # if there is no SN or no orbit then skip
                if not np.any(mask) or orbit is None or not show_obj:           # pragma: no cover
                    continue

                # find the time of the SN the closest position before it occurs
                sn_time = rows["tphys"][mask].iloc[0] * u.Myr
                sn_pos = orbit[orbit.t - orbit.t[0] <= sn_time][-1]

                # don't plot the SN if it is outside the time range
                if "t_max" in kwargs and kwargs["t_max"] is not None and sn_time > kwargs["t_max"]:
                    continue

                # some default style settings
                full_sn_kwargs = {
                    "marker": (10, 2, 0) if "supernova" in l else "P",
                    "color": colour,
                    "s": 100,
                    "label": l,
                    "zorder": 10,
                }
                full_sn_kwargs.update(sn_kwargs)

                # plot individual SN positions
                axes[0].scatter(sn_pos.x, sn_pos.y, **full_sn_kwargs)
                axes[1].scatter(sn_pos.x, sn_pos.z, **full_sn_kwargs)
                axes[2].scatter(sn_pos.y, sn_pos.z, **full_sn_kwargs)

        if show_legend:
            handles, labels = axes[-1].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=16)
            fig.subplots_adjust(top=0.9)

        if show:
            plt.show()

        return fig, axes

    def to_legwork_sources(self, distances=None, assume_mw_galactocentric=False):
        """Convert the final state of the population to a LEGWORK Source class (only including bound binaries)

        Parameters
        ----------
        distances : :class:`np.ndarray`, optional
            Custom distance to each source, by default None
        assume_mw_galactocentric : `bool`, optional
            Assume population is in Milky Way and galactocentric coordinate system - such that distances can
            be calculated assuming the present location of the Earth, by default False

        Returns
        -------
        sources : :class:`~legwork.sources.Source`
            LEGWORK sources

        Raises
        ------
        ValueError
            When distances aren't provided and assume_mw_galactocentric=False
        """
        assert check_dependencies(["legwork"])
        import legwork as lw
        self.__citations__.append("legwork")

        # calculate distances as necessary
        if distances is None and not assume_mw_galactocentric:
            raise ValueError("Must either provide distances or set assume_mw_galactocentric=True")
        elif distances is None:
            final_coords = coords.SkyCoord(x=self.final_pos[:, 0], y=self.final_pos[:, 1],
                                           z=self.final_pos[:, 2], frame="galactocentric", unit=u.kpc,
                                           representation_type="cartesian")
            distances = final_coords.icrs.distance[:len(self)]

        # mask on only bound binaries
        binaries = self.final_bpp["sep"] > 0.0

        # construct LEGWORK sources
        return lw.source.Source(m_1=self.final_bpp["mass_1"].values[binaries] * u.Msun,
                                m_2=self.final_bpp["mass_2"].values[binaries] * u.Msun,
                                a=self.final_bpp["sep"].values[binaries] * u.Rsun,
                                ecc=self.final_bpp["ecc"].values[binaries],
                                dist=distances[binaries], interpolate_g=binaries.sum() > 1000)

    def save(self, file_name, overwrite=False):
        """Save a Population to disk as an HDF5 file.

        Parameters
        ----------
        file_name : `str`
            A file name to use. Either no file extension or ".h5".
        overwrite : `bool`, optional
            Whether to overwrite any existing files, by default False

        Raises
        ------
        FileExistsError
            If `overwrite=False` and files already exist
        """
        if file_name[-3:] != ".h5":
            file_name += ".h5"
        if os.path.isfile(file_name):
            if overwrite:
                os.remove(file_name)
            else:
                raise FileExistsError((f"{file_name} already exists. Set `overwrite=True` to overwrite "
                                       "the file."))

        # save initial binaries, preferably the initC table
        if self._initC is not None:
            save_initC(file_name, self._initC, key="initC", settings_key="initC_settings")
        elif self._initial_binaries is not None:
            self._initial_binaries.to_hdf(file_name, key="initial_binaries")

        if self._bpp is not None:
            self._bpp.to_hdf(file_name, key="bpp")
        if self._bcm is not None:
            self._bcm.to_hdf(file_name, key="bcm")
        if self._kick_info is not None:
            self._kick_info.to_hdf(file_name, key="kick_info")

        with h5.File(file_name, "a") as f:
            f.attrs["potential_dict"] = yaml.dump(potential_to_dict(self.galactic_potential),
                                                  default_flow_style=None)
            
        if self._initial_galaxy is not None:
            self.initial_galaxy.save(file_name, key="initial_galaxy")

        # save the orbits if they have been calculated/loaded
        if self._orbits is not None:
            # go through the orbits calculate their lengths (and therefore offsets in the file)
            orbit_lengths = [len(orbit.pos) for orbit in self.orbits]
            orbit_lengths_total = sum(orbit_lengths)
            offsets = np.insert(np.cumsum(orbit_lengths), 0, 0)

            # start some empty arrays to store the data
            orbits_data = {"offsets": offsets,
                           "pos": np.zeros((3, orbit_lengths_total)),
                           "vel": np.zeros((3, orbit_lengths_total)),
                           "t": np.zeros(orbit_lengths_total)}

            # save each orbit to the arrays with the same units
            for i, orbit in enumerate(self.orbits):
                orbits_data["pos"][:, offsets[i]:offsets[i + 1]] = orbit.pos.xyz.to(u.kpc).value
                orbits_data["vel"][:, offsets[i]:offsets[i + 1]] = orbit.vel.d_xyz.to(u.km / u.s).value
                orbits_data["t"][offsets[i]:offsets[i + 1]] = orbit.t.to(u.Myr).value

            # save the orbits arrays to the file
            with h5.File(file_name, "a") as file:
                orbits = file.create_group("orbits")
                for key in orbits_data:
                    orbits[key] = orbits_data[key]

        with h5.File(file_name, "a") as file:
            numeric_params = np.array([self.n_binaries, self.n_binaries_match, self.processes, self.m1_cutoff,
                                       self.v_dispersion.to(u.km / u.s).value,
                                       self.max_ev_time.to(u.Gyr).value, self.timestep_size.to(u.Myr).value,
                                       self.mass_singles, self.mass_binaries, self.n_singles_req,
                                       self.n_bin_req])
            num_par = file.create_dataset("numeric_params", data=numeric_params)
            num_par.attrs["store_entire_orbits"] = self.store_entire_orbits

            num_par.attrs["final_kstar1"] = self.final_kstar1
            num_par.attrs["final_kstar2"] = self.final_kstar2
            num_par.attrs["timestep_conditions"] = self.bcm_timestep_conditions
            num_par.attrs["bpp_columns"] = np.array(self.bpp_columns, dtype="S")
            num_par.attrs["bcm_columns"] = np.array(self.bcm_columns, dtype="S")
            num_par.attrs["error_file_path"] = self.error_file_path

            # save BSE settings
            d = file.create_dataset("BSE_settings", data=[])
            for key in self.BSE_settings:
                d.attrs[key] = self.BSE_settings[key]

            # save sampling params
            d = file.create_dataset("sampling_params", data=[])
            d.attrs["dict"] = yaml.dump(self.sampling_params, default_flow_style=None)
            
            # save integrator settings
            d = file.create_dataset("integrator_settings", data=[])
            d.attrs["integrator"] = self.integrator.__name__
            d.attrs["integrator_kwargs"] = yaml.dump(self.integrator_kwargs, default_flow_style=None)

            # save the version of cogsworth, COSMIC, and gala that was used
            file.attrs["cogsworth_version"] = __version__
            file.attrs["COSMIC_version"] = cosmic_version
            file.attrs["gala_version"] = gala_version


def load(file_name, parts=["initial_binaries", "initial_galaxy", "stellar_evolution"]):
    """Load a Population from a series of files

    Parameters
    ----------
    file_name : `str`
        Base name of the files to use. Should either have no file extension or ".h5"
    parts : `list`, optional
        Which parts of the Population to load immediately, the rest are loaded as necessary. Any of
        ["initial_binaries", "initial_galaxy", "stellar_evolution", "galactic_orbits"], by default
        ["initial_binaries", "initial_galaxy", "stellar_evolution"]

    Returns
    -------
    pop : `Population`
        The loaded Population
    """
    if file_name[-3:] != ".h5":
        file_name += ".h5"

    BSE_settings = {}
    sampling_params = {}
    with h5.File(file_name, "r") as file:
        if "numeric_params" not in file.keys():
            raise ValueError((f"{file_name} is not a Population file, "
                             "perhaps you meant to use `cogsworth.sfh.load`?"))
        
        # warn users about version mismatches
        for key, version in [("cogsworth_version", __version__),
                             ("COSMIC_version", cosmic_version),
                             ("gala_version", gala_version)]:
            if key in file.attrs:
                saved_version = file.attrs[key]
                if saved_version != version:
                    logging.getLogger("cogsworth").warning(
                        f"cogsworth warning: file was saved with {key.split('_')[0]} v{saved_version} "
                        f"but you are using v{version}"
                    )

        numeric_params = file["numeric_params"][...]

        store_entire_orbits = file["numeric_params"].attrs["store_entire_orbits"]
        final_kstars = [file["numeric_params"].attrs["final_kstar1"],
                        file["numeric_params"].attrs["final_kstar2"]]
        bcm_tc = file["numeric_params"].attrs["timestep_conditions"].tolist()
        bpp_columns = file["numeric_params"].attrs["bpp_columns"]
        bcm_columns = file["numeric_params"].attrs["bcm_columns"]
        error_file_path = (file["numeric_params"].attrs["error_file_path"]
                           if "error_file_path" in file["numeric_params"].attrs else None)

        # convert columns to None if empty
        bpp_columns = None if isinstance(bpp_columns, bytes) and bpp_columns == b'None' else bpp_columns
        bcm_columns = None if isinstance(bcm_columns, bytes) and bcm_columns == b'None' else bcm_columns

        # decode byte strings as necessary
        bpp_columns = [col.decode('utf-8') for col in bpp_columns] if bpp_columns is not None else None
        bcm_columns = [col.decode('utf-8') for col in bcm_columns] if bcm_columns is not None else None

        # load in BSE settings
        for key in file["BSE_settings"].attrs:
            BSE_settings[key] = file["BSE_settings"].attrs[key]

        sampling_params = yaml.load(file["sampling_params"].attrs["dict"], Loader=yaml.Loader)

        integrator_name = file.get("integrator", None)
        integrator_kwargs = file.get("integrator_kwargs", {})
        integrator = gi.DOPRI853Integrator if integrator_name is None else getattr(gi, integrator_name)

    with h5.File(file_name, 'r') as f:
        galactic_potential = potential_from_dict(yaml.load(f.attrs["potential_dict"], Loader=yaml.Loader))

    p = Population(n_binaries=int(numeric_params[0]), processes=int(numeric_params[2]),
                   m1_cutoff=numeric_params[3], final_kstar1=final_kstars[0], final_kstar2=final_kstars[1],
                   sfh_model=sfh.StarFormationHistory, galactic_potential=galactic_potential,
                   v_dispersion=numeric_params[4] * u.km / u.s, max_ev_time=numeric_params[5] * u.Gyr,
                   timestep_size=numeric_params[6] * u.Myr, BSE_settings=BSE_settings,
                   sampling_params=sampling_params, store_entire_orbits=store_entire_orbits,
                   bcm_timestep_conditions=bcm_tc, bpp_columns=bpp_columns, bcm_columns=bcm_columns,
                   error_file_path=error_file_path,
                   integrator=integrator, integrator_kwargs=integrator_kwargs)

    p._file = file_name
    p.n_binaries_match = int(numeric_params[1])
    p._mass_singles = numeric_params[7]
    p._mass_binaries = numeric_params[8]
    p._n_singles_req = numeric_params[9]
    p._n_bin_req = numeric_params[10]

    # load parts as necessary
    if "initial_binaries" in parts:
        try:
            p.initC
        except KeyError:
            p.initial_binaries

    if "initial_galaxy" in parts:
        p.initial_galaxy

    if "stellar_evolution" in parts:
        p.kick_info
        p.bcm
        p.bpp

    if "galactic_orbits" in parts:
        p.orbits

    return p


def concat(*pops):
    """Concatenate multiple populations into a single population

    .. note::
    
        The final population will have the same settings as the first population in the list (but data
        from all populations)

    Parameters
    ----------
    pops : `list` of :class:`~cogsworth.pop.Population` or :class:`~cogsworth.pop.EvolvedPopulation`
        List of populations to concatenate

    Returns
    -------
    total_pop : :class:`~cogsworth.pop.Population` or :class:`~cogsworth.pop.EvolvedPopulation`
        The concatenated population
    """
    # ensure the input is a list of populations
    pops = list(pops)
    assert all([isinstance(pop, Population) for pop in pops])

    # if there's only one population then just return it
    if len(pops) == 1:
        return pops[0]
    elif len(pops) == 0:
        raise ValueError("No populations provided to concatenate")
    
    # warn about orbits if necessary
    if any([pop._orbits is not None for pop in pops]):
        logging.getLogger("cogsworth").warning(
            "cogsworth warning: Concatenating populations with orbits is not supported yet - "
            "the final population will not have orbits. PRs are welcome!"
        )

    # get the offset for the bin numbers
    bin_num_offset = max(pops[0].bin_nums) + 1

    # create a new population to store the final population (just a copy of the first population)
    final_pop = pops[0][:]

    # store the final positions and velocities if they were loaded
    # separate into bound and disrupted for easier stacking later
    bound_pos, bound_vel = None, None
    disrupted_pos, disrupted_vel = None, None
    if final_pop._final_pos is not None:
        bound_pos = final_pop._final_pos[:len(final_pop)]
        bound_vel = final_pop._final_vel[:len(final_pop)]
        disrupted_pos = final_pop._final_pos[len(final_pop):]
        disrupted_vel = final_pop._final_vel[len(final_pop):]

    # reset auto-calculated class variables
    final_pop._bin_nums = None
    final_pop._classes = None
    final_pop._final_pos = None
    final_pop._final_vel = None
    final_pop._final_bpp = None
    final_pop._disrupted = None
    final_pop._escaped = None
    final_pop._observables = None

    # loop over the remaining populations
    for pop in pops[1:]:
        # sum the total numbers of binaries
        final_pop.n_binaries += pop.n_binaries

        # combine the star formation history distributions
        if final_pop._initial_galaxy is not None:
            if pop._initial_galaxy is None:
                raise ValueError(f"Population {pop} does not have an initial galaxy, but the first does")

            final_pop._initial_galaxy += pop._initial_galaxy

        # loop through pandas tables that may need to be copied
        for table in ["_initial_binaries", "_initC", "_bpp", "_bcm", "_kick_info"]:
            # only copy if the table exists in the main population
            if getattr(final_pop, table) is not None:
                # if the table doesn't exist in the new population then raise an error
                if getattr(pop, table) is None:
                    raise ValueError(f"Population {pop} does not have a {table} table, but the first does")

                # otherwise copy the table and update the bin nums
                new_table = getattr(pop, table).copy()
                new_table.index += bin_num_offset

                # if the table has a "bin_num" column then update it
                if "bin_num" in new_table.columns:
                    new_table["bin_num"] += bin_num_offset
                setattr(final_pop, table, pd.concat([getattr(final_pop, table), new_table]))

        # sum the sampling numbers
        final_pop._n_singles_req += pop._n_singles_req
        final_pop._n_bin_req += pop._n_bin_req
        final_pop._mass_singles += pop._mass_singles
        final_pop._mass_binaries += pop._mass_binaries
        final_pop.n_binaries_match += pop.n_binaries_match

        # combine the final positions and velocities if they were loaded
        if bound_pos is not None:
            if pop._final_pos is None:
                raise ValueError(f"Population {pop} does not have final positions, but the first does")
            bound_pos = np.vstack((bound_pos, pop._final_pos[:len(pop)]))
            bound_vel = np.vstack((bound_vel, pop._final_vel[:len(pop)]))
            disrupted_pos = np.vstack((disrupted_pos, pop._final_pos[len(pop):]))
            disrupted_vel = np.vstack((disrupted_vel, pop._final_vel[len(pop):]))

        final_pop._bin_nums = None
        bin_num_offset = max(final_pop.bin_nums) + 1

    # set the final positions and velocities if they were loaded
    if bound_pos is not None:
        final_pop._final_pos = np.vstack((bound_pos, disrupted_pos))
        final_pop._final_vel = np.vstack((bound_vel, disrupted_vel))

    return final_pop


def _init_pool(potential, t2, store_all, quiet, integrator, integrator_kwargs):
    _GLOBAL["potential"] = potential
    _GLOBAL["t2"] = t2
    _GLOBAL["store_all"] = store_all
    _GLOBAL["quiet"] = quiet
    _GLOBAL["integrator"] = integrator
    _GLOBAL["integrator_kwargs"] = integrator_kwargs

def _orbit_worker(w0, t1, dt, events):
    return integrate_orbit_with_events(
        w0, t1, _GLOBAL["t2"], dt, _GLOBAL["potential"], events, _GLOBAL["store_all"], _GLOBAL["quiet"],
        _GLOBAL["integrator"], _GLOBAL["integrator_kwargs"]
    )

class EvolvedPopulation(Population):
    def __init__(self, n_binaries, mass_singles=None, mass_binaries=None, n_singles_req=None, n_bin_req=None,
                 bpp=None, bcm=None, initC=None, kick_info=None, **pop_kwargs):
        super().__init__(n_binaries=n_binaries, **pop_kwargs)

        self._mass_singles = mass_singles
        self._mass_binaries = mass_binaries
        self._n_singles_req = n_singles_req
        self._n_bin_req = n_bin_req
        self._bpp = bpp
        self._bcm = bcm
        self._initC = initC
        self._kick_info = kick_info

    def sample_initial_binaries(self):
        raise NotImplementedError("`EvolvedPopulation` cannot sample new binaries, use `Population` instead")

    def perform_stellar_evolution(self):
        raise NotImplementedError("`EvolvedPopulation` cannot do stellar evolution, use `Population` instead")

    def create_population(self, with_timing=True):
        """Create an entirely evolved population of binaries with sampling or stellar evolution

        This will sample the initial galaxy and then perform the :py:mod:`gala` evolution.

        Parameters
        ----------
        with_timing : `bool`, optional
            Whether to print messages about the timing, by default True
        """
        if with_timing:
            start = time.time()
            print(f"Run for {self.n_binaries} binaries")

        self.sample_initial_galaxy()
        if with_timing:
            print(f"[{time.time() - start:1.0e}s] Sample initial galaxy")
            lap = time.time()

        self._ensure_pool(quiet=False)
        self.perform_galactic_evolution(progress_bar=with_timing)
        if with_timing:
            print(f"[{time.time() - lap:1.1f}s] Get orbits (run gala)")
        self._close_pool()

        if with_timing:
            print(f"Overall: {time.time() - start:1.1f}s")
