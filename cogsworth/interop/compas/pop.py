import logging
import subprocess
import tempfile
import os

from ...pop import Population
from .runner import pythonProgramOptions as COMPAS_command_creator
from .file import get_bpp, get_kick_info, get_initial_binaries
from .utils import add_kicks_to_initial_binaries


__all__ = ["COMPASPopulation"]


class COMPASPopulation(Population):
    """A Population class which uses COMPAS to perform binary stellar evolution instead of COSMIC
    
    This class extends the generic Population class to use COMPAS as the binary stellar evolution
    engine. It requires a COMPAS installation to be available.

    Parameters
    ----------
    n_binaries : int
        The number of binaries to simulate
    config_file : str, optional
        The path to the COMPAS configuration file to use. If None, a default config file included with
        cogsworth is used.
    logfile_definitions : str, optional
        The path to the COMPAS logfile definitions file to use. If None, a default logfile definitions file
        included with cogsworth is used.
        TODO: list which columns are necessary
    output_directory : str, optional
        The directory to output the COMPAS results to. If the directory already exists, a suffix
        _1, _2 etc. will be appended to create a new directory. By default, "./COMPAS_Output" is used.
    **kwargs : dict
        Additional keyword arguments to pass to the Population class constructor
    """
    def __init__(self, n_binaries, config_file=None, logfile_definitions=None,
                 output_directory="./COMPAS_Output", **kwargs):
        # set to the default file, which is in the same directory as this file
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), 'compas_config.yaml')
        self.config_file = config_file

        if logfile_definitions is None:
            logfile_definitions = os.path.join(os.path.dirname(__file__), 'switchdefs.txt')
        self.logfile_definitions = logfile_definitions

        # check if the output directory exists, if so need to append _1, _2 etc.
        counter = 1
        while os.path.exists(output_directory):
            # if dir ends in _<number>, remove it before appending new number
            if output_directory.rstrip().endswith(f"_{counter-1}"):
                output_directory = output_directory.rstrip()[:-len(f"_{counter-1}")]
            output_directory = f"{output_directory}_{counter}"
            counter += 1
        self.output_directory = output_directory

        if "use_default_BSE_settings" not in kwargs:
            kwargs["use_default_BSE_settings"] = True

        super().__init__(n_binaries=n_binaries, **kwargs)

    @classmethod
    def from_COMPAS_output(cls, compas_output_file, lookback_times=None, **kwargs):
        """Create a COMPASPopulation from an existing COMPAS output file

        This class method creates a COMPASPopulation object from an existing COMPAS output file which has
        already been generated with COMPAS. The initial binaries, BPP and kick information are read from the
        output file.

        Parameters
        ----------
        compas_output_file : str
            The path to the COMPAS output file (HDF5 format)
        lookback_times : array-like, optional
            An array of lookback times (in Myr) to use for each binary when reading the initial conditions.
            If None, the maximum evolution times stored in the COMPAS output file are used (found in the
            PO_Max_Evolution_Time column). If that column is not present, a warning is issued and a default
            of 13.7 Gyr is assumed for all systems.
        **kwargs : dict
            Additional keyword arguments to pass to the COMPASPopulation constructor

        Returns
        -------
        pop : `~cogsworth.interop.compas.COMPASPopulation`
            The COMPASPopulation object created from the COMPAS output file
        """
        initial_binaries = get_initial_binaries(compas_output_file, tphysf=lookback_times)
        pop = cls(n_binaries=len(initial_binaries), **kwargs)
        pop._initial_binaries = initial_binaries
        pop._bpp = get_bpp(compas_output_file)
        pop._kick_info = get_kick_info(compas_output_file)
        pop._append_kicks()
        pop.output_directory = os.path.dirname(compas_output_file)
        return pop
    
    def _append_kicks(self):
        """Add the kick information from COMPAS to the initial binaries dataframe"""
        if self._initial_binaries is None or self._kick_info is None:
            raise ValueError("Either initial_binaries or kick_info is None, cannot append kicks.")
        self._initial_binaries = add_kicks_to_initial_binaries(self._initial_binaries, self._kick_info)
        return self._initial_binaries

    def initial_binaries_to_gridfile(self, grid_filename):
        """Write the initial binaries to a COMPAS grid file
        
        Parameters
        ----------
        grid_filename : str
            The path to the grid file to write the initial binaries to
        """
        with open(grid_filename, "w") as f:
            f.write('\n'.join(self.initial_binaries.apply(_stringify_initC, axis=1).values))

    def perform_stellar_evolution(self):
        """Perform the (binary) stellar evolution of the sampled binaries"""
        # delete any cached variables
        self._final_bpp = None
        self._observables = None
        self._bin_nums = None
        self._disrupted = None
        self._escaped = None

        # if no initial binaries have been sampled then we need to create some
        if self._initial_binaries is None and self._initC is None:
            logging.getLogger("cogsworth").warning(("cogsworth warning: Initial binaries not yet sampled, "
                                                    "performing sampling now."))
            self.sample_initial_binaries()

        # create temporary directory to hold grid file
        with tempfile.TemporaryDirectory() as tmpdir:
            grid_filename = f"{tmpdir}/temp_grid_file.txt"
            self.initial_binaries_to_gridfile(grid_filename)
            self.COMPAS_command = COMPAS_command_creator(config_file=self.config_file,
                                                         grid_filename=grid_filename,
                                                         logfile_definitions=self.logfile_definitions,
                                                         output_directory=self.output_directory).shellCommand
            subprocess.call(self.COMPAS_command + " > /dev/null", shell=True)

        self._bpp = get_bpp(f"{self.output_directory}/COMPAS_Output.h5")
        self._kick_info = get_kick_info(f"{self.output_directory}/COMPAS_Output.h5")
        self._initC = self.initial_binaries

        self.initial_binaries["bin_num"] = self.final_bpp["bin_num"].values
        self.initial_binaries.index = self.final_bpp["bin_num"].values
        self._append_kicks()
        self._initC = self.initial_binaries

    def to_Population(self, **kwargs):
        """Convert this COMPASPopulation to a generic Population object that uses COSMIC"""
        use_defaults = kwargs.pop("use_default_BSE_settings", True)
        pop = Population(self.n_binaries, use_default_BSE_settings=use_defaults, **kwargs)
        attrs_to_copy = ["n_binaries", "n_binaries_match", "processes", "final_kstar1", "final_kstar2",
                         "sfh_model", "sfh_params", "galactic_potential", "v_dispersion", "max_ev_time",
                         "timestep_size", "pool", "store_entire_orbits", "bpp_columns", "bcm_columns",
                         "_file", "_initial_binaries", "_initial_galaxy", "_mass_singles", "_mass_binaries",
                         "_n_singles_req", "_n_bin_req", "_bpp", "_bcm", "_kick_info",
                         "_orbits", "_classes", "_final_pos", "_final_vel", "_final_bpp", "_disrupted",
                         "_escaped", "_observables", "_bin_nums", "BSE_settings",
                         "sampling_params", "bcm_timestep_conditions"]
        for attr in attrs_to_copy:
            if attr not in kwargs:
                setattr(pop, attr, getattr(self, attr))

        # check whether kicks were calculated from COMPAS and whether they might be overwritten
        kick_cols = ["natal_kick_1", "natal_kick_2", "phi_1", "theta_1", "phi_2", "theta_2",
                     "mean_anomaly_1", "mean_anomaly_2"]
        any_were_present = False
        for col in kick_cols:
            if col in self.initial_binaries.columns:
                any_were_present = True
        
        # for defaults, just remove any natal kick settings so that COSMIC uses the table
        if any_were_present and use_defaults and "natal_kick_array" in pop.BSE_settings:
            del pop.BSE_settings["natal_kick_array"]

        # if not using defaults, warn the user that their settings will overwrite COMPAS kicks
        elif any_were_present and not use_defaults:
            logging.getLogger("cogsworth").warning(
                "cogsworth warning: Natal kick settings found in BSE_settings will overwrite "
                "the kicks calculated by COMPAS."
            )

        return pop


def _stringify_initC(df):
    """Convert a row of the initial conditions dataframe to a COMPAS grid file line string"""
    return " ".join([
        f'--initial-mass-1 {df["mass_1"]}',
        f'--initial-mass-2 {df["mass_2"]}',
        f'--orbital-period {df["porb"]}',
        f'--eccentricity {df["ecc"]}',
        f'--metallicity {df["metallicity"]}',
        f'--maximum-evolution-time {df["tphysf"]}'])
