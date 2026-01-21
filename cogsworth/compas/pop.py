import logging
import subprocess
from .runSubmit import pythonProgramOptions as COMPAS_command_creator
import tempfile
import os

from ..pop import Population
from .utils import create_bpp_from_COMPAS_file, create_kick_info_from_COMPAS_file


class COMPASPopulation(Population):
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
        super().__init__(n_binaries=n_binaries, **kwargs)

    def initial_binaries_to_gridfile(self, grid_filename):
        """Write the initial conditions to a COMPAS grid file"""
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

        self._bpp = create_bpp_from_COMPAS_file(f"{self.output_directory}/COMPAS_Output.h5")
        self._kick_info = create_kick_info_from_COMPAS_file(f"{self.output_directory}/COMPAS_Output.h5")
        self._initC = self.initial_binaries

        self.initial_binaries["bin_num"] = self.final_bpp["bin_num"].values
        self.initial_binaries.index = self.final_bpp["bin_num"].values
        self._initC = self.initial_binaries


def _stringify_initC(df):
    return " ".join([
        f'--initial-mass-1 {df["mass_1"]}',
        f'--initial-mass-2 {df["mass_2"]}',
        f'--orbital-period {df["porb"]}',
        f'--eccentricity {df["ecc"]}',
        f'--metallicity {df["metallicity"]}',
        f'--maximum-evolution-time {df["tphysf"]}'])
