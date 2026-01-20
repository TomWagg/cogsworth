import logging
import subprocess
from .runSubmit import pythonProgramOptions as COMPAS_command_creator

from ..pop import Population


class COMPASPopulation(Population):
    def __init__(self, n_binaries, compas_config_path, **kwargs):
        self.compas_config_path = compas_config_path
        super().__init__(n_binaries=n_binaries, **kwargs)

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

        print(self.initial_binaries["metallicity"].max())

        GRID_PATH = "temp_grid_file.txt"
        with open(GRID_PATH, "w") as f:
            f.write('\n'.join(self.initial_binaries.apply(_stringify_initC, axis=1).values))

        self.COMPAS_command = COMPAS_command_creator(self.compas_config_path, GRID_PATH).shellCommand

        subprocess.call(self.COMPAS_command, shell=True)

        print("Binary evolution complete")


def _stringify_initC(df):
    return " ".join([
        # f'--stellar-type-1 {df["kstar_1"]}',
        # f'--stellar-type-2 {df["kstar_2"]}',
        f'--initial-mass-1 {df["mass_1"]}',
        f'--initial-mass-2 {df["mass_2"]}',
        f'--orbital-period {df["porb"]}',
        f'--eccentricity {df["ecc"]}',
        f'--metallicity {df["metallicity"]}',
        f'--maximum-evolution-time {df["tphysf"]}'])
