import unittest
import cogsworth
from cogsworth import COMPASPopulation
import os
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test(unittest.TestCase):
    def test_imports(self):
        """Test that the imports work"""
        try:
            from cogsworth import interop
            from cogsworth.interop import compas
            from cogsworth.interop.compas import file
            from cogsworth.interop.compas import pop
            from cogsworth.interop.compas import runner
            from cogsworth.interop.compas import utils
        except ImportError:
            self.fail("Failed to import interop modules")

        try:
            from cogsworth.interop.compas import nonsense
        except ImportError:
            pass
        else:
            self.fail("Should have failed to import nonsense")

    def test_compas_pop_conversion(self):
        """Test that we can convert to a COMPAS population and back"""
        pop = cogsworth.pop.Population(10, use_default_BSE_settings=True)
        pop.create_population()

        compas_pop = pop.to_COMPASPopulation()

        pop_converted_back = compas_pop.to_Population()

        self.assertTrue(np.all(pop.bin_nums == pop_converted_back.bin_nums))
        self.assertTrue(np.all(pop.final_bpp["mass_1"] == pop_converted_back.final_bpp["mass_1"]))

    def test_creating_COMPASPopulation(self):
        """Test that we can create a COMPASPopulation from parameters"""
        compas_pop = COMPASPopulation(
            n_binaries=42,
            output_directory=os.path.join(THIS_DIR, "test_data/COMPAS_Output"),
        )

        it_worked = True
        try:
            compas_pop._append_kicks()
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked, "Should have failed to append kicks because no data")

        self.assertEqual(compas_pop.n_binaries, 42)
        self.assertEqual(compas_pop.output_directory, os.path.join(THIS_DIR, "test_data/COMPAS_Output_2"))
    
    def test_creating_COMPASPopulation_from_COMPAS_output(self):
        """Test that we can create a COMPASPopulation from a COMPAS output directory"""
        compas_pop = COMPASPopulation.from_COMPAS_output(
            os.path.join(THIS_DIR, "test_data/COMPAS_Output_1/COMPAS_Output.h5")
        )

        self.assertEqual(compas_pop.n_binaries, 101)
        self.assertEqual(compas_pop.output_directory, os.path.join(THIS_DIR, "test_data/COMPAS_Output_1"))

    def test_gridfile_creation(self):
        """Test that we can create a grid file from a COMPAS population"""
        compas_pop = COMPASPopulation.from_COMPAS_output(
            os.path.join(THIS_DIR, "test_data/COMPAS_Output_1/COMPAS_Output.h5")
        )

        # delete bin nums for testing
        compas_pop.initial_binaries.drop(columns=["bin_num"], inplace=True)

        grid_filename = os.path.join(THIS_DIR, "test_data/test_grid_file.txt")
        compas_pop.initial_binaries_to_gridfile(grid_filename)

        self.assertTrue(os.path.isfile(grid_filename), "Grid file was not created")

        # check that there are the right number of lines and that the masses are correct
        with open(grid_filename, 'r') as f:
            lines = f.readlines()

        self.assertEqual(len(lines), 101, "Grid file should have 101 lines")
        for line, (_, row) in zip(lines, compas_pop.initial_binaries.iterrows()):
            split_line = line.strip().split()
            mass_1 = float(split_line[1])
            self.assertAlmostEqual(mass_1, row["mass_1"], places=5)

        # clean up
        os.remove(grid_filename)