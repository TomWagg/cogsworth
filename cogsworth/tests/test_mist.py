import numpy as np
import unittest
import cogsworth.obs.mist as mist
import astropy.units as u


class Test(unittest.TestCase):
    def test_bad_band(self):
        """Test that it breaks when given a bad band"""
        it_worked = True
        try:
            grid = mist.MISTBolometricCorrectionGrid(bands=("NOT A BAND",))
        except KeyError:
            it_worked = False
        self.assertFalse(it_worked)

    def test_build(self):
        """Test building and loading the HDF5 files"""
        grid = mist.MISTBolometricCorrectionGrid(bands=("Gaia_G_EDR3",), rebuild=True)
        h5_path = grid.build_hdf5("Gaia_EDR3")
        self.assertTrue(h5_path.exists())

        df = grid.load_hdf5("Gaia_EDR3")
        self.assertIsInstance(df, type(grid.read_filter_set("Gaia_EDR3")))
