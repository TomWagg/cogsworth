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
        h5_path = grid.build_hdf5("UBVRIplus")
        self.assertTrue(h5_path.exists())

        grid = mist.MISTBolometricCorrectionGrid(bands=("Gaia_G_EDR3",), rebuild=False)
        grid.download_filter_set("UBVRIplus")
        grid.extract_filter_set("UBVRIplus")
        grid.build_hdf5("UBVRIplus")

        self.assertTrue(h5_path.exists())

    def test_interpolator(self):
        """Test that the interpolator works as expected"""
        grid = mist.MISTBolometricCorrectionGrid(bands=("Gaia_G_EDR3",))
        
        # pick random points in the grid
        sample = grid.bc_grid.sample(10)
        teff, logg, feh, av = np.transpose(sample.index.tolist())
        expected = sample["Gaia_G_EDR3"].values

        self.assertTrue(np.allclose(
            grid.interp(teff, logg, feh, av)["Gaia_G_EDR3"].values,
            expected
        ))
        
        self.assertTrue(np.isclose(
            grid.interp(teff[0], logg[0], feh[0], av[0])["Gaia_G_EDR3"],
            expected[0]
        ))
