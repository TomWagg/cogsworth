import numpy as np
import unittest
import cogsworth.obs.mist as mist
import astropy.units as u

# temp directories
import tempfile
import io
import contextlib

class Test(unittest.TestCase):
    def test_bad_band(self):
        """Test that it breaks when given a bad band"""
        it_worked = True
        try:
            grid = mist.MISTBolometricCorrectionGrid(bands=("NOT A BAND",))
        except KeyError:
            it_worked = False
        self.assertFalse(it_worked)

    def test_check_filters(self):
        """Test that filter checking passes valid filters and gives helpful suggestions otherwise"""
        mist.check_filters(["Gaia_G_EDR3", "NIRCAM_F070W"])
        mist.check_filters("Gaia_G_EDR3")

        with self.assertRaises(mist.MISTFilterError) as cm:
            mist.check_filters(["Gaia_G_EDR3", "F070W", "IRAC_3.6", "JWST"])
        msg = str(cm.exception)
        self.assertIn("NIRCAM_F070W", msg)
        self.assertIn("IRAC_36", msg)
        self.assertIn("this is a filter set", msg)
        self.assertIn("list_filters()", msg)
        self.assertNotIn("'Gaia_G_EDR3'", msg)

        # old MIST v1 names should point to their v2 equivalents
        with self.assertRaises(mist.MISTFilterError) as cm:
            mist.check_filters(["F070W", "F470N", "IRAC_3.6", "HST_ACSWF"])
        msg = str(cm.exception)
        self.assertIn("MIST v1 filter name", msg)
        for new in ["'NIRCAM_F070W'", "'NIRCAM_F470W'", "'IRAC_36'"]:
            self.assertIn(new, msg)
        self.assertIn("'HST_ACSWF' (this is a filter set", msg)

        # every v1 filter is either still valid or has a valid v2 rename
        v2_filters = {f for fs in mist.MIST_FILTER_SETS.values() for f in fs}
        for filters in mist.MIST_FILTER_SETS_V1.values():
            for f in filters:
                self.assertIn(mist.MIST_FILTER_RENAMES_V1_TO_V2.get(f, f), v2_filters)
        for old_set, new_set in mist.MIST_FILTER_SET_RENAMES_V1_TO_V2.items():
            self.assertIn(new_set, mist.MIST_FILTER_SETS)
            self.assertNotIn(old_set, mist.MIST_FILTER_SETS)

        # should still be catchable as a KeyError for backwards compatibility
        self.assertTrue(issubclass(mist.MISTFilterError, KeyError))

    def test_list_filters(self):
        """Test the printing of filter sets and filters"""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mist.list_filter_sets()
        for filter_set in mist.MIST_FILTER_SETS:
            self.assertIn(filter_set, out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mist.list_filters("SPITZER")
        self.assertIn("SPITZER (4 filters)", out.getvalue())
        self.assertIn("    IRAC_36", out.getvalue())
        self.assertNotIn("WISE", out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mist.list_filters(["WISE", "LSST"])
        self.assertIn("WISE_W1", out.getvalue())
        self.assertIn("LSST_u", out.getvalue())

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mist.list_filters("all", width=80)
        for filter_set, filters in mist.MIST_FILTER_SETS.items():
            self.assertIn(filter_set, out.getvalue())
            for f in filters:
                self.assertIn(f, out.getvalue())
        self.assertTrue(all(len(line) <= 80 for line in out.getvalue().splitlines()))

        with self.assertRaises(mist.MISTFilterError):
            mist.list_filters("NOT A SET")
        with self.assertRaises(mist.MISTFilterError) as cm:
            mist.list_filters("HST_ACSWF")
        self.assertIn("'HST_ACS_WFC'", str(cm.exception))

    def test_build(self):
        """Test building and loading the HDF5 files"""

        with tempfile.TemporaryDirectory() as tmpdir:
            grid = mist.MISTBolometricCorrectionGrid(bands=("Gaia_G_EDR3",), cache_dir=tmpdir, rebuild=True)
            h5_path = grid.build_hdf5("UBVRIplus")
            self.assertTrue(h5_path.exists())

            grid = mist.MISTBolometricCorrectionGrid(bands=("Gaia_G_EDR3",), cache_dir=tmpdir, rebuild=False)
            grid.download_filter_set("UBVRIplus")
            grid.extract_filter_set("UBVRIplus")
            grid.build_hdf5("UBVRIplus")

            self.assertTrue(h5_path.exists())

    def test_interpolator(self):
        """Test that the interpolator works as expected"""

        with tempfile.TemporaryDirectory() as tmpdir:
            grid = mist.MISTBolometricCorrectionGrid(bands=("Gaia_G_EDR3",), cache_dir=tmpdir)
        
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

            # bands that weren't loaded into the grid should raise an error
            with self.assertRaises(KeyError):
                grid.interp(teff[0], logg[0], feh[0], av[0], bands=["Gaia_BP_EDR3"])
            with self.assertRaises(mist.MISTFilterError):
                grid.interp(teff[0], logg[0], feh[0], av[0], bands=["NOT A BAND"])
