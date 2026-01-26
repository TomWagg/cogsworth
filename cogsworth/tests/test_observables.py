import numpy as np
import unittest
import cogsworth.obs.observables as obs
import astropy.units as u


class Test(unittest.TestCase):
    def test_magnitudes(self):
        """Test the magnitude functions"""
        # check using ints and lists is the same as floats and arrays
        from_int_list = obs.add_mags(1, 2, [1, 2])
        from_float_array = obs.add_mags(1.0, 2.0, np.array([1, 2]))
        self.assertTrue(np.all(from_int_list == from_float_array))

        # catch the case of strings
        it_broke = False
        try:
            obs.add_mags("not a mag", "NOPE")
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        # check NaNs
        with_nan = obs.add_mags(1, 2, np.nan, remove_nans=True)
        without_nan = obs.add_mags(1, 2, remove_nans=True)
        self.assertTrue(with_nan == without_nan)

        # check mag conversion
        m_app = np.array([20, 21, 22])
        distance = [2, 2, 2] * u.kpc
        M_abs = obs.get_absolute_mag(m_app=m_app, distance=distance)
        m_app_converted = obs.get_apparent_mag(M_abs=M_abs, distance=distance)
        print(m_app)
        print(m_app_converted)
        self.assertTrue(np.all(m_app == m_app_converted))

    def test_bad_input(self):
        """Test that it breaks when it should"""
        it_worked = True
        try:
            obs.get_photometry(filters=["Gaia_G_EDR3"])
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        it_worked = True
        try:
            obs.get_photometry(filters=["Gaia_G_EDR3"], population="dummy")
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        it_worked = True
        try:
            obs.get_photometry(filters=["Gaia_G_EDR3"], population="dummy", distances="dummy")
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)
