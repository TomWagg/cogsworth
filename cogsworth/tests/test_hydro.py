import unittest
import cogsworth
import astropy.units as u


class Test(unittest.TestCase):
    def test_bad_inputs(self):
        """Ensure failure with bad input"""
        it_broke = False