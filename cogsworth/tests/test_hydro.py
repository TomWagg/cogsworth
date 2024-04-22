import unittest
import cogsworth
import astropy.units as u
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test(unittest.TestCase):
    def test_snapshot_prep_changa(self):
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        self.assertTrue(snap["r"].mean().in_units("kpc") < 0.1)

    def test_snapshot_prep_fire(self):
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test_like_FIRE"))
        self.assertTrue(snap["r"].mean().in_units("kpc") < 0.1)

    def test_potential_calculation(self):
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        cogsworth.hydro.potential.get_snapshot_potential(snap, verbose=True, out_path="test_pot.yml")
        self.assertTrue(os.path.exists("test_pot.yml"))
        os.remove("test_pot.yml")
