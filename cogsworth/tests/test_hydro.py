import unittest
import cogsworth
import astropy.units as u
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


class Test(unittest.TestCase):
    def test_snapshot_prep_changa(self):
        """Test that we can prepare a snapshot of a hydrodynamical simulation in ChaNGa format"""
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        self.assertTrue(snap["r"].mean().in_units("kpc") < 0.1)

    def test_snapshot_prep_fire(self):
        """Test that we can prepare a snapshot of a hydrodynamical simulation in FIRE format"""
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test_like_FIRE"))
        self.assertTrue(snap["r"].mean().in_units("kpc") < 0.1)

    def test_potential_calculation(self):
        """Test that we can compute the potential of a snapshot of a hydrodynamical simulation"""
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        cogsworth.hydro.potential.get_snapshot_potential(snap, verbose=True, out_path="test_pot.yml")
        self.assertTrue(os.path.exists("test_pot.yml"))
        os.remove("test_pot.yml")

    def test_rewind(self):
        """Test that we can rewind a snapshot of a hydrodynamical simulation"""
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        pot = cogsworth.hydro.potential.get_snapshot_potential(snap)
        subsnap = snap.s[0, 1, 2]
        df_single_thread = cogsworth.hydro.rewind.rewind_to_formation(subsnap, pot, processes=1)
        df_multiprocessing = cogsworth.hydro.rewind.rewind_to_formation(subsnap, pot, processes=3)

        self.assertTrue((df_single_thread == df_multiprocessing).all().all())

        snap_no_massform = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test_like_FIRE"))
        pot = cogsworth.hydro.potential.get_snapshot_potential(snap_no_massform)
        subsnap = snap_no_massform.s[0, 1, 2]
        df = cogsworth.hydro.rewind.rewind_to_formation(subsnap, pot, processes=1)
        self.assertTrue((df["mass"] == subsnap["mass"].in_units("Msol")).all())
