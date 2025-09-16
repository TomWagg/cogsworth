import unittest
import cogsworth
import os
import numpy as np
import pytest
import astropy.units as u

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

    def test_pop_init(self):
        """Test that we can create a population from a snapshot of a hydrodynamical simulation"""
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        pot = cogsworth.hydro.potential.get_snapshot_potential(snap)

        subsnap = snap.s[:15]
        init_particles = cogsworth.hydro.rewind.rewind_to_formation(subsnap, pot, processes=1)

        for subset, length in [(None, len(init_particles)), (10, 10), ([0, 1, 2], 3)]:
            pop = cogsworth.hydro.pop.HydroPopulation(init_particles, galactic_potential=pot,
                                                      subset=subset, snapshot_type="FIRE",
                                                      use_default_BSE_settings=True)
            self.assertEqual(len(pop._subset_inds), length)
        self.assertTrue("star particles" in pop.__repr__())

        # also test citations
        it_broke = False
        try:
            pop.get_citations("")
        except Exception:
            it_broke = True
        self.assertFalse(it_broke)

    @pytest.mark.filterwarnings("ignore:.*duplicate")
    def test_pop_sample_and_mask(self):
        """Test that we can sample and mask a population of a hydrodynamical simulation"""
        snap = cogsworth.hydro.utils.prepare_snapshot(os.path.join(THIS_DIR, "test_data/hydro_test"))
        pot = cogsworth.hydro.potential.get_snapshot_potential(snap)

        subsnap = snap.s[:3]
        init_particles = cogsworth.hydro.rewind.rewind_to_formation(subsnap, pot, processes=1)
        init_particles["mass"] = np.random.uniform(10, 20, size=len(init_particles))

        p = cogsworth.hydro.pop.HydroPopulation(init_particles, galactic_potential=pot,
                                                max_ev_time=100 * u.Myr, snapshot_type="FIRE", processes=1,
                                                use_default_BSE_settings=True)
        p.create_population()

        self.assertTrue("star particles" in p.__repr__() and "evolved systems" in p.__repr__())

        p.disrupted
        p.final_bpp

        inds = [int(np.random.choice(p.bin_nums, replace=False)),
                np.random.choice(p.bin_nums, size=2, replace=False),
                list(np.random.choice(p.bin_nums, size=2, replace=False)),
                slice(0, 1, 1),
                [0, 1, 1, 1, 0]]

        # mock up some data so it tests the indexing
        p._observables = p.final_bpp
        p._classes = p.final_bpp
        p._final_pos = np.zeros(len(p._orbits))
        p._final_vel = np.zeros(len(p._orbits))

        for ind in inds:
            p_ind = p[ind]
            if isinstance(ind, slice):
                ind = list(range(ind.stop)[ind])
            elif isinstance(ind, int):
                ind = [ind]
            og_m1 = p.final_bpp.loc[ind]["mass_1"].values
            self.assertTrue(np.all(og_m1 == p_ind.final_bpp["mass_1"].values))
