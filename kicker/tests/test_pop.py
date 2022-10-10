import numpy as np
import unittest
import kicker.pop as pop
import os


class Test(unittest.TestCase):
    def test_bad_inputs(self):
        """Ensure the class fails with bad input"""
        it_broke = False
        try:
            pop.Population(n_binaries=-100)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            pop.Population(n_binaries=0)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_io(self):
        """Check that a population can be saved and re-loaded"""
        p = pop.Population(2)
        p.create_population()

        p.save("testing-pop-io", overwrite=True)

        p_loaded = pop.load("testing-pop-io")

        self.assertTrue(np.all(p.bpp == p_loaded.bpp))

        # attempt overwrite without setting flag
        it_broke = False
        try:
            p.save("testing-pop-io")
        except FileExistsError:
            it_broke = True
        self.assertTrue(it_broke)

        # attempt overwrite now WITH the flag
        it_broke = False
        try:
            p.save("testing-pop-io", overwrite=True)
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)

        os.remove("testing-pop-io.h5")
        os.remove("testing-pop-io-galaxy-params.txt")
        os.remove("testing-pop-io-orbits.npy")
        os.remove("testing-pop-io-potential.txt")

    def test_orbit_storage(self):
        """Test that we can control how orbits are stored"""
        p = pop.Population(2, processes=1, store_entire_orbits=True)
        p.create_population()

        first_orbit = p.orbits[0][0] if isinstance(p.orbits[0], list) else p.orbits[0]
        self.assertTrue(first_orbit.shape[0] >= 1)

        p = pop.Population(2, processes=1, store_entire_orbits=False)
        p.create_population()

        first_orbit = p.orbits[0][0] if isinstance(p.orbits[0], list) else p.orbits[0]

        self.assertTrue(first_orbit.shape[0] == 1)

    def test_overly_stringent_cutoff(self):
        """Make sure that it crashes if the m1_cutoff is too large to create anything"""
        p = pop.Population(10, m1_cutoff=10000)

        it_broke = False
        try:
            p.create_population()
        except ValueError:
            it_broke = True

        self.assertTrue(it_broke)

    def test_interface(self):
        """Test the interface of this class with the other modules"""
        p = pop.Population(10, final_kstar1=[13, 14], store_entire_orbits=False)
        p.create_population()

        # ensure we get something that disrupts to ensure coverage
        MAX_REPS = 5
        i = 0
        while not p.disrupted.any() and i < MAX_REPS:
            p = pop.Population(10, final_kstar1=[13, 14])
            p.create_population()
            i += 1
        if i == MAX_REPS:
            raise ValueError("Couldn't make anything disrupt :/")

        # test we can get the final distances properly
        self.assertTrue(np.all(p.final_coords[0].icrs.distance.value >= 0.0))

        # test that classes can be identified
        self.assertTrue(p.classes.shape[0] == p.n_binaries_match)

        # test that observable table is done right
        p.observables

        # cheat and make sure at least one binary is bright enough
        p.observables["G_app_1"].iloc[0] = 18.0
        p.get_gaia_observed_bin_nums()

        p.plot_map(coord="C", show=False)
        p.plot_map(coord="G", show=False)

    def test_getters(self):
        """Test the property getters"""
        p = pop.Population(2, store_entire_orbits=False)

        # test getters from sampling
        p.mass_singles
        p._mass_binaries = None
        p.mass_binaries

        p._n_singles_req = None
        p.n_singles_req

        p._n_bin_req = None
        p.n_bin_req

        # test getters from stellar evolution
        p.bpp
        p._bcm = None
        p.bcm

        p._initC = None
        p.initC

        p._kick_info = None
        p.kick_info

        # test getters for galactic evolution
        p.orbits

    def test_singles_evolution(self):
        """Check everything works well when evolving singles"""
        p = pop.Population(2, BSE_settings={"binfrac": 0.0})
        p.create_population(with_timing=False)

        self.assertTrue((p.final_bpp["sep"] == 0.0).all())

    def test_from_initC(self):
        """Check it can handle only having an initC rather than initial_binaries"""
        p = pop.Population(2)
        p.sample_initial_binaries()
        p.perform_stellar_evolution()
        p._initial_binaries = None
        p.perform_stellar_evolution()

    def test_none_orbits(self):
        """Ensure final_coords still works when there is an Orbit with None"""
        p = pop.Population(2)
        p._orbits = [None, None]
        self.assertTrue(p.final_coords[0].x[0].value == np.inf)
