import numpy as np
import unittest
import cogsworth.pop as pop
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

        # test that observable table is done right (with or without extinction)
        p.observables
        p.get_observables(ignore_extinction=True)

        # cheat and make sure at least one binary is bright enough
        p.observables["G_app_1"].iloc[0] = 18.0
        p.get_gaia_observed_bin_nums()

        p.plot_map(coord="C", show=False)
        p.plot_map(coord="G", show=False)

        p[0]

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

        p._initial_galaxy = None
        p.initial_galaxy

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

        p.escaped

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

    def test_indexing(self):
        """Ensure that indexing works correctly (reprs too)"""
        p = pop.Population(10)
        print(p)
        p.create_population()
        print(p)

        # make sure it fails for strings
        it_worked = True
        try:
            p["absolute nonsense mate"]
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        inds = [np.random.randint(p.n_binaries_match),
                np.random.randint(p.n_binaries_match, size=4),
                list(np.random.randint(p.n_binaries_match, size=2)),
                slice(0, 7, 3)]

        for ind in inds:
            p_ind = p[ind]
            if isinstance(ind, slice):
                ind = list(range(ind.stop)[ind])
            og_m1 = p.final_bpp[p.final_bpp["bin_num"].isin(np.atleast_1d(ind))]["mass_1"].values
            self.assertTrue(np.all(og_m1 == p_ind.final_bpp["mass_1"].values))

        # make sure it fails for bin_nums that don't exist
        it_worked = True
        try:
            p[-42]
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

    def test_evolved_pop(self):
        """Check that the EvolvedPopulation class works as it should"""
        p = pop.Population(10)
        p.create_population()

        ep = pop.EvolvedPopulation(n_binaries=p.n_binaries_match, mass_singles=p.mass_singles,
                                   mass_binaries=p.mass_binaries, n_singles_req=p.n_singles_req,
                                   n_bin_req=p.n_bin_req, bpp=p.bpp, bcm=p.bcm, initC=p.initC,
                                   kick_info=p.kick_info)

        cant_do_that = False
        try:
            ep.sample_initial_binaries()
        except NotImplementedError:
            cant_do_that = True
        self.assertTrue(cant_do_that)

        cant_do_that = False
        try:
            ep.perform_stellar_evolution()
        except NotImplementedError:
            cant_do_that = True
        self.assertTrue(cant_do_that)

        ep.create_population()

    def test_bin_nums(self):
        """Check that we are creating the correct bin_nums"""
        p = pop.Population(10)

        # can't get bin_nums before evolution
        it_failed = False
        try:
            p.bin_nums
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

        p.create_population()

        initC_bin_nums = p.initC["bin_num"].unique()
        bpp_bin_nums = p.bpp["bin_num"].unique()

        self.assertTrue(np.all(p.bin_nums == initC_bin_nums))
        self.assertTrue(np.all(p.bin_nums == bpp_bin_nums))

        self.assertTrue(len(p) == len(p.bin_nums))
