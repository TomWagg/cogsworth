from more_itertools import first
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
        av_1 = np.nan_to_num(p.observables["Av_1"])
        self.assertTrue(av_1.min() >= 0.0)
