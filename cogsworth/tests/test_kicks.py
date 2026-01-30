import unittest
import cogsworth
import numpy as np


class Test(unittest.TestCase):
    def test_duplicated_timesteps(self):
        """Ensure that the locations at timesteps in orbits are never duplicated

        Motivated by a bug that occurred during the stitching of timesteps in the orbit of a kicked binary,
        so that the orbit was not continuous. This test ensures that the orbit is continuous and only tests
        the kicked binaries since the rest are unaffected"""

        # create a population likely to have lots of disruptions
        p = cogsworth.pop.Population(100, final_kstar1=[13, 14], final_kstar2=[13, 14],
                                     use_default_BSE_settings=True)
        p.create_population()

        sn_1 = np.isin(p.bin_nums, p.bpp[p.bpp["evol_type"] == 15]["bin_num"].unique())
        sn_2 = np.isin(p.bin_nums, p.bpp[p.bpp["evol_type"] == 16]["bin_num"].unique())

        primary_kick_orbits = p.primary_orbits[sn_1 | sn_2]
        secondary_kick_orbits = p.secondary_orbits[sn_1 | sn_2]
        kick_orbits = np.concatenate((primary_kick_orbits, secondary_kick_orbits))

        valid_orbit = np.repeat(True, len(kick_orbits))
        for i in range(len(kick_orbits)):
            orbit = kick_orbits[i]
            if np.any(np.diff(orbit.t) == 0.0):
                valid_orbit[i] = False
            if np.any(np.diff(orbit.x) == 0.0):
                valid_orbit[i] = False
            if np.any(np.diff(orbit.y) == 0.0):
                valid_orbit[i] = False
            if np.any(np.diff(orbit.z) == 0.0):
                valid_orbit[i] = False

        self.assertTrue(np.all(valid_orbit))

    def test_saved_inc_phase(self):
        """Test that the inclination and phase are saved correctly in the kick events - such that the same
        population at present day is fully recreated"""

        p = cogsworth.pop.Population(5, final_kstar1=[13, 14], processes=1, BSE_settings={"binfrac": 1.0},
                                     use_default_BSE_settings=True)
        p.create_population()
        first_pos = p.final_pos.copy()

        p.perform_galactic_evolution()
        second_pos = p.final_pos.copy()

        self.assertTrue(np.allclose(first_pos, second_pos))
