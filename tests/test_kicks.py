import unittest
import cogsworth
import numpy as np
import gala.dynamics as gd
import astropy.units as u
import pandas as pd

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

        p = cogsworth.pop.Population(5, final_kstar1=[13, 14], processes=1,
                                     use_default_BSE_settings=True)
        p.create_population()
        first_pos = p.final_pos.copy()

        p.perform_galactic_evolution()
        second_pos = p.final_pos.copy()

        self.assertTrue(np.allclose(first_pos, second_pos))

    def test_events_outside_range(self):
        """Test that events outside the integration range don't affect an orbit"""

        w0 = gd.PhaseSpacePosition(pos=[8, 0, 0] * u.kpc, vel=[0, 220, 0] * u.km / u.s)
        t1 = 0 * u.Myr
        t2 = 50 * u.Myr
        dt = 1 * u.Myr

        # the second event is outside the integration range and should be ignored
        events = pd.DataFrame({
            "tphys": [10, 60],
            "delta_vsys_x": [0, 0],
            "delta_vsys_y": [0, 0],
            "delta_vsys_z": [0, 100],
            "inc": [0, 0],
            "phase": [0, 0]
        })

        orbit = cogsworth.kicks.integrate_orbit_with_events(
            w0, t1, t2, dt, events=events,
        )

        # the orbit should still be close to the plane if nothing went wrong
        self.assertTrue(
            np.all(orbit.z.to(u.kpc) < 0.1 * u.kpc)
        )

    def test_out_of_order_events(self):
        """Test that events that are out of order in time are still applied correctly"""

        w0 = gd.PhaseSpacePosition(pos=[8, 0, 0] * u.kpc, vel=[0, 220, 0] * u.km / u.s)
        t1 = 0 * u.Myr
        t2 = 50 * u.Myr
        dt = 1 * u.Myr

        # the second event is outside the integration range and should be ignored
        events = pd.DataFrame({
            "tphys": [10, 20],
            "delta_vsys_x": [0, 0],
            "delta_vsys_y": [10, 0],
            "delta_vsys_z": [0, 10],
            "inc": [0, 0],
            "phase": [0, 0]
        })

        backwards_events = events.iloc[::-1].reset_index(drop=True)

        orbit = cogsworth.kicks.integrate_orbit_with_events(
            w0, t1, t2, dt, events=events,
        )
        backwards_orbit = cogsworth.kicks.integrate_orbit_with_events(
            w0, t1, t2, dt, events=backwards_events,
        )

        # the orbits should be the same regardless of the order of the events
        self.assertTrue(
            np.allclose(orbit.x.to(u.kpc), backwards_orbit.x.to(u.kpc))
        )
