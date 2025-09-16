import unittest
import cogsworth
import pandas as pd


class Test(unittest.TestCase):
    def test_various_events(self):
        """Ensure functions work for different kinds of events

        I'm going to use fake bpp and kick_info tables for this. There will be four binaries
            - One with no events
            - One with a bound binary and a kick
            - One with a disruption from the first SN
            - One with a disruption from the second SN
        """

        bpp_dict = {
            "mass_1": [1, 10, 20, 1.4, 20, 20, 8],
            "mass_2": [1, 10, 20, 20, 20, 20, 1.4],
            "tphys": [0, 0, 0, 1, 0, 1, 2],
            "sep": [10, 10, 10, -1.0, 10, 10, -1.0],
            "ecc": [0, 0, 0, -1.0, 0, 0.1, -1.0],
            "evol_type": [-1, 15, 15, 11, 15, 16, 11],
            "bin_num": [0, 1, 2, 2, 3, 3, 3],
        }
        bpp = pd.DataFrame(data=bpp_dict)
        bpp.set_index("bin_num", drop=False, inplace=True)

        kick_info_dict = {
            "star": [0, 1, 1, 1, 2],
            "disrupted": [0, 0, 1, 0, 1],
            "delta_vsysx_1": [0, 0, 0, 0, 0],
            "delta_vsysy_1": [0, 0, 0, 0, 0],
            "delta_vsysz_1": [0, 0, 0, 0, 0],
            "delta_vsysx_2": [0, 0, 0, 0, 0],
            "delta_vsysy_2": [0, 0, 0, 0, 0],
            "delta_vsysz_2": [0, 0, 0, 0, 0],
            "bin_num": [0, 1, 2, 3, 3],
        }
        kick_info = pd.DataFrame(data=kick_info_dict)
        kick_info.set_index("bin_num", drop=False, inplace=True)

        p = cogsworth.pop.Population(4, use_default_BSE_settings=True)
        p._bpp = bpp
        p._kick_info = kick_info
        p._initC = pd.DataFrame(data={"metallicity": [1e-2, 1e-2, 1e-2, 1e-2]})
        p.final_bpp

        primary_events, secondary_events = cogsworth.events.identify_events(p)

        self.assertTrue(len(primary_events) == 4)
        self.assertTrue(len(secondary_events) == 4)
        self.assertTrue(primary_events[0] is None)
        self.assertTrue(secondary_events[1] is None)
        self.assertTrue(secondary_events[2] is not None)
        self.assertTrue(len(secondary_events[3]) > len(secondary_events[2]))
