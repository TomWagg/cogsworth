import unittest
import kicker
import pandas as pd


class Test(unittest.TestCase):
    def test_various_events(self):
        """Ensure functions work for different kinds of events

        I'm going to use fake bpp and kick_info tables for this. There will have four binaries
            - One with no events
            - One with a bound binary and a kick
            - One with a disruption from the first SN
            - One with a disruption from the second SN
        """

        bpp_dict = {
            "mass_1": [1, 10, 20, 20, 20],
            "mass_2": [1, 10, 20, 20, 20],
            "tphys": [0, 0, 0, 0, 1],
            "sep": [10, 10, 10, 10, 10],
            "ecc": [0, 0, 0, 0, 0.1],
            "evol_type": [-1, 15, 15, 15, 16],
            "bin_num": [1, 2, 3, 4, 4],
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
            "bin_num": [1, 2, 3, 4, 4],
        }
        kick_info = pd.DataFrame(data=kick_info_dict)
        kick_info.set_index("bin_num", drop=False, inplace=True)

        events = kicker.events.identify_events(bpp, kick_info)

        self.assertTrue(events[0] is None)
        self.assertTrue(len(events[1]) == 1)
        self.assertTrue(len(events[2]) == 2)
        self.assertTrue(len(events[3]) == 2)
