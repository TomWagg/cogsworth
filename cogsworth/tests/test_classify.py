import unittest
import cogsworth
import astropy.units as u


class Test(unittest.TestCase):
    def test_bad_inputs(self):
        """Ensure failure with bad input"""
        it_broke = False
        try:
            cogsworth.classify.determine_final_classes()
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            cogsworth.classify.get_eddington_rate(10 * u.Msun)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_list(self):
        """Test listing classes"""
        it_broke = False
        try:
            cogsworth.classify.list_classes()
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)

    def test_x_rays(self):
        """Test x-ray luminosity calculation"""
        p = cogsworth.pop.Population(10, final_kstar1=[13, 14],bcm_timestep_conditions=[['dtp=100000.0']],
                                     use_default_BSE_settings=True)
        p.perform_stellar_evolution()
        bcm = p.bcm.drop_duplicates(subset="bin_num", keep='last')
        it_broke = False
        try:
            cogsworth.classify.get_x_ray_lum(bcm["mass_1"].values * u.Msun, bcm["rad_1"].values * u.Rsun,
                                             bcm["deltam_1"].values * u.Msun / u.yr,
                                             bcm["porb"].values * u.day,
                                             bcm["kstar_1"].values,
                                             bcm["mass_2"].values * u.Msun, bcm["RRLO_2"].values)
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)

        it_broke = False
        try:
            cogsworth.classify.get_eddington_rate(10 * u.Msun, radius=1 * u.Rsun)
        except Exception as e:
            print(e)
            it_broke = True
        self.assertFalse(it_broke)
