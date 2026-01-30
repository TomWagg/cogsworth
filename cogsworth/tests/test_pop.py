import numpy as np
import unittest
import cogsworth.pop as pop
import cogsworth.sfh as sfh
import cogsworth.obs.observables as obs
import h5py as h5
import os
import pytest
import astropy.units as u
import matplotlib.pyplot as plt
import tempfile


class Test(unittest.TestCase):
    def test_bad_inputs(self):
        """Ensure the class fails with bad input"""
        it_broke = False
        try:
            pop.Population(n_binaries=-100, use_default_BSE_settings=True)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

        it_broke = False
        try:
            pop.Population(n_binaries=0, use_default_BSE_settings=True)
        except ValueError:
            it_broke = True
        self.assertTrue(it_broke)

    def test_io(self):
        """Check that a population can be saved and re-loaded"""

        with tempfile.TemporaryDirectory() as tmpdir:
            # do one with just initial sampling
            p = pop.Population(2, processes=1, sampling_params={"qmin": 0.5}, use_default_BSE_settings=True)
            p.sample_initial_galaxy()
            p.sample_initial_binaries()
            p.save(os.path.join(tmpdir, "testing-pop-io"), overwrite=True)

            p_loaded = pop.load(os.path.join(tmpdir, "testing-pop-io"), 
                                parts=["initial_binaries", "initial_galaxy"])

            self.assertTrue(np.all(p.initial_binaries["mass_1"] == p_loaded.initial_binaries["mass_1"]))

            # again with everything
            p = pop.Population(2, processes=1, bcm_timestep_conditions=[['dtp=100000.0']],
                            sampling_params={"qmin": 0.5}, use_default_BSE_settings=True)
            p.create_population()

            p.save(os.path.join(tmpdir, "testing-pop-io"), overwrite=True)

            p_loaded = pop.load(os.path.join(tmpdir, "testing-pop-io"),
                                parts=["initial_binaries", "initial_galaxy",
                                       "stellar_evolution", "galactic_orbits"])

            self.assertTrue(np.all(p.bpp == p_loaded.bpp))
            self.assertTrue(np.all(p.final_pos == p_loaded.final_pos))
            self.assertTrue(np.all(p.orbits[0].pos == p_loaded.orbits[0].pos))
            self.assertTrue(np.all(p.initial_galaxy.v_R == p_loaded.initial_galaxy.v_R))
            self.assertTrue(p.sampling_params == p_loaded.sampling_params)

            # attempt overwrite without setting flag
            it_broke = False
            try:
                p.save(os.path.join(tmpdir, "testing-pop-io"))
            except FileExistsError:
                it_broke = True
            self.assertTrue(it_broke)

            # attempt overwrite now WITH the flag
            it_broke = False
            try:
                p.save(os.path.join(tmpdir, "testing-pop-io"), overwrite=True)
            except Exception as e:
                print(e)
                it_broke = True
            self.assertFalse(it_broke)

    def test_io_types(self):
        """Ensure that certain variables still have the same type after saving and loading"""
        p = pop.Population(2, processes=1, use_default_BSE_settings=True)
        p.create_population()

        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(os.path.join(tmpdir, "testing-pop-io-types"), overwrite=True)
            p_loaded = pop.load(os.path.join(tmpdir, "testing-pop-io-types"))

        self.assertTrue(type(p.bcm_timestep_conditions) == type(p_loaded.bcm_timestep_conditions))
        self.assertTrue(type(p.bpp_columns) == type(p_loaded.bpp_columns))
        self.assertTrue(type(p.bcm_columns) == type(p_loaded.bcm_columns))

    def test_io_cols(self):
        """Ensure that loading columns is possible"""
        p = pop.Population(2, processes=1, use_default_BSE_settings=True,
                           bpp_columns=["tphys", "mass_1", "mass_2", "sep", "evol_type"],
                           bcm_columns=["tphys", "mass_1", "mass_2", "porb", "ecc", "sep"],
                           bcm_timestep_conditions=[['dtp=0.0']])
        p.create_population()

        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(os.path.join(tmpdir, "testing-pop-io-cols"), overwrite=True)
            p_loaded = pop.load(os.path.join(tmpdir, "testing-pop-io-cols"))

        self.assertTrue(set(p.bpp.columns) == set(p_loaded.bpp.columns))
        self.assertTrue(set(p.bcm.columns) == set(p_loaded.bcm.columns))

    def test_io_versions(self):
        """Check that version mismatches are warned about"""
        p = pop.Population(2, processes=1, use_default_BSE_settings=True)
        p.create_population()

        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(os.path.join(tmpdir, "testing-pop-io-versions"), overwrite=True)

            # load the file and mess with the versions
            with h5.File(os.path.join(tmpdir, "testing-pop-io-versions.h5"), "a") as f:
                f.attrs["cogsworth_version"] = "0.0.0"
                f.attrs["COSMIC_version"] = "0.0.0"
                f.attrs["gala_version"] = "0.0.0"

            with self.assertLogs("cogsworth", level="WARNING") as cm:
                p_loaded = pop.load(os.path.join(tmpdir, "testing-pop-io-versions"))
            self.assertIn("file was saved with", cm.output[0])

    def test_save_complicated_sampling(self):
        """Check that you can save a population with complicated sampling params"""
        p = pop.Population(2, processes=1,
                           sampling_params={
                               "qmin": 0.5,
                               "porb_model": {
                                    "min": 0.15,
                                    "max": 5,
                                    "slope": 0.0
                                }
                           }, use_default_BSE_settings=True)
        p.create_population()

        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(os.path.join(tmpdir, "testing-pop-io-sampling"), overwrite=True)
            p_loaded = pop.load(os.path.join(tmpdir, "testing-pop-io-sampling"), parts=["initial_binaries"])

        # sort columns in both initC to ensure they match
        p._initC = p.initC.reindex(sorted(p.initC.columns), axis=1)
        p_loaded._initC = p_loaded.initC.reindex(sorted(p_loaded.initC.columns), axis=1)

        self.assertTrue(np.all(p.initC == p_loaded.initC))

    def test_lazy_io(self):
        """Check that a population can be saved and re-loaded lazily"""
        p = pop.Population(2, processes=1, bcm_timestep_conditions=[['dtp=100000.0']],
                           sampling_params={"qmin": 0.5}, use_default_BSE_settings=True)
        p.create_population()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(os.path.join(tmpdir, "testing-lazy-io"), overwrite=True)
            p_loaded = pop.load(os.path.join(tmpdir, "testing-lazy-io"), parts=[])

            # sort columns in both initC to ensure they match
            p._initC = p.initC.reindex(sorted(p.initC.columns), axis=1)
            p_loaded._initC = p_loaded.initC.reindex(sorted(p_loaded.initC.columns), axis=1)

            self.assertTrue(np.all(p.initC == p_loaded.initC))
            self.assertTrue(np.all(p.bpp == p_loaded.bpp))
            self.assertTrue(np.all(p.final_pos == p_loaded.final_pos))
            self.assertTrue(np.all(p.orbits[0].pos == p_loaded.orbits[0].pos))
            self.assertTrue(np.all(p.initial_galaxy.v_R == p_loaded.initial_galaxy.v_R))
            self.assertTrue(p.sampling_params == p_loaded.sampling_params)


    def test_load_no_orbits(self):
        """Check that a population can be saved without orbits, and raises an error if trying to load them"""
        p = pop.Population(2, processes=1, bcm_timestep_conditions=[['dtp=100000.0']],
                           sampling_params={"qmin": 0.5}, use_default_BSE_settings=True)
        p.sample_initial_galaxy()
        p.sample_initial_binaries()
        p.perform_stellar_evolution()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            p.save(os.path.join(tmpdir, "testing-no-orbits"), overwrite=True)
            p_loaded = pop.load(os.path.join(tmpdir, "testing-no-orbits"), parts=[])

            it_broke = False
            try:
                p_loaded.orbits
            except ValueError:
                it_broke = True
            self.assertTrue(it_broke)


    def test_wrong_load_function(self):
        """Check that errors are properly raised when the wrong load function is used"""
        g = sfh.Wagg2022(10000)

        with tempfile.TemporaryDirectory() as tmpdir:
            g.save(os.path.join(tmpdir, "test-sfh-for-load"))

            it_broke = False
            try:
                pop.load(os.path.join(tmpdir, "test-sfh-for-load"))
            except ValueError:
                it_broke = True
            self.assertTrue(it_broke)

            p = pop.Population(2, processes=1, use_default_BSE_settings=True)
            p.create_population()
            p.save(os.path.join(tmpdir, "test-pop-for-load"), overwrite=True)
            it_broke = False
            try:
                sfh.load(os.path.join(tmpdir, "test-pop-for-load"))
            except ValueError:
                it_broke = True
            self.assertTrue(it_broke)

    def test_orbit_storage(self):
        """Test that we can control how orbits are stored"""
        p = pop.Population(20, final_kstar1=[13, 14], processes=1, store_entire_orbits=True,
                           use_default_BSE_settings=True)
        p.create_population()

        first_orbit = p.orbits[0][0] if isinstance(p.orbits[0], list) else p.orbits[0]
        self.assertTrue(first_orbit.shape[0] >= 1)

        p = pop.Population(20, final_kstar1=[13, 14], processes=1, store_entire_orbits=False,
                           use_default_BSE_settings=True)
        p.create_population()

        first_orbit = p.orbits[0][0] if isinstance(p.orbits[0], list) else p.orbits[0]

        self.assertTrue(first_orbit.shape[0] == 1)

    def test_overly_stringent_cutoff(self):
        """Make sure that it crashes if the m1_cutoff is too large to create anything"""
        p = pop.Population(10, processes=1, m1_cutoff=10000,
                           use_default_BSE_settings=True)

        it_broke = False
        try:
            p.create_population()
        except ValueError:
            it_broke = True

        self.assertTrue(it_broke)

    def test_interface(self):
        """Test the interface of this class with the other modules"""
        p = pop.Population(10, processes=1, final_kstar1=[13, 14], store_entire_orbits=False,
                           use_default_BSE_settings=True)
        p.create_population()

        # ensure we get something that disrupts to ensure coverage
        MAX_REPS = 5
        i = 0
        while not p.disrupted.any() and i < MAX_REPS:
            p = pop.Population(10, processes=1, final_kstar1=[13, 14], use_default_BSE_settings=True)
            p.create_population()
            i += 1
        if i == MAX_REPS:
            raise ValueError("Couldn't make anything disrupt :/")

        # test that classes can be identified
        self.assertTrue(p.classes.shape[0] == p.n_binaries_match)

        # test that observable table is done right (with or without extinction)
        try:
            p.observables
        except ValueError:
            pass
        p.get_observables(filters=["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"],
                          assume_mw_galactocentric=True, ignore_extinction=True)
        obs.get_photometry(filters=["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"], final_bpp=p.final_bpp,
                           final_pos=p.final_pos, assume_mw_galactocentric=True, ignore_extinction=True)
        p.observables

        # cheat and make sure at least one binary is bright enough
        p.observables["Gaia_G_EDR3_app_1"].iloc[0] = 18.0
        p.get_gaia_observed_bin_nums(ra="auto", dec="auto")

        it_worked = True
        try:
            p.get_healpix_inds()
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

        p.plot_map(ra="auto", dec="auto", coord="C", show=False)
        p.plot_map(ra="auto", dec="auto", coord="G", show=False)
        p.plot_sky_locations(show=False)

        # find a binary that disrupted
        bn = p.bin_nums[p.disrupted][0]
        p.plot_orbit(bn, show=False)
        p.plot_orbit(bn, t_max=0.1 * u.Myr, show=False)
        plt.close("all")

    def test_initial_binaries_replace_initC(self):
        """Test that initial binaries returns initC if present and initial_binaries is not"""

        p = pop.Population(2, processes=1, use_default_BSE_settings=True)
        p.sample_initial_binaries()
        p.sample_initial_galaxy()
        p.perform_stellar_evolution()

        p._initial_binaries = None
        self.assertTrue(p.initial_binaries is p.initC)

    def test_getters(self):
        """Test the property getters"""
        p = pop.Population(2, processes=1, store_entire_orbits=False,
                           bcm_timestep_conditions=[['dtp=1000.0']], use_default_BSE_settings=True)
        p.create_population()

        # test getters from sampling
        p.mass_singles
        p._mass_singles = None
        try:
            p.mass_singles
        except ValueError:
            pass

        p.mass_binaries
        p._mass_binaries = None
        try:
            p.mass_binaries
        except ValueError:
            pass

        p.n_singles_req
        p._n_singles_req = None
        try:
            p.n_singles_req
        except ValueError:
            pass

        p.n_bin_req
        p._n_bin_req = None
        try:
            p.n_bin_req
        except ValueError:
            pass

        p.initial_galaxy
        p._initial_galaxy = None
        try:
            p.initial_galaxy
        except ValueError:
            pass

        # test getters from stellar evolution
        p.bpp
        p._bpp = None
        try:
            p.bpp
        except ValueError:
            pass

        p.bcm
        p._bcm = None
        try:
            p.bcm
        except ValueError:
            pass

        p.initC
        p._initC = None
        try:
            p.initC
        except ValueError:
            pass

        p.kick_info
        p._kick_info = None
        try:
            p.kick_info
        except ValueError:
            pass

        p.create_population()

        # test getters for galactic evolution
        p.orbits
        o = p.orbits.copy()
        p._orbits = None
        try:
            p.orbits
        except ValueError:
            pass
        p._orbits = o

        p.primary_orbits
        p.secondary_orbits

        p._final_vel = None
        p.final_vel

        p.get_final_mw_skycoord()

        p.escaped

    def test_singles_evolution(self):
        """Check everything works well when evolving singles"""
        p = pop.Population(2, processes=1, BSE_settings={"binfrac": 0.0},
                           sampling_params={'keep_singles': True, 'total_mass': 100,
                                            'sampling_target': 'total_mass'}, use_default_BSE_settings=True)
        p.create_population(with_timing=False)

        self.assertTrue((p.final_bpp["sep"] == 0.0).all())

    def test_singles_bad_input(self):
        """Test what happens when you mess up single stars"""
        it_failed = True
        p = pop.Population(1, processes=1, BSE_settings={"binfrac": 0.0},
                           sampling_params={'total_mass': 1000, 'sampling_target': 'total_mass'},
                           use_default_BSE_settings=True)
        try:
            p.sample_initial_binaries()
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

    def test_from_initC(self):
        """Check it can handle only having an initC rather than initial_binaries"""
        p = pop.Population(2, use_default_BSE_settings=True)
        p.sample_initial_binaries()
        p.perform_stellar_evolution()
        p._initial_binaries = None
        p.perform_stellar_evolution()

    def test_none_orbits(self):
        """Ensure final_pos/vel still works when there is an Orbit with None"""
        p = pop.Population(2, use_default_BSE_settings=True)
        p._orbits = [None, None]
        self.assertTrue(p.final_pos[0][0].value == np.inf)
        self.assertTrue(p.final_vel[0][0].value == np.inf)

    @pytest.mark.filterwarnings("ignore:.*duplicate")
    def test_indexing(self):
        """Ensure that indexing works as expected for proper types"""
        p = pop.Population(10, bcm_timestep_conditions=[['dtp=100000.0']], use_default_BSE_settings=True)
        p.create_population()
        inds = [int(np.random.choice(p.bin_nums, replace=False)),
                np.random.choice(p.bin_nums, size=4, replace=False),
                list(np.random.choice(p.bin_nums, size=2, replace=False)),
                slice(0, 7, 3),
                [0, 1, 1, 1, 0]]

        # mock up some data so it tests the indexing
        p._observables = p.final_bpp
        p._classes = p.final_bpp
        p._final_pos = np.zeros(len(p._orbits))
        p._final_vel = np.zeros(len(p._orbits))

        p.copy()

        for ind in inds:
            p_ind = p[ind]
            if isinstance(ind, slice):
                ind = list(range(ind.stop)[ind])
            elif isinstance(ind, int):
                ind = [ind]
            og_m1 = p.final_bpp.loc[ind]["mass_1"].values
            self.assertTrue(np.all(og_m1 == p_ind.final_bpp["mass_1"].values))

    def test_indexing_bad_type(self):
        """Ensure that indexing breaks on bad types (reprs too)"""
        p = pop.Population(10, use_default_BSE_settings=True)
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

    def test_indexing_bad_bin_num(self):
        """Ensure that indexing breaks on non-existent bin nums"""
        p = pop.Population(10, use_default_BSE_settings=True)
        p.create_population()

        # make sure it fails for bin_nums that don't exist
        it_worked = True
        try:
            p[-42]
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)

    def test_indexing_booleans(self):
        """Ensure that indexing allows a boolean mask, but breaks on a dodgy version"""
        p = pop.Population(10, use_default_BSE_settings=True)
        p.create_population()

        # make sure it fails for a list of bools of the wrong length
        it_worked = True
        try:
            p[[True, False, True]]
        except AssertionError:
            it_worked = False
        self.assertFalse(it_worked)

        # make sure it works for a proper mask
        it_worked = True
        try:
            p[[True for _ in range(p.n_binaries_match)]]
        except AssertionError:
            it_worked = False
        self.assertTrue(it_worked)

    def test_indexing_mixed_types(self):
        """Don't allow indexing with mixed types"""
        p = pop.Population(10, use_default_BSE_settings=True)
        p.create_population()

        it_worked = True
        try:
            p[[0, "absolute nonsense mate"]]
        except AssertionError:
            it_worked = False
        self.assertFalse(it_worked)

    def test_indexing_loaded_pop(self):
        """Test indexing fails when trying to slice a half-loaded population"""
        p = pop.Population(10, use_default_BSE_settings=True)
        p.perform_stellar_evolution()

        with h5.File("DUMMY.h5", "w") as f:
            f.create_dataset("orbits", data=[])
        p._file = "DUMMY.h5"
        p._orbits = None

        it_worked = True
        try:
            p[:5]
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)
        os.remove("DUMMY.h5")

    def test_evolved_pop(self):
        """Check that the EvolvedPopulation class works as it should"""
        p = pop.Population(10, use_default_BSE_settings=True)
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
        p = pop.Population(10, use_default_BSE_settings=True)

        # can't get bin_nums before evolution
        it_failed = False
        try:
            p.bin_nums
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

        p.sample_initial_binaries()
        p.bin_nums
        p._bin_nums = None

        p.perform_stellar_evolution()
        p._final_bpp = None
        p.bin_nums
        p._bin_nums = None

        p.final_bpp
        p.bin_nums
        p._bin_nums = None

        initC_bin_nums = p.initC["bin_num"].unique()
        bpp_bin_nums = p.bpp["bin_num"].unique()

        self.assertTrue(np.all(p.bin_nums == initC_bin_nums))
        self.assertTrue(np.all(p.bin_nums == bpp_bin_nums))

        self.assertTrue(len(p) == len(p.bin_nums))

    def test_sampling(self):
        """Ensure that changing sampling parameters actually has an effect"""
        # choose some random q's, ensure samples actually obey changes
        for qmin in np.random.uniform(0, 1, size=10):
            p = pop.Population(1000, sampling_params={"qmin": qmin}, use_default_BSE_settings=True)
            p.sample_initial_binaries()
            q = p._initial_binaries["mass_2"] / p._initial_binaries["mass_1"]
            self.assertTrue(min(q) >= qmin)

    def test_translation(self):
        """Ensure that COSMIC tables are being translated properly"""
        p = pop.Population(10, bcm_timestep_conditions=[['dtp=100000.0']], use_default_BSE_settings=True)
        p.perform_stellar_evolution()
        p.translate_tables(replace_columns=False, label_type="short")

        self.assertTrue(p.bpp["kstar_1"].dtype == np.int64)
        self.assertTrue((p.bpp["kstar_1_str"][p.bpp["kstar_1"] == 1] == "MS").all())

        p.translate_tables(replace_columns=True)
        self.assertFalse(p.bpp["kstar_1"].dtype == np.int64)

    def test_cartoon(self):
        """Ensure that the cartoon plot works"""
        p = pop.Population(10, final_kstar1=[14], use_default_BSE_settings=True)
        p.perform_stellar_evolution()

        for bin_num in p.bin_nums:
            p.plot_cartoon_binary(bin_num, show=False)
        plt.close("all")

    def test_sampling_with_initC(self):
        """Check we can sample from an initC table"""
        p = pop.Population(10, use_default_BSE_settings=True)
        p.perform_stellar_evolution()

        p.sample_initial_binaries(initC=p.initC,
                                  overwrite_initC_settings=True,
                                  reset_sampled_kicks=True)

    def test_legwork_conversion(self):
        """Check construction of LEGWORK sources"""
        p = pop.Population(100, processes=1, use_default_BSE_settings=True)
        p.create_population()

        it_failed = False
        try:
            p.to_legwork_sources(assume_mw_galactocentric=False)
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

        sources = p.to_legwork_sources(assume_mw_galactocentric=True)
        sources.get_merger_time()

    def test_galactic_pool(self):
        """Check that you can create a pool on the fly for galactic evolution"""
        p = pop.Population(10, processes=2, use_default_BSE_settings=True)
        p.sample_initial_binaries()
        p.sample_initial_galaxy()
        p.perform_stellar_evolution()
        p.perform_galactic_evolution()
        self.assertTrue(p.pool is None)

    def test_concat(self):
        """Check that we can concatenate populations"""
        p = pop.Population(10, use_default_BSE_settings=True)
        q = pop.Population(10, use_default_BSE_settings=True)
        p.perform_stellar_evolution()
        q.perform_stellar_evolution()

        r = p + q
        self.assertTrue(len(r) == len(p) + len(q))
        self.assertTrue(len(r.initC["bin_num"].unique()) == len(r))
        self.assertTrue(len(r.initial_galaxy) == len(r))

        self.assertTrue(len(pop.concat(p)) == len(p))
        self.assertTrue(len(sfh.concat(p.initial_galaxy)) == len(p.initial_galaxy))

    def test_concat_wrong_type(self):
        """Check that we can't concatenate with the wrong type"""
        p = pop.Population(10, use_default_BSE_settings=True)
        it_failed = False
        try:
            p + 1
        except AssertionError:
            it_failed = True
        self.assertTrue(it_failed)

    def test_concat_empty(self):
        it_failed = False
        try:
            sfh.concat()
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

        it_failed = False
        try:
            pop.concat()
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

    def test_concat_mismatch(self):
        """Check that we can't concatenate populations with different stuff"""
        p = pop.Population(10, use_default_BSE_settings=True)
        q = pop.Population(10, use_default_BSE_settings=True)
        p.perform_stellar_evolution()

        it_failed = False
        try:
            p + q
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

        q.sample_initial_galaxy()
        q._initial_binaries = None
        it_failed = False
        try:
            p + q
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

        q.sample_initial_binaries()
        it_failed = False
        try:
            p + q
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

    def test_concat_no_orbits(self):
        """Check that a warning is raised when concatenating populations with orbits"""
        p = pop.Population(10, use_default_BSE_settings=True)
        q = pop.Population(10, use_default_BSE_settings=True)
        p.create_population()
        q.create_population()

        with self.assertLogs("cogsworth", level="WARNING") as cm:
            r = pop.concat(p, q)
        self.assertIn("Concatenating populations with orbits is not supported yet", cm.output[0])

    def test_concat_bin_nums_consistent(self):
        """Check that bin_nums are consistent after concatenation"""
        pops = [
            pop.Population(10, use_default_BSE_settings=True, processes=1)
            for _ in range(3)
        ]
        for p in pops:
            p.sample_initial_binaries()
            p.perform_stellar_evolution()

        total = pop.concat(*pops)
        
        # there should be the same number of unique bin_nums in total as the sum of the individuals
        total_unique_bin_nums = total.initC["bin_num"].nunique()
        sum_individual_unique_bin_nums = sum(p.initC["bin_num"].nunique() for p in pops)
        self.assertEqual(total_unique_bin_nums, sum_individual_unique_bin_nums)

    def test_concat_final_pos(self):
        """Check that final_pos is consistent after concatenation"""
        pops = [
            pop.Population(100, final_kstar1=[13, 14], use_default_BSE_settings=True, processes=1,
                           store_entire_orbits=False)
            for _ in range(2)
        ]
        for p in pops:
            p.create_population()
            p.final_pos
            p._orbits = None

        total = pop.concat(*pops)

        # final_pos should have the correct length
        self.assertEqual(len(total.final_pos), sum(len(p.final_pos) for p in pops))

        # final_pos entries should match those from the individual populations, start with bound systems
        # and then the unbound systems like in a normal population
        index = 0
        for p in pops:
            for pos in p.final_pos[:len(p)]:
                self.assertTrue(np.array_equal(total.final_pos[index], pos))
                index += 1
        for p in pops:
            for pos in p.final_pos[len(p):]:
                self.assertTrue(np.array_equal(total.final_pos[index], pos))
                index += 1


    def test_concat_final_pos_bad_input(self):
        """Check that final_pos concatenation raises error when one population lacks final positions"""
        pops = [
            pop.Population(5, use_default_BSE_settings=True, processes=1,
                           store_entire_orbits=False)
            for _ in range(2)
        ]
        for p in pops:
            p.create_population()
            p.final_pos
            p._orbits = None

        pops[-1]._final_pos = None  # simulate not having final positions for one population

        it_failed = False
        try:
            total = pop.concat(*pops)
        except ValueError:
            it_failed = True
        self.assertTrue(it_failed)

    def test_changing_columns(self):
        """Check that a different choice of bpp and bcm columns works"""
        TEST_COLS = ["mass_1", "mass_2", "tphys", "porb", "sep", "ecc", "evol_type"]
        p = pop.Population(10, processes=1, bpp_columns=TEST_COLS, bcm_columns=TEST_COLS,
                           bcm_timestep_conditions=[["mass_1 < 100", 'dtp=100000.0']],
                           use_default_BSE_settings=True)
        p.create_population()

        # bin_num is always added
        TEST_COLS += ["bin_num"]
        self.assertTrue(set(p.bpp.columns) == set(TEST_COLS))
        self.assertTrue(set(p.bcm.columns) == set(TEST_COLS))

    def test_bad_settings(self):
        """Check that passing settings incorrectly raises errors"""
        it_worked = True
        try:
            p = pop.Population(10, use_default_BSE_settings=False, BSE_settings={})
            p.create_population()
        except ValueError:
            it_worked = False
        self.assertFalse(it_worked)
