**************
Full changelog
**************

This page tracks all of the changes that have been made to ``cogsworth``. We follow the standard versioning convention of A.B.C, where C is a patch/bugfix, B is a large bugfix or new feature and A is a major new breaking change. B/C are backwards compatible but A changes may be breaking.

3.6.1
=====
- Enhancement: Allow users to specify a directory for saving bad orbits when integrating populations by adding an `error_file_path` argument to the Population class. If set, bad orbits will be saved to this directory instead of the current working directory. If set to None, bad orbits will not be saved. This provides more flexibility in managing output files.
- Development: Default BSE settings are drawn directly from the COSMIC cosmic-settings.json data file
- Development: Use new initC IO functions from COSMIC that compress the data a LOT
- Development: Add testing for Python 3.11, 3.12, and 3.13

3.6.0
=====
- New feature: Move observables to obs module and added obs.mist module to interpolate MIST bolometric corrections. This means we can avoid depending on isochrones which is largely unmaintained at this point.

3.5.1
=====
- Enhancement: Avoid pickling static arguments unnecessarily when evolving orbits, this should speed things up and reduce memory usage.

3.5.0
=====
- New feature: Added support for the latest version of Gala (v1.11.0). This means time-evolving potentials can now be used when integrating orbits for populations, as well as connections to EXP and new interface with the MilkyWayPotential class.
- New feature: When saving populations, the versions of `cogsworth`, `COSMIC`, and `gala` used to create the population are now stored as attributes in the saved file. This allows for better tracking of software versions for reproducibility.
- Bug fix: When loading populations, ensure that `bpp_columns` and `bcm_columns` are properly decoded from byte strings to regular strings. This resolves issues when these attributes were not of the expected type after loading a saved population.

3.4.0
=====
- New feature: Populations that have had orbits integrated can now be concatenated together using the `pop.concat` function or simply the `+` operator. Note that the resulting population will not have orbits, as combining orbits from different populations is non-trivial and not yet implemented. However, the final positions and velocities are available as `.final_pos` and `.final_vel` attributes if they were loaded for the original populations.
- Bug fix: Ensure that when concatenating populations, the bin_nums are updated correctly to avoid overlaps and that everything remains unique.

3.3.1
=====
- Bug fix: Change `ConstantPlummerSphere` to directly accept the Plummer scale radius and mass rather than a Gala potential instance. This avoids indexing issues

3.3.0
=====
- New feature: Added a new star formation history in the `cogsworth.sfh` module called `ConstantPlummerSphere`, which allows users to create a population of stars formed at a constant rate within a Plummer sphere potential. This SFH samples stellar positions and velocities according to the Plummer model, providing a more realistic spatial and kinematic distribution for certain astrophysical scenarios.

3.2.3
=====
- Bug fix: When indexing or copying populations, ensure that `bpp_columns` and `bcm_timestep_conditions` are preserved correctly. This resolves issues where these attributes were lost or incorrectly set after such operations.
- Bug fix: When loading populations, ensure that `bpp_columns` and `bcm_columns` are only converted to `None` if they are explicitly set to the string 'None'. Don't perform check on lists.
- Bug fix: Saving/loading a SandersBinney2015 sfh now works correctly, previously failed due to bad saving of precomputed interpolations and potential.

3.2.2
=====
- Bug fix: When loading populations, ensure that `bpp_columns` and `bcm_timestep_conditions` are properly converted back to lists from numpy arrays. This resolves issues when these attributes were not of the expected type after loading a saved population.

3.2.1
=====
- Bug fix: Ensure masks used in plotting functions are always the correct length to avoid broadcasting issues. Problems arose when orbits were removed after bad integration.

3.2.0
=====
- New feature: Added distribution function based star formation histories in the `cogsworth.sfh` module, allowing users to create more realistic SFHs based on analytic distribution functions for stellar systems in equilibrium.

3.1.0
=====
- Bug fix: Inclinations of binaries relative to the galactic plane are now drawn from a uniform in cos(i) distribution rather than uniform in i

3.0.0
=====
- Major breaking change: `cogsworth` no longer allows you to use the default BSE settings unless you explicitly set `use_default_BSE_settings=True` when creating a Population. This is to avoid users passing settings without acknowledging that they are making choices about the binary physics.
    - This also fixes an issue where the settings in an initC table were being overwritten by BSE_settings
- Update default `kickflag` to match COSMIC `v3.6.1` with the Disberg distribution instead of Hobbs

2.1.1
=====
- Allow choice of `bpp` and `bcm` columns as I implemented in COSMIC (see [#86](https://github.com/TomWagg/cogsworth/issues/86))

2.1.0
=====

- New feature: `plot_orbit` now shows the location of mergers (either stellar or GW), this can be turned off with the `show_merger` argument
- Bug fix: `plot_orbit` hides the SN/merger locations if they occur after `t_max` (previously they were shown even though the orbit might not reach them)
- Bug fix ([#154](https://github.com/TomWagg/cogsworth/issues/154)): Ensure secondary SNe are still shown for bound binaries in `plot_orbit`

2.0.4
=====
- `cogsworth` is now published 🎉 Citations are updated to match

2.0.3
=====
- Update default `kickflag` to match COSMIC `v3.5.0`

2.0.2
=====
- Add some more flexbility to the ``plot_cartoon_binary`` function in terms of fontsize
- Allow use of pynbody 2.0 with cogsworth

2.0.1
=====

- Add clearer warning when pynbodyrc is not set up for FIRE (https://github.com/TomWagg/cogsworth/issues/137) 

2.0.0
=====

Major release to go with the release paper submission! 🎉
- New feature: ``plot_cartoon_binary`` will now adjust the width of the binary based on the orbital separation and label simultaneous timesteps more clearly, also add a marker for disruption events
- Major change: calls like ``p.bpp`` will now raise an error if sampling is not yet done to avoid confusion
- Bug fix: Can now save and load unevolved populations
- Bug fix: Saving ``sampling_params`` now works correctly when you have a ``sampling_params`` object that includes a dictionary (nested dictionaries were causing crashes before)

1.2.0
=====

* Add some new simple SFHs by @TomWagg in https://github.com/TomWagg/cogsworth/pull/120
* Add new orbit plotting functionality by @TomWagg in https://github.com/TomWagg/cogsworth/pull/121
* Add new case studies + minor fixes by @TomWagg in https://github.com/TomWagg/cogsworth/pull/122

1.1.2
=====

- Bug fix: saving ``sfh`` objects with ``save`` now works correctly (previously failed when they had custom parameters)

1.1.1
=====

- Bug fix: ``concat`` functions now accessible in the main namespace

1.1.0
=====

- New feature: Concatenate multiple populations together with ``pop.concat`` or simply `+` (see #116)

1.0.0
=====

Major release with many new features and bugfixes. The main new features are:

- Allow for optional dependencies by @TomWagg in https://github.com/TomWagg/cogsworth/pull/54
- Personalised citation information framework by @TomWagg in https://github.com/TomWagg/cogsworth/pull/63
- Make sample_params an option by @TomWagg in https://github.com/TomWagg/cogsworth/pull/60
- Indexing improvements by @TomWagg in https://github.com/TomWagg/cogsworth/pull/69
- No more Skycoords except where necessary by @TomWagg in https://github.com/TomWagg/cogsworth/pull/79
- Add translation and cartoon helpers by @TomWagg in https://github.com/TomWagg/cogsworth/pull/80
- Simplify the way in which we store disruptions to arrays instead of two lists by @TomWagg in https://github.com/TomWagg/cogsworth/pull/82
- Avoid pickled storage (speed up orbits save/load) by @TomWagg in https://github.com/TomWagg/cogsworth/pull/84
- Change to storing everything directly in a single file instead of 4 by @TomWagg in https://github.com/TomWagg/cogsworth/pull/85
- Allow dynamic timestep resolution for ``bcm``, don't save otherwise by @TomWagg in https://github.com/TomWagg/cogsworth/pull/89
- BUG: fix duplicate timestep bug in kicked orbits by @TomWagg in https://github.com/TomWagg/cogsworth/pull/90
- Add connections to ``LEGWORK`` by @TomWagg in https://github.com/TomWagg/cogsworth/pull/91
- Add ``hydro`` submodule for creating populations from hydrodynamical zoom-in simulations by @TomWagg in https://github.com/TomWagg/cogsworth/pull/96
- Change nomenclature from ``Galaxy`` to ``StarFormationHistory`` by @TomWagg in https://github.com/TomWagg/cogsworth/pull/100
- Improve cluster velocity dispersion by @TomWagg in https://github.com/TomWagg/cogsworth/pull/101
- Allow the use of ``ini`` files to specify ``BSE_settings`` by @TomWagg in https://github.com/TomWagg/cogsworth/pull/107
- Add initial velocity storage by @TomWagg in https://github.com/TomWagg/cogsworth/pull/109
- Ensure sampling params are saved in Populations by @TomWagg in https://github.com/TomWagg/cogsworth/pull/110
- Improve saving and lazy-loading of files by @TomWagg in https://github.com/TomWagg/cogsworth/pull/113
- Save binary inclination and phase at each SN to ensure reproducibility by @TomWagg in https://github.com/TomWagg/cogsworth/pull/115

0.3.0
=====

- Made several dependencies optional
- Allow users to specify `sampling_params` to pass to COSMIC and simplify drawing singles stars

0.2.0
=====

- Add new options for action-based initial galaxy distributions

0.1.0
=====

- Add option to copy initial conditions from another Population

0.0.0
=====

- Initial release (woop!)