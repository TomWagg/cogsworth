**************
Full changelog
**************

This page tracks all of the changes that have been made to ``cogsworth``. We follow the standard versioning convention of A.B.C, where C is a patch/bugfix, B is a large bugfix or new feature and A is a major new breaking change. B/C are backwards compatible but A changes may be breaking.

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