**************************************
Hydrodynamical simulations (``hydro``)
**************************************

The ``hydro`` module contains the functionality necessary for running simulations that use hydrodynamical
zoom-in simulations as their initial conditions. This includes the ability to read in snapshots,
centre and rotate them, estimate the galactic potential, get the initial conditions of star particles and
run a full ``cogsworth`` Population based on all that!

Populations (``hydro.pop``)
===========================

.. automodapi:: cogsworth.hydro.pop
    :no-inheritance-diagram:
    :no-heading:

Estimating potentials (``hydro.potential``)
===========================================

.. automodapi:: cogsworth.hydro.potential
    :no-heading:

Initial conditions (``hydro.rewind``)
=====================================

.. automodapi:: cogsworth.hydro.rewind
    :no-heading:

Utility functions (``hydro.utils``)
===================================

.. automodapi:: cogsworth.hydro.utils
    :no-heading: