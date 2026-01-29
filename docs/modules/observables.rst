*********************
Observables (``obs``)
*********************

The ``obs`` module contains functionality for calculating observables for stars and binaries in a 
``cogsworth`` :class:`~cogsworth.pop.Population`.

MIST bolometric corrections (``obs.mist``)
==========================================

The ``mist`` submodule contains functionality for
interpolating MIST bolometric corrections to get magnitudes in various filters.

.. automodapi:: cogsworth.obs.mist
    :no-heading:


General observables (``obs.observables``)
=========================================

In the ``observables`` submodule you'll find functions
:func:`~cogsworth.obs.observables.get_extinction` and :func:`~cogsworth.obs.observables.get_photometry`, 
but there are also some helper functions for dealing with magnitudes.

.. automodapi:: cogsworth.obs.observables
    :no-heading:
