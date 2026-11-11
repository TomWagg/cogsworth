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


Dust maps (``obs.dust``)
========================

The ``dust`` submodule specifies the 3D dust maps from :mod:`dustmaps` that can be used to calculate extinction
(see :func:`~cogsworth.obs.dust.list_dustmaps`), along with functions for loading and querying them for
E(B-V). Most users will want to use :func:`~cogsworth.obs.observables.get_extinction` (or the ``dustmaps``
argument of :meth:`~cogsworth.pop.Population.get_observables`) rather than these functions directly.

.. automodapi:: cogsworth.obs.dust
    :no-heading:


Gaia astrometry (``obs.gaia``)
==============================

The ``gaia`` submodule uses :mod:`gaiamock` to predict Gaia astrometric solutions for binaries with a
luminous star and a dark companion. Most users will want to use
:meth:`~cogsworth.pop.Population.get_gaia_astrometry` rather than these functions directly.

.. automodapi:: cogsworth.obs.gaia
    :no-heading:
