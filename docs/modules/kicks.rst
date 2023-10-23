*****************************
Orbit integration (``kicks``)
*****************************

In ``kicks`` you'll find functions for applying the effect of supernova kicks to orbital integration.
This is mostly handled by :func:`~cogsworth.kicks.integrate_orbit_with_events`, which provides a wrapper on
Gala's :meth:`~gala.potential.potential.PotentialBase.integrate_orbit` to additionally handle the kick events.

.. automodapi:: cogsworth.kicks
    :no-heading:
