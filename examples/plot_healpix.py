"""
Healpix Plot
============

Create a healpix map of your final population

This example shows how you can use the :meth:`~cogsworth.pop.Population.plot_map` function in 
:class:`~cogsworth.pop.Population` class to create a healpix map after
running your population. Try increasing ``n_binaries`` or ``nside`` to see a higher resolution plot with more
data. Also feel free to turn ``with_timing`` back on.
"""

import cogsworth
p = cogsworth.pop.Population(1000, processes=1, BSE_settings={"binfrac": 1.0})
p.create_population(with_timing=False)

p.plot_map(ra="auto", dec="auto", nside=16, norm="log")
