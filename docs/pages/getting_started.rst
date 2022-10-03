***************
Getting Started
***************

Welcome! You've found the documentation for ``kicker``, good for you! This is an epic crossover episode of a
package in which we bring the worlds of rapid population synthesis and galactic dynamics together and see what
happens.

You should know that package relies heavily on :mod:`cosmic` and :mod:`gala` and I recommend you check out
their respective documentation. Note that if you look at the :mod:`gala` documentation and think "huh, this
seems familiar", yet that is because I have `shamelessly` stolen this template from Adrian.

Okay, let's get into this!

Basic imports
=============

For more of the examples that I give in this documentation you're probably going to need the following basic
imports. You can always read more about :mod:`numpy` and :mod:`astropy.units` at those respective links.

    import numpy as np
    import astropy.units as u


Create your first population
============================

The central class in this package is :class:`~kicker.pop.Population`::

    p = kicker.pop.Population(n_binaries=1000)

Subtitle
--------

Yet another Subtitle
--------------------

Now for a subsubtitle
^^^^^^^^^^^^^^^^^^^^^

Other links
===========

Some text

* `Link <index.rst>`

.. contents:: Table of Contents
    :depth: 3
