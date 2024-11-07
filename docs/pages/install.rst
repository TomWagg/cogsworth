*******
Install
*******

Package installation
====================

.. tab-set::

    .. tab-item:: Stable (with conda)

        This is our recommend installation method! Follow the steps below to start using ``cogsworth``!

        #. Create a new empty conda environment for ``cogsworth``::

                conda create --name cogsworth python=3.10

        #. Activate the environment by running::

                conda activate cogsworth

        #. Install ``cogsworth`` by running::

                pip install cogsworth

        #. **OPTIONAL:** If you want to install some of the ``cogsworth`` extras (**this is necessary for some tutorials and examples**, particularly those on observables predictions and postprocessing hydrodynamical simulations) then you can do so by running::

                pip install cogsworth[extras]

        and you should be all set! Now it's time to learn about `Getting Started <getting_started.ipynb>`_ with ``cogsworth``.

    .. tab-item:: Stable (without conda)

        We don't recommend installing ``cogsworth`` without a conda environment but if you prefer to do it this
        way then all you need to do is run::

            pip install cogsworth

        **OPTIONALLY** if you want to install some of the ``cogsworth`` extras (**this is necessary for some tutorials and examples**, particularly those on observables predictions and postprocessing hydrodynamical simulations) then you can do so by instead running::

            pip install cogsworth[extras]

        and you should be all set! Now it's time to learn about `Getting Started <getting_started.ipynb>`_ with ``cogsworth``.

    .. tab-item:: Development (from GitHub)
        
        .. warning::

            We don't guarantee that there won't be mistakes or bugs in the development version, use at your own risk!

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/TomWagg/cogsworth>`_. To start, clone the repository onto your machine: ::
        
            git clone https://github.com/TomWagg/cogsworth
            cd cogsworth

        Next, we recommend that you create a Conda environment for working with ``cogsworth``.
        You can do this by running::

            conda create --name cogsworth python=3.10

        And then activate the environment by running::

            conda activate cogsworth

        At this point, all that's left to do is install ``cogsworth``!::

            pip install .

        **OPTIONALLY** if you want to install some of the ``cogsworth`` extras (**this is necessary for some tutorials and examples**, particularly those on observables predictions and postprocessing hydrodynamical simulations) then you can do so by instead running::

            pip install .[extras]

        and you should be all set! Now it's time to learn about `Getting Started <getting_started.ipynb>`_ with ``cogsworth``.

.. tip::
    If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!


Dependencies
============

.. grid:: 1 1 2 2

    .. grid-item::

        .. card::
            
            .. div:: sd-text-center sd-fs-4 sd-text-primary sd-font-weight-bolder

                Core Dependencies
            
            .. div:: sd-text-center sd-fs-6 sd-font-italic

                Install via 'pip install cogsworth'

            ^^^^^^^^^^^^^^^^^

            The core dependencies for a basic ``cogsworth`` installation are listed below.

            - :mod:`cosmic` for binary population synthesis
            - :mod:`gala` for galactic dynamics
            - :mod:`numpy` for vectorised operations
            - :mod:`pandas` for handling ``COSMIC`` dataframes
            - :mod:`matplotlib` for plotting
            - :mod:`scipy` for statistical distrubtions and integration
            - :mod:`astropy` for coordinate transformations

        .. card::
            
            .. div:: sd-text-center sd-fs-4 sd-text-primary sd-font-weight-bolder

                Development Dependencies
            
            .. div:: sd-text-center sd-fs-6 sd-font-italic

                Install via 'pip install cogsworth[all]'

            ^^^^^^^^^^^^^^^^^^^^^^^^

            For developers of ``cogsworth`` there are also additional dependencies for testing (``pytest``, ``coverage``, etc.) and documentation building (``sphinx``, ``nbspinx``, etc.).
            
            Most users do **not** need these dependencies.

    .. grid-item::

        .. card::
            
            .. div:: sd-text-center sd-fs-4 sd-text-primary sd-font-weight-bolder

                Optional Dependencies
            
            .. div:: sd-text-center sd-fs-6 sd-font-italic

                Install via 'pip install cogsworth[extras]'

            ^^^^^^^^^^^^^^^^^^^^^^^^

            In addition to the core dependencies, there are a number of optional dependencies that are required for some of the tutorials and examples in the documentation.

            **Observables predictions**:

            - :mod:`isochrones` for applying bolometric corrections using stellar isochrones
            - :mod:`dustmaps` for accounting for dust extinction
            - :mod:`gaiaunlimited` for applying the empirical Gaia selection function

            **Postprocessing hydrodynamical simulations**:

            - :mod:`pynbody` for reading and manipulating hydrodynamical simulations

            **LISA gravitational wave sources**:

            - :mod:`legwork` for calculating LISA gravitational wave signals
            
            **Action-based galactic potentials**:

            - :mod:`agama` for action-based galactic potentials


Data downloads for observables
==============================

If you want to make predictions for observables with ``cogsworth`` then you'll need to download some data
files. Specifically, ``dustmaps`` requires the actual dust map files and ``gaiaunlimited`` needs data for the
empirical gaia selection function. Below are some code snippets for downloading these files.

Dust maps
---------

If you'd like to apply dust maps then you'll need to download the right files. This will depend on the dust
map that you use, but here's an example for Bayestar2019 that downloads the files if you don't have them
already::

    import os
    import dustmaps.bayestar
    from dustmaps.std_paths import data_dir
    bayestar_path = os.path.join(data_dir(), 'bayestar', '{}.h5'.format("bayestar2019"))
    if not os.path.exists(bayestar_path):
        dustmaps.bayestar.fetch()

Gaia empirical selection function
---------------------------------

If you'd like to use ``cogsworth`` to make predictions for which stars are observable by Gaia then you'll need
to run the following to ensure there's a directory for the files::
    
    import os
    gaia_unlimited_path = os.path.join(os.path.expanduser('~'), ".gaiaunlimited")
    if not os.path.isdir(gaia_unlimited_path):
        os.mkdir(gaia_unlimited_path)
