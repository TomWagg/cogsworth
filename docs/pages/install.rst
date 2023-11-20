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

        #. **OPTIONAL:** If you want to install some of the ``cogsworth`` extras (necessary for features such as observables predictions, action-based galactic potentials and more) then you can do so by running::

                pip install cogsworth[extras]

        and you should be all set! Now it's time to learn about `Getting Started <getting_started>`_ with ``cogsworth``.

    .. tab-item:: Stable (without conda)

        We don't recommend installing ``cogsworth`` without a conda environment but if you prefer to do it this
        way then all you need to do is run::

            pip install cogsworth

        **OPTIONALLY** if you want to install some of the ``cogsworth`` extras (necessary for features such as observables predictions, action-based galactic potentials and more) then you can do so by instead running::

            pip install cogsworth[extras]

        and you should be all set! Now it's time to learn about `Getting Started <getting_started>`_ with ``cogsworth``.

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

        **OPTIONALLY** if you want to install some of the ``cogsworth`` extras (necessary for features such as observables predictions, action-based galactic potentials and more) then you can do so by instead running::

            pip install .[extras]

        and you should be all set! Now it's time to learn about `Getting Started <getting_started>`_ with ``cogsworth``.

.. tip::
    If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!


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
