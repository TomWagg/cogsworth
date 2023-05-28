Install
=======

.. tab-set::

    .. tab-item:: Stable (with conda)

        This is our recommend installation method! Follow the steps below to start using ``PACKAGE NAME``!

        #. :download:`Download the environment.yml file from our repository <https://raw.githubusercontent.com/TomWagg/cosmic-gala/main/environment.yml>`
        #. Create a new conda environment using this file::

                conda env create -f path/to/environment.yml

        #. Activate the environment by running::

                conda activate cosmic-gala

        and you should be all set! Now it's time to learn how about `Getting Started <getting_started>`__ with ``PACKAGE NAME``.

        .. note::
            If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!

    .. tab-item:: Stable (without conda)

        We don't recommend installing ``PACKAGE NAME`` without a conda environment but if you prefer to do it this
        way then all you need to do is run::

            pip install PACKAGE NAME

        and you should be all set! Now it's time to learn how about `Getting Started <getting_started>`__ with ``PACKAGE NAME``.

        .. note::
            If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!
    .. tab-item:: Development (from GitHub)
        
        .. warning::

            We don't guarantee that there won't be mistakes or bugs in the development version, use at your own risk!

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/TomWagg/cosmic-gala>`_. To start, clone the repository onto your machine: ::
        
            git clone https://github.com/TomWagg/cosmic-gala
            cd LEGWORK

        Next, we recommend that you create a Conda environment for working with LEGWORK.
        You can do this by running::

            conda env create -f environment.yml

        And then activate the environment by running::

            conda activate cosmic-gala

        At this point, all that's left to do is install ``PACKAGE NAME``!::

            pip install .

        and you should be all set! Now it's time to learn how about `Getting Started <getting_started>`__ with ``PACKAGE NAME``.

        .. note::
            If you also want to work with Jupyter notebooks then you'll also need to install jupyter/ipython to this environment!