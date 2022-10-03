Installation
============

.. tab-set::

    .. tab-item:: Stable (with conda)

        This is our recommend installation method! Follow the steps below to start using ``LEGWORK``!

        #. :download:`Download the environment.yml file from our repository <https://raw.githubusercontent.com/TeamLEGWORK/LEGWORK/main/environment.yml>`
        #. Create a new conda environment using this file::

                conda env create -f path/to/environment.yml

        #. Activate the environment by running::

                conda activate legwork

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.
        Note that if you also want to work with the notebooks in the tutorials and/or demos you'll also need to install jupyter/ipython in this environment!

    .. tab-item:: Stable (without conda)

        We don't recommend installing ``LEGWORK`` without a conda environment but if you prefer to do it this
        way then all you need to do is run::

            pip install legwork

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.
        Note that if you also want to work with the notebooks in the tutorials and/or demos you'll also need to install jupyter/ipython in this environment!

    .. tab-item:: Development (from GitHub)
        
        .. warning::

            We don't guarantee that there won't be mistakes or bugs in the development version, use at your own risk!

        The latest development version is available directly from our `GitHub Repo
        <https://github.com/TeamLEGWORK/LEGWORK>`_. To start, clone the repository onto your machine: ::
        
            git clone https://github.com/TeamLEGWORK/LEGWORK
            cd LEGWORK

        Next, we recommend that you create a Conda environment for working with LEGWORK.
        You can do this by running::

            conda create --name legwork "python>=3.7" pip "numba>=0.50" "numpy>=1.16" "astropy>=4.0" "scipy>=1.5.0" "matplotlib>=3.3.2" "seaborn>=0.11.1" "schwimmbad>=0.3.2" -c conda-forge -c defaults

        And then activate the environment by running::

            conda activate legwork

        At this point, all that's left to do is install LEGWORK!::

            pip install .

        and you should be all set! Check out our `quickstart tutorial <notebooks/Quickstart.ipynb>`__ to learn some LEGWORK basics.
        Note that if you also want to work with the notebooks in the tutorials and/or demos you'll also need to install jupyter/ipython in this environment!