.. _utils:

****************************
Helper functions (``utils``)
****************************

The ``utils`` module contains a number of helper functions that are used throughout ``cogsworth``. One of the
most useful functionalities is translating COSMIC tables and plotting them as a cartoon.

.. automodapi:: cogsworth.utils
    :no-heading:


.. grid:: 1 1 2 2

    .. grid-item::

        .. list-table:: Translation of stellar types
            :header-rows: 1

            * - Stellar type ID
              - Short name
              - Long name
            * - 0
              - MS < 0.7
              - Main Sequence (Low mass)
            * - 1
              - MS
              - Main Sequence
            * - 2
              - HG
              - Hertzsprung Gap
            * - 3
              - FGB
              - First Giant Branch
            * - 4
              - CHeB
              - Core Helium Burning
            * - 5
              - EAGB
              - Early Asymptotic Giant Branch
            * - 6
              - TPAGB
              - Thermally Pulsing Asymptotic Giant Branch
            * - 7
              - HeMS
              - Helium Main Sequence
            * - 8
              - HeHG
              - Helium Hertsprung Gap
            * - 9
              - HeGB
              - Helium Giant Branch
            * - 10
              - HeWD
              - Helium White Dwarf
            * - 11
              - COWD
              - Carbon/Oxygen White Dwarf
            * - 12
              - ONeWD
              - Oxygen/Neon White Dwarf
            * - 13
              - NS
              - Neutron Star
            * - 14
              - BH
              - Black Hole
            * - 15
              - MR
              - Massless Remnant
            * - 16
              - CHE
              - Chemically Homogeneous



    .. grid-item::

        .. list-table:: Translation of evolutionary stages
            :header-rows: 1

            * - Evolutionary stage ID
              - Short name
              - Long name
            * - 1
              - Init
              - Initial state
            * - 2
              - Kstar change
              - Stellar type changed
            * - 3
              - RLOF start
              - Roche lobe overflow started
            * - 4
              - RLOF end
              - Roche lobe overflow ended
            * - 5
              - Contact
              - Binary entered contact phase
            * - 6
              - Coalescence
              - Binary coalesced
            * - 7
              - CE start
              - Common-envelope started
            * - 8
              - CE end
              - Common-envelope ended
            * - 9
              - No remnant
              - No remnant
            * - 10
              - Max evol time
              - Maximum evolution time reached
            * - 11
              - Disruption
              - Binary disrupted
            * - 12
              - Begin symbiotic phase
              - Begin symbiotic phase
            * - 13
              - End symbiotic phase
              - End symbiotic phase
            * - 14
              - Blue straggler
              - Blue straggler
            * - 15
              - SN1
              - Supernova of primary
            * - 16
              - SN2
              - Supernova of secondary
