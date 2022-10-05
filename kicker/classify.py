import pandas as pd
import numpy as np
import astropy.units as u
import gala.dynamics as gd

__all__ = ["determine_final_classes", "list_classes"]


def determine_final_classes(population=None, bpp=None, bcm=None, kick_info=None, orbits=None, potential=None):
    """Determine the classes of each member of a population at the last point in the evolution (usually
    present day).

    Either supply a Population class or each individual table separately

    Parameters
    ----------
    population : :class:`~kicker.pop.Population`, optional
        A full population class created from the pop module, by default None
    bpp : :class:`~pandas.DataFrame`
        Evolutionary history of each binary
    bcm : :class:`~pandas.DataFrame`
        Final state of each binary
    initC : :class:`~pandas.DataFrame`
        Initial conditions for each binary
    kick_info : :class:`~pandas.DataFrame`
        Information about the kicks that occur for each binary
    orbits : `list` of :class:`~gala.dynamics.Orbit`
        The orbits of each binary within the galaxy. Disrupted binaries should have two entries
        (for both stars).
    galactic_potential : :class:`~gala.potential.potential.PotentialBase`, optional
        Galactic potential to use for evolving the orbits of binaries

    Returns
    -------
    classes : :class:`~pandas.DataFrame`
        A DataFrame with a boolean column for each class (see :meth:`list_classes`) and a row for each binary

    Raises
    ------
    ValueError
        If either `population` is None OR any another parameter is None
    """
    # ensure that there's enough input
    if population is None and (bpp is None or bcm is None or kick_info is None
                               or orbits is None or potential is None):
        raise ValueError("Either `population` must be supplied or all other parameters")

    # split up the input so that I can use a single interface
    if population is not None:
        bpp, bcm, kick_info, orbits, potential = population.bpp, population.bcm, population.kick_info,\
            population.orbits, population.galactic_potential

    # get the binary indices and also reduce the tables to just the final row in each
    final_bpp = bpp[~bpp.index.duplicated(keep="last")]
    final_bcm = bcm[~bcm.index.duplicated(keep="last")]
    final_kick_info = kick_info.sort_values(["bin_num", "star"]).drop_duplicates(subset="bin_num",
                                                                                 keep="last")

    # set up an empty dataframe
    columns = ["dco", "co-1", "co-2", "xrb", "walkaway-t-1", "walkaway-t-2", "runaway-t-1", "runaway-t-2",
               "walkaway-o-1", "walkaway-o-2", "runaway-o-1", "runaway-o-2", "widow-1", "widow-2",
               "stellar-merger-co-1", "stellar-merger-co-2", "pisn-1", "pisn-2"]
    data = np.zeros(shape=(len(final_bpp), len(columns))).astype(bool)
    classes = pd.DataFrame(data=data, columns=columns)

    # match the index to the final bpp bin_nums
    classes.index = final_bpp["bin_num"].index

    # check the state of the binary
    bound = final_bpp["sep"] > 0.0
    merger = final_bpp["sep"] == 0.0

    # HACK: three different markers and only assume it is disrupted if ALL are True
    disrupted = (final_bpp["bin_num"].isin(kick_info[kick_info["disrupted"] == 1.0]["bin_num"].unique())
                 & (final_bpp["sep"] < 0.0)
                 & final_bpp["bin_num"].isin(bpp[bpp["evol_type"] == 11.0]["bin_num"]))

    # set some flags for the end conditions of each component
    primary_is_star = final_bpp["kstar_1"] <= 9
    secondary_is_star = final_bpp["kstar_2"] <= 9
    primary_is_bh_ns = final_bpp["kstar_1"].isin([13, 14])
    secondary_is_bh_ns = final_bpp["kstar_2"].isin([13, 14])

    # check if there was ever a bound CO
    primary_ever_bound_bh_ns = final_bpp["bin_num"].isin(bpp[bpp["kstar_1"].isin([13, 14])
                                                             & bpp["sep"] > 0.0]["bin_num"].unique())
    secondary_ever_bound_bh_ns = final_bpp["bin_num"].isin(bpp[bpp["kstar_2"].isin([13, 14])
                                                               & bpp["sep"] > 0.0]["bin_num"].unique())

    # create masks based on conditions listed in `list_classes`
    classes["dco"] = bound & primary_is_bh_ns & secondary_is_bh_ns
    classes["co-1"] = ~merger & primary_is_bh_ns
    classes["co-2"] = ~merger & secondary_is_bh_ns
    classes["stellar-merger-co-1"] = merger & primary_is_bh_ns
    classes["stellar-merger-co-2"] = merger & secondary_is_bh_ns
    classes["xrb"] = bound & ((primary_is_bh_ns & secondary_is_star) | (secondary_is_bh_ns & primary_is_star))

    classes["widow-1"] = ~merger & primary_is_star & secondary_ever_bound_bh_ns
    classes["widow-2"] = ~merger & secondary_is_star & primary_ever_bound_bh_ns

    classes["pisn-1"] = final_bcm["SN_1"].isin([6, 7])
    classes["pisn-2"] = final_bcm["SN_2"].isin([6, 7])

    classes["walkaway-t-1"] = disrupted & (final_kick_info["vsys_1_total"] < 30.0) & primary_is_star
    classes["walkaway-t-2"] = disrupted & (final_kick_info["vsys_2_total"] < 30.0) & secondary_is_star
    classes["runaway-t-1"] = disrupted & (final_kick_info["vsys_1_total"] >= 30.0) & primary_is_star
    classes["runaway-t-2"] = disrupted & (final_kick_info["vsys_2_total"] >= 30.0) & secondary_is_star

    # calculate relative speeds for observed walk/runaways
    if disrupted.any():
        rel_speed_1 = _get_rel_speed(orbits=orbits[disrupted], potential=potential, which_star=0)
        rel_speed_2 = _get_rel_speed(orbits=orbits[disrupted], potential=potential, which_star=1)

        # set the classes based on the relative speeds (non-disrupted as all left as False by default)
        classes.loc[disrupted, "walkaway-o-1"] = (rel_speed_1 < 30.0) & primary_is_star[disrupted]
        classes.loc[disrupted, "walkaway-o-2"] = (rel_speed_2 < 30.0) & secondary_is_star[disrupted]
        classes.loc[disrupted, "runaway-o-1"] = (rel_speed_1 >= 30.0) & primary_is_star[disrupted]
        classes.loc[disrupted, "runaway-o-2"] = (rel_speed_2 >= 30.0) & secondary_is_star[disrupted]

    return classes


def _get_rel_speed(orbits, potential, which_star):
    """Calculate the relative speed of a set of stars at the end of their orbits compared to the circular velocity at their
    final positions (given a galactic potential)

    Parameters
    ----------
    orbits : `list`
        List of gala Orbits
    potential : :class:`gala.potential.potential.PotentialBase`
        The galactic potential used for finding the circular velocity
    which_star : `int`
        Index of which star to consider, either 0 or 1

    Returns
    -------
    rel_speed : `float`
        Relative speed in km / s
    """
    # get final positions and velocities
    posf = [None for _ in range(len(orbits))]
    velf = [None for _ in range(len(orbits))]
    for i, orbit in enumerate(orbits):
        posf[i] = orbit[which_star][-1].pos.xyz.to(u.kpc).value
        velf[i] = orbit[which_star][-1].vel.d_xyz.to(u.km / u.s)
    posf *= u.kpc
    velf *= u.km / u.s

    # create gala phase space positions based on them
    wf = gd.PhaseSpacePosition(pos=posf.T, vel=velf.T)

    # calculate the circular velocities at those locations
    v_circ = potential.circular_velocity(q=wf.pos.xyz)

    # get the cylindrical velocities and calculate the relative speeds compared to the circular velocity
    v_R = wf.represent_as("cylindrical").vel.d_rho
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        v_T = wf.represent_as("cylindrical").vel.d_phi.to(1/u.Myr) * wf.represent_as("cylindrical").rho
    v_z = wf.represent_as("cylindrical").vel.d_z
    rel_speed = (((v_R - v_circ)**2 + v_T**2 + v_z**2)**(0.5)).to(u.km / u.s).value

    return rel_speed


def list_classes():
    """List out the available classes that are used in other functions"""
    classes = [
        {
            "name": "runaway-t",
            "full_name": "Theory Runaway",
            "condition": ("Any star from a disrupted binary that has an instantaneous velocity > 30 km/s in "
                          "the frame of the binary")
        },
        {
            "name": "runaway-o",
            "full_name": "Observation runaway",
            "condition": ("Any star from a disrupted binary that is moving with a Galactocentric velocity "
                          "> 30km/s relative to the local circular velocity at its location")
        },
        {
            "name": "walkaway-t",
            "full_name": "Theory Runaway",
            "condition": ("Any star from a disrupted binary that has an instantaneous velocity < 30 km/s in "
                          "the frame of the binary")
        },
        {
            "name": "walkaway-o",
            "full_name": "Observation walkaway",
            "condition": ("Any star from a disrupted binary that is moving with a Galactocentric velocity "
                          "< 30km/s relative to the local circular velocity at its location")
        },
        {
            "name": "widow",
            "full_name": "Widowed Star",
            "condition": ("Any star, or binary containing a star, that is/was a companion to a compact "
                          "object")
        },
        {
            "name": "xrb",
            "full_name": "X-ray binary",
            "condition": ("Any binary with a star that is a companion to a compact object")
        },
        {
            "name": "co",
            "full_name": "Compact object",
            "condition": ("Any compact object or binary containing a compact object")
        },
        {
            "name": "stellar-merger-co",
            "full_name": "Compact object from merger",
            "condition": ("Any compact object resulting from a stellar merger")
        },
        {
            "name": "dco",
            "full_name": "Double compact object",
            "condition": ("Any bound binary of two compact objects")
        },
        {
            "name": "sdIa",
            "full_name": "Single degenerate type Ia",
            "condition": ("Any disrupted binary that contains a massless remnant that was once a white dwarf")
        },
        {
            "name": "pisn",
            "full_name": "Pair Instability Supernova",
            "condition": ("Any binary that had a star with a pair instability supernova")
        },
    ]

    print("Any class with a suffix '-1' or '-2' applies to only the primary or secondary")
    print("Available classes")
    print("-----------------")
    for c in classes:
        print(f'{c["full_name"]} ({c["name"]})')
        print(f'    {c["condition"]}\n')
