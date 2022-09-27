import pandas as pd
import numpy as np
import astropy.units as u
import gala.dynamics as gd


def determine_final_classes(population=None, bpp=None, bcm=None, kick_info=None, orbits=None, potential=None):
    """Determine the classes of each member of a population at the last point in the evolution (usually
    present day).

    Either supply a Population class or each individual table separately

    Parameters
    ----------
    population : `pop.Population`, optional
        A full population class created from the pop module, by default None
    bpp : `pandas.DataFrame`
        Evolutionary history of each binary
    bcm : `pandas.DataFrame`
        Final state of each binary
    initC : `pandas.DataFrame`
        Initial conditions for each binary
    kick_info : `pandas.DataFrame`
        Information about the kicks that occur for each binary
    orbits : `list of gala.dynamics.Orbit`
        The orbits of each binary within the galaxy. Disrupted binaries should have two entries
        (for both stars).
    galactic_potential : `gala.potential.PotentialBase`, optional
        Galactic potential to use for evolving the orbits of binaries

    Returns
    -------
    classes : `list`
        A list of classes pertaining to each binary. Each entry will either be a list of strings that apply
        (see `list_classes` for options) or None if there are no matching classes.

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
    bin_nums = bpp["bin_num"].unique()
    final_bpp = bpp[~bpp.index.duplicated(keep="last")]
    final_bcm = bcm[~bcm.index.duplicated(keep="last")]

    # set up an empty class list
    classes = [None for _ in range(len(bin_nums))]

    # go through each binary in the population
    for i, bin_num in enumerate(bin_nums):
        # set up class list for this binary and get the bpp row for just this binary
        binary_classes = []
        this_bpp = final_bpp.loc[bin_num]

        # check the binary state
        bound = this_bpp["sep"] > 0.0
        merger = this_bpp["sep"] == 0.0
        disrupted = this_bpp["sep"] == -1.0

        # check the state of the primary/secondary stars
        primary_is_star = this_bpp["kstar_1"] <= 9
        primary_is_wd = this_bpp["kstar_1"] in [10, 11, 12]
        primary_is_bh_ns = this_bpp["kstar_1"] in [13, 14]
        secondary_is_star = this_bpp["kstar_2"] <= 9
        secondary_is_wd = this_bpp["kstar_2"] in [10, 11, 12]
        secondary_is_bh_ns = this_bpp["kstar_2"] in [13, 14]

        if bound:
            if primary_is_bh_ns and secondary_is_bh_ns:
                binary_classes.append("dco")
            if primary_is_bh_ns and secondary_is_wd:
                binary_classes.append("co-1")
            if secondary_is_bh_ns and primary_is_wd:
                binary_classes.append("co-2")
            if (primary_is_bh_ns and secondary_is_star) or (secondary_is_bh_ns and primary_is_star):
                # TODO: improve this, maybe with this paper:
                # https://ui.adsabs.harvard.edu/abs/2022arXiv220905505M/abstract
                binary_classes.append("xrb")

        if disrupted:
            if primary_is_bh_ns:
                binary_classes.append("co-1")
            if secondary_is_bh_ns:
                binary_classes.append("co-2")

            # TODO: sdIa

            disruption_kick = kick_info[kick_info["disrupted"] == 1.0].loc[bin_num]
            if isinstance(disruption_kick, pd.DataFrame):
                disruption_kick = disruption_kick.iloc[-1]
            if primary_is_star and disruption_kick["vsys_1_total"] < 30.0:
                binary_classes.append("walkaway-t-1")
            elif primary_is_star:
                binary_classes.append("runaway-t-1")

            if secondary_is_star and disruption_kick["vsys_2_total"] < 30.0:
                binary_classes.append("walkaway-t-2")
            elif secondary_is_star:
                binary_classes.append("runaway-t-2")

            for ind in [1, 2]:
                with u.set_enabled_equivalencies(u.dimensionless_angles()):
                    v_R = orbits[i][ind - 1][-1].represent_as("cylindrical").vel.d_rho
                    v_T = orbits[i][ind - 1][-1].represent_as("cylindrical").vel.d_phi.to(1 / u.Myr)\
                        * orbits[i][ind - 1][-1].represent_as("cylindrical").rho
                    v_z = orbits[i][ind - 1][-1].represent_as("cylindrical").vel.d_z

                    rel_v_R = v_R - potential.circular_velocity(q=orbits[i][ind - 1][-1].pos.xyz)[0]

                    rel_speed = ((rel_v_R**2 + v_T**2 + v_z**2)**(0.5)).to(u.km / u.s).value
                    star_flag = primary_is_star if ind == 1 else secondary_is_star
                    if rel_speed < 30.0 and star_flag:
                        binary_classes.append(f"walkaway-o-{ind}")
                    elif star_flag:
                        binary_classes.append(f"runaway-o-{ind}")

        # widow stars
        if bound or disrupted:
            if primary_is_star and bpp.loc[bin_num]["kstar_2"].isin([13, 14]).any():
                binary_classes.append("widow-1")
            if secondary_is_star and bpp.loc[bin_num]["kstar_1"].isin([13, 14]).any():
                binary_classes.append("widow-2")

        # check for any compact objects coming from mergers
        if merger:
            if primary_is_bh_ns:
                binary_classes.append("stellar-merger-co-1")
            if secondary_is_bh_ns:
                binary_classes.append("stellar-merger-co-2")

        # look for pisn
        if final_bcm["SN_1"].loc[bin_num] in [6, 7]:
            binary_classes.append("pisn-1")
        if final_bcm["SN_2"].loc[bin_num] in [6, 7]:
            binary_classes.append("pisn-2")

        # either add the classes or just leave it as None if there weren't any
        classes[i] = binary_classes if binary_classes != [] else None

    return classes


def determine_final_classes_pandas(population=None, bpp=None, bcm=None, kick_info=None, orbits=None, potential=None):
    """Determine the classes of each member of a population at the last point in the evolution (usually
    present day).

    Either supply a Population class or each individual table separately

    Parameters
    ----------
    population : `pop.Population`, optional
        A full population class created from the pop module, by default None
    bpp : `pandas.DataFrame`
        Evolutionary history of each binary
    bcm : `pandas.DataFrame`
        Final state of each binary
    initC : `pandas.DataFrame`
        Initial conditions for each binary
    kick_info : `pandas.DataFrame`
        Information about the kicks that occur for each binary
    orbits : `list of gala.dynamics.Orbit`
        The orbits of each binary within the galaxy. Disrupted binaries should have two entries
        (for both stars).
    galactic_potential : `gala.potential.PotentialBase`, optional
        Galactic potential to use for evolving the orbits of binaries

    Returns
    -------
    classes : `list`
        A list of classes pertaining to each binary. Each entry will either be a list of strings that apply
        (see `list_classes` for options) or None if there are no matching classes.

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
    disrupted = final_bpp["sep"] == -1.0

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

    classes["walkaway-t-1"] = disrupted & (final_kick_info["vsys_1_total"] < 30.0)
    classes["walkaway-t-2"] = disrupted & (final_kick_info["vsys_2_total"] < 30.0)
    classes["runaway-t-1"] = disrupted & (final_kick_info["vsys_1_total"] >= 30.0)
    classes["runaway-t-2"] = disrupted & (final_kick_info["vsys_2_total"] >= 30.0)

    # observed run/walkways are more complicated
    # first get the final positions and velocities of each component of a disrupted binary
    posf_1 = [None for _ in range(len(orbits[disrupted]))]
    posf_2 = [None for _ in range(len(orbits[disrupted]))]
    velf_1 = [None for _ in range(len(orbits[disrupted]))]
    velf_2 = [None for _ in range(len(orbits[disrupted]))]
    for i, orbit in enumerate(orbits[disrupted]):
        posf_1[i] = orbit[0][-1].pos.xyz.to(u.kpc).value
        posf_2[i] = orbit[1][-1].pos.xyz.to(u.kpc).value

        velf_1[i] = orbit[0][-1].vel.d_xyz.to(u.km / u.s)
        velf_2[i] = orbit[1][-1].vel.d_xyz.to(u.km / u.s)
    posf_1 *= u.kpc
    posf_2 *= u.kpc
    velf_1 *= u.km / u.s
    velf_2 *= u.km / u.s

    # create gala phase space positions based on them
    wf_1 = gd.PhaseSpacePosition(pos=posf_1.T, vel=velf_1.T)
    wf_2 = gd.PhaseSpacePosition(pos=posf_2.T, vel=velf_2.T)

    # calculate the circular velocities at those locations
    v_circ_1 = potential.circular_velocity(q=wf_1.pos.xyz)
    v_circ_2 = potential.circular_velocity(q=wf_2.pos.xyz)

    # get the cylindrical velocities and calculate the relative speeds compared to the circular velocity
    v_R_1 = wf_1.represent_as("cylindrical").vel.d_rho
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        v_T_1 = wf_1.represent_as("cylindrical").vel.d_phi.to(1/u.Myr) * wf_1.represent_as("cylindrical").rho
    v_z_1 = wf_1.represent_as("cylindrical").vel.d_z
    rel_speed_1 = (((v_R_1 - v_circ_1)**2 + v_T_1**2 + v_z_1**2)**(0.5)).to(u.km / u.s).value

    # same for secondary
    v_R_2 = wf_2.represent_as("cylindrical").vel.d_rho
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        v_T_2 = wf_2.represent_as("cylindrical").vel.d_phi.to(1/u.Myr) * wf_2.represent_as("cylindrical").rho
    v_z_2 = wf_2.represent_as("cylindrical").vel.d_z
    rel_speed_2 = (((v_R_2 - v_circ_2)**2 + v_T_2**2 + v_z_2**2)**(0.5)).to(u.km / u.s).value

    # set the classes based on the relative speeds (non-disrupted as all left as False by default)
    classes["walkaway-o-1"][disrupted] = rel_speed_1 < 30.0
    classes["walkaway-o-2"][disrupted] = rel_speed_2 < 30.0
    classes["runaway-o-1"][disrupted] = rel_speed_1 >= 30.0
    classes["runaway-o-2"][disrupted] = rel_speed_2 >= 30.0

    return classes


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
