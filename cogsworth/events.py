import astropy.units as u

__all__ = ["identify_events"]


def identify_events_slow(p):
    """Identify any events that occur in the stellar evolution that would affect the galactic evolution

    .. note::
        This function currently only considers supernovae when identifying events

    Parameters
    ----------
    p : :class:`~cogsworth.pop.Population`
        A ``cogsworth`` population

    Returns
    -------
    primary_events_list : `list`
        A list of events for each bound binary or primary star. `None` is returned if there are no pertinent
        events, otherwise a list of dicts
    secondary_events_list : `list`
        As ``primary_events_lists`` but for disrupted secondaries
    """
    # remove anything that doesn't get a kick
    full_kick_info = p.kick_info[p.kick_info["star"] > 0.0]

    # mask for the rows that contain supernova events
    full_bpp = p.bpp[p.bpp["evol_type"].isin([15, 16])]

    # reduce to just the supernova rows and ensure we have the same length in each table
    assert len(full_kick_info) == len(full_bpp)

    primary_events_list = [None for _ in range(len(p.bin_nums))]
    secondary_events_list = [None for _ in range(len(p.bin_nums))]
    for i, bin_num in enumerate(p.bin_nums):
        if bin_num not in full_bpp["bin_num"]:
            continue

        bpp = full_bpp.loc[[bin_num]]
        kick_info = full_kick_info.loc[[bin_num]]
        initC = p.initC.loc[bin_num]

        # for primaries and bound binaries we just need a simple list
        primary_events_list[i] = [{
            "time": bpp.iloc[j]["tphys"] * u.Myr,
            "delta_v_sys_xyz": [kick_info.iloc[j]["delta_vsysx_1"],
                                kick_info.iloc[j]["delta_vsysy_1"],
                                kick_info.iloc[j]["delta_vsysz_1"]] * u.km / u.s,
            "inc": initC[f"inc_sn_{j + 1:0d}"] if f"inc_sn_{j + 1:0d}" in initC else None,
            "phase": initC[f"phase_sn_{j + 1:0d}"] if f"phase_sn_{j + 1:0d}" in initC else None,
        } for j in range(len(kick_info))]

        # iterate over the kick file and store the relevant information (for both if disruption will occur)
        if p.disrupted[i]:
            secondary_events_list[i] = []
            for j in range(len(kick_info)):
                # for rows including the disruption or after it then the secondary component is what we want
                col_suffix = "_1" if kick_info.iloc[j]["disrupted"] == 0.0 else "_2"
                secondary_events_list[i].append({
                    "time": bpp.iloc[j]["tphys"] * u.Myr,
                    "delta_v_sys_xyz": [kick_info.iloc[j][f'delta_vsysx{col_suffix}'],
                                        kick_info.iloc[j][f'delta_vsysy{col_suffix}'],
                                        kick_info.iloc[j][f'delta_vsysz{col_suffix}']] * u.km / u.s,
                    "inc": initC[f"inc_sn_{j + 1:0d}"] if f"inc_sn_{j + 1:0d}" in initC else None,
                    "phase": initC[f"phase_sn_{j + 1:0d}"] if f"phase_sn_{j + 1:0d}" in initC else None,
                })
    return primary_events_list, secondary_events_list

def identify_events(p):

    # reduce kick info to just the rows that contain supernova events
    sn_kicks = p.kick_info[p.kick_info["star"] > 0.0][
        ["star", "disrupted", "delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1",
        "delta_vsysx_2", "delta_vsysy_2", "delta_vsysz_2", "bin_num"]
    ]

    # reduce bpp to just SN rows
    sn_bpp = p.bpp[p.bpp["evol_type"].isin([15, 16])][["tphys", "evol_type", "bin_num", "sep"]]

    # add a star column for joining to kick_info, evol_type = 15 -> star 1 and evol_type = 16 -> star 2
    sn_bpp["star"] = sn_bpp["evol_type"].map({15: 1, 16: 2})

    # same for initC
    sn_initC = p.initC.loc[sn_bpp.index][["inc_sn_1", "phase_sn_1", "inc_sn_2", "phase_sn_2", "bin_num"]]
    sn_initC["star"] = sn_bpp["star"].values

    # put inc_sn and phase_sn in their own columns based on star
    sn_initC["inc_sn"] = None
    sn_initC["phase_sn"] = None
    sn_initC.loc[sn_initC["star"] == 1, "inc_sn"] = sn_initC.loc[sn_initC["star"] == 1, "inc_sn_1"]
    sn_initC.loc[sn_initC["star"] == 1, "phase_sn"] = sn_initC.loc[sn_initC["star"] == 1, "phase_sn_1"]
    sn_initC.loc[sn_initC["star"] == 2, "inc_sn"] = sn_initC.loc[sn_initC["star"] == 2, "inc_sn_2"]
    sn_initC.loc[sn_initC["star"] == 2, "phase_sn"] = sn_initC.loc[sn_initC["star"] == 2, "phase_sn_2"]

    # drop the originals
    sn_initC = sn_initC.drop(columns=["inc_sn_1", "phase_sn_1", "inc_sn_2", "phase_sn_2"])

    # join the tables together to get all the relevant info in one place
    sn_info = sn_bpp.merge(sn_kicks, on=["bin_num", "star"]).merge(sn_initC, on=["bin_num", "star"])
    sn_info.index = sn_info["bin_num"].values

    # primary stars use delta_vsysx_1 for every supernova
    primary_cols = ["tphys", "delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1", "inc_sn", "phase_sn"]
    primary_event_rows = sn_info[primary_cols].copy().rename(
        columns={
            "delta_vsysx_1": "delta_vsysx", "delta_vsysy_1": "delta_vsysy", "delta_vsysz_1": "delta_vsysz"
        }
    )
    
    # secondaries mostly use delta_vsysx_2
    secondary_cols = ["tphys", "delta_vsysx_2", "delta_vsysy_2", "delta_vsysz_2", "inc_sn", "phase_sn"]
    secondary_event_rows = sn_info[secondary_cols].copy().rename(
        columns={
            "delta_vsysx_2": "delta_vsysx", "delta_vsysy_2": "delta_vsysy", "delta_vsysz_2": "delta_vsysz"
        }
    )

    # but for binaries bound after the supernova, secondaries need to use delta_vsysx_1 instead
    bound_delta_vsys = sn_info.loc[sn_info["disrupted"] == 0,
                                   ["delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1"]].values
    secondary_event_rows.loc[sn_info["disrupted"] == 0,
                             ["delta_vsysx", "delta_vsysy", "delta_vsysz"]] = bound_delta_vsys
    
    # finally reduce to just the disrupted binaries for the secondary events
    secondary_event_rows = secondary_event_rows.loc[p.bin_nums[p.disrupted]]
    
    return primary_event_rows, secondary_event_rows