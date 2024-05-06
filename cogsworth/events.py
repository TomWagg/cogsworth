import astropy.units as u

__all__ = ["identify_events"]


def identify_events(p):
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
