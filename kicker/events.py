import astropy.units as u

__all__ = ["identify_events"]


def identify_events(full_bpp, full_kick_info):
    """Identify any events that occur in the stellar evolution that would affect the galactic evolution

    .. note::
        This function currently only considers supernovae when identifying events

    Parameters
    ----------
    full_bpp : :class:`pandas.DataFrame`
        Table of evolution phase events from COSMIC
    full_kick_info : :class:`pandas.DataFrame`
        Table of information about kicks from COSMIC

    Returns
    -------
    events : `list`
        A list of events for each binary. `None` is returned if there are no pertinent events, a list of
        dicts for a bound system and two lists for a disrupted system (to track each component)
    """
    bin_nums = full_bpp["bin_num"].unique()
    # remove anything that doesn't get a kick
    full_kick_info = full_kick_info[full_kick_info["star"] > 0.0]

    # mask for the rows that contain supernova events
    full_bpp = full_bpp[full_bpp["evol_type"].isin([15, 16])]

    # reduce to just the supernova rows and ensure we have the same length in each table
    assert len(full_kick_info) == len(full_bpp)

    events_list = [None for _ in range(len(bin_nums))]
    for i, bin_num in enumerate(bin_nums):
        if bin_num not in full_bpp.index:
            continue

        bpp = full_bpp.loc[[bin_num]]
        kick_info = full_kick_info.loc[[bin_num]]

        # check if the the binary is going to disrupt at any point
        it_will_disrupt = (kick_info["disrupted"] == 1.0).any()

        # iterate over the kick file and store the relevant information (for both if disruption will occur)
        if it_will_disrupt:
            events_1, events_2 = [], []
            for j in range(len(kick_info)):
                default_event = {
                    "time": bpp.iloc[j]["tphys"] * u.Myr,
                    "m_1": bpp.iloc[j]["mass_1"] * u.Msun,
                    "m_2": bpp.iloc[j]["mass_2"] * u.Msun,
                    "a": bpp.iloc[j]["sep"] * u.Rsun,
                    "ecc": bpp.iloc[j]["ecc"],
                    "delta_v_sys_xyz": [kick_info.iloc[j]["delta_vsysx_1"],
                                        kick_info.iloc[j]["delta_vsysy_1"],
                                        kick_info.iloc[j]["delta_vsysz_1"]] * u.km / u.s
                }
                events_1.append(default_event)

                # for rows including the disruption or after it then the secondary component needs editing
                if kick_info.iloc[j]["disrupted"] == 1.0:
                    events_2.append({
                        "time": bpp.iloc[j]["tphys"] * u.Myr,
                        "m_1": bpp.iloc[j]["mass_1"] * u.Msun,
                        "m_2": bpp.iloc[j]["mass_2"] * u.Msun,
                        "a": bpp.iloc[j]["sep"] * u.Rsun,
                        "ecc": bpp.iloc[j]["ecc"],
                        "delta_v_sys_xyz": [kick_info.iloc[j]["delta_vsysx_2"],
                                            kick_info.iloc[j]["delta_vsysy_2"],
                                            kick_info.iloc[j]["delta_vsysz_2"]] * u.km / u.s
                    })
                else:
                    # otherwise just append the regular stuff
                    events_2.append(default_event)
            events_list[i] = [events_1, events_2]
        else:
            # for bound binaries we just need a single list
            events_list[i] = [{
                "time": bpp.iloc[j]["tphys"] * u.Myr,
                "m_1": bpp.iloc[j]["mass_1"] * u.Msun,
                "m_2": bpp.iloc[j]["mass_2"] * u.Msun,
                "a": bpp.iloc[j]["sep"] * u.Rsun,
                "ecc": bpp.iloc[j]["ecc"],
                "delta_v_sys_xyz": [kick_info.iloc[j]["delta_vsysx_1"],
                                    kick_info.iloc[j]["delta_vsysy_1"],
                                    kick_info.iloc[j]["delta_vsysz_1"]] * u.km / u.s
            } for j in range(len(kick_info))]
    return events_list
