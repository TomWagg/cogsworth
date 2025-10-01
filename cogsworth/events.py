import unxt as u
import astropy.units as u_astropy
import numpy as np

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
    events : `dict`
        A dictionary with keys ``time`` and ``delta_v`` that contain lists of event times and velocity kicks
        respectively. The shape of this array is (N, 2, 2) for times and (N, 2, 2, 3) for delta_v, where
        N is the number of binaries in the population, the second dimension is for primary and secondary,
        and the third dimension is for multiple events (if they occur).
    """
    # remove anything that doesn't get a kick
    full_kick_info = p.kick_info[p.kick_info["star"] > 0.0]

    # mask for the rows that contain supernova events
    full_bpp = p.bpp[p.bpp["evol_type"].isin([15, 16])]

    # reduce to just the supernova rows and ensure we have the same length in each table
    assert len(full_kick_info) == len(full_bpp)

    # randomly drawn phase and inclination angles as necessary
    for col in ["phase_sn_1", "phase_sn_2"]:
        if col not in p.initC:
            p.initC[col] = np.random.uniform(0, 2 * np.pi, len(p.initC))
    for col in ["inc_sn_1", "inc_sn_2"]:
        if col not in p.initC:
            p.initC[col] = np.arccos(2 * np.random.rand(len(p.initC)) - 1.0)

    t0 = (p.max_ev_time - p.initial_galaxy.tau).to(u_astropy.Myr).value
    events = {
        "time": np.ones((len(p.bin_nums), 2, 2)) * t0[:, np.newaxis, np.newaxis],
        "delta_v": np.zeros((len(p.bin_nums), 2, 2, 3)),
    }
    inc = np.zeros((len(p.bin_nums), 2, 2))
    phase = np.zeros((len(p.bin_nums), 2, 2))

    for i, bin_num in enumerate(p.bin_nums):
        if bin_num not in full_bpp["bin_num"]:
            continue

        bpp = full_bpp.loc[[bin_num]]
        kick_info = full_kick_info.loc[[bin_num]]
        initC = p.initC.loc[bin_num]

        for j in range(len(kick_info)):
            bound = kick_info.iloc[j]["disrupted"] == 0.0
            col_suffix = "_1" if bound else "_2"
            # time is the same for both stars in the event
            events["time"][i, :, j:] = t0[i] + bpp.iloc[j]["tphys"]
            
            # primary star always gets the first kick
            events["delta_v"][i, 0, j, :] = [kick_info.iloc[j]["delta_vsysx_1"],
                                             kick_info.iloc[j]["delta_vsysy_1"],
                                             kick_info.iloc[j]["delta_vsysz_1"]]
            
            # secondary star still gets primary kick if bound, otherwise gets its own kick
            events["delta_v"][i, 1, j, :] = [kick_info.iloc[j][f'delta_vsysx{col_suffix}'],
                                             kick_info.iloc[j][f'delta_vsysy{col_suffix}'],
                                             kick_info.iloc[j][f'delta_vsysz{col_suffix}']]
            
            # inclination and phase only apply to bound systems
            inc_mask = f"inc_sn_{j + 1:0d}" in initC and bound
            inc[i, :, j] = initC[f"inc_sn_{j + 1:0d}"] if inc_mask else 0.0
            phase_mask = f"phase_sn_{j + 1:0d}" in initC and bound
            phase[i, :, j] = initC[f"phase_sn_{j + 1:0d}"] if phase_mask else 0.0
            
    # apply inclination and phase rotations to the kicks
    v_X = events["delta_v"][:, :, :, 0] * np.cos(phase)\
        - events["delta_v"][:, :, :, 1] * np.sin(phase) * np.cos(inc)\
        + events["delta_v"][:, :, :, 2] * np.sin(phase) * np.sin(inc)
    v_Y = events["delta_v"][:, :, :, 0] * np.sin(phase)\
        + events["delta_v"][:, :, :, 1] * np.cos(phase) * np.cos(inc)\
        - events["delta_v"][:, :, :, 2] * np.cos(phase) * np.sin(inc)
    v_Z = events["delta_v"][:, :, :, 1] * np.sin(inc)\
        + events["delta_v"][:, :, :, 2] * np.cos(inc)
    
    events["delta_v"] = np.stack((v_X, v_Y, v_Z), axis=-1)

    events["time"] = u.Quantity(events["time"], "Myr")
    events["delta_v"] = u.Quantity(events["delta_v"], "km/s")
            
    return events
