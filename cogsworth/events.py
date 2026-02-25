import numpy as np

__all__ = ["identify_events"]


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

    # randomly drawn phase and inclination angles as necessary
    for col in ["phase_sn_1", "phase_sn_2"]:
        if col not in p.initC:
            p.initC[col] = np.random.uniform(0, 2 * np.pi, len(p.initC))
    for col in ["inc_sn_1", "inc_sn_2"]:
        if col not in p.initC:
            p.initC[col] = np.arccos(2 * np.random.rand(len(p.initC)) - 1.0)

    # same for initC
    sn_initC = p.initC.loc[sn_bpp.index][["inc_sn_1", "phase_sn_1", "inc_sn_2", "phase_sn_2", "bin_num"]]
    sn_initC["star"] = sn_bpp["star"].values

    # put inc_sn and phase_sn in their own columns based on star
    sn_initC["inc"] = None
    sn_initC["phase"] = None
    sn_initC.loc[sn_initC["star"] == 1, "inc"] = sn_initC.loc[sn_initC["star"] == 1, "inc_sn_1"]
    sn_initC.loc[sn_initC["star"] == 1, "phase"] = sn_initC.loc[sn_initC["star"] == 1, "phase_sn_1"]
    sn_initC.loc[sn_initC["star"] == 2, "inc"] = sn_initC.loc[sn_initC["star"] == 2, "inc_sn_2"]
    sn_initC.loc[sn_initC["star"] == 2, "phase"] = sn_initC.loc[sn_initC["star"] == 2, "phase_sn_2"]

    # drop the originals
    sn_initC = sn_initC.drop(columns=["inc_sn_1", "phase_sn_1", "inc_sn_2", "phase_sn_2"])

    # join the tables together to get all the relevant info in one place
    sn_info = sn_bpp.merge(sn_kicks, on=["bin_num", "star"]).merge(sn_initC, on=["bin_num", "star"])
    sn_info.index = sn_info["bin_num"].values

    # primary stars use delta_vsysx_1 for every supernova
    primary_cols = ["tphys", "delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1", "inc", "phase"]
    primary_event_rows = sn_info[primary_cols].copy().rename(
        columns={
            "delta_vsysx_1": "delta_vsys_x", "delta_vsysy_1": "delta_vsys_y", "delta_vsysz_1": "delta_vsys_z"
        }
    )
    
    # secondaries mostly use delta_vsysx_2
    secondary_cols = ["tphys", "delta_vsysx_2", "delta_vsysy_2", "delta_vsysz_2", "inc", "phase"]
    secondary_event_rows = sn_info[secondary_cols].copy().rename(
        columns={
            "delta_vsysx_2": "delta_vsys_x", "delta_vsysy_2": "delta_vsys_y", "delta_vsysz_2": "delta_vsys_z"
        }
    )

    # but for binaries bound after the supernova, secondaries need to use delta_vsysx_1 instead
    bound_delta_vsys = sn_info.loc[sn_info["disrupted"] == 0,
                                   ["delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1"]].values
    secondary_event_rows.loc[sn_info["disrupted"] == 0,
                             ["delta_vsys_x", "delta_vsys_y", "delta_vsys_z"]] = bound_delta_vsys
    
    # finally reduce to just the disrupted binaries for the secondary events
    secondary_event_rows = secondary_event_rows.loc[p.bin_nums[p.disrupted]]
    
    return primary_event_rows, secondary_event_rows