import h5py as h5
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as const
from cosmic.sample import InitialBinaryTable
import logging


__all__ = ["get_initial_binaries", "get_bpp", "get_kick_info"]


BPP_COLUMNS = ["Time", "Mass(1)", "Mass(2)", "Stellar_Type(1)", "Stellar_Type(2)",
                "SemiMajorAxis", "Eccentricity", "Radius(1)", "Radius(2)", "SEED"]
INIT_COLS = ["Mass@ZAMS(1)", "Mass@ZAMS(2)", "Eccentricity@ZAMS", "SemiMajorAxis@ZAMS",
                "Stellar_Type@ZAMS(1)", "Stellar_Type@ZAMS(2)", "Radius@ZAMS(1)", "Radius@ZAMS(2)", "SEED"]
INIT_COL_TRANSLATOR = {
    "Mass@ZAMS(1)": "Mass(1)", "Mass@ZAMS(2)": "Mass(2)", "Eccentricity@ZAMS": "Eccentricity",
    "SemiMajorAxis@ZAMS": "SemiMajorAxis", "Stellar_Type@ZAMS(1)": "Stellar_Type(1)",
    "Stellar_Type@ZAMS(2)": "Stellar_Type(2)",
    "Radius@ZAMS(1)": "Radius(1)", "Radius@ZAMS(2)": "Radius(2)",
}


def get_initial_binaries(filename, tphysf=None):
    # open the COMPAS file and read DF
    with h5.File(filename, "r") as f:
        bse_sys = pd.DataFrame({k: f["BSE_System_Parameters"][k][...]
                                for k in f["BSE_System_Parameters"].keys()})
        bse_sys.index = bse_sys["SEED"].values

    REQ_COLS = ["Mass@ZAMS(1)", "Mass@ZAMS(2)", "Eccentricity@ZAMS", "SemiMajorAxis@ZAMS",
                "Stellar_Type@ZAMS(1)", "Stellar_Type@ZAMS(2)", "Metallicity@ZAMS(1)", "SEED"]
    missing_cols = [col for col in REQ_COLS if col not in bse_sys.columns]
    if missing_cols:
        raise ValueError(
            f"Input COMPAS file is missing required columns for initial binaries: {missing_cols}"
        )
    
    if "PO_Max_Evolution_Time" not in bse_sys.columns and tphysf is None:
        logging.getLogger("cogsworth").warning(
            "cogsworth warning: PO_Max_Evolution_Time not found in COMPAS output, and no lookback times"
            "/tphysf provided. Assuming default of 13.7 Gyr for all systems."
        )
        tphysf = 13700.0
    elif "PO_Max_Evolution_Time" in bse_sys.columns and tphysf is None:
        tphysf = bse_sys["PO_Max_Evolution_Time"].values

    # convert initial separation to Rsun, use evol_type = 1
    bse_sys["SemiMajorAxis@ZAMS"] = bse_sys["SemiMajorAxis@ZAMS"].values * u.AU.to(u.Rsun)
    bse_sys["porb@ZAMS"] = _get_porb_from_a(
        bse_sys["SemiMajorAxis@ZAMS"].values,
        bse_sys["Mass@ZAMS(1)"].values,
        bse_sys["Mass@ZAMS(2)"].values
    )

    tphysf = np.full(len(bse_sys), tphysf) if np.isscalar(tphysf) else tphysf

    initial_binaries = InitialBinaryTable.InitialBinaries(
        m1=bse_sys["Mass@ZAMS(1)"].values,
        m2=bse_sys["Mass@ZAMS(2)"].values,
        porb=bse_sys["porb@ZAMS"].values,
        ecc=bse_sys["Eccentricity@ZAMS"].values,
        tphysf=tphysf,
        kstar1=bse_sys["Stellar_Type@ZAMS(1)"].values,
        kstar2=bse_sys["Stellar_Type@ZAMS(2)"].values,
        metallicity=bse_sys["Metallicity@ZAMS(1)"].values
    )
    initial_binaries.index = bse_sys["SEED"].values
    initial_binaries["bin_num"] = bse_sys["SEED"].values

    return initial_binaries


def get_bpp(filename):
    """Create a COSMIC-like bpp table from a COMPAS output file

    This function combines several different COMPAS output tables to create a bpp-like
    table that contains rows for each significant stellar evolution event in the life of
    each binary system.

    The function currently implements the following events:
    - Initial conditions and final state (from BSE_System_Parameters)
    - Changes in stellar type (from BSE_Switch_Log)
    - Start and end of mass transfer episodes (from BSE_RLOF)
    - Start and end of common-envelope events (from BSE_Common_Envelopes)
    - Supernova events (from BSE_Supernovae)

    This code is built upon work by Eugene Shang.

    Parameters
    ----------
    filename : `str`
        Path to the COMPAS output file in HDF5 format.

    Returns
    -------
    bpp : `pandas.DataFrame`
        A DataFrame containing the bpp table with rows for each stellar evolution event.
    """

    has_subfile = {"BSE_System_Parameters": False, "BSE_Switch_Log": False, "BSE_RLOF": False,
                   "BSE_Common_Envelopes": False, "BSE_Supernovae": False}

    # open the COMPAS file and read DFs
    with h5.File(filename, "r") as f:
        for subfile in has_subfile.keys():
            if subfile in f.keys():
                has_subfile[subfile] = True

        if has_subfile["BSE_System_Parameters"]:
            bse_sys = pd.DataFrame({k: f["BSE_System_Parameters"][k][...]
                                    for k in f["BSE_System_Parameters"].keys()})
            bse_sys.index = bse_sys["SEED"].values
        if has_subfile["BSE_Switch_Log"]:
            kstar_change = pd.DataFrame({k: f["BSE_Switch_Log"][k][...] for k in f["BSE_Switch_Log"].keys()})
            kstar_change.index = kstar_change["SEED"].values
        if has_subfile["BSE_RLOF"]:
            rlof = pd.DataFrame({k: f["BSE_RLOF"][k][...] for k in f["BSE_RLOF"].keys()})
            rlof.index = rlof["SEED"].values
        if has_subfile["BSE_Common_Envelopes"]:
            ce = pd.DataFrame({k: f["BSE_Common_Envelopes"][k][...]
                               for k in f["BSE_Common_Envelopes"].keys()})
            ce.index = ce["SEED"].values
        if has_subfile["BSE_Supernovae"]:
            sn = pd.DataFrame({k: f["BSE_Supernovae"][k][...] for k in f["BSE_Supernovae"].keys()})
            sn.index = sn["SEED"].values

    # ----------------------------------
    # INITIAL CONDITIONS AND FINAL STATE
    # ----------------------------------

    if has_subfile["BSE_System_Parameters"]:
        # rename columns to look like other tables
        init = bse_sys[INIT_COLS].rename(INIT_COL_TRANSLATOR, axis=1)

        # convert initial separation to Rsun, use evol_type = 1
        init["SemiMajorAxis"] = init["SemiMajorAxis"].values * u.AU.to(u.Rsun)
        init[["Time", "evol_type"]] = [0.0, 1]
        
        final = bse_sys[BPP_COLUMNS].copy()
        final["evol_type"] = 10
    
    # --------------------
    # STELLAR TYPE CHANGES
    # --------------------

    if has_subfile["BSE_Switch_Log"]:
        # remove any rows where the stellar type doesn't actually change, or mergers
        kstar_change = kstar_change[(kstar_change["Switching_From"] != kstar_change["Switching_To"])
                                    & (kstar_change["Switching_To"] != 15)]
        
        # keep bpp columns, set evol_type = 2
        kstar_change = kstar_change.loc[:, BPP_COLUMNS]
        kstar_change["evol_type"] = 2

    # --------------------
    # MASS TRANSFER EVENTS
    # --------------------

    if has_subfile["BSE_RLOF"]:
        # first we need to isolate separate episodes of MT. It seems COMPAS can sometimes create *MANY* rows
        # where the mass transfer gets split up because the rate changes slightly - we only care about
        # the start and end here
        rlof.sort_values(["SEED", "Time<MT"], inplace=True)
        rlof["prev_SEED"] = rlof["SEED"].shift(1, fill_value=-1)
        rlof["next_SEED"] = rlof["SEED"].shift(-1, fill_value=-1)
        rlof["prev_Time>MT"] = rlof["Time>MT"].shift(1, fill_value=-1)
        rlof["next_Time<MT"] = rlof["Time<MT"].shift(-1, fill_value=-1)

        # find rows where MT starts or ends
        rlof["mt_start"] = (rlof["SEED"] != rlof["prev_SEED"]) | (rlof["Time<MT"] != rlof["prev_Time>MT"])
        rlof["mt_end"] = (rlof["SEED"] != rlof["next_SEED"]) | (rlof["Time>MT"] != rlof["next_Time<MT"])

        # split the DFs into two, with a row for the start and end of RLOF
        # starting cols are marked with <MT, ending cols with >MT
        start_cols = ([c + "<MT" if c != "SEED" else c for c in BPP_COLUMNS]
                      + ["Radius(1)|RL<step", "Radius(2)|RL<step"])
        end_cols = ([c + ">MT" if c != "SEED" else c for c in BPP_COLUMNS]
                    + ["Radius(1)|RL>step", "Radius(2)|RL>step"])

        # grab the start rows, stripping off the <MT suffix, these are evol_type 3
        mt_start_rows = rlof.loc[rlof["mt_start"], start_cols].rename(
            {s: s.replace("<MT", "") for s in start_cols}
            | {"Radius(1)|RL<step": "RRLO(1)", "Radius(2)|RL<step": "RRLO(2)"}, axis=1)
        mt_start_rows["evol_type"] = 3

        # grab the end rows, stripping off the >MT suffix, these are evol_type 4
        mt_end_rows = rlof.loc[rlof["mt_end"], end_cols].rename(
            {s: s.replace(">MT", "") for s in end_cols}
            | {"Radius(1)|RL>step": "RRLO(1)", "Radius(2)|RL>step": "RRLO(2)"}, axis=1)
        mt_end_rows["evol_type"] = 4

        # combine the mass transfer start and end rows into a single dataframe
        mt_rows = pd.concat([mt_start_rows, mt_end_rows]).sort_values(["SEED", "Time", "evol_type"],
                                                                      ascending=[True, True, False])

        # get rid of those extra columns we just created
        mt_rows = mt_rows.loc[:, BPP_COLUMNS + ["evol_type", "RRLO(1)", "RRLO(2)"]]

    # ----------------------
    # COMMON-ENVELOPE EVENTS
    # ----------------------

    if has_subfile["BSE_Common_Envelopes"]:
        # as before, convert the DF into two, with a row for the start and end of CE
        start_ce_cols = [c + "<CE" if c not in ["Time", "SEED"] else c for c in BPP_COLUMNS]

        # end_cols is the same as BPP_COLUMNS but with >CE suffixes, except for Time and Stellar_Type columns
        end_ce_cols = [c + ">CE" if c not in ["Time", "Stellar_Type(1)", "Stellar_Type(2)", "SEED"] else c
                    for c in BPP_COLUMNS]
        ce_start_rows = ce[start_ce_cols].rename({s: s.replace("<CE", "") for s in start_ce_cols}, axis=1)
        ce_start_rows["evol_type"] = 7
        ce_end_rows = ce[end_ce_cols + ["Merger"]].rename({s: s.replace(">CE", "") for s in end_ce_cols}, axis=1)
        ce_end_rows["evol_type"] = 8

        # if a merger occurs, turn the other star into a massless remnant and set the separation to 0.0
        # (COMPAS doesn't do this...?)
        # for now let's just insert a row that does the coalescence that has sep = 0.0
        merger_mask = ce_end_rows["Merger"] == 1
        if merger_mask.any():
            merger_rows = ce_end_rows.loc[merger_mask].copy()
            merger_rows[["SemiMajorAxis", "Eccentricity", "evol_type"]] = [0.0, 0.0, 6]
            ce_end_rows = pd.concat([ce_end_rows, merger_rows])
        ce_end_rows.drop(columns=["Merger"], inplace=True)

        # combine all ce_rows
        ce_rows = pd.concat([ce_start_rows, ce_end_rows]).sort_values(["SEED", "Time"])

    # ----------
    # SUPERNOVAE
    # ----------

    if has_subfile["BSE_Supernovae"]:
        # no RLOF during supernovae
        sn[["RLOF(1)", "RLOF(2)"]] = [0.0, 0.0]

        # set evol_type based on Supernova_State
        sn_type = sn["Supernova_State"].copy()
        sn = sn.loc[:, BPP_COLUMNS]
        sn["evol_type"] = np.where(sn_type == 1, 15, 16)

        # add rows for disrupted systems
        disruption_rows = sn.loc[sn["SemiMajorAxis"] < 0.0].copy()
        disruption_rows["evol_type"] = 11

        # remove any duplicate disruption rows, just keep the first instance
        disruption_rows = disruption_rows.drop_duplicates(subset=["SEED"], keep="first")

        # combine back into SN DF
        sn = pd.concat([sn, disruption_rows])

    # -------------
    # CONSTRUCT BPP
    # -------------

    to_concat = []
    if has_subfile["BSE_System_Parameters"]:
        to_concat.append(init)
        to_concat.append(final)
    if has_subfile["BSE_Switch_Log"]:
        to_concat.append(kstar_change)
    if has_subfile["BSE_RLOF"]:
        to_concat.append(mt_rows)
    if has_subfile["BSE_Common_Envelopes"]:
        to_concat.append(ce_rows)
    if has_subfile["BSE_Supernovae"]:
        to_concat.append(sn)

    bpp = pd.concat(to_concat, sort=False)
    
    # sort by SEED, then by Time, then by evol_type based on priority
    evol_type_priority = [1, 3, 7, 2, 15, 16, 11, 8, 4, 6, 10]
    order = {v: i for i, v in enumerate(evol_type_priority)}
    bpp["_evol_key"] = bpp["evol_type"].map(order).fillna(len(order)).astype(int)
    bpp = (
        bpp.sort_values(
            by=["SEED", "Time", "_evol_key"],
        ).drop(columns="_evol_key")
    )

    # find all rows where evol_type == 4 and the previous row had evol_type == 8, state should be same
    bpp["row_inds"] = np.arange(len(bpp))
    bpp["prev_evol_type"] = bpp["evol_type"].shift(1)
    mask = (bpp["evol_type"] == 4) & (bpp["prev_evol_type"] == 8)
    cols = BPP_COLUMNS + ["row_inds"]
    bpp.loc[mask, cols] = bpp.loc[bpp["row_inds"].isin(bpp.loc[mask, cols]["row_inds"] - 1), cols]

    # reduce to just the columns we want
    cols = BPP_COLUMNS + ["evol_type", "RRLO(1)", "RRLO(2)"]
    bpp = bpp.loc[:, cols]

    # delete any duplicate rows that may have been created
    bpp = bpp.drop_duplicates()

    # COMPAS sets SemiMajorAxis and Eccentricity to NaN for disrupted systems, we want -1.0
    bpp.loc[bpp["SemiMajorAxis"].isna(), ["SemiMajorAxis", "Eccentricity"]] = -1.0

    # calculate porb and match SemiMajorAxis/Eccentricity conventions
    merged = (bpp["SemiMajorAxis"] == np.inf) | (bpp["SemiMajorAxis"] == 0.0)
    disrupted = bpp["SemiMajorAxis"] < 0.0
    bound = ~(disrupted | merged)
    bpp.loc[bound, "porb"] = _get_porb_from_a(bpp.loc[bound, "SemiMajorAxis"].values,
                                              bpp.loc[bound, "Mass(1)"].values,
                                              bpp.loc[bound, "Mass(2)"].values)
    bpp.loc[disrupted, ["SemiMajorAxis", "Eccentricity", "porb"]] = [-1.0, -1.0, -1.0]
    bpp.loc[merged, ["SemiMajorAxis", "Eccentricity", "porb"]] = [0.0, 0.0, 0.0]

    # rename columns to COSMIC bpp style
    bpp.rename({
        "Time": "tphys",
        "Mass(1)": "mass_1",
        "Mass(2)": "mass_2",
        "Stellar_Type(1)": "kstar_1",
        "Stellar_Type(2)": "kstar_2",
        "SemiMajorAxis": "sep",
        "Eccentricity": "ecc",
        "RRLO(1)": "RRLO_1",
        "RRLO(2)": "RRLO_2",
        "Radius(1)": "rad_1",
        "Radius(2)": "rad_2",
        "SEED": "bin_num"
    }, axis=1, inplace=True)
    
    bpp.loc[~bound, ["RRLO_1", "RRLO_2"]] = [0.0, 0.0]
    mask = bound & (bpp["RRLO_1"].isna()) & (bpp["RRLO_2"].isna())
    for l, r in [("1", "2"), ("2", "1")]:
        f_Roche = _calculate_roche_lobe_factor(bpp.loc[mask, f"mass_{r}"] / bpp.loc[mask, f"mass_{l}"])
        bpp.loc[mask, f"RRLO_{l}"] = bpp.loc[mask, f"rad_{l}"] / (f_Roche * bpp.loc[mask, "sep"])

    return bpp

def get_kick_info(filename):
    """Create a kick_info table from a COMPAS output file
    
    This function reads the BSE_Supernovae table from a COMPAS output file and reformats
    it into a kick_info table similar to that used in COSMIC.

    Parameters
    ----------
    filename : `str`
        Path to the COMPAS output file in HDF5 format.

    Returns
    -------
    kick_info : `pandas.DataFrame`
        A DataFrame containing the kick_info table.
    """
    any_kicks = False
    KICK_INFO_COLS = ['disrupted', 'natal_kick', 'phi', 'theta', 'mean_anomaly',
                      'delta_vsysx_1', 'delta_vsysy_1', 'delta_vsysz_1', 'vsys_1_total',
                      'delta_vsysx_2', 'delta_vsysy_2', 'delta_vsysz_2', 'vsys_2_total']
    # open the COMPAS file and read the Supernovae DF
    with h5.File(filename, "r") as f:
        all_seeds = f["BSE_System_Parameters"]["SEED"][...]
        if "BSE_Supernovae" in f.keys():
            any_kicks = True
            df_dict = {k: f["BSE_Supernovae"][k][...] for k in f["BSE_Supernovae"].keys()}
            sn = pd.DataFrame(df_dict)
            sn.index = sn["SEED"].values

    if any_kicks:

        # determine which star is undergoing the SN
        sn["star"] = np.where(sn["Supernova_State"] == 1, 1, 2)

        # extract relevant columns
        proto_kick_info = sn[["star", "Unbound", "Applied_Kick_Magnitude(SN)",
                            "SN_Kick_Theta(SN)", "SN_Kick_Phi(SN)", "SN_Kick_Mean_Anomaly(SN)",
                            "VelocityX(SN)", "VelocityY(SN)", "VelocityZ(SN)", "ComponentSpeed(SN)",
                            "VelocityX(CP)", "VelocityY(CP)", "VelocityZ(CP)", "ComponentSpeed(CP)", "SEED"]]

        # COMPAS reports the total velocity change imparted to each star, but we need to adjust
        # the values for the second SN so that it's just a *delta* from the first SN as in COSMIC
        which_kick = proto_kick_info.groupby("SEED").cumcount()
        first_kicks = proto_kick_info[which_kick == 0]
        second_kicks = proto_kick_info[which_kick == 1]

        # find any system that had two kicks
        two_kick_seeds = np.intersect1d(first_kicks["SEED"], second_kicks["SEED"])

        # adjust the second kick velocities to be relative to the first kick
        # the SN star gets the CP velocity subtracted, the companion gets the SN velocity subtracted since they
        # swap which star is exploding between the rows
        SN_VEL_COLS = ["VelocityX(SN)", "VelocityY(SN)", "VelocityZ(SN)"]
        CP_VEL_COLS = ["VelocityX(CP)", "VelocityY(CP)", "VelocityZ(CP)"]
        second_kicks.loc[two_kick_seeds, SN_VEL_COLS] -= first_kicks.loc[two_kick_seeds, CP_VEL_COLS].values
        second_kicks.loc[two_kick_seeds, CP_VEL_COLS] -= first_kicks.loc[two_kick_seeds, SN_VEL_COLS].values

        # recombine the adjusted kicks
        kick_info = pd.concat([first_kicks, second_kicks]).sort_values("SEED")

        # now the problem is COSMIC wants the kick info in a different format, with star 1 and star 2 columns
        # rather than SN and CP columns - so we need to rearrange things again
        NEW_COLS = ["delta_vsysx_1", "delta_vsysy_1", "delta_vsysz_1", "vsys_1_total",
                        "delta_vsysx_2", "delta_vsysy_2", "delta_vsysz_2", "vsys_2_total"]

        # for default case (star = 1), star 1 gets SN values, star 2 gets CP values
        kick_info[NEW_COLS] = kick_info[["VelocityX(SN)", "VelocityY(SN)", "VelocityZ(SN)", "ComponentSpeed(SN)",
                                        "VelocityX(CP)", "VelocityY(CP)", "VelocityZ(CP)", "ComponentSpeed(CP)"]]

        # for star = 2, star 2 gets SN values, star 1 gets CP values
        is_star_2 = kick_info["star"] == 2
        kick_info.loc[is_star_2, NEW_COLS] = kick_info.loc[
            is_star_2,
            ["VelocityX(CP)", "VelocityY(CP)", "VelocityZ(CP)", "ComponentSpeed(CP)",
            "VelocityX(SN)", "VelocityY(SN)", "VelocityZ(SN)", "ComponentSpeed(SN)"]
        ].values

        # rename columns to COSMIC kick_info style
        kick_info.rename({
            "Unbound": "disrupted",
            "Applied_Kick_Magnitude(SN)": "natal_kick",
            "SN_Kick_Phi(SN)": "theta",                 # <-- this is a purposeful swap to match COSMIC convention
            "SN_Kick_Theta(SN)": "phi",                 # <-- this is a purposeful swap to match COSMIC convention
            "SN_Kick_Mean_Anomaly(SN)": "mean_anomaly",
            "SEED": "bin_num"
        }, axis=1, inplace=True)

        # drop the extra columns
        kick_info.drop(columns=["VelocityX(SN)", "VelocityY(SN)", "VelocityZ(SN)", "ComponentSpeed(SN)",
                                "VelocityX(CP)", "VelocityY(CP)", "VelocityZ(CP)", "ComponentSpeed(CP)"],
                    inplace=True)
        # track which systems were kicked (see below)
        kick_info["WAS_KICKED"] = 1

        # adjust angles to degrees from radians
        kick_info["phi"] = np.degrees(kick_info["phi"])
        kick_info["theta"] = np.degrees(kick_info["theta"])
        kick_info["mean_anomaly"] = np.degrees(kick_info["mean_anomaly"])

        # index kick_info on both the bin_num and star columns
        kick_info.set_index(["bin_num", "star"], inplace=True)
    
    # the full kick_info has TWO rows per binary system, it's just a row of zeros if an SN didn't occur
    # indices are from the all_seeds list, star are []
    n_binaries = len(all_seeds)
    full_index = pd.MultiIndex.from_product([all_seeds, [1, 2]], names=["bin_num", "star"])
    full_kick_info = pd.DataFrame(np.zeros((n_binaries * 2, len(KICK_INFO_COLS))),
                                  columns=KICK_INFO_COLS, index=full_index)
    # fill in the kick_info rows we have
    if any_kicks:
        full_kick_info["WAS_KICKED"] = 0
        full_kick_info.update(kick_info)

    full_kick_info.reset_index(inplace=True)
    full_kick_info.index = full_kick_info["bin_num"].values

    # set star to 0 where WAS_KICKED is 0, drop WAS_KICKED
    if any_kicks:
        full_kick_info.loc[full_kick_info["WAS_KICKED"] == 0, "star"] = 0
        full_kick_info.drop(columns=["WAS_KICKED"], inplace=True)
    else:
        full_kick_info["star"] = 0

    return full_kick_info

def _get_porb_from_a(a, m1, m2):
    """Calculate the orbital period based on the semi-major axis and stellar masses.

    Parameters
    ----------
    a : `float`
        Semi-major axis in solar radii
    m1 : `float`
        Mass of the first star in solar masses
    m2 : `float`
        Mass of the second star in solar masses

    Returns
    =======
    porb : `float`
        Orbital period in days
    """
    return (2 * np.pi * np.sqrt((a * u.Rsun)**3 / (const.G * (m1 + m2) * u.M_sun))).to(u.day).value


def _calculate_roche_lobe_factor(q):
    """Apply Eggleton approximation to calculate the Roche Lobe factor for a star in the binary

    Parameters
    ----------
    q : `float`
        Mass ratio of the binary system (m2/m1 if you want r1 and vice/versa)

    """
    return (0.49 * q**(2/3)) / (0.6 * q**(2/3) + np.log(1 + q**(1/3)))
