import h5py as h5
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as const


__all__ = ["create_bpp_from_COMPAS_files"]


def create_bpp_from_COMPAS_files(filename):
    """Create a COSMIC-like bpp table from a COMPAS output file

    This function combines several different COMPAS output tables to create a bpp-like
    table that contains rows for each significant stellar evolution event in the life of
    each binary system.

    The function currently implements the following events:
    - Changes in stellar type (from BSE_Switch_Log)
    - Start and end of mass transfer episodes (from BSE_RLOF)
    - Start and end of common-envelope events (from BSE_Common_Envelopes)
    - Supernova events (from BSE_Supernovae)

    Parameters
    ----------
    filename : `str`
        Path to the COMPAS output file in HDF5 format.

    Returns
    -------
    bpp : `pandas.DataFrame`
        A DataFrame containing the bpp table with rows for each stellar evolution event.
    """
    BPP_COLUMNS = ["Time", "Mass(1)", "Mass(2)", "Stellar_Type(1)", "Stellar_Type(2)",
                   "SemiMajorAxis", "Eccentricity", "RLOF(1)", "RLOF(2)", "Radius(1)", "Radius(2)"]
    INIT_COLS = ["Mass@ZAMS(1)", "Mass@ZAMS(2)", "Eccentricity@ZAMS", "SemiMajorAxis@ZAMS",
                 "Stellar_Type@ZAMS(1)", "Stellar_Type@ZAMS(2)"]

    # open the COMPAS file
    with h5.File(filename, "r") as f:

        # ------------------
        # INITIAL CONDITIONS
        # ------------------

        bse_sys = pd.DataFrame({k: f["BSE_System_Parameters"][k][...] for k in f["BSE_System_Parameters"].keys()})
        bse_sys.set_index("SEED", inplace=True)

        init_col_translator = {
            "Mass@ZAMS(1)": "Mass(1)", "Mass@ZAMS(2)": "Mass(2)", "Eccentricity@ZAMS": "Eccentricity",
            "SemiMajorAxis@ZAMS": "SemiMajorAxis", "Stellar_Type@ZAMS(1)": "Stellar_Type(1)",
            "Stellar_Type@ZAMS(2)": "Stellar_Type(2)"
        }
        init = bse_sys[INIT_COLS].rename(init_col_translator, axis=1)

        init[["Time", "RLOF(1)", "RLOF(2)",
              "Radius(1)", "Radius(2)", "evol_type"]] = [0.0, 0.0, 0.0, np.nan, np.nan, 1]

        # create a DF just with the moments where a star changes stellar type
        df_dict = {k: f["BSE_Switch_Log"][k][...] for k in f["BSE_Switch_Log"].keys()}
        switch_df = pd.DataFrame(df_dict)
        switch_df.set_index("SEED", inplace=True)

        # but remove any rows where the stellar type doesn't actually change, or mergers
        switch_df = switch_df[(switch_df["Switching_From"] != switch_df["Switching_To"])
                              & (switch_df["Switching_To"] != 15)]
        switch_df = switch_df.loc[:, BPP_COLUMNS]
        switch_df["evol_type"] = 2

        # create a DF with the moments where a star fills its Roche lobe
        df_dict = {k: f["BSE_RLOF"][k][...] for k in f["BSE_RLOF"].keys()}
        rlof_df = pd.DataFrame(df_dict)
        rlof_df.set_index("SEED", inplace=True)

        # convert the DF into two, with a row for the start and end of RLOF
        mt_cols = ["Time", "Mass(1)", "Mass(2)", "Stellar_Type(1)", "Stellar_Type(2)",
                "SemiMajorAxis", "Eccentricity", "Radius(1)", "Radius(2)"]
        start_cols = [c + "<MT" for c in mt_cols]
        end_cols = [c + ">MT" for c in mt_cols] + ["CEE>MT"]
        mt_start_rows = rlof_df[start_cols].rename({s: s.replace("<MT", "") for s in start_cols}, axis=1)
        mt_start_rows["evol_type"] = 3
        mt_end_rows = rlof_df[end_cols].rename({s: s.replace(">MT", "") for s in end_cols if s != "CEE>MT"},
                                               axis=1)
        mt_end_rows["evol_type"] = 4
        # mt_end_rows = mt_end_rows[~mt_end_rows["CEE>MT"] == 1]
        mt_end_rows.drop(columns=["CEE>MT"], inplace=True)

        # combine the mass transfer start and end rows into a single dataframe
        mt_rows = pd.concat([mt_start_rows, mt_end_rows]).sort_values(["SEED", "Time", "evol_type"],
                                                                    ascending=[True, True, False])

        # go through mt_rows dataframe, merge subsequent mass transfer episodes
        mt_rows["SEED"] = mt_rows.index
        for col in ["evol_type", "SEED", "Time"]:
            for prefix, dir in [("prev_", 1), ("next_", -1)]:
                mt_rows[f"{prefix}{col}"] = mt_rows[col].shift(dir)


        mt_rows = mt_rows[~(((mt_rows["evol_type"] == 4) & (mt_rows["next_evol_type"] == 3)
                            & (mt_rows["next_SEED"] == mt_rows["SEED"]) & (mt_rows["next_Time"] == mt_rows["Time"]))
                            | ((mt_rows["evol_type"] == 3) & (mt_rows["prev_evol_type"] == 4)
                            & (mt_rows["prev_SEED"] == mt_rows["SEED"]) & (mt_rows["prev_Time"] == mt_rows["Time"])))]

        # get rid of those extra columns
        mt_rows = mt_rows.loc[:, mt_cols + ["evol_type"]]

        # create a DF with the moments where a star causes a common-envelope event
        df_dict = {k: f["BSE_Common_Envelopes"][k][...] for k in f["BSE_Common_Envelopes"].keys()}
        ce_df = pd.DataFrame(df_dict)
        ce_df.set_index("SEED", inplace=True)

        # as before, convert the DF into two, with a row for the start and end of CE
        ce_cols = ["Time", "Mass(1)", "Mass(2)", "Stellar_Type(1)", "Stellar_Type(2)", "SemiMajorAxis",
                   "Eccentricity", "Radius(1)", "Radius(2)"]
        start_cols = [c + "<CE" if c != "Time" else c for c in ce_cols]
        end_cols = [c + ">CE" if c not in ["Time", "Stellar_Type(1)", "Stellar_Type(2)"] else c for c in ce_cols]
        ce_start_rows = ce_df[start_cols].rename({s: s.replace("<CE", "") for s in start_cols}, axis=1)
        ce_start_rows["evol_type"] = 7
        ce_end_rows = ce_df[end_cols + ["Merger"]].rename({s: s.replace(">CE", "") for s in end_cols}, axis=1)

        # set to -42 so that sorting works out nicely (this is just a magic negative number)
        ce_end_rows["evol_type"] = 6.9

        # if a merger occurs, turn the other star into a massless remnant (COMPAS doesn't do this...?)
        ce_end_rows.loc[(ce_end_rows["Merger"] == 1) & (ce_end_rows["Mass(1)"] > 1e10),
                        ["Mass(1)", "Stellar_Type(1)", "Radius(1)"]] = [0.0, 15, 0.0]
        ce_end_rows.loc[(ce_end_rows["Merger"] == 1) & (ce_end_rows["Mass(2)"] > 1e10),
                        ["Mass(2)", "Stellar_Type(2)", "Radius(2)"]] = [0.0, 15, 0.0]
        ce_end_rows.drop(columns=["Merger"], inplace=True)

        # combine all ce_rows
        ce_rows = pd.concat([ce_start_rows, ce_end_rows]).sort_values(["SEED", "Time"])

        df_dict = {k: f["BSE_Supernovae"][k][...] for k in f["BSE_Supernovae"].keys()}
        sn_df = pd.DataFrame(df_dict)
        sn_df[["RLOF(1)", "RLOF(2)"]] = [0.0, 0.0]
        sn_df.set_index("SEED", inplace=True)
        sn_type = sn_df["Supernova_State"].copy()
        sn_df = sn_df.loc[:, BPP_COLUMNS]
        sn_df["evol_type"] = np.where(sn_type == 1, 15, 16)

        disruption_rows = sn_df.loc[sn_df["SemiMajorAxis"] < 0.0].copy()
        disruption_rows["evol_type"] = 11
        sn_df = pd.concat([sn_df, disruption_rows])

    bpp = pd.concat([init, switch_df, mt_rows, ce_rows, sn_df], sort=False)
    bpp = bpp.sort_values(["SEED", "Time", "evol_type"], ascending=[True, True, False])

    # change CE end rows back to evol_type 8
    bpp["evol_type"] = bpp["evol_type"].replace({6.9: 8}).astype(int)

    # find all rows where evol_type == 4 and the previous row had evol_type == 8
    bpp["prev_evol_type"] = bpp["evol_type"].shift(1)
    mask = (bpp["evol_type"] == 4) & (bpp["prev_evol_type"] == 8)
    cols = BPP_COLUMNS + ["row_inds"]
    bpp["row_inds"] = np.arange(len(bpp))
    bpp.loc[mask, cols] = bpp.loc[bpp["row_inds"].isin(bpp.loc[mask, cols]["row_inds"] - 1), cols]

    cols = BPP_COLUMNS + ["evol_type", "bin_num"]
    bpp["bin_num"] = bpp.index
    bpp.index = bpp["bin_num"]
    bpp = bpp.loc[:, cols]

    # calculate porb and match SemiMajorAxis conventions
    merged = bpp["SemiMajorAxis"] == np.inf
    disrupted = bpp["SemiMajorAxis"] < 0.0
    bpp["porb"] = _get_porb_from_a(bpp["SemiMajorAxis"].values, bpp["Mass(1)"].values, bpp["Mass(2)"].values)
    bpp.loc[disrupted, ["SemiMajorAxis", "porb"]] = [-1.0, -1.0]
    bpp.loc[merged, ["SemiMajorAxis", "porb"]] = [0.0, 0.0]

    bpp.rename({
        "Time": "tphys",
        "Mass(1)": "mass_1",
        "Mass(2)": "mass_2",
        "Stellar_Type(1)": "kstar_1",
        "Stellar_Type(2)": "kstar_2",
        "SemiMajorAxis": "sep",
        "Eccentricity": "ecc",
        "RLOF(1)": "RRLO_1",
        "RLOF(2)": "RRLO_2",
        "Radius(1)": "rad_1",
        "Radius(2)": "rad_2",
    }, axis=1, inplace=True)

    bound = bpp["sep"] > 0.0
    bpp.loc[~bound, ["RRLO_1", "RRLO_2"]] = [0.0, 0.0]
    for l, r in [("1", "2"), ("2", "1")]:
        f_Roche = _calculate_roche_lobe_factor(bpp.loc[bound, f"mass_{r}"] / bpp.loc[bound, f"mass_{l}"])
        bpp.loc[bound, f"RRLO_{l}"] = bpp.loc[bound, f"rad_{l}"] / (f_Roche * bpp.loc[bound, "sep"])

    return bpp

def create_kick_info_from_COMPAS_file(filename):
    """Create a kick_info table from a COMPAS output file

    The kick_info table needs the following columns, in this order:
        star, disrupted, natal_kick, phi, theta, mean_anomaly

    Parameters
    ----------
    filename : `str`
        Path to the COMPAS output file in HDF5 format.

    Returns
    -------
    kick_info : `pandas.DataFrame`
        A DataFrame containing the kick_info table.
    """
    with h5.File(filename, "r") as f:
        df_dict = {k: f["BSE_Supernovae"][k][...] for k in f["BSE_Supernovae"].keys()}
        supernovae = pd.DataFrame(df_dict)
        supernovae.set_index("SEED", inplace=True)

    return kick_info

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
