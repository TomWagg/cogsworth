import pandas as pd
import numpy as np
import cogsworth
from astropy import constants as const
from astropy import units as u
import h5py as h5

def grab_h5_data(filename, groupname, set_index=False, fields=None):
    """Grab data from an HDF5 file and return as a pandas DataFrame

    Parameters
    ----------
    filename : `str`
        Path to the HDF5 file.
    groupname : `str`
        Name of the group within the HDF5 file to read data from.
    set_index : `bool`, optional
        Whether to set the index of the DataFrame to the 'SEED' column. Default is False.
    fields : `list` of `str`, optional
        List of fields (columns) to read from the group. If None, all fields are read. Default is None.

    Returns
    -------
    df : `pandas.DataFrame`
        DataFrame containing the data from the specified group in the HDF5 file.
    """
    with h5.File(filename, "r") as f:
        if "SEED" not in fields and fields is not None:
            fields.append("SEED")
        data_dict = {k: f[groupname][k][...] for k in f[groupname].keys() if fields is None or k in fields}
        df = pd.DataFrame(data_dict)
        # if set_index and 'SEED' in df.columns:
        #     df.set_index('SEED', inplace=True, drop=False)
    return df

# Function to convert separation to orbital period
def convert_a_to_P(separation, M1, M2):
    # convert to astropy units
    separation = np.abs(separation) * u.AU
    M1 = M1 * u.solMass
    M2 = M2 * u.solMass

    # Input validation
    if not separation.unit.is_equivalent(u.m):
        raise ValueError("Separation must have length units (e.g., AU, m).")
    if not M1.unit.is_equivalent(u.kg) or not M2.unit.is_equivalent(u.kg):
        raise ValueError("Masses must have mass units (e.g., solar masses, kg).")
    
    G = const.G  # Gravitational constant with correct units
    
    mu = G * (M1 + M2)
    
    period = 2 * np.pi * np.sqrt(separation**3 / mu)
    period = np.float64(period.to(u.day))  # Convert to seconds (or any desired time unit)
    
    return period

# Initial state
def initial(compas_output):
    bpp = grab_h5_data(compas_output, "BSE_System_Parameters", True, fields=["Mass@ZAMS(1)", "Mass@ZAMS(2)", "Stellar_Type@ZAMS(1)", "Stellar_Type@ZAMS(2)", "SemiMajorAxis@ZAMS"])
    cols_sys = {"Mass@ZAMS(1)": "mass_1", 
                "Mass@ZAMS(2)": "mass_2", 
                "Stellar_Type@ZAMS(1)": "kstar_1", 
                "Stellar_Type@ZAMS(2)": "kstar_2",
                "SemiMajorAxis@ZAMS": "sep"}
    bpp = bpp.rename(columns=cols_sys)
    bpp["tphys"] = 0.0
    bpp["evol_type"] = 1
    bpp[["RRLO_1", "RRLO_2"]] = np.nan
    return bpp

# RLOF
def rlof(compas_output):
    rlof = grab_h5_data(compas_output, "BSE_RLOF", True, fields=["Time<MT", "Mass(1)<MT", "Mass(2)<MT", "Stellar_Type(1)<MT", "Stellar_Type(2)<MT", "SemiMajorAxis<MT", "Radius(1)|RL<step", "Radius(2)|RL<step",
        "Time>MT", "Mass(1)>MT", "Mass(2)>MT", "Stellar_Type(1)>MT", "Stellar_Type(2)>MT", "SemiMajorAxis>MT", "Radius(1)|RL>step", "Radius(2)|RL>step"])
    rlof_before = rlof[["Time<MT", "Mass(1)<MT", "Mass(2)<MT", "Stellar_Type(1)<MT", "Stellar_Type(2)<MT", "SemiMajorAxis<MT", "Radius(1)|RL<step", "Radius(2)|RL<step"]]
    rlof_after = rlof[["Time>MT", "Mass(1)>MT", "Mass(2)>MT", "Stellar_Type(1)>MT", "Stellar_Type(2)>MT", "SemiMajorAxis>MT", "Radius(1)|RL>step", "Radius(2)|RL>step"]]
    cols_rlof_before = {"Time<MT": "tphys",
                        "Mass(1)<MT": "mass_1", 
                        "Mass(2)<MT": "mass_2", 
                        "Stellar_Type(1)<MT": "kstar_1", 
                        "Stellar_Type(2)<MT": "kstar_2", 
                        "SemiMajorAxis<MT": "sep",
                        "Radius(1)|RL<step": "RRLO_1",
                        "Radius(2)|RL<step": "RRLO_2"}
    cols_rlof_after = { "Time>MT": "tphys",
                        "Mass(1)>MT": "mass_1", 
                        "Mass(2)>MT": "mass_2", 
                        "Stellar_Type(1)>MT": "kstar_1", 
                        "Stellar_Type(2)>MT": "kstar_2", 
                        "SemiMajorAxis>MT": "sep",
                        "Radius(1)|RL>step": "RRLO_1",
                        "Radius(2)|RL>step": "RRLO_2"}
    rlof_before = rlof_before.rename(columns=cols_rlof_before)
    rlof_after = rlof_after.rename(columns=cols_rlof_after)

    rlof_before["evol_type"] = np.full(len(rlof_before), 3)
    rlof_after["evol_type"] = np.full(len(rlof_after), 4)

    bpp = pd.concat([rlof_before, rlof_after])
    return bpp

# Supernovae
def supernovae(compas_output):
    bpp = grab_h5_data(compas_output, "BSE_Supernovae", True, fields=["Time", "Mass(SN)", "Mass(CP)", "Stellar_Type(SN)", "Stellar_Type(CP)", "SemiMajorAxis", "Supernova_State"])
    
    def create_sn_rows(row):
        if row["Supernova_State"] == 1:
            row["evol_type"] = 15
        elif row["Supernova_State"] == 2:
            row["evol_type"] = 16
        else:
            print("extremely rare")
            exit()
        sn_star, cp_star = int(row["Supernova_State"]), 1 if row["Supernova_State"] == 2 else 2
        cols_sn = {
                "Stellar_Type(SN)": f"kstar_{sn_star}",
                "Stellar_Type(CP)": f"kstar_{cp_star}",
                "Mass(SN)": f"mass_{sn_star}",
                "Mass(CP)": f"mass_{cp_star}"
            }
        row = row.rename(cols_sn)
        return row

    bpp = bpp.apply(create_sn_rows, axis=1)
    bpp.drop("Supernova_State", axis=1, inplace=True)
    bpp.rename(columns={"SemiMajorAxis": "sep", "Time": "tphys"}, inplace=True)

    return bpp

# Stellar type changes
def stc(compas_output):
    bpp = grab_h5_data(compas_output, "BSE_Switch_Log", True, fields=["Time", "Star_Switching", "Switching_To"])
    bpp = bpp.rename(columns={"Time": "tphys"})
    def create_kstar(row):
        row[f"kstar_{int(row['Star_Switching'])}"] = row["Switching_To"]
        return row
    bpp = bpp.apply(create_kstar, axis=1)
    bpp.drop(["Star_Switching", "Switching_To"], axis=1, inplace=True)
    bpp["evol_type"] = 2

    return bpp

# Generate full bpp
def generate_full_bpp(compas_output):
    initial_bpp = initial(compas_output)
    rlof_bpp = rlof(compas_output)
    supernovae_bpp = supernovae(compas_output)
    stc_bpp = stc(compas_output)
    bpp = pd.concat([initial_bpp, rlof_bpp, supernovae_bpp, stc_bpp])

    # Carry previous values to each evol_type=2 event
    bpp.sort_values(by=["SEED", "tphys"], axis=0, inplace=True)
    bpp.ffill(inplace=True)

    # delete redundant type changes that are already logged by RLOF
    diffs = bpp[["kstar_1", "kstar_2"]].diff()
    bpp = bpp[~((bpp["evol_type"]==2) & (diffs["kstar_1"]==0.0) & (diffs["kstar_2"]==0.0))]
    
    # Calculate porb
    bpp["porb"] = bpp.apply(lambda x: convert_a_to_P(x["sep"], x["mass_1"], x["mass_2"]), axis=1)

    # Assign bin_num for each seed
    bpp["bin_num"], _ = pd.factorize(bpp.index)
    bpp = bpp.reset_index().set_index("bin_num")

    # Sort bpp
    def order_mt(col):
        if col.name != "evol_type":
            return col
        # Give priority in this order: Begin CE, End CE, End RLOF
        col.replace({7: 1, 8: 2, 4: 3}, inplace=True)
        return col
    bpp.sort_values(by=["bin_num", "tphys", "evol_type"], key=order_mt, inplace=True)

    return bpp

def generate_cartoon_plot(compas_output, seed):
    bpp = generate_full_bpp(compas_output)
    num = bpp.index[bpp["SEED"] == seed][0]
    cogsworth.plot.plot_cartoon_evolution(bpp, bin_num=num)