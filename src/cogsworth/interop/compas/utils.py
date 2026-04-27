import numpy as np
import pandas as pd

__all__ = ["add_kicks_to_initial_binaries"]

KICK_COLS = ['natal_kick', 'phi', 'theta', 'mean_anomaly', 'randomseed']


def add_kicks_to_initial_binaries(initial_binaries, kick_info):
    """Add the kick information from a kick_info dataframe to the initial_binaries dataframe"""
    new_cols = {
        f"{col}_{i}": np.full(len(initial_binaries), (0.0 if col == 'randomseed' else -100.0)) 
        for i in [1, 2] for col in KICK_COLS
    }
    initial_kick_df = pd.DataFrame(new_cols, index=initial_binaries.index)

    # if randomseed is missing, fill with bin_num (SEED)
    if 'randomseed' not in kick_info.columns:
        kick_info['randomseed'] = kick_info['bin_num']

    # update the kick dataframe with the kick information where it exists
    for i in [1, 2]:
        mask = kick_info['star'] == i
        sn_nums = kick_info[mask]["bin_num"]
        for col in KICK_COLS:
            initial_kick_df.loc[sn_nums, f"{col}_{i}"] = kick_info[mask][col].values
            
    # delete any existing kick columns to avoid duplication
    for i in [1, 2]:
        initial_binaries.drop(
            columns=[f"{col}_{i}" for col in KICK_COLS if f"{col}_{i}" in initial_binaries.columns],
            errors='ignore', inplace=True
        )

    # concatenate the new kick columns to the initial binaries dataframe
    initial_binaries = pd.concat([initial_binaries, initial_kick_df], axis=1)

    return initial_binaries