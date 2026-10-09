from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import shutil
import logging
import difflib

import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import numpy as np

from cogsworth.utils import check_dependencies

__all__ = [
    "MISTBolometricCorrectionGrid", "list_filter_sets", "list_filters", "check_filters",
    "MISTFilterError"
]


# mapping of photometric systems to the bands available in MIST v1 BC tables (currently unused)
MIST_FILTER_SETS_V1 = {
    "UBVRIplus": [
        "Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I", "2MASS_J", "2MASS_H", "2MASS_Ks",
        "Kepler_Kp", "Kepler_D51", "Hipparcos_Hp", "Tycho_B", "Tycho_V",
        "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev", "Gaia_RP_DR2Rev", "Gaia_G_MAW", "Gaia_BP_MAWf", "Gaia_BP_MAWb",
        "Gaia_RP_MAW", "TESS", "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"
    ],
    "WISE":["WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4"],
    "CFHT":["CFHT_u", "CFHT_g", "CFHT_r", "CFHT_i_new", "CFHT_i_old", "CFHT_z"],
    "DECam":["DECam_u", "DECam_g", "DECam_r", "DECam_i", "DECam_z", "DECam_Y"],
    "GALEX":["GALEX_FUV", "GALEX_NUV"],
    "JWST":[
        "F070W", "F090W", "F115W", "F140M", "F150W2", "F150W", "F162M", "F164N", "F182M", "F187N",
        "F200W", "F210M", "F212N", "F250M", "F277W", "F300M", "F322W2", "F323N", "F335M", "F356W",
        "F360M", "F405N", "F410M", "F430M", "F444W", "F460M", "F466N", "F470N", "F480M",
    ],
    "LSST": ["LSST_u", "LSST_g", "LSST_r", "LSST_i", "LSST_z", "LSST_y"],
    "PanSTARRS": ["PS_g", "PS_r", "PS_i", "PS_z", "PS_y", "PS_w", "PS_open"],
    "SkyMapper": ["SkyMapper_u", "SkyMapper_v", "SkyMapper_g", "SkyMapper_r", "SkyMapper_i", "SkyMapper_z"],
    "SPITZER": ["IRAC_3.6", "IRAC_4.5", "IRAC_5.8", "IRAC_8.0"],
    "UKIDSS": ["UKIDSS_Z", "UKIDSS_Y", "UKIDSS_J", "UKIDSS_H", "UKIDSS_K"],
    "SDSSugriz": ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
    "HST_ACSWF": ["ACS_WFC_F435W", "ACS_WFC_F475W", "ACS_WFC_F502N", "ACS_WFC_F550M", "ACS_WFC_F555W",
                  "ACS_WFC_F606W", "ACS_WFC_F625W", "ACS_WFC_F658N", "ACS_WFC_F660N", "ACS_WFC_F775W",
                  "ACS_WFC_F814W", "ACS_WFC_F850LP", "ACS_WFC_F892N"],
    "HST_ACSHR": ["ACS_HRC_F220W", "ACS_HRC_F250W", "ACS_HRC_F330W", "ACS_HRC_F344N", "ACS_HRC_F435W",
                  "ACS_HRC_F475W", "ACS_HRC_F502N", "ACS_HRC_F550M", "ACS_HRC_F555W", "ACS_HRC_F606W",
                  "ACS_HRC_F625W", "ACS_HRC_F658N", "ACS_HRC_F660N", "ACS_HRC_F775W", "ACS_HRC_F814W",
                  "ACS_HRC_F850LP", "ACS_HRC_F892N"],
    "HST_WFC3": ["WFC3_UVIS_F200LP", "WFC3_UVIS_F218W", "WFC3_UVIS_F225W", "WFC3_UVIS_F275W",
                 "WFC3_UVIS_F280N", "WFC3_UVIS_F300X", "WFC3_UVIS_F336W", "WFC3_UVIS_F343N",
                 "WFC3_UVIS_F350LP", "WFC3_UVIS_F373N", "WFC3_UVIS_F390M", "WFC3_UVIS_F390W",
                 "WFC3_UVIS_F395N", "WFC3_UVIS_F410M", "WFC3_UVIS_F438W", "WFC3_UVIS_F467M",
                 "WFC3_UVIS_F469N", "WFC3_UVIS_F475W", "WFC3_UVIS_F475X", "WFC3_UVIS_F487N",
                 "WFC3_UVIS_F502N", "WFC3_UVIS_F547M", "WFC3_UVIS_F555W", "WFC3_UVIS_F600LP",
                 "WFC3_UVIS_F606W", "WFC3_UVIS_F621M", "WFC3_UVIS_F625W", "WFC3_UVIS_F631N",
                 "WFC3_UVIS_F645N", "WFC3_UVIS_F656N", "WFC3_UVIS_F657N", "WFC3_UVIS_F658N",
                 "WFC3_UVIS_F665N", "WFC3_UVIS_F673N", "WFC3_UVIS_F680N", "WFC3_UVIS_F689M",
                 "WFC3_UVIS_F763M", "WFC3_UVIS_F775W", "WFC3_UVIS_F814W", "WFC3_UVIS_F845M",
                 "WFC3_UVIS_F850LP", "WFC3_UVIS_F953N", "WFC3_IR_F098M", "WFC3_IR_F105W", "WFC3_IR_F110W",
                 "WFC3_IR_F125W", "WFC3vIR_F126N", "WFC3_IR_F127M", "WFC3_IR_F128N", "WFC3_IR_F130N",
                 "WFC3_IR_F132N", "WFC3_IR_F139M", "WFC3_IR_F140W", "WFC3_IR_F153M", "WFC3_IR_F160W",
                 "WFC3_IR_F164N", "WFC3_IR_F167N" ],
    "HST_WFPC2": ["WFPC2_F218W", "WFPC2_F255W", "WFPC2_F300W", "WFPC2_F336W", "WFPC2_F439W", "WFPC2_F450W",
                  "WFPC2_F555W", "WFPC2_F606W", "WFPC2_F622W", "WFPC2_F675W", "WFPC2_F791W", "WFPC2_F814W",
                  "WFPC2_F850LP" ]
}

# mapping of photometric systems to the bands available in MIST v2 BC tables
MIST_FILTER_SETS = {
    "UBVRIplus": [
        "Bessell_U", "Bessell_B", "Bessell_V", "Bessell_R", "Bessell_I", "2MASS_J", "2MASS_H", "2MASS_Ks",
        "Kepler_Kp", "Kepler_D51", "Hipparcos_Hp", "Tycho_B", "Tycho_V", "Gaia_G_DR2Rev", "Gaia_BP_DR2Rev",
        "Gaia_RP_DR2Rev", "Gaia_G_MAW", "Gaia_BP_MAWb", "Gaia_BP_MAWf", "Gaia_RP_MAW", "TESS", "Gaia_G_EDR3",
        "Gaia_BP_EDR3", "Gaia_RP_EDR3", "Gemini_NIRI_BrG", "WIYN_NESSI_NB832"
    ],
    "CFHTugriz": ["CFHT_u", "CFHT_CaHK", "CFHT_g", "CFHT_r", "CFHT_i_new", "CFHT_i_old", "CFHT_z"],
    "DECam": ["DECam_u", "DECam_g", "DECam_r", "DECam_i", "DECam_z", "DECam_Y"],
    "Euclid": ["Euclid_VIS", "Euclid_Y", "Euclid_J", "Euclid_H"],
    "GALEX": ["GALEX_FUV", "GALEX_NUV"],
    "HSC": ["hsc_g", "hsc_r", "hsc_i", "hsc_z", "hsc_y", "hsc_nb816", "hsc_nb921"],
    "HST_ACS_HRC": [
        "ACS_HRC_F220W", "ACS_HRC_F250W", "ACS_HRC_F330W", "ACS_HRC_F344N", "ACS_HRC_F435W", "ACS_HRC_F475W",
        "ACS_HRC_F502N", "ACS_HRC_F550M", "ACS_HRC_F555W", "ACS_HRC_F606W", "ACS_HRC_F625W", "ACS_HRC_F658N",
        "ACS_HRC_F660N", "ACS_HRC_F775W", "ACS_HRC_F814W", "ACS_HRC_F850LP", "ACS_HRC_F892N"
    ],
    "HST_ACS_SBC": [
        "ACS_SBC_F115LP", "ACS_SBC_F122M", "ACS_SBC_F125LP", "ACS_SBC_F140LP", "ACS_SBC_F150LP",
        "ACS_SBC_F165LP", "ACS_SBC_PR110L", "ACS_SBC_PR130L"
    ],
    "HST_ACS_WFC": [
        "ACS_WFC_F435W", "ACS_WFC_F475W", "ACS_WFC_F502N", "ACS_WFC_F550M", "ACS_WFC_F555W", "ACS_WFC_F606W",
        "ACS_WFC_F625W", "ACS_WFC_F658N", "ACS_WFC_F660N", "ACS_WFC_F775W", "ACS_WFC_F814W", "ACS_WFC_F850LP",
        "ACS_WFC_F892N"
    ],
    "HST_WFC3": [
        "WFC3_UVIS_F200LP", "WFC3_UVIS_F218W", "WFC3_UVIS_F225W", "WFC3_UVIS_F275W", "WFC3_UVIS_F280N",
        "WFC3_UVIS_F300X", "WFC3_UVIS_F336W", "WFC3_UVIS_F343N", "WFC3_UVIS_F350LP", "WFC3_UVIS_F373N",
        "WFC3_UVIS_F390M", "WFC3_UVIS_F390W", "WFC3_UVIS_F395N", "WFC3_UVIS_F410M", "WFC3_UVIS_F438W",
        "WFC3_UVIS_F467M", "WFC3_UVIS_F469N", "WFC3_UVIS_F475W", "WFC3_UVIS_F475X", "WFC3_UVIS_F487N",
        "WFC3_UVIS_F502N", "WFC3_UVIS_F547M", "WFC3_UVIS_F555W", "WFC3_UVIS_F600LP", "WFC3_UVIS_F606W",
        "WFC3_UVIS_F621M", "WFC3_UVIS_F625W", "WFC3_UVIS_F631N", "WFC3_UVIS_F645N", "WFC3_UVIS_F656N",
        "WFC3_UVIS_F657N", "WFC3_UVIS_F658N", "WFC3_UVIS_F665N", "WFC3_UVIS_F673N", "WFC3_UVIS_F680N",
        "WFC3_UVIS_F689M", "WFC3_UVIS_F763M", "WFC3_UVIS_F775W", "WFC3_UVIS_F814W", "WFC3_UVIS_F845M",
        "WFC3_UVIS_F850LP", "WFC3_UVIS_F953N", "WFC3_IR_F098M", "WFC3_IR_F105W", "WFC3_IR_F110W",
        "WFC3_IR_F125W", "WFC3_IR_F126N", "WFC3_IR_F127M", "WFC3_IR_F128N", "WFC3_IR_F130N", "WFC3_IR_F132N",
        "WFC3_IR_F139M", "WFC3_IR_F140W", "WFC3_IR_F153M", "WFC3_IR_F160W", "WFC3_IR_F164N", "WFC3_IR_F167N"
    ],
    "HST_WFPC2": [
        "WFPC2_F218W", "WFPC2_F255W", "WFPC2_F300W", "WFPC2_F336W", "WFPC2_F439W", "WFPC2_F450W",
        "WFPC2_F555W", "WFPC2_F606W", "WFPC2_F622W", "WFPC2_F675W", "WFPC2_F791W", "WFPC2_F814W",
        "WFPC2_F850LP"
    ],
    "IPHAS": ["INT_IPHAS_gR", "INT_IPHAS_Ha", "INT_IPHAS_gI"],
    "JWST": [
        "NIRCAM_F070W", "NIRCAM_F090W", "NIRCAM_F115W", "NIRCAM_F140W", "NIRCAM_F150W", "NIRCAM_F150W2",
        "NIRCAM_F162M", "NIRCAM_F164N", "NIRCAM_F182M", "NIRCAM_F187N", "NIRCAM_F200W", "NIRCAM_F210M",
        "NIRCAM_F212N", "NIRCAM_F250M", "NIRCAM_F277W", "NIRCAM_F300M", "NIRCAM_F322W2", "NIRCAM_F323N",
        "NIRCAM_F335M", "NIRCAM_F356W", "NIRCAM_F360M", "NIRCAM_F405N", "NIRCAM_F410M", "NIRCAM_F430M",
        "NIRCAM_F444W", "NIRCAM_F460M", "NIRCAM_F466N", "NIRCAM_F470W", "NIRCAM_F480W"
    ],
    "LSST": ["LSST_u", "LSST_g", "LSST_r", "LSST_i", "LSST_z", "LSST_y"],
    "NIRISS": [
        "NIRISS_F090W", "NIRISS_F115W", "NIRISS_F140M", "NIRISS_F150W", "NIRISS_F158M", "NIRISS_F200W",
        "NIRISS_F277W", "NIRISS_F356W", "NIRISS_F380M", "NIRISS_F430M", "NIRISS_F444W", "NIRISS_F480M"
    ],
    "PanSTARRS": ["PS_g", "PS_r", "PS_i", "PS_z", "PS_y", "PS_w", "PS_open"],
    "RoboAO": ["LP600", "g", "r", "i", "z"],
    "Roman": [
        "Roman_F062", "Roman_F087", "Roman_F106", "Roman_F129", "Roman_F146", "Roman_F158", "Roman_F184",
        "Roman_F213", "Roman_Grism", "Roman_Prism"
    ],
    "SDSSugriz": ["SDSS_u", "SDSS_g", "SDSS_r", "SDSS_i", "SDSS_z"],
    "SPHEREx": [
        "SPHx_0", "SPHx_1", "SPHx_2", "SPHx_3", "SPHx_4", "SPHx_5", "SPHx_6", "SPHx_7", "SPHx_8", "SPHx_9",
        "SPHx_10", "SPHx_11", "SPHx_12", "SPHx_13", "SPHx_14", "SPHx_15", "SPHx_16", "SPHx_17", "SPHx_18",
        "SPHx_19", "SPHx_20", "SPHx_21", "SPHx_22", "SPHx_23", "SPHx_24", "SPHx_25", "SPHx_26", "SPHx_27",
        "SPHx_28", "SPHx_29", "SPHx_30", "SPHx_31", "SPHx_32", "SPHx_33", "SPHx_34", "SPHx_35", "SPHx_36",
        "SPHx_37", "SPHx_38", "SPHx_39", "SPHx_40", "SPHx_41", "SPHx_42", "SPHx_43", "SPHx_44", "SPHx_45",
        "SPHx_46", "SPHx_47", "SPHx_48", "SPHx_49", "SPHx_50", "SPHx_51", "SPHx_52", "SPHx_53", "SPHx_54",
        "SPHx_55", "SPHx_56", "SPHx_57", "SPHx_58", "SPHx_59", "SPHx_60", "SPHx_61", "SPHx_62", "SPHx_63",
        "SPHx_64", "SPHx_65", "SPHx_66", "SPHx_67", "SPHx_68", "SPHx_69", "SPHx_70", "SPHx_71", "SPHx_72",
        "SPHx_73", "SPHx_74", "SPHx_75", "SPHx_76", "SPHx_77", "SPHx_78", "SPHx_79", "SPHx_80", "SPHx_81",
        "SPHx_82", "SPHx_83", "SPHx_84", "SPHx_85", "SPHx_86", "SPHx_87", "SPHx_88", "SPHx_89", "SPHx_90",
        "SPHx_91", "SPHx_92", "SPHx_93", "SPHx_94", "SPHx_95", "SPHx_96", "SPHx_97", "SPHx_98", "SPHx_99",
        "SPHx_100", "SPHx_101"
    ],
    "SPITZER": ["IRAC_36", "IRAC_45", "IRAC_58", "IRAC_80"],
    "SPLUS": [
        "SPLUS_uJAVA", "SPLUS_gSDSS", "SPLUS_rSDSS", "SPLUS_iSDSS", "SPLUS_zSDSS", "SPLUS_J0340",
        "SPLUS_J0378", "SPLUS_J0395", "SPLUS_J0410", "SPLUS_J0515", "SPLUS_J0660", "SPLUS_J0861"
    ],
    "SkyMapper": ["SkyMapper_u", "SkyMapper_v", "SkyMapper_g", "SkyMapper_r", "SkyMapper_i", "SkyMapper_z"],
    "Swift": ["Swift_UVW2", "Swift_UVM2", "Swift_UVW1", "Swift_U", "Swift_B", "Swift_V"],
    "UKIDSS": ["UKIDSS_Z", "UKIDSS_Y", "UKIDSS_J", "UKIDSS_H", "UKIDSS_K"],
    "UVIT": [
        "UVIT_F148W", "UVIT_F154W", "UVIT_F169M", "UVIT_F172M", "UVIT_F242W", "UVIT_N219M", "UVIT_N245M",
        "UVIT_N263M", "UVIT_N279N"
    ],
    "VISTA": ["VISTA_Z", "VISTA_Y", "VISTA_J", "VISTA_H", "VISTA_Ks"],
    "WISE": ["WISE_W1", "WISE_W2", "WISE_W3", "WISE_W4"],
    "WashDDOuvby": [
        "Washington_C", "Washington_M", "Washington_T1", "Washington_T2", "DDO51_vac", "DDO51_f31",
        "Stromgren_u", "Stromgren_v", "Stromgren_b", "Stromgren_y"
    ],
}

# filter sets that were renamed between MIST v1 and v2
MIST_FILTER_SET_RENAMES_V1_TO_V2 = {"CFHT": "CFHTugriz", "HST_ACSWF": "HST_ACS_WFC", "HST_ACSHR": "HST_ACS_HRC"}

# filters that were renamed between MIST v1 and v2 (JWST gained a NIRCAM_ prefix, IRAC lost its decimal points)
MIST_FILTER_RENAMES_V1_TO_V2 = {
    **{f: f"NIRCAM_{f}" for f in MIST_FILTER_SETS_V1["JWST"]},
    # MIST v2 labels these NIRCam filters differently to v1 (the columns are in the same position)
    "F140M": "NIRCAM_F140W", "F470N": "NIRCAM_F470W", "F480M": "NIRCAM_F480W",
    **{f: f.replace(".", "") for f in MIST_FILTER_SETS_V1["SPITZER"]},
    "WFC3vIR_F126N": "WFC3_IR_F126N",
}


class MISTFilterError(KeyError):
    """Raised when a filter or filter set is not available in MIST (message printed without escaping)"""
    def __str__(self):
        return str(self.args[0])


def list_filter_sets():
    """Print the names of the available MIST filter sets

    Use :func:`list_filters` to see the filters available in each set.
    """
    print("Available MIST filter sets:")
    for filter_set in MIST_FILTER_SETS:
        print(f"    {filter_set} ({len(MIST_FILTER_SETS[filter_set])} filters)")


def list_filters(filter_set="all", width=100):
    """Print the filters available in one or more MIST filter sets

    Parameters
    ----------
    filter_set : `str` or `list` of `str`, optional
        Name of a filter set, a list of filter set names, or "all" for every set, by default "all".
        Use :func:`list_filter_sets` to see the available sets.
    width : `int`, optional
        Maximum line width of the printed output, by default 100

    Raises
    ------
    MISTFilterError
        If any of the requested filter sets does not exist (a subclass of KeyError)
    """
    if isinstance(filter_set, str):
        filter_sets = list(MIST_FILTER_SETS) if filter_set == "all" else [filter_set]
    else:
        filter_sets = list(filter_set)

    unknown = [fs for fs in filter_sets if fs not in MIST_FILTER_SETS]
    if unknown:
        msg = f"Unknown MIST filter set(s): {unknown}."
        for fs in unknown:
            if fs in MIST_FILTER_SET_RENAMES_V1_TO_V2:
                msg += (f" '{fs}' is a MIST v1 filter set name, the MIST v2 equivalent is "
                        f"'{MIST_FILTER_SET_RENAMES_V1_TO_V2[fs]}'.")
        raise MISTFilterError(msg + " Use `cogsworth.obs.mist.list_filter_sets()` to see the available sets.")

    for fs in filter_sets:
        filters = MIST_FILTER_SETS[fs]
        print(f"{fs} ({len(filters)} filters)")

        # arrange the filters in aligned columns, indented under the heading
        col_width = max(len(f) for f in filters) + 2
        n_cols = max(1, (width - 4) // col_width)
        for i in range(0, len(filters), n_cols):
            print("    " + "".join(f.ljust(col_width) for f in filters[i:i + n_cols]).rstrip())
        print()


def check_filters(filters):
    """Check that each filter is available in one of the MIST filter sets

    Parameters
    ----------
    filters : `list` of `str`
        Filters to check (e.g. ["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"])

    Raises
    ------
    MISTFilterError
        If any filter is not found (a subclass of KeyError), with suggestions for similar filters
    """
    if isinstance(filters, str):
        filters = [filters]

    all_filters = [f for fs in MIST_FILTER_SETS.values() for f in fs]
    unknown = [f for f in filters if f not in all_filters]
    if not unknown:
        return

    lines = ["The following filter(s) are not available in any MIST filter set:"]
    for f in unknown:
        if f in MIST_FILTER_SETS or f in MIST_FILTER_SET_RENAMES_V1_TO_V2:
            lines.append(f"    '{f}' (this is a filter set, not a filter)")
            continue
        if f in MIST_FILTER_RENAMES_V1_TO_V2:
            lines.append(f"    '{f}' (this is a MIST v1 filter name, cogsworth now uses MIST v2 - "
                         f"the equivalent is '{MIST_FILTER_RENAMES_V1_TO_V2[f]}')")
            continue

        # prefer filters that match with an instrument prefix (e.g. F070W -> NIRCAM_F070W), then fuzzy matches
        close = [a for a in all_filters if a.lower().endswith("_" + f.lower())]
        close += [a for a in difflib.get_close_matches(f, all_filters, n=3) if a not in close]
        lines.append(f"    '{f}'" + (f" (did you mean: {', '.join(close[:5])}?)" if close else ""))
    lines.append("Use `cogsworth.obs.mist.list_filter_sets()` to see the available filter sets and "
                 "`cogsworth.obs.mist.list_filters()` to see the filters in each set.")
    raise MISTFilterError("\n".join(lines))


@dataclass
class MISTBolometricCorrectionGrid:
    """
    Download, cache, and ingest MIST bolometric correction grids.

    Parameters
    ----------
    bands : tuple[str]
        tuple of photometric bands to include (e.g. ("LSST_u", "Gaia_G_EDR3")). Use
        :func:`list_filters` to see the available bands
    cache_dir
        directory where tarballs, extracted files, and HDF5s are stored
        (default: ~/.MIST_bc_grids)
    """
    bands: tuple[str] = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
    cache_dir: Path = Path("~/.MIST_bc_grids").expanduser()
    rebuild: bool = False

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        check_filters(self.bands)

        # find all of the necessary filter sets
        needed_filter_sets = set()
        for band in self.bands:
            for filter_set in MIST_FILTER_SETS:
                if band in MIST_FILTER_SETS[filter_set]:
                    needed_filter_sets.add(filter_set)
                    break

        self.needed_filter_sets = needed_filter_sets

        # load all necessary filter sets and concatenate, keeping only requested bands
        dfs = [self.load_hdf5(filter_set) for filter_set in self.needed_filter_sets]
        df_cols = ["Rv", *self.bands]
        bc_grid = pd.concat(dfs, axis=1, copy=False)[df_cols]
        self.bc_grid = bc_grid.loc[:, ~bc_grid.columns.duplicated()]

        self._build_interpolators()


    def download_filter_set(self, filter_set: str) -> Path:
        """
        Download the MIST BC tarball for a given filter set (e.g. 'LSST').
        """
        assert check_dependencies("requests")
        import requests

        tarball_path = self.cache_dir / f"{filter_set}.txz"

        if tarball_path.exists() and not self.rebuild:
            return tarball_path

        url = f"https://mist.science/BC_tables/v2/{filter_set}.txz"

        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tarball_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        return tarball_path

    def extract_filter_set(self, filter_set: str) -> Path:
        """
        Extract a downloaded tarball into a subdirectory of cache_dir.
        """
        extract_dir = self.cache_dir / filter_set

        if not self.rebuild and extract_dir.exists() and any(extract_dir.iterdir()):
            return extract_dir

        # remove stale files before re-extracting so old and new format files don't mix
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)

        tarball_path = self.download_filter_set(filter_set)

        with tarfile.open(tarball_path, mode="r:*") as tf:
            tf.extractall(path=extract_dir)

        return extract_dir

    def _iter_data_files(self, folder: Path):
        for p in folder.rglob("*"):
            if p.is_file() and not p.name.startswith("."):
                yield p

    def read_filter_set(self, filter_set: str) -> pd.DataFrame:
        """
        Read all BC files for a filter set and concatenate into one DataFrame.
        """
        extract_dir = self.extract_filter_set(filter_set)

        dfs: list[pd.DataFrame] = []

        for fp in sorted(self._iter_data_files(extract_dir)):
            # find the header line (starts with # and contains the first column name)
            with open(fp, "r") as f:
                for line in f:
                    stripped = line.strip().lstrip("#").strip()
                    if stripped.startswith("lgTef"):
                        header_line = stripped
                        break
            names = [s.replace("[Fe/H]", "feh").replace("Fe_H", "feh")
                     for s in header_line.split()]
            df = pd.read_csv(
                fp,
                sep="\\s+",
                comment="#",
                header=None,
                engine="python",
                names=names,
            )

            # convert from log10(Teff) to Teff and drop the lgTef column
            df["Teff"] = 10**df["lgTef"]
            df.drop(columns="lgTef", inplace=True)

            dfs.append(df)

        if not dfs:     # pragma: no cover
            raise FileNotFoundError(f"no BC files found in {extract_dir}")

        df = pd.concat(dfs, copy=False)

        # new MIST format includes an a_Fe (alpha enhancement) column; keep only solar alpha (0.0)
        if "a_Fe" in df.columns:
            df = df[df["a_Fe"] == 0.0].drop(columns="a_Fe")

        df.set_index(["Teff", "logg", "feh", "Av"], inplace=True)
        return df

    def build_hdf5(self, filter_set: str) -> Path:
        """
        Build (or rebuild) a single HDF5 file for a filter set.
        """
        h5_path = self.cache_dir / f"{filter_set}.h5"

        if h5_path.exists() and not self.rebuild:
            return h5_path

        df = self.read_filter_set(filter_set)
        df.to_hdf(
            h5_path,
            key="bc",
            mode="w",
        )

        return h5_path

    def load_hdf5(self, filter_set: str) -> pd.DataFrame:
        """
        Load a previously-built HDF5 BC grid.
        """
        h5_path = self.cache_dir / f"{filter_set}.h5"
        if not h5_path.exists() or self.rebuild:
            self.build_hdf5(filter_set)
        return pd.read_hdf(h5_path, key="bc")
    
    def _build_interpolators(self) -> None:
        df = self.bc_grid.sort_index()

        teff = np.asarray(df.index.get_level_values("Teff").unique(), dtype=float)
        logg = np.asarray(df.index.get_level_values("logg").unique(), dtype=float)
        feh = np.asarray(df.index.get_level_values("feh").unique(), dtype=float)
        av = np.asarray(df.index.get_level_values("Av").unique(), dtype=float)

        teff.sort()
        logg.sort()
        feh.sort()
        av.sort()

        full_index = pd.MultiIndex.from_product(
            [teff, logg, feh, av],
            names=["Teff", "logg", "feh", "Av"],
        )
        dense = df.reindex(full_index)

        self._grid_axes = (teff, logg, feh, av)
        self._interpolators: dict[str, RegularGridInterpolator] = {}

        n_teff, n_logg, n_feh, n_av = len(teff), len(logg), len(feh), len(av)

        for band in self.bands:
            values_1d = dense[band].to_numpy(dtype=float, copy=False)
            values_4d = values_1d.reshape(n_teff, n_logg, n_feh, n_av)

            self._interpolators[band] = RegularGridInterpolator(
                self._grid_axes,
                values_4d,
                method="linear",
            )


    def interp(
        self,
        teff: float | np.ndarray,
        logg: float | np.ndarray,
        feh: float | np.ndarray,
        av: float | np.ndarray,
        bands: tuple[str, ...] | None = None,
        silence_bounds_warning: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Interpolate BCs at (Teff, logg, feh, Av) in that order.

        Returns
        -------
        - Series if all inputs are scalar
        - DataFrame if any input is array-like (one row per broadcasted point)
        """
        use_bands = self.bands if bands is None else bands
        missing = [b for b in use_bands if b not in self._interpolators]
        if missing:
            check_filters(missing)
            raise KeyError(f"band(s) {missing} not loaded in this grid, include them in `bands` when "
                           "creating the MISTBolometricCorrectionGrid")

        teff_a = np.asarray(teff, dtype=float)
        logg_a = np.asarray(logg, dtype=float)
        feh_a = np.asarray(feh, dtype=float)
        av_a = np.asarray(av, dtype=float)

        teff_b, logg_b, feh_b, av_b = np.broadcast_arrays(teff_a, logg_a, feh_a, av_a)
        n = teff_b.size

        # warn the user if any points are out of bounds
        teff_min, teff_max = self._grid_axes[0][0], self._grid_axes[0][-1]
        logg_min, logg_max = self._grid_axes[1][0], self._grid_axes[1][-1]
        feh_min, feh_max = self._grid_axes[2][0], self._grid_axes[2][-1]
        av_min, av_max = self._grid_axes[3][0], self._grid_axes[3][-1]

        if not silence_bounds_warning:
            for var, label, min_val, max_val in [
                (teff_b, "Teff", teff_min, teff_max),
                (logg_b, "logg", logg_min, logg_max),
                (feh_b, "feh", feh_min, feh_max),
                (av_b, "Av", av_min, av_max),
            ]:
                n_out_of_bounds = ((var < min_val) | (var > max_val)).sum()
                if n_out_of_bounds > 0:
                    logging.getLogger("cogsworth").warning(
                        f"cogsworth warning: {n_out_of_bounds} out of bounds points for {label} when "
                        f"interpolating MIST BCs (valid range: {min_val} to {max_val}). Clipping to bounds."
                    )

        teff_b = np.clip(teff_b, teff_min, teff_max)
        logg_b = np.clip(logg_b, logg_min, logg_max)
        feh_b = np.clip(feh_b, feh_min, feh_max)
        av_b = np.clip(av_b, av_min, av_max)
                    
        pts = np.column_stack([
            teff_b.reshape(n),
            logg_b.reshape(n),
            feh_b.reshape(n),
            av_b.reshape(n),
        ])

        out = {b: self._interpolators[b](pts) for b in use_bands}

        # scalar -> Series
        if teff_b.shape == () and logg_b.shape == () and feh_b.shape == () and av_b.shape == ():
            return pd.Series({b: float(out[b][0]) for b in use_bands})

        # vectorised -> DataFrame
        return pd.DataFrame(out)
