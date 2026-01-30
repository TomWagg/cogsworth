from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tarfile
import logging

import requests
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import numpy as np


__all__ = ["MISTBolometricCorrectionGrid"]


# this is a mapping of photometric systems to the bands available in MIST BC tables
MIST_FILTER_SETS = {
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


@dataclass
class MISTBolometricCorrectionGrid:
    """
    Download, cache, and ingest MIST bolometric correction grids.

    Parameters
    ----------
    bands : tuple[str]
        tuple of photometric bands to include (e.g. ("LSST_u", "BP", "G", "RP"))
    cache_dir
        directory where tarballs, extracted files, and HDF5s are stored
        (default: ~/.MIST_bc_grids)
    """
    bands: tuple[str] = ("G", "BP", "RP")
    cache_dir: Path = Path("~/.MIST_bc_grids").expanduser()
    rebuild: bool = False

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # find all of the necessary filter sets
        needed_filter_sets = set()
        for band in self.bands:
            found_it = False
            for filter_set in MIST_FILTER_SETS:
                if band in MIST_FILTER_SETS[filter_set]:
                    needed_filter_sets.add(filter_set)
                    found_it = True
                    break
        
            if not found_it:
                raise KeyError(f"band '{band}' not found in any MIST filter set")

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
        tarball_path = self.cache_dir / f"{filter_set}.txz"

        if tarball_path.exists() and not self.rebuild:
            return tarball_path

        url = f"https://waps.cfa.harvard.edu/MIST/BC_tables/{filter_set}.txz"

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
        extract_dir.mkdir(parents=True, exist_ok=True)

        if not self.rebuild and any(extract_dir.iterdir()):
            return extract_dir

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
            # read the header row (row 5 from the file)
            with open(fp, "r") as f:
                for _ in range(5 + 1):
                    header_line = f.readline()
            names = [s.replace("[Fe/H]", "feh") for s in header_line.strip().lstrip("#").split()]
            df = pd.read_csv(
                fp,
                sep="\\s+",
                comment="#",
                header=None,
                engine="python",
                names=names,
            )
            dfs.append(df)

        if not dfs:     # pragma: no cover
            raise FileNotFoundError(f"no BC files found in {extract_dir}")

        df = pd.concat(dfs, copy=False)
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
            return pd.Series({b: float(out[b]) for b in use_bands})

        # vectorised -> DataFrame
        return pd.DataFrame(out)
