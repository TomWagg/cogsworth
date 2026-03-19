#!/usr/bin/env python
"""
cogsworth job worker — processes evolution jobs from the queue one at a time.

Usage:
    python webapp/worker.py            # normal operation
    python webapp/worker.py --reset-only   # reset stale jobs then exit
"""

import json
import logging
import os
import sys
import time
import traceback

# Make webapp/ importable when run from the repo root
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from astropy import units as u

import cogsworth.pop
import cogsworth.sfh as sfh

from queue_db import (
    deserialize_params,
    get_next_pending,
    set_done,
    set_failed,
    set_running,
    cleanup_old_results,
    reset_stale_running,
    RESULTS_DIR,
)
from potentials import POTENTIAL_CONSTRUCTORS

# ── Locked to 2 processes ──────────────────────
PROCESSES = 2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("cogsworth-worker")

SFH_MODEL_MAP = {
    "Wagg+2022 (default)": sfh.Wagg2022,
    "Burst Uniform Disc": sfh.BurstUniformDisc,
    "Constant Uniform Disc": sfh.ConstantUniformDisc,
    "Constant Plummer Sphere": sfh.ConstantPlummerSphere,
    "Sanders & Binney 2015": sfh.SandersBinney2015,
    "Carina Dwarf": sfh.CarinaDwarf,
}


# ──────────────────────────────────────────────
# Evolution helpers
# ──────────────────────────────────────────────

def run_population(sfh_name, sfh_params_dict, n_binaries, max_ev_time,
                   final_kstar1, final_kstar2, potential_name,
                   v_dispersion, evolve_orbits,
                   potential_params=None, bse_overrides=None,
                   full_bcm=False, sampling_params=None):
    pot_params = potential_params or {}
    bse_overrides = bse_overrides or {}
    sampling_params = sampling_params or {}
    bfm = sampling_params.get("binfrac_model", 0.5)
    pot = POTENTIAL_CONSTRUCTORS[potential_name](**pot_params)
    pop = cogsworth.pop.Population(
        n_binaries=n_binaries,
        sfh_model=SFH_MODEL_MAP[sfh_name],
        sfh_params=sfh_params_dict,
        galactic_potential=pot,
        max_ev_time=max_ev_time * u.Gyr,
        final_kstar1=final_kstar1,
        final_kstar2=final_kstar2,
        v_dispersion=v_dispersion * u.km / u.s,
        store_entire_orbits=evolve_orbits,
        processes=PROCESSES,
        BSE_settings={**bse_overrides},
        use_default_BSE_settings=True,
        error_file_path=None,               # don't write errors to disk
        bcm_default_timestep=0.0 if full_bcm else None,
        sampling_params=sampling_params,
    )
    pop.sample_initial_galaxy()
    pop.sample_initial_binaries()
    pop.perform_stellar_evolution()
    if evolve_orbits:
        pop.perform_galactic_evolution()
    return pop


def run_single_binary(mass_1, mass_ratio, porb, ecc, metallicity,
                      x, y, z, v_R, v_T, v_z, tau,
                      potential_name, evolve_orbits,
                      potential_params=None, bse_overrides=None):
    pot_params = potential_params or {}
    bse_overrides = bse_overrides or {}
    pot = POTENTIAL_CONSTRUCTORS[potential_name](**pot_params)
    mass_2 = mass_ratio * mass_1

    pop = cogsworth.pop.Population(
        n_binaries=1,
        sfh_model=sfh.Wagg2022,
        galactic_potential=pot,
        max_ev_time=tau * u.Gyr,
        final_kstar1=list(range(16)),
        final_kstar2=list(range(16)),
        store_entire_orbits=evolve_orbits,
        processes=1,
        BSE_settings={"binfrac": 1.0, "dtp": 0.0, **bse_overrides},
        use_default_BSE_settings=True,
        bcm_default_timestep=0.0
    )

    pop.sample_initial_galaxy()
    ig = pop._initial_galaxy
    ig._tau = np.array([tau]) * u.Gyr
    ig._Z = np.array([metallicity])
    ig._x = np.array([x]) * u.kpc
    ig._y = np.array([y]) * u.kpc
    ig._z = np.array([z]) * u.kpc
    ig._which_comp = np.array([ig._which_comp[0]])
    ig.v_R = np.array([v_R]) * u.km / u.s
    ig.v_T = np.array([v_T]) * u.km / u.s
    ig.v_z = np.array([v_z]) * u.km / u.s

    pop.sample_initial_binaries()

    ib = pop._initial_binaries
    idx = ib.index[0]
    for col in ["mass_1", "mass0_1"]:
        if col in ib.columns:
            ib.at[idx, col] = mass_1
    for col in ["mass_2", "mass0_2"]:
        if col in ib.columns:
            ib.at[idx, col] = mass_2
    ib.at[idx, "porb"] = porb
    ib.at[idx, "ecc"] = ecc
    if "metallicity" in ib.columns:
        ib.at[idx, "metallicity"] = metallicity

    pop.perform_stellar_evolution()
    if evolve_orbits:
        pop.perform_galactic_evolution()
    
    return pop


# ──────────────────────────────────────────────
# Job dispatcher
# ──────────────────────────────────────────────

def execute_job(job: dict) -> str:
    """Run one job, save results, return result path."""
    params = deserialize_params(json.loads(job["params_json"]))
    job_type = job["job_type"]
    result_path = os.path.join(RESULTS_DIR, f"{job['id']}.h5")

    if job_type == "population":
        pop = run_population(**params)
    elif job_type == "single_binary":
        pop = run_single_binary(**params)
    else:
        raise ValueError(f"Unknown job type: {job_type!r}")

    pop.save(result_path)
    return result_path


# ──────────────────────────────────────────────
# Main loop
# ──────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    reset_stale_running()
    log.info(f"Worker started (processes={PROCESSES})")

    while True:
        cleanup_old_results()
        job = get_next_pending()

        if job is None:
            time.sleep(1)
            continue

        log.info(f"Starting job {job['id']}  type={job['job_type']}")
        set_running(job["id"])

        try:
            result_path = execute_job(job)
            set_done(job["id"], result_path)
            log.info(f"Job {job['id']} done → {result_path}")
        except Exception:
            error = traceback.format_exc()
            set_failed(job["id"], error)
            log.error(f"Job {job['id']} failed:\n{error}")


if __name__ == "__main__":
    if "--reset-only" in sys.argv:
        reset_stale_running()
        log.info("Reset stale running jobs, exiting.")
        sys.exit(0)
    main()
