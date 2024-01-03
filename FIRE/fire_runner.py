import argparse
import numpy as np

import gala.potential as gp

import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
import pandas as pd

import cogsworth
from cosmic.sample.initialbinarytable import InitialBinaryTable

import sys
sys.path.append("/mnt/home/twagg/cogsworth/FIRE/helpers")

import tomFIRE


class FIREPop(cogsworth.pop.Population):
    def __init__(self, star_particles=None, particle_size=1 * u.pc, particle_boundedness=1.0,
                 sampling_params={"sampling_target": "total_mass",
                                  "trim_extra_samples": True,
                                  "keep_singles": True,}, **kwargs):
        self.star_particles = star_particles
        self.particle_size = particle_size
        self.particle_boundedness = particle_boundedness
        if "n_binaries" not in kwargs:
            kwargs["n_binaries"] = None
        super().__init__(sampling_params=sampling_params, **kwargs)

    def __getitem__(self, ind):
        if self._initC is not None and "particle_id" not in self._initC.columns:
            self._initC["particle_id"] = self._initial_binaries["particle_id"]
        new_pop = super().__getitem__(ind)
        new_pop.star_particles = self.star_particles
        new_pop.particle_size = self.particle_size
        new_pop.particle_boundedness = self.particle_boundedness
        return new_pop
        
    
    def sample_initial_binaries(self):
        assert self.star_particles is not None,\
            "`self.star_particles` is None, must provide star particles to sample from"
        initial_binaries_list = [None for _ in range(len(self.star_particles))]
        self._mass_singles, self._mass_binaries, self._n_singles_req, self._n_bin_req = 0.0, 0.0, 0, 0

        i = 0
        for id, particle in self.star_particles.iterrows():
            samples = InitialBinaryTable.sampler('independent', np.linspace(0, 15, 16), np.linspace(0, 15, 16),
                                                binfrac_model=self.BSE_settings["binfrac"],
                                                SF_start=self.max_ev_time.to(u.Myr).value - particle["t_form"] * 1000,
                                                SF_duration=0.0, met=particle["Z"],
                                                total_mass=particle["mass"],
                                                size=particle["mass"] * 0.8,
                                                **self.sampling_params)
        
            # apply the mass cutoff and record particle ID
            samples[0].reset_index(inplace=True)
            samples[0].drop(samples[0][samples[0]["mass_1"] < self.m1_cutoff].index, inplace=True)
            samples[0]["particle_id"] = np.repeat(id, len(samples[0]))

            # save samples
            initial_binaries_list[i] = samples[0]
            self._mass_singles += samples[1]
            self._mass_binaries += samples[2]
            self._n_singles_req += samples[3]
            self._n_bin_req += samples[4]

            i += 1

        self._initial_binaries = pd.concat(initial_binaries_list, ignore_index=True)

        self._initial_binaries = self._initial_binaries[self._initial_binaries["mass_1"] >= self.m1_cutoff]

        # same for this class
        self.n_binaries = len(self._initial_binaries)
        self.n_binaries_match = len(self._initial_binaries)

        # ensure metallicities remain in a range valid for COSMIC - original value still in initial_galaxy.Z
        self._initial_binaries.loc[self._initial_binaries["metallicity"] < 1e-4, "metallicity"] = 1e-4
        self._initial_binaries.loc[self._initial_binaries["metallicity"] > 0.03, "metallicity"] = 0.03

        self.sample_initial_galaxy()

    def sample_initial_galaxy(self):
        particles = self.star_particles.loc[self._initial_binaries["particle_id"]]

        x, y, z = np.random.normal([particles["x"].values,
                                    particles["y"].values,
                                    particles["z"].values],
                                    self.particle_size.to(u.kpc).value / np.sqrt(3),
                                    size=(3, self.n_binaries_match)) * u.kpc

        self._initial_galaxy = cogsworth.galaxy.Galaxy(self.n_binaries_match, immediately_sample=False)
        self._initial_galaxy._x = x
        self._initial_galaxy._y = y
        self._initial_galaxy._z = z
        self._initial_galaxy._tau = self._initial_binaries["tphysf"].values * u.Myr
        self._initial_galaxy._Z = self._initial_binaries["metallicity"].values
        self._initial_galaxy._which_comp = np.repeat("FIRE", len(self.initial_galaxy._tau))

        v_R = (particles["x"] * particles["v_x"] + particles["y"] * particles["v_y"])\
            / (particles["x"]**2 + particles["y"]**2)**0.5
        v_T = (particles["x"] * particles["v_y"] - particles["y"] * particles["v_x"])\
            / (particles["x"]**2 + particles["y"]**2)**0.5
        v_z = particles["v_z"]

        vel_units = u.km / u.s
        dispersion = dispersion_from_virial_parameter(self.particle_boundedness,
                                                      self.particle_size,
                                                      particles["mass"].values * u.Msun)
        v_R, v_T, v_z = np.random.normal([v_R.values, v_T.values, v_z.values], dispersion / np.sqrt(3),
                                         size=(3, self.n_binaries_match)) * vel_units
        
        self._initial_galaxy._v_R = v_R
        self._initial_galaxy._v_T = v_T
        self._initial_galaxy._v_z = v_z


def dispersion_from_virial_parameter(alpha_vir, R, M):
    return np.sqrt(alpha_vir * const.G * M / (5 * R)).to(u.km / u.s)


def run_boundedness_sim(alpha_vir, subset=None, processes=32, extra_time=200 * u.Myr, m1_cutoff=4 * u.Msun):
    """
    Runs a cogsworth simulation using the FIRE (Feedback In Realistic Environments) simulations, varying star
    particle boundedness.

    Parameters
    ----------
    alpha_vir : float
        The virial parameter used to determine star particle boundedness.
    subset : int, optional
        The number of star particles to randomly select from the dataset. If None, all are used.
    """
    # load recent stars, galactic potential, and stars at formation
    recent_stars = tomFIRE.FIRESnapshot(particle_type="star", min_t_form=13.6 * u.Gyr)
    pot = gp.load("/mnt/home/twagg/cogsworth/FIRE/m11h_potential.yml")
    stars_at_formation = pd.read_hdf("/mnt/home/twagg/cogsworth/FIRE/FIRE_star_particles.h5", key="df")

    # if subset is not None, randomly select stars from the dataset
    if subset is not None:
        stars_at_formation = stars_at_formation.iloc[np.random.choice(len(stars_at_formation), size=subset,
                                                                      replace=False)]

    # create a FIREPop object with the selected stars and other parameters
    p_fire = FIREPop(star_particles=stars_at_formation,
                     max_ev_time=recent_stars.snap_time + extra_time,
                     galactic_potential=pot,
                     m1_cutoff=4,
                     particle_boundedness=alpha_vir,
                     particle_size=1 * u.pc,
                     processes=processes)

    # sample initial binaries and perform stellar evolution
    p_fire.sample_initial_binaries()
    p_fire.perform_stellar_evolution()

    # select main sequence stars and perform galactic evolution
    ms_stars = (p_fire.final_bpp["kstar_1"] <= 1) | (p_fire.final_bpp["kstar_2"] <= 1)
    # p = p_fire[ms_stars]
    
    print(f"Sampled {len(p_fire)} systems")
    print(f"Would have trimmed to {ms_stars.sum()}")
    
    p_fire.perform_galactic_evolution()
    
    if p_fire._initC is not None and "particle_id" not in p_fire._initC.columns:
        p_fire._initC["particle_id"] = p_fire._initial_binaries["particle_id"]

    # save the results
    p_fire.save(f"/mnt/home/twagg/ceph/pops/boundedness/alpha_{alpha_vir}", overwrite=True)


def main():
    parser = argparse.ArgumentParser(description='Boundedness simulation runner')
    parser.add_argument('-a', '--alpha_vir', default=1.0, type=float,
                        help='Star particle virial parameter')
    parser.add_argument('-s', '--subset', default=None, type=int,
                        help='Size of subset of star particles to use')
    parser.add_argument('-p', '--processes', default=32, type=int,
                        help='Number of processes to use')
    parser.add_argument('-e', '--extra_time', default=200, type=int,
                        help='Extra time to evolve for (in Myr)')
    args = parser.parse_args()

    run_boundedness_sim(alpha_vir=args.alpha_vir, subset=args.subset, processes=args.processes, extra_time=args.extra_time * u.Myr)

if __name__ == "__main__":
    main()
