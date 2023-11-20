import cogsworth
import astropy.units as u
import numpy as np


class RecentHistory(cogsworth.galaxy.Wagg2022):
    def __init__(self, components=["low_alpha_disc"], component_masses=[1], **kwargs):
        super().__init__(components=components, component_masses=component_masses, **kwargs)

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        U = np.random.rand(size)
        norm = 1 / (self.tsfr * np.exp(-self.galaxy_age / self.tsfr) * (np.exp(200 * u.Myr / self.tsfr) - 1))
        tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr) + 1)

        return tau


p = cogsworth.pop.Population(10_000, processes=40, m1_cutoff=7, galaxy_model=RecentHistory,
                          max_ev_time=200 * u.Myr, timestep_size=0.5 * u.Myr, BSE_settings={"binfrac": 1.0,
                                                                                            'sigma': 265.0},
                          store_entire_orbits=False)

p.create_population()
p.save("/epyc/ssd/users/tomwagg/pops/recent-pop-265", overwrite=True)

p = cogsworth.pop.Population(10_000, processes=40, m1_cutoff=7, galaxy_model=RecentHistory,
                          max_ev_time=200 * u.Myr, timestep_size=0.5 * u.Myr, BSE_settings={"binfrac": 1.0,
                                                                                            'sigma': 30.0},
                          store_entire_orbits=False)

p.create_population()
p.save("/epyc/ssd/users/tomwagg/pops/recent-pop-30", overwrite=True)
