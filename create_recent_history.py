import kicker
import astropy.units as u
import numpy as np


class RecentHistory(kicker.galaxy.Frankel2018):
    def __init__(self, components=["low_alpha_disc"], component_masses=[1], **kwargs):
        super().__init__(components=components, component_masses=component_masses, **kwargs)

    def draw_lookback_times(self, size=None, component="low_alpha_disc"):
        if component != "low_alpha_disc":
            raise NotImplementedError()

        U = np.random.rand(size)
        norm = 1 / (self.tsfr * np.exp(-self.galaxy_age / self.tsfr) * (np.exp(200 * u.Myr / self.tsfr) - 1))
        tau = self.tsfr * np.log((U * np.exp(self.galaxy_age / self.tsfr)) / (norm * self.tsfr) + 1)

        return tau


p = kicker.pop.Population(100_000, processes=6, m1_cutoff=0, galaxy_model=RecentHistory,
                          max_ev_time=200 * u.Myr, timestep_size=0.5 * u.Myr, BSE_settings={"binfrac": 1.0},
                          store_entire_orbits=False)

p.create_population()
p.save("data/recent-pop", overwrite=True)
