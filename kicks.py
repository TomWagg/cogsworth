import gala as ga
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from scipy.stats import maxwell


class Kick():
    """Represents a supernova kick: magnitudes of kicks in particular directions"""

    def __init__(self, magnitude, phi, theta, t):
        """
        Parameters
        ----------
        magnitude : `float/array`
            Magnitude(s) of kicks corresponding to each object
        phi : `float/array`
            Magnitude(s) of kicks corresponding to each object
        theta : `float/array`
            Magnitude(s) of kicks corresponding to each object
        t : `float`
            Time at which the kick(s) occurred
        """
        self.magnitude = magnitude if isinstance(magnitude, u.quantity.Quantity) else magnitude * u.km / u.s
        self.phi = phi if isinstance(phi, u.quantity.Quantity) else phi * u.rad
        self.theta = theta if isinstance(theta, u.quantity.Quantity) else theta * u.rad
        self.t = t if isinstance(t, u.quantity.Quantity) else t * u.Gyr

    def __getitem__(self, slice_):
        if isinstance(slice_, np.ndarray) or isinstance(slice_, list):
            slice_ = (slice_,)

        try:
            slice_ = tuple(slice_)
        except TypeError:
            slice_ = (slice_,)

        return self.__class__(magnitude=self.magnitude[slice_],
                              phi=self.phi[slice_],
                              theta=self.theta[slice_],
                              t=self.t)

    def __repr__(self):
        if isinstance(self.magnitude.value, float):
            return "<Kick, magnitude={:1.2f}, phi={:1.2f}, theta={:1.2f}, time={:1.2f}>".format(self.magnitude, self.phi, self.theta, self.t)
        else:
            return "<Kick, n_objects={}>".format(len(self.magnitude))


def integrate_orbits_with_kicks(w0, potential=ga.potential.MilkyWayPotential(), kicks=None, kick_times=None,
                                maxwell_sigma=265 * u.km / u.s, same_angle=False, ret_kicks=False,
                                rng=np.random.default_rng(), **integrate_kwargs):
    """Integrate PhaseSpacePosition in a potential with kicks that occur at certain times

    Parameters
    ----------
    potential : `ga.potential.PotentialBase`
        Potential in which you which to integrate the orbits
    w0 : `ga.dynamics.PhaseSpacePosition`, optional
        Initial phase space position, by default the MilkyWayPotential()
    kicks : `list`, optional
        List of None, or list of kick magnitudes or list of tuples with kick magnitudes and angles,
        by default None
    kick_times : `list`, optional
        Times at which kicks occur, by default None
    maxwell_sigma : `float`, optional
        Sigma to use for the maxwellian for kick magnitudes, by default 265 km/s
    same_angle : `boolean`, optional
        Whether to use the same random kick angle for each individual orbit if several are provided,
        by default False
    ret_kicks : `boolean`, optional
        Whether to return the kicks that were used in the evolution in addition to the orbits,
        by default False
    rng : `NumPy RandomNumberGenerator`
        Which random number generator to use

    Returns
    -------
    full_orbits : `ga.orbit.Orbit`
        Orbits that have been integrated
    """
    # if there are no kicks then just integrate the whole thing
    if kicks is None and kick_times is None:
        return potential.integrate_orbit(w0, **integrate_kwargs)

    # otherwise make sure that both are there
    elif kick_times is None:
        raise ValueError("Kick times must be specified if kicks are used")

    # integrate using the kicks
    else:
        # create a list of None is nothing is given
        if kicks is None:
            kicks = [None for _ in range(len(kick_times))]

        # work out what the timesteps would be without kicks
        timesteps = ga.integrate.parse_time_specification(units=[u.s], **integrate_kwargs) * u.s

        # start the cursor at the smallest timestep
        time_cursor = timesteps[0]
        current_w0 = w0

        if ret_kicks:
            drawn_kicks = []

        # keep track of the orbit data throughout
        data = []
        for kick, kick_time in zip(kicks, kick_times):
            # find the timesteps that occur before the kick
            matching_timesteps = timesteps[np.logical_and(timesteps >= time_cursor, timesteps < kick_time)]

            # integrate the orbit over these timesteps
            orbits = potential.integrate_orbit(current_w0, t=matching_timesteps)

            # save the orbit data
            data.append(orbits.data)

            # adjust the time
            time_cursor = kick_time

            # get new PhaseSpacePosition(s)
            current_w0 = orbits[-1]

            if isinstance(kick, tuple):
                magnitude, phi, theta = kick
            else:
                # if there's only one orbit
                if current_w0.shape == ():
                    magnitude = kick if kick is not None\
                        else maxwell(scale=maxwell_sigma).rvs() * maxwell_sigma.unit
                    phi = rng.uniform(0, 2 * np.pi) * u.rad
                    theta = rng.uniform(-np.pi / 2, np.pi / 2) * u.rad
                else:
                    magnitude = kick if kick is not None else\
                        maxwell(scale=maxwell_sigma).rvs(current_w0.shape[0]) * maxwell_sigma.unit

                    if same_angle:
                        phi_0 = rng.uniform(0, 2 * np.pi)
                        theta_0 = rng.uniform(-np.pi / 2, np.pi / 2)
                        phi = np.repeat(phi_0, repeats=current_w0.shape[0]) * u.rad
                        theta = np.repeat(theta_0, repeats=current_w0.shape[0]) * u.rad
                    else:
                        phi = rng.uniform(0, 2 * np.pi, size=current_w0.shape[0]) * u.rad
                        theta = rng.uniform(-np.pi / 2, np.pi / 2, size=current_w0.shape[0]) * u.rad

            if ret_kicks:
                drawn_kicks.append(Kick(magnitude=magnitude, phi=phi, theta=theta, t=kick_time))

            d_x = magnitude * np.cos(phi) * np.sin(theta)
            d_y = magnitude * np.sin(phi) * np.sin(theta)
            d_z = magnitude * np.cos(theta)

            kick_differential = coord.CartesianDifferential(d_x, d_y, d_z)

            current_w0 = ga.dynamics.PhaseSpacePosition(pos=current_w0.pos,
                                                        vel=current_w0.vel + kick_differential,
                                                        frame=current_w0.frame)

        if time_cursor < timesteps[-1]:
            matching_timesteps = timesteps[timesteps >= time_cursor]
            orbits = potential.integrate_orbit(current_w0, t=matching_timesteps)
            data.append(orbits.data)

        data = coord.concatenate_representations(data)
        full_orbits = ga.dynamics.orbit.Orbit(pos=data.without_differentials(),
                                              vel=data.differentials["s"],
                                              t=timesteps.to(u.Myr))

        if ret_kicks:
            return full_orbits, drawn_kicks if len(drawn_kicks) > 1 else drawn_kicks[0]
        else:
            return full_orbits
