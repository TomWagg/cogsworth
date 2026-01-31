import numpy as np
import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp

import astropy.coordinates as coords
import astropy.units as u

__all__ = ["get_kick_differential", "integrate_orbit_with_events"]


def get_kick_differential(delta_v_sys_xyz, phase=None, inclination=None):
    """Calculate the :class:`~astropy.coordinates.CylindricalDifferential` from a combination of the natal
    kick, Blauuw kick and orbital motion.

    Parameters
    ----------
    delta_v_sys_xyz : :class:`~astropy.units.Quantity` [velocity]
        Change in systemic velocity due to natal and Blauuw kicks in BSE :math:`(v_x, v_y, v_z)` frame
        (see Fig A1 of `Hurley+02 <https://ui.adsabs.harvard.edu/abs/2002MNRAS.329..897H/abstract>`_)
    phase : `float`
        Orbital phase angle in radians
    inclination : `float`
        Inclination to the Galactic plane in radians

    Returns
    -------
    kick_differential : :class:`~astropy.coordinates.CylindricalDifferential`
        Kick differential
    """
    # orbital phase angle and inclination to Galactic plane
    theta = np.random.uniform(0, 2 * np.pi) if phase is None else phase
    phi = np.arccos(2 * np.random.rand() - 1.0) if inclination is None else inclination

    # rotate BSE (v_x, v_y, v_z) into Galactocentric (v_X, v_Y, v_Z)
    v_X = delta_v_sys_xyz[0] * np.cos(theta) - delta_v_sys_xyz[1] * np.sin(theta) * np.cos(phi)\
        + delta_v_sys_xyz[2] * np.sin(theta) * np.sin(phi)
    v_Y = delta_v_sys_xyz[0] * np.sin(theta) + delta_v_sys_xyz[1] * np.cos(theta) * np.cos(phi)\
        - delta_v_sys_xyz[2] * np.cos(theta) * np.sin(phi)
    v_Z = delta_v_sys_xyz[1] * np.sin(phi) + delta_v_sys_xyz[2] * np.cos(phi)

    kick_differential = coords.CartesianDifferential(v_X, v_Y, v_Z)

    return kick_differential


def integrate_orbit_with_events(w0, t1, t2, dt, potential=gp.MilkyWayPotential(version='v2'), events=None, 
                                store_all=True, quiet=False, 
                                integrator=gi.DOPRI853Integrator, integrator_kwargs={}):
    """Integrate :class:`~gala.dynamics.PhaseSpacePosition` in a 
    :class:`Potential <gala.potential.potential.PotentialBase>` with events that occur at certain times

    Parameters
    ----------
    w0 : :class:`~gala.dynamics.PhaseSpacePosition`
        Initial phase space position
    t1 : :class:`~astropy.units.Quantity` [time]
        Integration start time
    t2 : :class:`~astropy.units.Quantity` [time]
        Integration end time
    dt : :class:`~astropy.units.Quantity` [time]
        Integration initial timestep size (integrator may adapt timesteps)
    potential : :class:`Potential <gala.potential.potential.PotentialBase>`, optional
        Potential in which you which to integrate the orbits, by default the
        :class:`~gala.potential.potential.MilkyWayPotential`
    events : `varies`
        Events that occur during the orbit evolution (such as supernova resulting in kicks). If no events
        occur then set `events=None` (this will result in a simple call to `potential.integrate_orbit`). If
        this is a disrupted binary then supply a list of 2 lists of events. Each event list should contain
        the following parameters: `time`, `m_1`, `m_2`, `a`, `ecc`, `delta_v_sys_xyz` (and will be passed to
        `get_kick_differential`).
    store_all : `bool`, optional
        Whether to store the entire orbit, by default True. If not then only the final
        PhaseSpacePosition will be stored - this cuts down on memory usage.
    quiet : `bool`, optional
        Whether to silence warning messages about failing orbits
    integrator : :class:`~gala.integrate.Integrator`, optional
        The integrator used by gala for evolving the orbits of binaries in the galactic potential

    Returns
    -------
    full_orbit : :class:`~gala.dynamics.Orbit`
        Integrated orbit. If a disrupted binary with two event lists was supplied then two orbit classes will
        be returned. If the orbit integration failed for any reason then None is returned.
    """
    # ensure timestep isn't larger than integration time
    dt = min(dt, t2 - t1)

    # if there are no events then just integrate the whole thing
    if events is None:
        try:
            full_orbit = potential.integrate_orbit(
                w0, t1=t1, t2=t2, dt=dt, Integrator=integrator, Integrator_kwargs=integrator_kwargs
            )
        except RuntimeError:            # pragma: no cover
            return None
        # jettison everything but the final timestep if user says so
        if not store_all:
            full_orbit = full_orbit[-1:]
        return full_orbit

    # allow two retries with smaller timesteps
    MAX_DT_RESIZE = 2
    for n in range(MAX_DT_RESIZE):
        try:
            success = False

            # work out what the timesteps would be without kicks
            timesteps = gi.parse_time_specification(units=[u.s], t1=t1, t2=t2, dt=dt) * u.s

            # start the cursor at the first timestep
            time_cursor = timesteps[0]
            current_w0 = w0

            # keep track of the orbit data throughout
            orbit_data = []

            # loop over the events
            for event in events:
                # find the timesteps that occur before the kick
                timestep_mask = (timesteps >= time_cursor) & (timesteps < (t1 + event["time"]))

                # integrate up to the moment of the event (if there are any timesteps before it)
                if any(timestep_mask):
                    matching_timesteps = timesteps[timestep_mask]

                    # integrate the orbit over these timesteps
                    orbit = potential.integrate_orbit(current_w0, t=matching_timesteps,
                                                      Integrator=integrator,
                                                      Integrator_kwargs=integrator_kwargs)

                    # save the orbit data (minus the last timestep to avoid duplicates)
                    orbit_data.append(orbit.data[:-1])

                    # set new PhaseSpacePosition from the last timestep
                    current_w0 = orbit[-1]

                    # adjust the time to the last timestep
                    time_cursor = matching_timesteps[-1]
                else:           # pragma: no cover
                    # otherwise skip forward to the event
                    time_cursor = t1 + event["time"]

                # calculate the kick differential
                kick_differential = get_kick_differential(delta_v_sys_xyz=event["delta_v_sys_xyz"],
                                                          phase=event["phase"], inclination=event["inc"])

                # update the velocity of the current PhaseSpacePosition
                current_w0 = gd.PhaseSpacePosition(pos=current_w0.pos,
                                                   vel=current_w0.vel + kick_differential,
                                                   frame=current_w0.frame)

            # if we still have time left after the last event (very likely)
            if time_cursor < timesteps[-1]:
                # evolve the rest of the orbit out
                matching_timesteps = timesteps[timesteps >= time_cursor]
                orbit = potential.integrate_orbit(current_w0, t=matching_timesteps,
                                                  Integrator=integrator,
                                                  Integrator_kwargs=integrator_kwargs)
                orbit_data.append(orbit.data)

            data = coords.concatenate_representations(orbit_data) if len(orbit_data) > 1 else orbit_data[0]

            full_orbit = gd.orbit.Orbit(pos=data.without_differentials(),
                                        vel=data.differentials["s"],
                                        t=timesteps.to(u.Myr))
            success = True
            break

        except Exception:   # pragma: no cover
            dt /= 8.
            # if not quiet:
            #     print("Orbit is causing problems, attempting reduced timestep size", t1, dt)

    # if the orbit failed event after resizing then just return None
    if not success:   # pragma: no cover
        # if not quiet:
        #     print("ORBIT FAILED, returning None")
        return None

    # jettison everything but the final timestep if user says so
    if not store_all:
        full_orbit = full_orbit[-1:]

    return full_orbit
