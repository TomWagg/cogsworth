import numpy as np
# import galax.integrate as gi
import galax.dynamics as gd
import galax.potential as gp
import galax.coordinates as gc
import jax.numpy as jnp

import astropy.coordinates as coords
import astropy.units as u
import unxt as ux
import coordinax as cx

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

    return v_X, v_Y, v_Z


def integrate_orbit_with_events(w0, t1, t2, dt, potential=gp.MilkyWayPotential2022(), events=None,
                                store_all=True, quiet=False):
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

    Returns
    -------
    full_orbit : :class:`~gala.dynamics.Orbit`
        Integrated orbit. If a disrupted binary with two event lists was supplied then two orbit classes will
        be returned. If the orbit integration failed for any reason then None is returned.
    """
    # if there are no events then just integrate the whole thing
    if events is None:
        full_orbit = gd.evaluate_orbit(potential, w0,
                                       ux.Quantity(jnp.arange(t1.value,
                                                              t2.value + dt.value, dt.value), "Myr"))
        # full_orbit = potential.integrate_orbit(w0, t1=t1, t2=t2, dt=dt, Integrator=gi.DOPRI853Integrator)
        # jettison everything but the final timestep if user says so
        if not store_all:
            full_orbit = full_orbit[-1:]
        return full_orbit

    usys = ux.unitsystem("kpc", "Msun", "Myr", "rad")

    # allow two retries with smaller timesteps
    MAX_DT_RESIZE = 2
    for n in range(MAX_DT_RESIZE):
        try:
            success = False

            # work out what the timesteps would be without kicks
            # timesteps = gi.parse_time_specification(units=[u.s], t1=t1, t2=t2, dt=dt) * u.s
            timesteps = jnp.arange(t1.to(u.Myr).value,
                                   t2.to(u.Myr).value + dt.to(u.Myr).value,
                                   dt.to(u.Myr).value) * ux.unit("Myr")

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
                    orbit = gd.evaluate_orbit(potential, current_w0, matching_timesteps)
                    # orbit = potential.integrate_orbit(current_w0, t=matching_timesteps,
                    #                                   Integrator=gi.DOPRI853Integrator)

                    # save the orbit data (minus the last timestep to avoid duplicates)
                    orbit_data.append(orbit[:-1].data)

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

                applied_kick = cx.CartesianVel3D(
                    x=kick_differential[0].to(u.km / u.s).value * ux.unit("km / s"),
                    y=kick_differential[1].to(u.km / u.s).value * ux.unit("km / s"),
                    z=kick_differential[2].to(u.km / u.s).value * ux.unit("km / s")
                )

                # update the velocity of the current PhaseSpacePosition
                current_w0 = gc.PhaseSpaceCoordinate(q=current_w0.q,
                                                     p=current_w0.p + applied_kick.uconvert(usys),
                                                     t=time_cursor,
                                                     frame=current_w0.frame)

            # if we still have time left after the last event (very likely)
            if time_cursor < timesteps[-1]:
                # evolve the rest of the orbit out
                matching_timesteps = timesteps[timesteps >= time_cursor]
                orbit = gd.evaluate_orbit(potential, current_w0, matching_timesteps)
                # orbit = potential.integrate_orbit(current_w0, t=matching_timesteps,
                #                                   Integrator=gi.DOPRI853Integrator)
                orbit_data.append(orbit.data)

            if len(orbit_data) == 1:
                full_orbit = gd.orbit.Orbit(q=orbit_data[0]["length"],
                                            p=orbit_data[0]["speed"],
                                            t=timesteps,
                                            frame=current_w0.frame)
            else:
                # concatenate all the orbit data together
                # this is necessary because the timesteps may not be uniform
                # and so we need to concatenate the representations together
                # to get a single Orbit
                full_orbit = gd.orbit.Orbit(
                    q=cx.CartesianPos3D(
                        x=jnp.concatenate([od["length"].x.value for od in orbit_data]) * orbit_data[0]["length"].x.unit,
                        y=jnp.concatenate([od["length"].y.value for od in orbit_data]) * orbit_data[0]["length"].y.unit,
                        z=jnp.concatenate([od["length"].z.value for od in orbit_data]) * orbit_data[0]["length"].z.unit
                    ),
                    p= cx.CartesianVel3D(
                        x=jnp.concatenate([od["speed"].x.value for od in orbit_data]) * orbit_data[0]["speed"].x.unit,
                        y=jnp.concatenate([od["speed"].y.value for od in orbit_data]) * orbit_data[0]["speed"].y.unit,
                        z=jnp.concatenate([od["speed"].z.value for od in orbit_data]) * orbit_data[0]["speed"].z.unit
                    ),
                    t=timesteps,
                    frame=current_w0.frame,
                )
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
