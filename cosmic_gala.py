import numpy as np
import gala.integrate as gi
import gala.dynamics as gd
import gala.potential as gp

import astropy.coordinates as coords
import astropy.units as u
import astropy.constants as const

from galaxy import draw_lookback_times, draw_radii, draw_heights, R_exp

from multiprocessing import Pool


def get_kick_differential(delta_v_sys_xyz, m_1, m_2, a):
    """Calculate the Differential from a combination of the natal kick, Blauuw kick and orbital motion.

    Parameters
    ----------
    delta_v_sys_xyz : `float array`
        Change in systemic velocity due to natal and Blauuw kicks in BSE (v_x, v_y, v_z) frame (see Fig A1 of
        Hurley+02)
    m_1 : `float`
        Primary mass
    m_2 : `float`
        Secondary Mass
    a : `float`
        Binary separation

    Returns
    -------
    kick_differential : `CylindricalDifferential`
        Kick differential
    """
    # calculate the orbital velocity ASSUMING A CIRCULAR ORBIT
    if a.value > 0.0:
        v_orb = np.sqrt(const.G * (m_1 + m_2) / a)

        # adjust change in velocity by orbital motion of supernova star
        delta_v_sys_xyz -= v_orb

    # orbital phase angle and inclination to Galactic plane
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.random.uniform(0, 2 * np.pi)

    # rotate BSE (v_x, v_y, v_z) into Galactocentric (v_X, v_Y, v_Z)
    v_X = delta_v_sys_xyz[0] * np.cos(theta) - delta_v_sys_xyz[1] * np.sin(theta) * np.cos(phi)\
        + delta_v_sys_xyz[2] * np.sin(theta) * np.sin(phi)
    v_Y = delta_v_sys_xyz[0] * np.sin(theta) + delta_v_sys_xyz[1] * np.cos(theta) * np.cos(phi)\
        - delta_v_sys_xyz[2] * np.cos(theta) * np.sin(phi)
    v_Z = delta_v_sys_xyz[1] * np.sin(phi) + delta_v_sys_xyz[2] * np.cos(phi)

    kick_differential = coords.CartesianDifferential(v_X, v_Y, v_Z)

    return kick_differential


def integrate_orbit_with_events(w0, potential=gp.MilkyWayPotential(), events=None,
                                rng=np.random.default_rng(), **integrate_kwargs):
    """Integrate PhaseSpacePosition in a potential with events that occur at certain times

    Parameters
    ----------
    w0 : `ga.dynamics.PhaseSpacePosition`
        Initial phase space position
    potential : `ga.potential.PotentialBase`, optional
        Potential in which you which to integrate the orbits, by default the MilkyWayPotential()
    events : `list of objects`
        Events that occur during the orbit evolution. Should contain the following parameters: `time`, `m_1`,
        `m_2`, `a`, `ecc`, `delta_v_sys_xyz`
    rng : `NumPy RandomNumberGenerator`
        Which random number generator to use

    Returns
    -------
    full_orbit : `ga.orbit.Orbit`
        Orbit that have been integrated
    """
    # if there are no events then just integrate the whole thing
    if events is None:
        return potential.integrate_orbit(w0, **integrate_kwargs)

    # work out what the timesteps would be without kicks
    timesteps = gi.parse_time_specification(units=[u.s], **integrate_kwargs) * u.s

    # start the cursor at the smallest timestep
    time_cursor = timesteps[0]
    current_w0 = w0

    # keep track of the orbit data throughout
    orbit_data = []

    # loop over the events
    for event in events:
        # find the timesteps that occur before the kick
        timestep_mask = (timesteps >= time_cursor) & (timesteps < event["time"])

        # if any of them occur before the kick then do some integration
        if any(timestep_mask):
            matching_timesteps = timesteps[timestep_mask]

            # integrate the orbit over these timesteps
            orbit = potential.integrate_orbit(current_w0, t=matching_timesteps)

            # save the orbit data
            orbit_data.append(orbit.data)

            # get new PhaseSpacePosition(s)
            current_w0 = orbit[-1]

        # adjust the time
        time_cursor = event["time"]

        # calculate the kick differential
        kick_differential = get_kick_differential(delta_v_sys_xyz=event["delta_v_sys_xyz"],
                                                  m_1=event["m_1"], m_2=event["m_2"], a=event["a"])

        # update the velocity of the current PhaseSpacePosition
        current_w0 = gd.PhaseSpacePosition(pos=current_w0.pos,
                                           vel=current_w0.vel + kick_differential,
                                           frame=current_w0.frame)

    # if we still have time left after the last event (very likely)
    if time_cursor < timesteps[-1]:
        # evolve the rest of the orbit out
        matching_timesteps = timesteps[timesteps >= time_cursor]
        orbit = potential.integrate_orbit(current_w0, t=matching_timesteps)
        orbit_data.append(orbit.data)

    if len(orbit_data) > 1:
        orbit_data = coords.concatenate_representations(orbit_data)
    else:
        orbit_data = orbit_data[0]

    full_orbit = gd.orbit.Orbit(pos=orbit_data.without_differentials(),
                                vel=orbit_data.differentials["s"],
                                t=timesteps.to(u.Myr))

    return full_orbit


def fling_binary_through_galaxy(w0, potential, lookback_time, bpp, kick_info, bin_num,
                                max_ev_time=13.7 * u.Gyr, dt=1 * u.Myr):
    """Evolve the orbit of a binary through the Galaxy using information from COSMIC bpp and kick info tables

    Parameters
    ----------
    w0 : `gala.dynamics.PhaseSpacePosition`
        Phase space position of the binary
    potential : `gala.potential.PotentialBase`
        Galactic potential
    lookback_time : `float`
        Lookback time of binary
    bpp : `Pandas DataFrame`
        Table of evolution phase events from COSMIC
    kick_info : `Pandas DataFrame`
        Table of information about kicks from COSMIC
    bin_num : `int`
        Binary number of binary in COSMIC tables
    max_ev_time : `float`, optional
        Maximum evolution time in COSMIC simulation, by default 13.7*u.Gyr
    dt : `float`, optional
        Timestep size for integration, by default 1*u.Myr

    Returns
    -------
    orbit(s) : `gala.dynamics.Orbit or list`
        Either a single orbit of the binary or a list of two orbits (one for each star)
    """
    # reduce tables to just the given binary
    bpp = bpp.loc[bin_num]
    kick_info = kick_info.loc[bin_num]
    kick_info = kick_info[kick_info["star"] > 0.0]

    # mask for the rows that contain supernova events
    supernova_event_rows = bpp["evol_type"].isin([15, 16])

    # if no supernova occurs then just do regular orbit integration
    if not supernova_event_rows.any():
        return potential.integrate_orbit(w0, t1=lookback_time, t2=max_ev_time, dt=dt)

    # reduce to just the supernova rows and ensure we have the same length in each table
    bpp = bpp[supernova_event_rows]
    assert len(kick_info) == len(bpp)

    # check if the the binary is going to disrupt at any point
    it_will_disrupt = (kick_info["disrupted"] == 1.0).any()

    # iterate over the kick file and store the relevant information (for both stars if disruption will occur)
    if it_will_disrupt:
        events_1, events_2 = [], []
        for i in range(len(kick_info)):
            default_event = {
                "time": lookback_time + bpp.iloc[i]["tphys"] * u.Myr,
                "m_1": bpp.iloc[i]["mass_1"] * u.Msun,
                "m_2": bpp.iloc[i]["mass_2"] * u.Msun,
                "a": bpp.iloc[i]["sep"] * u.Rsun,
                "ecc": bpp.iloc[i]["ecc"],
                "delta_v_sys_xyz": [kick_info.iloc[i]["delta_vsysx_1"],
                                    kick_info.iloc[i]["delta_vsysy_1"],
                                    kick_info.iloc[i]["delta_vsysz_1"]] * u.km / u.s
            }
            if kick_info.iloc[i]["disrupted"] == 1.0:
                events_1.append(default_event)
                events_2.append({
                    "time": lookback_time + bpp.iloc[i]["tphys"] * u.Myr,
                    "m_1": bpp.iloc[i]["mass_1"] * u.Msun,
                    "m_2": bpp.iloc[i]["mass_2"] * u.Msun,
                    "a": bpp.iloc[i]["sep"] * u.Rsun,
                    "ecc": bpp.iloc[i]["ecc"],
                    "delta_v_sys_xyz": [kick_info.iloc[i]["delta_vsysx_2"],
                                        kick_info.iloc[i]["delta_vsysy_2"],
                                        kick_info.iloc[i]["delta_vsysz_2"]] * u.km / u.s
                })
            else:
                events_1.append(default_event)
                events_2.append(default_event)
        return [integrate_orbit_with_events(w0, potential=potential, events=events_1,
                                            t1=lookback_time, t2=max_ev_time, dt=dt),
                integrate_orbit_with_events(w0, potential=potential, events=events_2,
                                            t1=lookback_time, t2=max_ev_time, dt=dt)]

    else:
        events = [{
            "time": lookback_time + bpp.iloc[i]["tphys"] * u.Myr,
            "m_1": bpp.iloc[i]["mass_1"] * u.Msun,
            "m_2": bpp.iloc[i]["mass_2"] * u.Msun,
            "a": bpp.iloc[i]["sep"] * u.Rsun,
            "ecc": bpp.iloc[i]["ecc"],
            "delta_v_sys_xyz": [kick_info.iloc[i]["delta_vsysx_1"],
                                kick_info.iloc[i]["delta_vsysy_1"],
                                kick_info.iloc[i]["delta_vsysz_1"]] * u.km / u.s
        } for i in range(len(kick_info))]
        return integrate_orbit_with_events(w0, potential=potential, events=events,
                                           t1=lookback_time, t2=max_ev_time, dt=dt)


def evolve_binaries_in_galaxy(bpp, kick_info, galaxy_model=None,
                              galactic_potential=gp.MilkyWayPotential(),
                              dispersion=5 * u.km / u.s, max_ev_time=13.7 * u.Gyr, dt=1 * u.Myr,
                              pool=None, nproc=1):
    # work out how many binaries we are going to evolve
    bin_nums = bpp["bin_num"].unique()
    n_bin = len(bin_nums)

    vel_units = u.km / u.s

    # draw random positions and birth times in the galaxy
    # TODO: actually make this change based on the `galaxy model`
    lookback_time = draw_lookback_times(n_bin, tm=12 * u.Gyr, tsfr=6.8 * u.Gyr, component="low_alpha_disc")
    print(lookback_time.max())
    scale_length = R_exp(lookback_time, alpha=0.4)
    rho = draw_radii(n_bin, R_0=scale_length)
    scale_height = 0.3 * u.kpc
    z = draw_heights(n_bin, z_d=scale_height)
    phi = np.random.uniform(0, 2 * np.pi, size=n_bin) * u.rad

    # calculate the Galactic circular velocity at the given positions
    x, y = rho * np.cos(phi), rho * np.sin(phi)
    v_circ = gp.MilkyWayPotential().circular_velocity(q=[x, y, z]).to(vel_units)

    # add some velocity dispersion
    v_R, v_T, v_z = np.random.normal([np.zeros_like(v_circ), v_circ, np.zeros_like(v_circ)],
                                     dispersion.to(vel_units) / np.sqrt(3), size=(3, n_bin))
    v_R, v_T, v_z = v_R * vel_units, v_T * vel_units, v_z * vel_units

    # turn the drawn coordinates into an astropy representation
    rep = coords.CylindricalRepresentation(rho, phi, z)

    # create differentials based on the velocities (dimensionless angles allows radians conversion)
    with u.set_enabled_equivalencies(u.dimensionless_angles()):
        dif = coords.CylindricalDifferential(v_R, (v_T / rho).to(u.rad / u.Gyr), v_z)

    # combine the representation and differentials into a Gala PhaseSpacePosition
    w0s = gd.PhaseSpacePosition(rep.with_differentials(dif))

    # evolve the orbits from birth until present day
    if pool is not None or nproc > 1:
        pool_existed = pool is not None
        if not pool_existed:
            pool = Pool(nproc)

        args = [(w0s[i], galactic_potential, lookback_time[i],
                 bpp, kick_info, bin_num, max_ev_time, dt) for i, bin_num in enumerate(bin_nums)]

        orbits = pool.starmap(fling_binary_through_galaxy, args)
        if not pool_existed:
            pool.close()
            pool.join()
    else:
        orbits = []
        for i, bin_num in enumerate(bin_nums):
            orbits.append(fling_binary_through_galaxy(w0=w0s[i], potential=galactic_potential,
                                                      lookback_time=lookback_time[i], bpp=bpp,
                                                      kick_info=kick_info, bin_num=bin_num,
                                                      max_ev_time=max_ev_time, dt=dt))

    return orbits
