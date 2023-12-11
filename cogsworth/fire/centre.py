import numpy as np
import h5py as h5
import os
from readsnap import read_snapshot
import warnings


def find_centre(snap_dir, snap_num, out_path=None, theta=0.0, phi=0.0, project_ang_mom=True):
    """Find the centre of a galaxy by analysing star and gas particle positions and densities.

    Parameters
    ----------
    snap_dir : `str`
        The directory path of the snapshot.
    snap_num : `int`
        The snapshot number.
    out_path : `str`, optional
        The output directory path. If not provided, a folder will be created in the snapshot directory.
    theta : `float`, optional
        The theta angle for projection. Default is 0.0.
    phi : `float`, optional
        The phi angle for projection. Default is 0.0.
    project_ang_mom : `bool`, optional
        Whether to project to the plane perpendicular to the total angular momentum. Default is True.

    Returns
    -------
    pos_centre : :class:`~numpy.ndarray`
        The position of the galaxy centre.
    v_CM : :class:`~numpy.ndarray`
        The centre of mass velocity of the galaxy.
    normal_vector : `list`
        The normal vector of the projected plane.
    r_half : `float`
        The half-mass radius of the galaxy.
    """
    # kpc, radius in which total (stellar) mass should be defined (idk, just <Rvir, it doesn't matter)
    gridsize = 50

    # Pull up the star particle positions, and masses
    stars, _ = read_snapshot(snap_dir=snap_dir, snap_num=snap_num, ptype=4)
    pos_star = stars['p']
    vel_star = stars['v']
    mass_star = stars['m']

    # Pull up the gas particle positions, and number densities.
    gas, _ = read_snapshot(snap_dir=snap_dir, snap_num=snap_num, ptype=0)
    pos_gas = gas['p']
    rho_gas = gas['rho'] * 404      # num density
    # TODO: Ask Matt about this factor of 404

    for centre_func, message in zip([gaussfit_star_centre, calculate_star_centre],
                                    ["Failed to fit centre with gaussian, retrying with simpler fallback",
                                     "Failed to find centre with fallback, no centre found :("]):
        # calculate stellar center
        pos_centre = centre_func(pos_star, pos_gas, rho_gas)

        # Shift coordinates, re-centre
        pos_shifted = pos_star - pos_centre
        pos_shifted_centre = recentre(pos_shifted.T)

        r_half = half_mass_radius(pos_shifted, mass_star, pos_shifted_centre, gridsize)

        pos_shifted -= pos_shifted_centre

        if np.any(np.isnan(pos_shifted_centre)):
            warnings.warn(message)

        print(f"Centre: {pos_centre}")

    # if project_ang_mom, project to the plane perpendicular to the total angular momentum
    if project_ang_mom:
        J = AngularMomentum(pos_shifted, mass_star, vel_star, r_half)
        theta, phi = radial_vector_to_angular_coordinates(J)
        v_CM = get_v_CM(pos_shifted, mass_star, vel_star, 4 * r_half)

    # make the projection
    nx = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    ny = np.array([-np.sin(phi), np.cos(phi), 0])
    nz = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])

    # create the output path (and directory) if necessary
    if out_path is None:
        out_path = os.path.join(snap_dir, "centers")
        if not os.path.exists(out_path):
            os.path.mkdir(out_path)

    # write all of the information to the output file
    with h5.File(os.path.join(out_path, f"snap_{snap_num}_cents.hdf5"), 'a') as f:
        f.attrs["snap_num"] = snap_num
        f.attrs["Theta"] = theta
        f.attrs["Phi"] = phi
        f.attrs["Center"] = pos_centre
        f.attrs["StellarCMVel"] = v_CM
        f.attrs["StellarCenter"] = pos_centre + pos_shifted_centre
        f.attrs["NormalVector"] = [nx, ny, nz]
        f.attrs["Rhalfstar"] = r_half

    return pos_centre + pos_shifted_centre, v_CM, [nx, ny, nz], r_half


def calculate_star_centre(ps_p, pg_p, pg_rho, clip_size=2.e10, rho_cut=1.0e-5):
    # Calculates stellar center, provided there's enough star particles, else uses gas particles.
    # Returns center vector. very simple method, don't imagine it would work on major mergers.
    rgrid = np.array([1.0e10, 1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2.5, 1])
    rgrid = rgrid[rgrid <= clip_size]

    n_star = len(ps_p)
    cen = np.zeros(3)
    dense_enough = pg_rho > rho_cut
    for i_rcut in range(len(rgrid)):
        for _ in range(5):
            pos = ps_p.copy() if n_star > 1000 else pg_p[dense_enough, :].copy()
            pos -= cen[np.newaxis, :]
            r = np.sum(pos**2, axis=1)**0.5
            ok = r < rgrid[i_rcut]
            if ok.sum() > 1000:
                pos = pos[ok, :]
                if (i_rcut <= len(rgrid) - 5):
                    cen += np.median(pos, axis=0)
                else:
                    cen += np.mean(pos, axis=0)
            elif ok.sum() > 200:
                pos = pos[ok]
                cen += np.mean(pos, axis=0)
    return cen


##################################################################################################################################

def gaussfit_star_centre(ps_p,pg_p,pg_rho,cen=[0, 0, 0.],clip_size=2.e10,rho_cut=1.0e-5):
    # Calculates stellar center, provided there's enough star particles, else
    # uses gas particles.
    # Returns center vector. very simple method, don't imagine it would work on major mergers.
    import scipy.optimize
    rgrid = np.array([1.0e10, 1000, 700, 500, 300, 200, 100, 70, 50, 30, 20.])
    rgrid = rgrid[rgrid <= clip_size]

    # number of cells to tile, to fit gaussian to
    gridcells = 1000

    cen = np.array(cen)

    # fit a gaussian to the stellar radial density profile
    fitfunc = lambda p, x: p[0] * np.exp(-0.5 * ((x - p[1]) / p[2])**2)
    errfunc = lambda p, x, y: (y - fitfunc(p, x))

    # if there are at least 1000 star particles, use them, else use gas particles
    if len(ps_p) > 1000:
        x = ps_p[:, 0]
        y = ps_p[:, 1]
        z = ps_p[:, 2]
    else:
        dense_enough = np.array(pg_rho) > rho_cut
        x = pg_p[dense_enough, 0]
        y = pg_p[dense_enough, 1]
        z = pg_p[dense_enough, 2]

    # find initial "unresolved" center (i.e, max of 3d histogram, with 200 kpc resolution)
    unresolved_voxel_size = 99.9 # kpc
    rgrid = rgrid[rgrid <= 1.5*unresolved_voxel_size]
    init_bin = int(np.max([np.max(x), np.max(y), np.max(z)]) / unresolved_voxel_size)
    stardens3, cube_eds1 = np.histogramdd([x, y, z], bins=init_bin)
    maxinds = np.unravel_index(np.argmax(stardens3), shape=(init_bin, init_bin, init_bin))
    for dd in [0, 1, 2]:
        # grab coordinates of max star voxel, correcting for edges.
        cen[dd] = cube_eds1[dd][maxinds[dd]] + (cube_eds1[dd][1] - cube_eds1[dd][0]) / 2.

    # initialize smoothing lengths, 50 kpc
    sigmas = np.array([50, 50, 50.])
    for i_rcut in range(len(rgrid)):
        # grid resolution never below 50 pc
        gridcells = int(np.min([rgrid[i_rcut]/0.05, 1000.]))
        sigmas = np.array([np.min([sigmas[0], rgrid[i_rcut] / 2.]),
                           np.min([sigmas[1], rgrid[i_rcut] / 2.]),
                           np.min([sigmas[2], rgrid[i_rcut] / 2.])])
        ok = ((np.abs(x - cen[0]) < rgrid[i_rcut])
              & (np.abs(y - cen[1]) < rgrid[i_rcut])
              & (np.abs(z - cen[2]) < rgrid[i_rcut]))
        if (len(x[ok]) > 1000):
            x = x[ok]
            y = y[ok]
            z = z[ok]

            # fit in x
            xbins, xed = np.histogram(x, bins=gridcells)
            ybins, yed = np.histogram(y, bins=gridcells)
            zbins, zed = np.histogram(z, bins=gridcells)

            xcents = xed[:len(xed)-1]+(xed[1]-xed[0])/2.
            ycents = yed[:len(yed)-1]+(yed[1]-yed[0])/2.
            zcents = zed[:len(zed)-1]+(zed[1]-zed[0])/2.

            init = [np.average(xbins), cen[0], sigmas[0]]
            out = scipy.optimize.leastsq(errfunc, init, args=(xcents, xbins))
            gauslen_x = out[0][2]
            gauscent_x = out[0][1]

            init = [np.average(ybins), cen[1], sigmas[1]]
            out = scipy.optimize.leastsq(errfunc, init, args=(ycents, ybins))
            gauslen_y = out[0][2]
            gauscent_y = out[0][1]

            init = [np.average(zbins), cen[2], sigmas[2]]
            out = scipy.optimize.leastsq(errfunc, init, args=(zcents, zbins))
            gauslen_z = out[0][2]
            gauscent_z = out[0][1]

            cen = np.array([gauscent_x, gauscent_y, gauscent_z]);
            sigmas = np.array([gauslen_x, gauslen_y, gauslen_z])

        else:
            if (len(x[ok]) > 200):
                x = x[ok]
                y = y[ok]
                z = z[ok]
                cen = np.array([np.mean(x), np.mean(y), np.mean(z)])
    return cen


def half_mass_radius(pos_shifted, masses, pos_shifted_centre, r_out, ratio=0.5):
    """
    Calculate the half-stellar mass radius within a given radius.

    Parameters
    ----------
    pos_shifted : :class:`~numpy.ndarray`
        The shifted coordinates of star particles.
    masses : :class:`~numpy.ndarray`
        The masses of star particles.
    pos_shifted_centre : numpy.ndarray
        The center coordinates.
    r_out : `float`
        The radius within which the total mass is defined.
    ratio : `float`, optional
        The fraction of total mass within r_out. Defaults to 0.5.

    Returns
    -------
    r_half : `float`
        The half-stellar mass radius.

    Notes
    -----
    This function calculates the half-stellar mass radius, which is the radius within which half of the
    total stellar mass is contained.
    """
    rs = np.sum((pos_shifted - pos_shifted_centre)**2, axis=1)**0.5
    rs_in = rs[rs < r_out]
    ms_in = masses[rs < r_out]
    Mtotal = np.sum(ms_in)
    Mhalf = ratio * Mtotal

    if len(rs_in) < 10:
        return r_out
    order = np.argsort(rs_in)
    rs_in_sorted = rs_in[order]
    ms_in_sorted = ms_in[order]
    ms_in_cum = np.cumsum(ms_in_sorted)
    place = np.searchsorted(ms_in_cum, Mhalf)
    r_half = rs_in_sorted[place - 1]
    return r_half


def AngularMomentum(pos_shifted, masses, vel_star, r):
    """Calculate the angular momentum of particles within a given radius.

    Parameters
    ----------
    pos_shifted : :class:`~numpy.ndarray`
        The shifted coordinates of particles.
    masses : :class:`~numpy.ndarray`
        The masses of particles.
    vel_star : :class:`~numpy.ndarray`
        The velocities of particles.
    r : `float`
        The radius within which to calculate the angular momentum.

    Returns
    -------
    J : :class:`~numpy.ndarray`
        The angular momentum of particles within the given radius.
    """
    rs = np.sum(pos_shifted**2, axis=1)**0.5
    ok = rs < r
    return np.sum(masses[ok, np.newaxis] * np.cross(pos_shifted[ok, :], vel_star[ok, :]), axis=0)


def radial_vector_to_angular_coordinates(n):
    """Calculate the angular coordinates (theta, phi) from a radial vector.

    Parameters
    ----------
    n : :class:`~numpy.ndarray`
        Radial vector

    Returns
    -------
    theta, phi : `tuple`
        Angular coordinates (theta, phi)
    """
    n /= np.sum(n**2, axis=0)**0.5
    if (n[2] == 1.0):
        theta, phi = 0.0, 0.0
    elif (n[2] == -1.0):
        theta, phi = np.pi, np.pi
    else:
        theta = np.arccos(n[2])
        phi = np.arccos(n[0] / np.sin(theta))
        if (n[1] < 0):
            phi = 2 * np.pi - phi
    return theta, phi


def get_v_CM(pos_shifted, masses, vel_star, r):
    """Calculate the velocity of the centre of mass (CM) of particles within a given radius.

    Parameters:
    ----------
    pos_shifted : :class:`~numpy.ndarray`
        Array of particle positions.
    masses : :class:`~numpy.ndarray`
        Array of particle masses.
    vel_star : :class:`~numpy.ndarray`
        Array of particle velocities.
    r : `float`
        Radius within which to calculate the CM velocity.

    Returns:
    -------
    numpy.ndarray
        Velocity of the center of mass of particles within the given radius.
    """
    rs = np.sum(pos_shifted**2, axis=1)**0.5
    ok = rs < r
    return np.sum(masses[ok, np.newaxis] * vel_star[ok], axis=0) / np.sum(masses[ok])


def recentre(pos):
    """Recentre the coordinates of particles.

    Parameters
    ----------
    pos : :class:`~numpy.ndarray`
        Array of particle positions

    Returns
    -------
    centre : :class:`~numpy.ndarray`
        The centre of the particles
    """
    centre = np.zeros(3)
    for grid in [20.0, 10.0, 5.0, 2.0]:
        for _ in range(4):
            ok_star = np.all(np.abs(pos - centre[:, np.newaxis]) < grid, axis=0)
            centre = np.median(pos[:, ok_star], axis=1)
    return centre
