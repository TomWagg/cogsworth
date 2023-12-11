import numpy as np
import h5py as h5
import os
from readsnap import read_snapshot
import warnings


def find_centre(snap_dir, snap_num, out_path=None, theta=0.0, phi=0.0, J=True):
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
    J : `bool`, optional
        Whether to project to the plane perpendicular to the total angular momentum. Default is True.

    Returns
    -------
    pos_centre : `~numpy.ndarray`
        The position of the galaxy centre.
    v_CM : `~numpy.ndarray`
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
                                    ["Failed to fit centre with gaussian, retrying with simpler fallback"
                                     "Failed to find centre with fallback, no centre found :("]):
        # calculate stellar center
        pos_centre = centre_func(pos_star, pos_gas, rho_gas)

        # Shift coordinates, re-centre
        pos_shifted = pos_star - pos_centre
        pos_shifted_centre = Recent(pos_shifted.T)

        r_half = half_mass_radius(pos_shifted, mass_star, pos_shifted_centre, gridsize)

        pos_shifted -= pos_shifted_centre

        if np.any(np.isnan(pos_shifted_centre)):
            warnings.warn(message)

    # if J is set, project to the plane perpendicular to the total angular momentum
    if J:
        Jx, Jy, Jz = AngularMomentum(pos_shifted, mass_star, vel_star, r_half)
        theta, phi = RadialVector2AngularCoordiante(Jx, Jy, Jz)
        v_CM = CMvelocity(pos_shifted, mass_star, vel_star, 4 * r_half)

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


def calculate_star_centre(ps_p, pg_p, pg_rho, cen=[0, 0, 0.], clip_size=2.e10, rho_cut=1.0e-5):
    # Calculates stellar center, provided there's enough star particles, else uses gas particles.
    # Returns center vector. very simple method, don't imagine it would work on major mergers.
    rgrid = np.array([1.0e10, 1000, 700, 500, 300, 200, 100, 70, 50, 30, 20, 10, 5, 2.5, 1.])
    rgrid = rgrid[rgrid <= clip_size]
    
    n_new=len(ps_p)
    if (n_new > 1):
        pos=np.array(ps_p); x0s=pos[:,0]; y0s=pos[:,1]; z0s=pos[:,2];
    else: n_new=0
    rho=np.array(pg_rho);
    if (rho.shape[0] > 0):
        pos=np.array(pg_p); x0g=pos[:,0]; y0g=pos[:,1]; z0g=pos[:,2];
    cen=np.array(cen);
    
    for i_rcut in range(len(rgrid)):
        for j_looper in range(5):
            if (n_new > 1000):
                x=x0s; y=y0s; z=z0s;
            else:
                ok=(rho > rho_cut);
                x=x0g[ok]; y=y0g[ok]; z=z0g[ok];
            x=x-cen[0]; y=y-cen[1]; z=z-cen[2];
            r = np.sqrt(x*x + y*y + z*z);
            ok = (r < rgrid[i_rcut]);
            if (checklen(r[ok]) > 1000):
                x=x[ok]; y=y[ok]; z=z[ok];
                if (i_rcut <= len(rgrid)-5):
                    cen+=np.array([np.median(x),np.median(y),np.median(z)]);
                else:
                    cen+=np.array([np.mean(x),np.mean(y),np.mean(z)]);
            else:
                if (checklen(r[ok]) > 200):
                    x=x[ok]; y=y[ok]; z=z[ok];
                    cen+=np.array([np.mean(x),np.mean(y),np.mean(z)]);
    return cen;
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


def half_mass_radius(pos_shifted, masses, pos_shifted_centre, Rout, ratio=0.5):
    """
    Calculate the half-stellar mass radius within a given radius.

    Parameters
    ----------
    pos_shifted : `~numpy.ndarray`
        The shifted coordinates of star particles.
    masses : `~numpy.ndarray`
        The masses of star particles.
    pos_shifted_centre : numpy.ndarray
        The center coordinates.
    Rout : `float`
        The radius within which the total mass is defined.
    ratio : `float`, optional
        The fraction of total mass within Rout. Defaults to 0.5.

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
    rs_in = rs[rs < Rout]
    ms_in = masses[rs < Rout]
    Mtotal = np.sum(ms_in)
    Mhalf = ratio * Mtotal

    Rhalf = Rout
    if len(rs_in) < 10:
        return Rout
    order = np.argsort(rs_in)
    rs_in_sorted = rs_in[order]
    ms_in_sorted = ms_in[order]
    ms_in_cum = np.cumsum(ms_in_sorted)
    place = np.searchsorted(ms_in_cum, Mhalf)
    r_half = rs_in_sorted[place - 1]
    return r_half

def AngularMomentum(pos_shifted, ms, vel_star, r, cen=[0,0,0]):
    # calculate the angular momentum of all star particles within some radius
    # input:
    #    xs, ys, zs, ms, vxs, vys, vzs - coordiantes, masses and velocities
    #    r - the outermost radius we consider
    # keywords:
    #    cen - the center
    xs, ys, zs = pos_shifted.T
    vxs, vys, vzs = vel_star.T
    xc = cen[0]; yc = cen[1]; zc = cen[2]
    rs = np.sqrt((xs-xc)**2+(ys-yc)**2+(zs-zc)**2)
    ok = rs < r
    Jx = np.sum(ms[ok]*((ys[ok]-yc)*vzs[ok]-(zs[ok]-zc)*vys[ok]))
    Jy = np.sum(ms[ok]*((zs[ok]-zc)*vxs[ok]-(xs[ok]-xc)*vzs[ok]))
    Jz = np.sum(ms[ok]*((xs[ok]-xc)*vys[ok]-(ys[ok]-yc)*vxs[ok]))
    return Jx, Jy, Jz


def RadialVector2AngularCoordiante(nx, ny, nz):
    # convert radial vector to angular coordinate
    # input:
    #    nx, ny, nz - component of radial vector
    length = np.sqrt(nx**2 + ny**2 + nz**2)
    nx /= length; ny /= length; nz /= length
    if (nz == 1.0):
        theta, phi = 0.0, 0.0
    elif (nz == -1.0):
        theta, phi = np.pi, np.pi
    else:
        theta = np.arccos(nz)
    phi = np.arccos(nx/np.sin(theta))
    if (ny < 0):
        phi = 2*np.pi - phi
    return theta, phi


def CMvelocity(pos_shifted, ms, vel_star, r):
    # calculate center of mass velocity of all star particles within some radius
    # input:
    #    xs, ys, zs, ms, vxs, vys, vzs - coordiantes, masses and velocities
    #    r - the outermost radius we consider
    # keywords:
    #    cen - the center
    # xs, ys, zs = pos_shifted.T
    rs = np.sum(pos_shifted**2, axis=1)**0.5
    ok = rs < r
    return np.sum(ms[ok, np.newaxis] * vel_star[ok], axis=0) / np.sum(ms[ok])


def Recent(Xs):
    # recalculate the center of a galaxy
    xc, yc, zc = 0, 0, 0
    for grid in [20.0, 10.0, 5.0, 2.0]:
        for _ in range(4):
            ok_star = (np.abs(Xs[0] - xc) < grid) & (np.abs(Xs[1] - yc) < grid) & (np.abs(Xs[2] - zc) < grid)
            xc = np.median(Xs[0][ok_star])
            yc = np.median(Xs[1][ok_star])
            zc = np.median(Xs[2][ok_star])
    return (xc, yc, zc)

