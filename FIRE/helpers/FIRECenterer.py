import numpy as np
import h5py
import os
import sys
import timeit as tm
from gadget import *
from utilities import *

#######
#  Top-level environment for FIREMapper projects
#
#   Goal: make modular set of python scripts to generate FIRE galaxy maps
#
#   List of included functions, and descriptions:
#       TheMap - top-level function which calls various sub-functions to map the snapshot
#       CenFinder - finds the stellar center of the snapshot, if a 'centering' file already exists, use that
#       calculate_star_center - routine to find stellar center, called by CenFinder
#       Recent - routine to check that the center found by calculate_star_center is good
#       StarMapper - mapping routine to make map of stellar properties
#       GasMapper - mapping routine to map the gas properties
#       OmegaMapper - mapping routine to map the 1/dynamical time
#

##################################################################################################################################

def WriteCents(RunName, SnapNum, theta=0.0, phi=0.0, Flag_J=1, Force=0, Verbose=0):
    
    # Calculate center/normal vector for snapshot
    # Input:
    #    RunName - which run?
    #    SnapNum - which snapshot?
    # Keyword:
    #    theta, phi - angular coordinates, x-y projection by default
    #    Flag_J - if 1, select the direction of total star angular momentum,
    #		and in this case theta and phi will be ignored
    #    Verbose - if set, mapping and functions print out statuses
    
    if Verbose: print("Make Galaxy Center File: %s, snapshot %03d." % (RunName, SnapNum))
    
    #######
    #  These directories point the code to where the snapshots live,
    #   and where maps will be saved! Edit these appropriately!
    #
    #bDir = "/mnt/home/chayward/firesims/fire3_suite_done/" # base directory, all analysis scripts/snapshots live below here
    #SnapDir = bDir+RunName+"/"+RunName+"_MHD_fire3_fireBH_Sep182021_hr_crdiffc690_sdp1e10_gacc31_fa0.5/output"

    #bDir = '/mnt/home/chayward/firesims/fire2/metaldiff/' # base directory, all analysis scripts/snapshots live below here
    bDir = '/mnt/home/morr/ceph/simulations/m12i_res7100_tests/' # base directory, all analysis scripts/snapshots live below here
    SnapDir = bDir+RunName+"/output"

    
    ##############
    # Mapping Functions here.
    #
    
    # Call CenFinder to find gal center, and projection vector. Writes these to CenFile
    tic=tm.default_timer()
    
    CenFinder(SnapDir, SnapNum, theta=0.0, phi=0.0, J=Flag_J, force=Force, loud=Verbose)
    
    toc=tm.default_timer()
    
    if Verbose: print("Center coords & normal vector found in %.2f sec." % (toc-tic))
    sys.stdout.flush()
    
    return

##################################################################################################################################

def CenFinder(SnapDir, SnapNum, theta=0.0, phi=0.0, J=1, force=0, loud=0):
    # CenFinder function.
    #
    #   Finds the stellar center of the snapshot, unless one is already known
    #
    #   Input:
    #       SnapDir - directory of snapshot to read
    #       MapFile - map file to read/append
    #       theta, phi - angular coordinates, x-y projection by default
    #       J - if 1, select the direction of total star angular momentum,
    #           and in this case theta and phi will be ignored
    #       force - if 1, force CenFinder to run, even if
    #       loud - if 1, verbose output regarding performance
    #   Return:
    #       Dictionary.
    #       'n' - list of unit vectors to project positions into
    #               in form: X_new = x_old*nx_x + y_old*nx_y + z_old*nx_z  ...and so on for ny_(x..y..z), nz_(x..y..z)
    #       'Xc' - vector of coordinates of stellar center of snapshot (x,y,z).
    #               They are in old coordinate system, must translate stars/gas before transforming using 'n' vector
    
    #outputbase = "/work2/03454/tg827535/stampede2/visualization/frames/"
    #outputbase = "/mnt/home/morr/ceph/analysis/sfr/galmap/FIRE-3/"
    outputbase = "/mnt/home/morr/ceph/simulations/m12i_res7100_tests/"
    gridsize = 50. # kpc, radius in which total (stellar) mass should be defined (idk, just <Rvir, it doesn't matter)
    
    # Check if center has already been found
    s0=SnapDir.split("/"); RunName=s0[len(s0)-2];#-2 if there's /output
    if(len(RunName)==0): RunName=s0[len(s0)-3];
    print("RunName:"+RunName)
    cenPath = outputbase+RunName+"/centers/"
    cenFile = cenPath+"snap_"+str(SnapNum)+"_cents.hdf5"
    #cenFile = SnapDir+"/centers/snap_"+str(SnapNum)+"_cents.hdf5"
    if (os.path.exists(cenFile) and not force):
        if loud: print("Center and normal vector previously calculated, found.")
        sys.stdout.flush()
        return
    # Otherwise, do the hard work yourself.
    else:
        # Pull up the star particle positions, and masses
        toc1=tm.default_timer()
        stars = readsnap_simple(SnapDir, SnapNum, 4, cosmological=1)
        Header = readsnap_simple(SnapDir, SnapNum, 4, cosmological=1, header_only=1)
        toc2=tm.default_timer()
        if loud: print("Loaded star particles (simple) in %.2f sec." % (toc2-toc1))
        sys.stdout.flush()
        ps=stars['p'] #position
        vs=stars['v'] #velocity
        ms=stars['m'] #mass
        # Pull up the gas particle positions, and number densities.
        toc3=tm.default_timer()
        gas = readsnap_simple(SnapDir, SnapNum, 0, cosmological=1)
        toc4=tm.default_timer()
        if loud: print("Loaded gas particles (simple) in %.2f sec." % (toc4-toc3))
        sys.stdout.flush()
        pg =gas['p']         #position
        rho =gas['rho']*404 #num density
        
        # Calcualte stellar center
        toc5=tm.default_timer()
        #xc, yc, zc = calculate_star_center(ps,ms,pg,rho)
        xc, yc, zc = gaussfit_star_center(ps,ms,pg,rho) # faster method, dead nuts on too.
        toc6=tm.default_timer()
        if loud: print("Function: calculate_star_center ran in %.2f sec." % (toc6-toc5))
        sys.stdout.flush()
        # Shift coordinates, recenter
        xs = ps[:,0] - xc; ys = ps[:,1] - yc; zs = ps[:,2] - zc
        toc7=tm.default_timer()
        (xsc, ysc, zsc) = Recent([xs, ys, zs], ms)
        toc8=tm.default_timer()
        if loud: print("Function: Recent ran in %.2f sec." % (toc8-toc7))
        rhalf = half_mass_radius(xs, ys, zs, ms, xsc, ysc, zsc, gridsize)
        toc9=tm.default_timer()
        if loud: print("Function: half_mass_radius ran in %.2f sec." % (toc9-toc8))
        if loud: print("Half-mass radius %.2f kpc" % (rhalf))
        sys.stdout.flush()
        xs -= xsc; ys -= ysc; zs -= zsc
        if np.any(np.isnan([xsc,ysc,zsc])):
            print("****gaussfit_star_center CATASTROPHIC FAILURE****")
            print(" ")
            print(" RETRYING: calculate_star_center")
        
            xc, yc, zc = calculate_star_center(ps,ms,pg,rho)
            xs = ps[:,0] - xc; ys = ps[:,1] - yc; zs = ps[:,2] - zc
            (xsc, ysc, zsc) = Recent([xs, ys, zs], ms)
            rhalf = half_mass_radius(xs, ys, zs, ms, xsc, ysc, zsc, gridsize)
            xs -= xsc; ys -= ysc; zs -= zsc
            if np.any(np.isnan([xsc,ysc,zsc])):
                print("****SUBSEQUENT calculate_star_center CATASTROPHIC FAILURE****")
                print("CENTERING FAILED.")
                sys.stdout.flush()
                return
            else: print("****SUBSEQUENT calculate_star_center SUCCESS****")
            sys.stdout.flush()
        # if J is set, project to the plane perpendicular to the total angular momentum
        if (J):
            toc11=tm.default_timer()
            Jx, Jy, Jz = AngularMomentum(xs, ys, zs, ms, vs[:,0], vs[:,1], vs[:,2], rhalf)
            theta, phi = RadialVector2AngularCoordiante(Jx, Jy, Jz)
            toc12=tm.default_timer()
            if loud: print("Function: AngularMomentum ran in %.2f sec." % (toc12-toc11))
            sys.stdout.flush()
            tocCM=tm.default_timer()
            Vx_cm, Vy_cm, Vz_cm = CMvelocity(xs, ys, zs, ms, vs[:,0], vs[:,1], vs[:,2], 4.*rhalf)
            ticCM=tm.default_timer()
            if loud: print("Function: CMvelocity ran in %.2f sec." % (ticCM-tocCM))
            sys.stdout.flush()

        # make the projection
        nx = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        ny = np.array([-np.sin(phi), np.cos(phi), 0])
        nz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])


        if (not os.path.exists(cenPath)): os.system("mkdir "+cenPath)
        cfile = h5py.File(cenFile, 'a')
        # Write Map/gal Attributes to cenFile, to be referenced in future.
        cfile.attrs["RunName"] = RunName
        cfile.attrs["SnapNum"] = SnapNum
        cfile.attrs["Time"] = Header["time"]
        Redshift = 1.0/Header["time"] - 1.0
        if (Redshift<0): Redshift = 0.0
        cfile.attrs["Redshift"] = Redshift
        cfile.attrs["Theta"] = theta
        cfile.attrs["Phi"] = phi
        cfile.attrs["Center"] = [xc, yc, zc]
        cfile.attrs["StellarCMVel"] = [Vx_cm, Vy_cm, Vz_cm]
        cfile.attrs["StellarCenter"] = [xc+xsc, yc+ysc, zc+zsc]
        cfile.attrs["NormalVector"] = [nx, ny, nz]
        cfile.attrs["Rhalfstar"] = rhalf
        
        cfile.close()



    return

##################################################################################################################################

def calculate_star_center(ps_p,ps_m,pg_p,pg_rho,cen=[0.,0.,0.],clip_size=2.e10,rho_cut=1.0e-5):
    # Calculates stellar center, provided there's enough star particles, else
    # uses gas particles.
    # Returns center vector. very simple method, don't imagine it would work on major mergers.
    rgrid=np.array([1.0e10,1000.,700.,500.,300.,200.,100.,70.,50.,30.,20.,10.,5.,2.5,1.]);
    rgrid=rgrid[rgrid <= clip_size];
    
    n_new=np.array(ps_m).shape[0];
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

def gaussfit_star_center(ps_p,ps_m,pg_p,pg_rho,cen=[0.,0.,0.],clip_size=2.e10,rho_cut=1.0e-5):
    # Calculates stellar center, provided there's enough star particles, else
    # uses gas particles.
    # Returns center vector. very simple method, don't imagine it would work on major mergers.
    import scipy.optimize
    rgrid=np.array([1.0e10,1000.,700.,500.,300.,200.,100.,70.,50.,30.,20.]) #,10.,5.,2.5,1.]);
    rgrid=rgrid[rgrid <= clip_size];
    gridcells = 1000 # number of cells to tile, to fit gaussian to
    
    n_new=len(ps_m);
    if (n_new > 1):
        pos=np.array(ps_p); x0s=pos[:,0]; y0s=pos[:,1]; z0s=pos[:,2];
    else: n_new=0
    rho=np.array(pg_rho);
    if (rho.shape[0] > 0):
        pos=np.array(pg_p); x0g=pos[:,0]; y0g=pos[:,1]; z0g=pos[:,2];
    cen=np.array(cen);
    # fit a guassian to the stellar radial density profile
    fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2) #+p[3]
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))

    if (n_new > 1000):
        x=x0s; y=y0s; z=z0s;
    else:
        ok=(rho > rho_cut);
        x=x0g[ok]; y=y0g[ok]; z=z0g[ok]

    # find initial "unresolved" center (i.e., max of 3d histogram, with 200 kpc resolution)
    unresolved_voxel_size=99.9 # kpc
    rgrid=rgrid[rgrid<=1.5*unresolved_voxel_size]
    init_bin = int(np.max([np.max(x),np.max(y),np.max(z)])/unresolved_voxel_size)
    stardens3, cube_eds1 = np.histogramdd([x,y,z],bins=init_bin)
    maxinds = np.unravel_index(np.argmax(stardens3),shape=(init_bin,init_bin,init_bin))
    for dd in [0,1,2]:
        # grab coordinates of max star voxel, correcting for edges.
        cen[dd] = cube_eds1[dd][maxinds[dd]]+(cube_eds1[dd][1]-cube_eds1[dd][0])/2.
    sigmas=np.array([50.,50.,50.]) # initialize smoothing lengths, 50 kpc.
    for i_rcut in range(len(rgrid)):
        gridcells = int(np.min([rgrid[i_rcut]/0.05, 1000.])) # grid resolution never below 50 pc.
        sigmas=np.array([np.min([sigmas[0],rgrid[i_rcut]/2.]),np.min([sigmas[1],rgrid[i_rcut]/2.]),np.min([sigmas[2],rgrid[i_rcut]/2.])])
        ok = ((np.abs(x-cen[0]) < rgrid[i_rcut]) & (np.abs(y-cen[1]) < rgrid[i_rcut]) & (np.abs(z-cen[2]) < rgrid[i_rcut]))
        if (len(x[ok]) > 1000):
            x=x[ok]; y=y[ok]; z=z[ok]
            
            # fit in x
            xbins, xed = np.histogram(x,bins=gridcells)
            ybins, yed = np.histogram(y,bins=gridcells)
            zbins, zed = np.histogram(z,bins=gridcells)
            
            xcents = xed[:len(xed)-1]+(xed[1]-xed[0])/2.
            ycents = yed[:len(yed)-1]+(yed[1]-yed[0])/2.
            zcents = zed[:len(zed)-1]+(zed[1]-zed[0])/2.
            
            init  = [np.average(xbins), cen[0], sigmas[0]]
            out = scipy.optimize.leastsq( errfunc, init, args=(xcents, xbins))
            gauslen_x = out[0][2];gauscent_x = out[0][1]
            
            init  = [np.average(ybins), cen[1], sigmas[1]]
            out = scipy.optimize.leastsq( errfunc, init, args=(ycents, ybins))
            gauslen_y = out[0][2];gauscent_y = out[0][1]
            
            init  = [np.average(zbins), cen[2], sigmas[2]]
            out = scipy.optimize.leastsq( errfunc, init, args=(zcents, zbins))
            gauslen_z = out[0][2];gauscent_z = out[0][1]
            
            cen=np.array([gauscent_x,gauscent_y,gauscent_z]);
            sigmas=np.array([gauslen_x,gauslen_y,gauslen_z])
        
        else:
            if (len(x[ok]) > 200):
                x=x[ok]; y=y[ok]; z=z[ok];
                cen=np.array([np.mean(x),np.mean(y),np.mean(z)]);
    return cen;

#############################################################################
def half_mass_radius(xs, ys, zs, ms, xc, yc, zc, Rout, ratio=0.5):
    # calculate the half-stellar mass radius (ratio=0.5)
    # input:
    #    xs, ys, zs, ms - coordinates and masses of star particles
    #    xc, yc, zc - the center
    #    Rout - the radius within which the total mass is defined
    # keyword:
    #    ratio - a number between 0 and 1, what fraction of total mass within Rout,
    #         0.5 by default
    rs = np.sqrt((xs-xc)**2+(ys-yc)**2+(zs-zc)**2)
    rs_in = rs[rs < Rout]; ms_in = ms[rs < Rout]
    Mtotal = np.sum(ms_in); Mhalf = ratio*Mtotal
    # I require 10 star particles at least to further calculation, otherwise,
    # it will just return Rout
    Rhalf = Rout
    if (len(rs_in)>10):
        # sort the particles according to the distance
        order = np.argsort(rs_in)
        rs_in_sorted = rs_in[order]; ms_in_sorted = ms_in[order]
        ms_in_cum = np.cumsum(ms_in_sorted)
        place = np.searchsorted(ms_in_cum, Mhalf)
        Rhalf = rs_in_sorted[place-1]
    return Rhalf
#############################################################################
def AngularMomentum(xs, ys, zs, ms, vxs, vys, vzs, r, cen=[0,0,0]):
    # calculate the angular momentum of all star particles within some radius
    # input:
    #    xs, ys, zs, ms, vxs, vys, vzs - coordiantes, masses and velocities
    #    r - the outermost radius we consider
    # keywords:
    #    cen - the center
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
    length = np.sqrt(nx**2+ny**2+nz**2)
    nx /= length; ny /= length; nz /= length
    if (nz == 1.0):
        theta = 0.0; phi = 0.0
    elif (nz == -1.0):
        theta = np.pi; phi = np.pi
    else:
        theta = np.arccos(nz)
    phi = np.arccos(nx/np.sin(theta))
    if (ny < 0):
        phi = 2*np.pi - phi
    return theta, phi

def CMvelocity(xs, ys, zs, ms, vxs, vys, vzs, r, cen=[0,0,0]):
    # calculate center of mass velocity of all star particles within some radius
    # input:
    #    xs, ys, zs, ms, vxs, vys, vzs - coordiantes, masses and velocities
    #    r - the outermost radius we consider
    # keywords:
    #    cen - the center
    xc = cen[0]; yc = cen[1]; zc = cen[2]
    rs = np.sqrt((xs-xc)**2+(ys-yc)**2+(zs-zc)**2)
    ok = rs < r
    Mtot = np.sum(ms[ok])
    Vx = np.sum(ms[ok]*vxs[ok])/Mtot
    Vy = np.sum(ms[ok]*vys[ok])/Mtot
    Vz = np.sum(ms[ok]*vzs[ok])/Mtot
    return Vx, Vy, Vz

##################################################################################################################################

def Recent(Xs, ms, cen=[0.,0.,0.]):
    # recalculate the center of a galaxy
    xc = cen[0]; yc = cen[1]; zc = cen[2]
    for grid in [20.0, 10.0, 5.0, 2.0]:
        for i in range(4):
            ok_star = (np.abs(Xs[0]-xc)<grid) & (np.abs(Xs[1]-yc)<grid) & (np.abs(Xs[2]-zc)<grid)
            xc = np.median(Xs[0][ok_star])
            yc = np.median(Xs[1][ok_star])
            zc = np.median(Xs[2][ok_star])
    return (xc, yc, zc)

##################################################################################################################################

