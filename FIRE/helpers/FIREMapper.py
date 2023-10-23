import numpy as np
import h5py
import os
import sys
import timeit as tm
from meo_python_lib import *
from colors_sps.colors_table import *

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

def TheMap(RunName, SnapNum, theta=0.0, phi=0.0, Flag_J=1, res=1.0,\
           gridsize=30.0, Nx_min=1, Nx_max=3000, Flag_Rvir=0, Flag_Rhalf=0, frac=1.0,\
           pSuite="",sVarName="",\
           VSpace=0, maxv=100., dv=1.0, CSpace=0, Verbose=0):
    
    # Make a 2D galaxy map
    # Input:
    #    RunName - which run?
    #    SnapNum - which snapshot?
    # Keyword:
    #    theta, phi - angular coordinates, x-y projection by default
    #    Flag_J - if 1, select the direction of total star angular momentum,
    #		and in this case theta and phi will be ignored
    #    res - spatial resolution of pixels (units:kpc)
    #    gridsize - manually set (size of grid, units:kpc), if flag_rvir=0 and flag_rhalf=0
    #    Nx_min - minimum cell number along each direction (mostly sanity check)
    #    Nx_max - maximum cell number along each direction (mostly sanity check)
    #    Flag_Rvir - if set, gridsize=frac*rvir
    #    Flag_Rhalf - if set, gridsize=frac*rhalf
    #    VSpace - if set, map some gas, star properties in velocity space
    #    maxv - if mapping in velocity space, this is the cut +/-, in km/s
    #    CSpace - if set, take CHIMES outputs and map the molecular mass.
    #               (if VSpace set, map this in velocity space too)
    #    Verbose - if set, mapping and functions print out statuses
    
    if Verbose: print("Make 2D Galaxy Map %s, snapshot %03d (at %.2f kpc)." % (RunName, SnapNum, res))
    
    bigtic=tm.default_timer()
    
    #######
    #  These directories point the code to where the snapshots live,
    #   and where maps will be saved! Edit these appropriately!
    #
    subvariant_name = sVarName
    mBase = "/mnt/home/twagg/ceph/FIREmaps/" # base for maps, where they're saved
    SnapDir = f"/mnt/home/chayward/firesims/fire2/public_release/core/{RunName}/output/"
    #######

    Header = readsnap(SnapDir, SnapNum, 4, header_only=1, cosmological=1)

    Time = Header["time"]; ascale = Time
    Redshift = 1.0/ascale - 1.0
    Hubble = Header["hubble"]; hinv = 1.0/Hubble
    if (Redshift<0): Redshift = 0.0
    Boxsize = Header["boxsize"]
    #print not(Header['metals_key'])
    #if not(len(Header['metals_key'])) and not(Header['metals_key']):
    #    NMetal = len(Header['metals_key'][Header['metals_key'] >= 0])
    #else: NMetal = 11
    NMetal = 11
    # set up the grid.
    # gridsize - if Flag_Rvir=1, gridsize=frac*rvir, if Flag_Rhalf=1, gridsize=frac*rhalf
    # resolution - keep Nx at least Nx_min, at most Nx_max, otherwise use gridsize/res
    if (Flag_Rhalf): gridsize = frac*rhalf; Nx_max = 200
    if (Flag_Rvir): gridsize = frac*rvir; Nx_max = 400
    Nx = np.min([Nx_max, np.max([int(gridsize/res), Nx_min])])
    cellsize = gridsize/Nx
    print("Gridsize = %.2f kpc, Nx = %d" % (gridsize, Nx))
    
    MapDir = mBase + RunName + subvariant_name
    for path in [os.path.join(mBase, RunName), MapDir, os.path.join(mBase, RunName, "centers")]:
        if not os.path.exists(path):
            os.mkdir(path)
    MapFile = MapDir+"/map_%03d_%03dpc.hdf5" % (SnapNum, cellsize*1e3)
    file = h5py.File(MapFile, 'a')
    # Map/gal Attributes
    file.attrs["RunName"] = RunName
    file.attrs["SnapNum"] = SnapNum
    file.attrs["Time"] = Time
    file.attrs["Redshift"] = Redshift
    file.attrs["NMetal"] = NMetal
    file.attrs["BoxSize"] = Boxsize
    file.attrs["GridSize"] = gridsize
    file.attrs["Nx"] = Nx
    file.attrs["CellSize"] = cellsize
    if VSpace:
        file.attrs["maxv"] = maxv
        file.attrs["dv"] = dv
    file.close()
    sys.stdout.flush() 
    ##############
    # Mapping Functions here.
    #
    
    # Call CenFinder to find gal center, and projection vector. Writes these to MapFile
    # If doing multiple map resolutions, only need to call this once?
    tic=tm.default_timer()
    
    CenFinder(SnapDir, MapFile, J=Flag_J, force=0, loud=Verbose)
    
    toc=tm.default_timer()
    
    if Verbose: print("Center coords & normal vector found in %.2f sec." % (toc-tic))
    sys.stdout.flush() 
    # Call StarMapper to calculate star particle properties. Writes these to MapFile
    tic=tm.default_timer()

    # StarMapper(SnapDir, MapFile, vSpace=VSpace, loud=Verbose)
    
    toc=tm.default_timer()
    if Verbose: print("Stellar properties mapped in %.2f sec." % (toc-tic))
    sys.stdout.flush() 
    # Call GasMapper to calculate gas particle properties. Writes these to MapFile
    tic=tm.default_timer()
    
    #GasMapper(SnapDir, MapFile, vSpace=VSpace, chemSpace=CSpace, loud=Verbose)
    
    toc=tm.default_timer()
    if Verbose: print("Gas properties mapped in %.2f sec." % (toc-tic))
    sys.stdout.flush() 
    # Call OmegaMapper to calculate dynamical time. Writes these to MapFile
    tic=tm.default_timer()
    
    # OmegaMapper(SnapDir, MapFile, loud=Verbose)
    
    toc=tm.default_timer()
    if Verbose:
        print("Dynamical time mapped in %.2f sec." % (toc-tic))
        bigtoc=tm.default_timer()
        print("Total Mapping time: %.2f sec." % (bigtoc-bigtic))
    sys.stdout.flush()
    
    return

##################################################################################################################################

def CenFinder(SnapDir, MapFile, theta=0.0, phi=0.0, J=1, force=0, loud=0):
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
    
    # Open the mapfile - read necessary data
    file = h5py.File(MapFile, 'r+')
    SnapNum = file.attrs["SnapNum"]
    gridsize = file.attrs["GridSize"]
    
    # Check if center has already been found
    
    cenFile = os.path.split(MapFile)[0]+"/centers/snap_"+str(SnapNum)+"_cents.hdf5"
    
    if (os.path.exists(cenFile) and not force):
        ticky = tm.default_timer()
        cfile = h5py.File(cenFile, 'r+')
        # Write to file and close.
        file.attrs["Theta"] = cfile.attrs["Theta"]
        file.attrs["Phi"] = cfile.attrs["Phi"]
        file.attrs["Center"] = cfile.attrs["Center"]
        file.attrs["StellarCenter"] = cfile.attrs["StellarCenter"]
        file.attrs["StellarCMVel"] = cfile.attrs["StellarCMVel"]
        file.attrs["NormalVector"] = cfile.attrs["NormalVector"]
        file.attrs["Rhalfstar"] = cfile.attrs["Rhalfstar"]
        file.close()
        cfile.close()
        tocky = tm.default_timer()
        
        if loud: print("Center and normal vector previously found, loaded in %.2f sec." % (tocky-ticky))
    # Otherwise, do the hard work yourself.
    else:
        # Pull up the star particle positions, and masses
        toc1=tm.default_timer()
        stars = readsnap_simple(SnapDir, SnapNum, 4, cosmological=1)
        toc2=tm.default_timer()
        ps=stars['p'] #position
        vs=stars['v'] #velocity
        ms=stars['m'] #mass
        # Pull up the gas particle positions, and number densities.
        toc3=tm.default_timer()
        gas = readsnap_simple(SnapDir, SnapNum, 0, cosmological=1)
        toc4=tm.default_timer()
        pg =gas['p']         #position
        rho =gas['rho']*404 #num density
        
        # Calcualte stellar center
        toc5=tm.default_timer()
        #xc, yc, zc = calculate_star_center(ps,ms,pg,rho)
        xc, yc, zc = gaussfit_star_center(ps,ms,pg,rho) # faster method, dead nuts on too.
        toc6=tm.default_timer()
        # Shift coordinates, recenter
        xs = ps[:,0] - xc; ys = ps[:,1] - yc; zs = ps[:,2] - zc
        toc7=tm.default_timer()
        (xsc, ysc, zsc) = Recent([xs, ys, zs], ms)
        toc8=tm.default_timer()
        rhalf = half_mass_radius(xs, ys, zs, ms, xsc, ysc, zsc, gridsize)
        toc9=tm.default_timer()
        if loud: print("Half-mass radius %.2f kpc" % (rhalf))
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
            tocCM=tm.default_timer()
            Vx_cm, Vy_cm, Vz_cm = CMvelocity(xs, ys, zs, ms, vs[:,0], vs[:,1], vs[:,2], 4.*rhalf)
            ticCM=tm.default_timer()
            if loud: print("Function: CMvelocity ran in %.2f sec." % (ticCM-tocCM))
        if loud: print("Loaded star particles (simple) in %.2f sec." % (toc2-toc1))
        if loud: print("Loaded gas particles (simple) in %.2f sec." % (toc4-toc3))
        if loud: print("Function: calculate_star_center ran in %.2f sec." % (toc6-toc5))
        if loud: print("Function: Recent ran in %.2f sec." % (toc8-toc7))
        if loud: print("Function: half_mass_radius ran in %.2f sec." % (toc9-toc8))
        # make the projection
        nx = np.array([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)])
        ny = np.array([-np.sin(phi), np.cos(phi), 0])
        nz = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        
        # Write to file and close.
        file.attrs["Theta"] = theta
        file.attrs["Phi"] = phi
        file.attrs["Center"] = [xc, yc, zc]
        file.attrs["StellarCenter"] = [xc+xsc, yc+ysc, zc+zsc]
        file.attrs["StellarCMVel"] = [Vx_cm, Vy_cm, Vz_cm]
        file.attrs["NormalVector"] = [nx, ny, nz]
        file.attrs["Rhalfstar"] = rhalf

        if (not os.path.exists(os.path.split(MapFile)[0]+"/centers")): os.system("mkdir "+os.path.split(MapFile)[0]+"/centers")
        cfile = h5py.File(cenFile, 'a')
        # Write Map/gal Attributes to cenFile, to be referenced in future.
        cfile.attrs["RunName"] = file.attrs["RunName"]
        cfile.attrs["SnapNum"] = SnapNum
        cfile.attrs["Time"] = file.attrs["Time"]
        cfile.attrs["Redshift"] = file.attrs["Redshift"]
        cfile.attrs["Theta"] = theta
        cfile.attrs["Phi"] = phi
        cfile.attrs["Center"] = [xc, yc, zc]
        cfile.attrs["StellarCMVel"] = [Vx_cm, Vy_cm, Vz_cm]
        cfile.attrs["StellarCenter"] = [xc+xsc, yc+ysc, zc+zsc]
        cfile.attrs["NormalVector"] = [nx, ny, nz]
        cfile.attrs["Rhalfstar"] = rhalf
        
        cfile.close()

        file.close()

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

def StarMapper(SnapDir, MapFile, vSpace=0, loud=0):
    # StarMapper function.
    #   Maps properties of star particles in snapshot
    #   adds those to the given map.

    #   Input:
    #       Nx - number of pixels along a dimension of the maps
    #       Xc - len-3 vector, of the coordinates of stellar center

    # Open the MapFile, read relavent data
    file = h5py.File(MapFile, 'r+')
    
    SnapNum = file.attrs["SnapNum"]
    Xsc = file.attrs["StellarCenter"]
    (nx, ny, nz) = file.attrs["NormalVector"]
    Vcm = file.attrs["StellarCMVel"]
    gridsize = file.attrs["GridSize"]
    cellsize = file.attrs["CellSize"]
    Nx = file.attrs["Nx"]
    NMetal = file.attrs["NMetal"]
    if vSpace:
        maxv=file.attrs["maxv"]
        dv=file.attrs["dv"]
        Nv = int(2.*maxv/dv)
    # Load star particles from snapshot
    tic=tm.default_timer()
    star = readsnap(SnapDir, SnapNum, 4, cosmological=1)
    Header = readsnap(SnapDir, SnapNum, 4, header_only=1, cosmological=1)
    toc=tm.default_timer()
    if loud: print("Loaded star particles in %.2f sec." % (toc-tic))
    ps = star['p'] # position
    ms = 1.0e10*star['m'] # mass in solar mass
    Z_s = star['z'][:,0:NMetal] # metallicity
    #NMetal= len(Z_s[1,:])
    vs = star['v'] # velocity
    age = get_stellar_ages(star, Header, cosmological=1) # stellar age
    
    ls = colors_table(age, Z_s[:,0]/0.02, band_name='sdss_i', QUIET=1)*ms # SDSS i band luminosity in solar units

    # shift the coordinates, trim out all those that can't possibly be mapped
    xs = ps[:,0]-Xsc[0]; ys = ps[:,1]-Xsc[1]; zs = ps[:,2]-Xsc[2]
    
    r2 =xs*xs+ys*ys+zs*zs
    ok = r2<(gridsize*gridsize*0.75)
    xs=xs[ok];ys=ys[ok];zs=zs[ok]
    # remove the CM component of velocity field. ('peculiar vel. of galaxy')
    vxs=vs[:,0][ok]-Vcm[0];vys=vs[:,1][ok]-Vcm[1];vzs=vs[:,2][ok]-Vcm[2]
    ms=ms[ok];age=age[ok];Z_s=Z_s[ok];ls=ls[ok]

    # make the projection
    # Project star coords
    X_s = [xs*nx[0] + ys*nx[1] + zs*nx[2],\
           xs*ny[0] + ys*ny[1] + zs*ny[2],\
           xs*nz[0] + ys*nz[1] + zs*nz[2]]
    V_s = [vxs*nx[0] + vys*nx[1] + vzs*nx[2],\
           vxs*ny[0] + vys*ny[1] + vzs*ny[2],\
           vxs*nz[0] + vys*nz[1] + vzs*nz[2]]
    # Convert velocity to cylindrical coordinates:
    V_cyl= [(X_s[0]*V_s[0]+X_s[1]*V_s[1])/(X_s[0]**2.+X_s[1]**2.)**0.5,\
            (X_s[0]*V_s[1]-X_s[1]*V_s[0])/(X_s[0]**2.+X_s[1]**2.)**0.5,\
            V_s[2][:]] # v_r, v_phi, v_z
    
    # Star Formation Rates
    SFR_ave_010 = np.zeros((Nx,Nx)) # averaged SFR over 10 Myr
    SFR_ave_040 = np.zeros((Nx,Nx)) # averaged SFR over 40 Myr
    SFR_ave_100 = np.zeros((Nx,Nx)) # averaged SFR over 100 Myr
    
    # Stellar Properties
    StarMass = np.zeros((Nx,Nx)) # Stellar mass
    #StarNpart = np.zeros((Nx,Nx)) # number of star particle
    #StarNpart_010 = np.zeros((Nx,Nx)) # number of star particle, age < 0.01 Gyr
    #StarNpart_040 = np.zeros((Nx,Nx)) # number of star particle, age < 0.04 Gyr
    #StarNpart_100 = np.zeros((Nx,Nx)) # number of star particle, age < 0.10 Gyr
    
    StarLum = np.zeros((Nx,Nx)) # Stellar luminosity (SDSS i band)
    
    #StarLum_010 = np.zeros((Nx,Nx)) # Stellar luminosity (SDSS i band)
    
    StarZ = np.zeros((Nx,Nx,NMetal))
    #StarZ_l = np.zeros((Nx,Nx,NMetal)) # stellar metallicity, weighted by luminosity
    StarV_m = np.zeros((Nx,Nx,3))
    #StarV_l = np.zeros((Nx,Nx,3)) # stellar velocity, weighted by mass/lum
    StarV_010_m = np.zeros((Nx,Nx,3)) # Young star velocity, < 10 Myr, weighted by mass. Sim to HII regions (X, Y, Z) for each pixel
    StarVDisp_010_m = np.zeros((Nx,Nx,3)) # Young star velocity dispersion, < 10 Myr, weighted by lum. Sim to HII regions
    #StarV_010_l = np.zeros((Nx,Nx,3)) # Young star velocity, < 10 Myr, weighted by lum. Sim to HII regions (X, Y, Z) for each pixel
    #StarVDisp_010_l = np.zeros((Nx,Nx,3)) # Young star velocity dispersion, < 10 Myr, weighted by mass. Sim to HII regions
    StarVDisp_m = np.zeros((Nx,Nx,3))
    #StarVDisp_l = np.zeros((Nx,Nx,3)) # stellar velocity dispersion, weighted by mass/lum
    
    StarAge_m = np.zeros((Nx,Nx)) # average stellar age, weighted by mass/lum
    #StarAge_l = np.zeros((Nx,Nx))
    StarAgeDisp_m = np.zeros((Nx,Nx)) # stellar age dispersion, weighted by mass/lum
    #StarAgeDisp_l = np.zeros((Nx,Nx))
    StarAvgz = np.zeros((Nx,Nx)) # stellar avg z-coordinate ..not saved.
    StarsH = np.zeros((Nx,Nx)) # stellar rms-z-coordinate (scale Height).
    
    if vSpace:
        vStarMass = np.zeros((Nx,Nx,Nv))
        #vStarMass_sqr = np.zeros((Nx,Nx,Nv))
        vStarAge_m = np.zeros((Nx,Nx,Nv))
        vStarAgeDisp_m = np.zeros((Nx,Nx,Nv))
        vSFR_ave_010 = np.zeros((Nx,Nx,Nv))
        vSFR_ave_100 = np.zeros((Nx,Nx,Nv))
    # calculate the grid
    # stars
    tics=tm.default_timer()
    star_id = np.arange(len(ms))
    tic=tm.default_timer()
    DRANGE = range(3)
    if vSpace: star_in_mask = ((np.abs(X_s[0])<gridsize/2) & (np.abs(X_s[1])<gridsize/2) & (np.abs(X_s[2])<gridsize/2) & (np.abs(V_s[2])<maxv)) # throw in the additional cut in vel. space
    else: star_in_mask = ((np.abs(X_s[0])<gridsize/2) & (np.abs(X_s[1])<gridsize/2) & (np.abs(X_s[2])<gridsize/2))
    toc=tm.default_timer()
    count=0; starnum=len(star_id[star_in_mask])
    if loud:
        print("Masked mappable star particles in %.2f sec." % (toc-tic))
        chkpts=np.array([0.25,0.5,0.75,1.0])
        chknum=np.fix(chkpts*starnum)
        print("Star particles in mapping region: "+str(starnum)+", %2.f %% of total." % ( np.fix(100.*starnum/len(ps))))
    strlist=star_id[star_in_mask].tolist()
    for i in strlist:
        count+=1
        if loud:
            if np.any(chknum==count):
                print("Stars calculation is "+str(int(100*float(count)/starnum))+"% done ("+str(count)+" particles).")
        # which cell the particle belongs to?
        ix = int((X_s[0][i]+gridsize/2)/cellsize)
        iy = int((X_s[1][i]+gridsize/2)/cellsize)
        if vSpace:
            iv = int((V_s[2][i]+maxv)/dv)
            
            vStarAge_m[ix, iy, iv], vStarAgeDisp_m[ix, iy, iv] = SimpleRunningAvg_n_Std(vStarAge_m[ix, iy, iv],\
                    vStarAgeDisp_m[ix, iy, iv], vStarMass[ix, iy, iv], age[i], ms[i])
            vStarMass[ix, iy, iv] += ms[i]

        
        for dd in DRANGE:
            StarV_m[ix, iy, dd], StarVDisp_m[ix, iy, dd] = SimpleRunningAvg_n_Std(StarV_m[ix, iy, dd],\
                        StarVDisp_m[ix, iy, dd], StarMass[ix, iy], V_cyl[dd][i], ms[i])
        StarAge_m[ix, iy], StarAgeDisp_m[ix, iy] = SimpleRunningAvg_n_Std(StarAge_m[ix, iy],\
                StarAgeDisp_m[ix, iy], StarMass[ix, iy], age[i], ms[i])
        StarAvgz[ix, iy], StarsH[ix, iy] = SimpleRunningAvg_n_Std(StarAvgz[ix, iy],\
                StarsH[ix, iy], StarMass[ix, iy], X_s[2][i], ms[i])
        
        
        if (age[i] < 0.1):
            SFR_ave_100[ix, iy] += ms[i]
            if vSpace: vSFR_ave_100[ix, iy, iv] += ms[i]
            if (age[i] < 0.04):
                SFR_ave_040[ix, iy] += ms[i]
                if (age[i] < 0.01):
                    for dd in DRANGE:
                        StarV_010_m[ix, iy, dd], StarVDisp_010_m[ix, iy, dd] = SimpleRunningAvg_n_Std(StarV_010_m[ix, iy, dd],\
                            StarVDisp_010_m[ix, iy, dd], SFR_ave_010[ix, iy], V_cyl[dd][i], ms[i])
                    SFR_ave_010[ix, iy] += ms[i];
                    if vSpace: vSFR_ave_010[ix, iy, iv] += ms[i]

        # other regular quantities
        StarMass[ix, iy] += ms[i]

        StarLum[ix, iy] += ls[i]
        
        StarZ[ix, iy, :] += ms[i]*Z_s[i,:]
        
    tocs=tm.default_timer()
    if loud: print("Star particle calculations done in %.2f sec." % (tocs-tics))
    
    for i in range(NMetal):
        StarZ[StarMass>0,i] /= StarMass[StarMass>0]
    # check all dispersions are greater than zero, set zero if necessary.
    StarVDisp_m[StarVDisp_m<0.]=0.
    StarVDisp_010_m[StarVDisp_010_m<0.]=0.
    StarAgeDisp_m[StarAgeDisp_m<0.]=0.
    StarsH[StarsH<0.]=0.
    for dd in DRANGE:
        StarVDisp_m[:,:,dd] = np.sqrt(StarVDisp_m[:,:,dd])
        StarVDisp_010_m[:,:,dd] = np.sqrt(StarVDisp_010_m[:,:,dd])

    StarAgeDisp_m = np.sqrt(StarAgeDisp_m)
    StarsH = np.sqrt(StarsH)
    # turn masses in bins to rates, by dividing out its dt.
    SFR_ave_010 /= 1.0e7
    SFR_ave_040 /= 4.0e7
    SFR_ave_100 /= 1.0e8

    # Write data to MapFile, and close it.

    file.create_dataset("SFR_ave_010", data=SFR_ave_010)
    file.create_dataset("SFR_ave_040", data=SFR_ave_040)
    file.create_dataset("SFR_ave_100", data=SFR_ave_100)

    file.create_dataset("StarMass", data=StarMass)

    file.create_dataset("StarLum", data=StarLum)
    file.create_dataset("StarZ", data=StarZ)
    file.create_dataset("StarV_m", data=StarV_m)
    file.create_dataset("StarV_010_m", data=StarV_010_m)
    #file.create_dataset("StarV_010_l", data=StarV_010_l) # NOTE: not sure which band to use for L.. from <10 Myr sources
    file.create_dataset("StarVDisp_m", data=StarVDisp_m)
    #file.create_dataset("StarVDisp_l", data=StarVDisp_l)
    file.create_dataset("StarVDisp_010_m", data=StarVDisp_010_m)
    #file.create_dataset("StarVDisp_010_l", data=StarVDisp_010_l) # NOTE: not sure which band to use for L.. from <10 Myr sources
    file.create_dataset("StarAge_m", data=StarAge_m)
    file.create_dataset("StarAgeDisp_m", data=StarAgeDisp_m)
    file.create_dataset("StarsH", data=StarsH)
    if vSpace:
        # post-process velocity space stellar quantities.
        vSFR_ave_010 /= 1.0e7
        vSFR_ave_100 /= 1.0e8
        vStarAgeDisp_m[vStarAgeDisp_m<0.]=0.
        vStarAgeDisp_m = np.sqrt(vStarAgeDisp_m)
        file.create_dataset("vStarMass", data=vStarMass)
        file.create_dataset("vStarAge_m", data=vStarAge_m)
        file.create_dataset("vStarAgeDisp_m", data=vStarAgeDisp_m)
        file.create_dataset("vSFR_ave_100", data=vSFR_ave_100)
        file.create_dataset("vSFR_ave_010", data=vSFR_ave_010)

    file.close()

    return

##################################################################################################################################

def GasMapper(SnapDir, MapFile, vSpace=0, chemSpace=0, loud=0):
    # GasMapper - generates map of gas properties.
    # Inputs:
    #       SnapDir - directory where snapshot to map lives.
    #       MapFile - file to write gas map quantities to.
    # Flags:
    #       vSpace - enable velocity space mapping, i.e. 3rd dim to maps (default off).
    #       dv - velocity space resolution, in km/s (default 1 km/s).
    #       maxv - +/- value to cut at in velocity space (default +/-100 km/s).
    #       loud - verbose running of function, tell me how long it takes to run! (default quiet)
    
    
    # Open the MapFile, read relavent data
    file = h5py.File(MapFile, 'r+')
    
    SnapNum = file.attrs["SnapNum"]
    Xsc = file.attrs["StellarCenter"]
    (nx, ny, nz) = file.attrs["NormalVector"]
    Vcm = file.attrs["StellarCMVel"]
    gridsize = file.attrs["GridSize"]
    cellsize = file.attrs["CellSize"]
    Nx = file.attrs["Nx"]
    NMetal = file.attrs["NMetal"]
    boxsize = file.attrs["BoxSize"] # boxsize of parent snap, for meshoid routine
    if vSpace:
        maxv=file.attrs["maxv"]
        dv=file.attrs["dv"]
        Nv = int(2.*maxv/dv)
    # gas
    tic=tm.default_timer()
    gas = readsnap(SnapDir, SnapNum, 0, cosmological=1)
    toc=tm.default_timer()
    if loud: print("Loaded gas particles in "+str(toc-tic)+" sec.")
    sys.stdout.flush()

    pg = gas['p'] # position
    mg = 1.0e10*gas['m'] # mass in solar mass
    sfr = gas['sfr']
    Z_g = gas['z'][:,0:NMetal] # metallicity
    #NMetal= len(Z_g[1,:])
    vg = gas['v'] # velocity
    temp = gas_temperature(gas['u'],gas['ne']) # temperature
    nh = gas['nh'] # neutral hydrogen fraction
    ne = gas['ne']*mg*(1-Z_g[:,0]-Z_g[:,1])*1.19e57 # N_free-electrons
    mh = nh*mg*(1-Z_g[:,0]-Z_g[:,1]) # neutral hydrogen mass (He corrected)
    rho = gas['rho']*404.367*(1-Z_g[:,0]-Z_g[:,1]) # number density (xchma had it multiplied by 407.8)
    h_g = gas['h']  # smoothing lengths of gas particles

    #alpha_const = 1.17e-13 
    alpha_draine = 1.17e-13*(temp/1e4)**(-0.942-0.031*np.log(temp/1e4))
    #E_H_alpha_erg = 3.03e-12
    E_H_alpha_Lsuns = 7.88e-46 
    recomb_rate = 2.94e64*(gas['m']/gas['rho'])*gas['ne']*(1.-nh)*rho**2.  # units: 1/cm^3 crappy, proportional to Halpha!
    L_Halpha_Lsun = recomb_rate * alpha_draine * E_H_alpha_Lsuns
    
    if chemSpace:
        tic=tm.default_timer()
        chemFileName = SnapDir+'/snapshot_'+str(SnapNum)+'_chem_reduced.hdf5'
        if SnapNum < 100:
            chemFileName = SnapDir+'/snapshot_0'+str(SnapNum)+'_chem_reduced.hdf5'
        chemFile = h5py.File(chemFileName,'r');
        chemKeys =['CI', 'CII', 'CO', 'H2', 'HI', 'HII', 'HeI', 'NII', 'elec'] #..not using all: chemFile.keys()
        # atomic mass for the aforementioned species
        molMasses = [12.,12.,28.,2.,1.,1.,4.,14.,1.] #electron given 1. since we are just getting NUMBER/hydrogen atom not mass of electrons
        assert len(molMasses)==len(chemKeys) # better be true..
        Xchem = np.zeros([len(chemFile[chemKeys[0]]),len(chemKeys)])
        for jj in range(len(chemKeys)):
            Xchem[:,jj] = chemFile[chemKeys[jj]]*mg*(1-Z_g[:,0]-Z_g[:,1])
        toc=tm.default_timer()
        if loud:print("Loaded chemical abundances in "+str(toc-tic)+" sec.")

    sys.stdout.flush() 
    # Gas coords
    # shift the coordinates, trim out all those that can't possibly be mapped
    xg = pg[:,0]-Xsc[0]; yg = pg[:,1]-Xsc[1]; zg = pg[:,2]-Xsc[2]

    # Trim out all particles that can't possibly be mapped
    r2 =xg*xg+yg*yg+zg*zg
    ok = r2<(gridsize*gridsize*0.75)
    xg=xg[ok];yg=yg[ok];zg=zg[ok]
    vxg=vg[:,0][ok]-Vcm[0];vyg=vg[:,1][ok]-Vcm[1];vzg=vg[:,2][ok]-Vcm[2]
    mg=mg[ok];sfr=sfr[ok];Z_g=Z_g[ok];temp=temp[ok]
    nh=nh[ok];mh=mh[ok];rho=rho[ok];ne=ne[ok]
    L_Halpha_Lsun=L_Halpha_Lsun[ok]
    if chemSpace: Xchem=Xchem[ok]

    # project the coordinates
    X_g = [xg*nx[0] + yg*nx[1] + zg*nx[2],\
           xg*ny[0] + yg*ny[1] + zg*ny[2],\
           xg*nz[0] + yg*nz[1] + zg*nz[2]]
    # hacky make it edge-on ### REMOVE WHEN DONE
    #tempX = X_g[0].copy()
    #X_g[0]=X_g[1].copy()
    #X_g[1]=X_g[2].copy()
    #X_g[2]=tempX.copy()
    #tempX=None
    ####

    
    V_g = [vxg*nx[0] + vyg*nx[1] + vzg*nx[2],\
           vxg*ny[0] + vyg*ny[1] + vzg*ny[2],\
           vxg*nz[0] + vyg*nz[1] + vzg*nz[2]] #collect into a nice vector
    # Convert velocity to cylindrical coordinates:
    V_cyl= [(X_g[0]*V_g[0]+X_g[1]*V_g[1])/(X_g[0]**2.+X_g[1]**2.)**0.5,\
            (X_g[0]*V_g[1]-X_g[1]*V_g[0])/(X_g[0]**2.+X_g[1]**2.)**0.5,\
            V_g[2][:]] # v_r, v_phi, v_z
    
    # Initialize map properties
    # Gas Properties
    
    # -- All gas (no cut)
    GasMass = np.zeros((Nx,Nx)) # Gas Mass
    GasV = np.zeros((Nx,Nx,3)) # Gas velocity, (r, phi, Z) for each pixel
    GasVDisp = np.zeros((Nx,Nx,3)) # Gas velocity dispersion
    GasZ = np.zeros((Nx,Nx,NMetal)) # metallicity
    GasTemp = np.zeros((Nx,Nx)) # Gas Temp (mean, mass-weighted)
    GasRho = np.zeros((Nx,Nx)) # Gas Density (mean, mass-weighted)
    # -- "cold" gas: <10^(4.05 - 0.33333/2)
    GasColdMass = np.zeros((Nx,Nx)) # Cold gas mass
    GasColdV = np.zeros((Nx,Nx,3)) # cold gas velocity
    GasColdVDisp = np.zeros((Nx,Nx,3)) # cold gas velocity dispersion
    GasColdZ = np.zeros((Nx,Nx,NMetal)) # cold gas metallicity

    # -- "neutral" gas: f_ion * mh (no temp cut though)
    GasNeutMass = np.zeros((Nx,Nx)) # Neutral hydrogen gas mass, He/metals corrected
    GasNeutV = np.zeros((Nx,Nx,3)) # cold gas velocity
    GasNeutVDisp = np.zeros((Nx,Nx,3)) # cold gas velocity dispersion
    #GasNeutZ = np.zeros((Nx,Nx,NMetal)) # cold gas metallicity

    # -- "Halpha/HII" gas: temp cut   10^(4.05-.3333/2) < T/K < 10^(4.05+.3333/2), ionized.
    GasHIIMass = np.zeros((Nx,Nx)) # HII gas mass
    GasHIIV = np.zeros((Nx,Nx,3)) # HII gas velocity
    GasHIIVDisp = np.zeros((Nx,Nx,3)) # HII gas velocity dispersion
    GasHIIZ = np.zeros((Nx,Nx,NMetal)) # HII gas metallicity

    # -- "cold & dense" gas: temp cut <500 K, dens > 0.01/cc (very nominal density cut)
    GasColdDensMass = np.zeros((Nx,Nx)) # Cold dense gas mass
    GasColdDensV = np.zeros((Nx,Nx,3)) # cold dense gas velocity
    GasColdDensVDisp = np.zeros((Nx,Nx,3)) # cold dense gas velocity dispersion
    GasColdDensZ = np.zeros((Nx,Nx,NMetal)) # cold dense gas metallicity

    # -- "warm neutral" gas: temp cut   500 K < T < 10^(4+.3333/2) K, neutral.
    GasWNMass = np.zeros((Nx,Nx)) # WN gas mass
    GasWNV = np.zeros((Nx,Nx,3)) # WN gas velocity
    GasWNVDisp = np.zeros((Nx,Nx,3)) # WN gas velocity dispersion
    GasWNZ = np.zeros((Nx,Nx,NMetal)) # WN gas metallicity

    SFR = np.zeros((Nx,Nx)) # instantaneous SFR

    GasNe = np.zeros((Nx,Nx)) # free electron column density
    
    ###TESTTESTTEST
    Halpha = np.zeros((Nx,Nx)) #stab at Halpha propto
    HalphaVDisp = np.zeros((Nx,Nx,3)) #stab at Halpha V (lums) 
    HalphaV = np.zeros((Nx,Nx,3)) #stab at Halpha Vdisps 
    
    if vSpace:
        vGasMass = np.zeros((Nx,Nx,Nv))
        vGasNeutMass = np.zeros((Nx,Nx,Nv))
        vGasMoleMass = np.zeros((Nx,Nx,Nv))
        vSFR = np.zeros((Nx,Nx,Nv))
    if chemSpace:
        GasChemMass = np.zeros((Nx,Nx,len(chemKeys)))
        if vSpace: vGasChemMass = np.zeros((Nx,Nx,Nv,len(chemKeys)))
    # chem masses are Msol, electrons are total number.

    # gas
    tics=tm.default_timer()
    gas_id = np.arange(len(mg))

    DRANGE = range(3)
    if vSpace: gas_in_mask = ((np.abs(X_g[0])<gridsize/2) & (np.abs(X_g[1])<gridsize/2) & (np.abs(X_g[2])<gridsize/2) & (np.abs(V_g[2])<maxv)) # throw in the additional cut in vel. space
    else: gas_in_mask = ((np.abs(X_g[0])<gridsize/2) & (np.abs(X_g[1])<gridsize/2) & (np.abs(X_g[2])<gridsize/2))

    count=0; gasnum=len(gas_id[gas_in_mask])
    if loud:
        chkpts=np.array([0.25,0.5,0.75,1.0])
        chknum=np.fix(chkpts*gasnum)
        print("Gas particles in mapping region: "+str(gasnum)+", %3.f %% of total." % ( np.fix(100.*gasnum/len(pg))))
    for i in gas_id[gas_in_mask].tolist():
        count+=1
        if loud:
            if np.any(chknum==count): print("Gas calculation is "+str(int(100*float(count)/gasnum))+"% done ("+str(count)+" particles).")
        # which cell the particle belongs to?
        ix = int((X_g[0][i]+gridsize/2)/cellsize)
        iy = int((X_g[1][i]+gridsize/2)/cellsize)
        if vSpace:
            iv = int((V_g[2][i]+maxv)/dv)
            vGasMass[ix, iy, iv] += mg[i]
            vGasNeutMass[ix, iy, iv] += mh[i]
            vSFR[ix, iy, iv] += sfr[i]
            if chemSpace: vGasChemMass[ix, iy, iv, :] += Xchem[i,:]
        
        # first update the velocity dispersion .. 3 dim now
        for dd in DRANGE:
            GasV[ix, iy, dd], GasVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasV[ix, iy, dd],\
                        GasVDisp[ix, iy, dd], GasMass[ix, iy], V_cyl[dd][i], mg[i])
            GasNeutV[ix, iy, dd], GasNeutVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasNeutV[ix, iy, dd],\
                        GasNeutVDisp[ix, iy, dd], GasNeutMass[ix, iy], V_cyl[dd][i], mh[i])
            HalphaV[ix, iy, dd], HalphaVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(HalphaV[ix, iy, dd], HalphaVDisp[ix, iy, dd], Halpha[ix, iy], V_cyl[dd][i], L_Halpha_Lsun[i])
        # other regular quantities
        GasNeutMass[ix, iy] += mh[i]
        GasMass[ix, iy] += mg[i]
        GasTemp[ix, iy] += temp[i]*mg[i]
        GasRho[ix, iy] += rho[i]*mg[i]
        GasZ[ix, iy, :] += mg[i]*Z_g[i,:]
        SFR[ix, iy] += sfr[i]
        GasNe[ix, iy] += ne[i]
        ###TESTTEST
        Halpha[ix, iy] += L_Halpha_Lsun[i]
        
        if chemSpace: GasChemMass[ix, iy, :] += Xchem[i,:]

        if (temp[i] < 1.6468e4):
            if (temp[i] > 7.644e3):
                # HII Gas Properties
                for dd in DRANGE:
                    GasHIIV[ix, iy, dd], GasHIIVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasHIIV[ix, iy, dd], GasHIIVDisp[ix, iy, dd], GasHIIMass[ix, iy], V_cyl[dd][i], mg[i]*(1.-nh[i]))
                GasHIIMass[ix, iy] += mg[i]*(1.-nh[i]) # HII gas mass
                GasHIIZ[ix, iy, :] += mg[i]*(1.-nh[i])*Z_g[i,:] # HII gas metallicity
                
                # WNM Properties (in the 'HII' band)
                for dd in DRANGE:
                    GasWNV[ix, iy, dd], GasWNVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasWNV[ix, iy, dd], GasWNVDisp[ix, iy, dd], GasWNMass[ix, iy], V_cyl[dd][i], mg[i]*nh[i])
                GasWNMass[ix, iy] += mg[i]*nh[i] # WN gas mass
                GasWNZ[ix, iy, :] += mg[i]*nh[i]*Z_g[i,:] # WN gas metallicity

            elif (temp[i] > 500.):
                # WNM Properties (in the 'warm' band)
                for dd in DRANGE:
                    GasWNV[ix, iy, dd], GasWNVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasWNV[ix, iy, dd], GasWNVDisp[ix, iy, dd], GasWNMass[ix, iy], V_cyl[dd][i], mg[i]*nh[i])
                GasWNMass[ix, iy] += mg[i]*nh[i] # WN gas mass
                GasWNZ[ix, iy, :] += mg[i]*nh[i]*Z_g[i,:] # WN gas metallicity

                # 'Cold' Gas Properties (in 'very cold' band)
                for dd in DRANGE:
                    GasColdV[ix, iy, dd], GasColdVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasColdV[ix, iy, dd], GasColdVDisp[ix, iy, dd], GasColdMass[ix, iy], V_cyl[dd][i], mg[i])
                    GasColdMass[ix, iy] += mg[i]
                    GasColdZ[ix, iy, :] += mg[i]*Z_g[i,:]

            elif ((temp[i] < 500.) and (rho[i] >= 0.1)):
                # C&D Gas Properties
                for dd in DRANGE:
                    GasColdDensV[ix, iy, dd], GasColdDensVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasColdDensV[ix, iy, dd], GasColdDensVDisp[ix, iy, dd], GasColdDensMass[ix, iy], V_cyl[dd][i], mg[i])
                GasColdDensMass[ix, iy] += mg[i]
                GasColdDensZ[ix, iy, :] += mg[i]*Z_g[i,:]
                if vSpace: vGasMoleMass[ix, iy, iv] += mh[i]

                # 'Cold' Gas Properties (in 'very cold' band)
                for dd in DRANGE:
                    GasColdV[ix, iy, dd], GasColdVDisp[ix, iy, dd] = SimpleRunningAvg_n_Std(GasColdV[ix, iy, dd], GasColdVDisp[ix, iy, dd], GasColdMass[ix, iy], V_cyl[dd][i], mg[i])
                GasColdMass[ix, iy] += mg[i]
                GasColdZ[ix, iy, :] += mg[i]*Z_g[i,:]
        
    tocs=tm.default_timer()
    if loud: print("Gas particle calculations done in "+str(tocs-tics)+" sec.")
    # post-processing
    GasTemp[GasMass>0] /= GasMass[GasMass>0]
    GasRho[GasMass>0] /= GasMass[GasMass>0]
    for i in range(NMetal):
        GasZ[GasMass>0,i] /= GasMass[GasMass>0]
        GasColdZ[GasColdMass>0,i] /= GasColdMass[GasColdMass>0]
        GasColdDensZ[GasColdDensMass>0,i] /= GasColdDensMass[GasColdDensMass>0]
        GasHIIZ[GasHIIMass>0,i] /= GasHIIMass[GasHIIMass>0]
        GasWNZ[GasWNMass>0,i] /= GasWNMass[GasWNMass>0]
    GasNe /= (3.086e21*cellsize)**2 # into e-/cm^2
    if chemSpace:
        for i in range(len(molMasses)):
            GasChemMass[:, :, i] *= molMasses[i]
            if vSpace: vGasChemMass[:, :, :, i] *= molMasses[i]
    # make sure all dispersions are positive.
    GasVDisp[GasVDisp<0.] = 0.
    GasColdVDisp[GasColdVDisp<0.] = 0.
    GasColdDensVDisp[GasColdDensVDisp<0.] = 0.
    GasNeutVDisp[GasNeutVDisp<0.] = 0.
    GasHIIVDisp[GasHIIVDisp<0.] = 0.
    GasWNVDisp[GasWNVDisp<0.] = 0.

    for dd in DRANGE:
        GasVDisp[:,:,dd] = np.sqrt(GasVDisp[:,:,dd])
        GasColdVDisp[:,:,dd] = np.sqrt(GasColdVDisp[:,:,dd])
        GasColdDensVDisp[:,:,dd] = np.sqrt(GasColdDensVDisp[:,:,dd])
        GasNeutVDisp[:,:,dd] = np.sqrt(GasNeutVDisp[:,:,dd])
        GasHIIVDisp[:,:,dd] = np.sqrt(GasHIIVDisp[:,:,dd])
        GasWNVDisp[:,:,dd] = np.sqrt(GasWNVDisp[:,:,dd])
    # Write data to MapFile, and close it.
    # Gas Data:
    # All gas --
    file.create_dataset("GasMass", data=GasMass)
    file.create_dataset("GasTemp", data=GasTemp)
    file.create_dataset("GasRho", data=GasRho)
    file.create_dataset("GasNe", data=GasNe)
    file.create_dataset("GasZ", data=GasZ)
    file.create_dataset("GasV", data=GasV)
    file.create_dataset("GasVDisp", data=GasVDisp)
    # 'Cold gas' -
    file.create_dataset("GasColdMass", data=GasColdMass)
    file.create_dataset("GasColdZ", data=GasColdZ)
    file.create_dataset("GasColdV", data=GasColdV)
    file.create_dataset("GasColdVDisp", data=GasColdVDisp)
    # 'C&D gas' --
    file.create_dataset("GasColdDensMass", data=GasColdDensMass)
    file.create_dataset("GasColdDensZ", data=GasColdDensZ)
    file.create_dataset("GasColdDensV", data=GasColdDensV)
    file.create_dataset("GasColdDensVDisp", data=GasColdDensVDisp)
    # HII gas --
    file.create_dataset("GasHIIMass", data=GasHIIMass)
    file.create_dataset("GasHIIZ", data=GasHIIZ)
    file.create_dataset("GasHIIV", data=GasHIIV)
    file.create_dataset("GasHIIVDisp", data=GasHIIVDisp)
    # WN gas --
    file.create_dataset("GasWNMass", data=GasWNMass)
    file.create_dataset("GasWNZ", data=GasWNZ)
    file.create_dataset("GasWNV", data=GasWNV)
    file.create_dataset("GasWNVDisp", data=GasWNVDisp)
    
    # Taking neutral frac, H only. neut is from all gas, mole is from C&D.
    file.create_dataset("GasNeutMass", data=GasNeutMass)
    file.create_dataset("GasNeutV", data=GasNeutV)
    file.create_dataset("GasNeutVDisp", data=GasNeutVDisp)

    file.create_dataset("Halpha", data=Halpha)
    file.create_dataset("HalphaV", data=HalphaV)
    file.create_dataset("HalphaVDisp", data=HalphaVDisp)
    if chemSpace:
        for ii in range(len(chemKeys)):
            file.create_dataset("Gas"+str(chemKeys[ii])+"Mass_chimes",data=GasChemMass[:,:,ii])
            if vSpace:
                file.create_dataset("vGas"+str(chemKeys[ii])+"Mass_chimes",data=vGasChemMass[:,:,:,ii])
    # Star Data:
    # Star formation rates, averages in Myrs.
    file.create_dataset("SFR", data=SFR)
    if vSpace:
        file.create_dataset("vGasMass", data=vGasMass)
        file.create_dataset("vGasNeutMass", data=vGasNeutMass)
        file.create_dataset("vGasMoleMass", data=vGasMoleMass)
        file.create_dataset("vSFR", data=vSFR)
    file.close()
    return

##################################################################################################################################

def OmegaMapper(SnapDir, MapFile, loud=0):
    # Open the MapFile, read relavent data
    file = h5py.File(MapFile, 'a')
    
    SnapNum = file.attrs["SnapNum"]
    try:
        Xsc = file.attrs["StellarCenter"]
        (nx, ny, nz) = file.attrs["NormalVector"]
    except:
        print("Error: No Center in MapFile, running CenFinder.")
        CenFinder(SnapDir, MapFile, J=1, loud=0)
        Xsc = file.attrs["StellarCenter"]
        (nx, ny, nz) = file.attrs["NormalVector"]
    gridsize = file.attrs["GridSize"]
    cellsize = file.attrs["CellSize"]
    Nx = file.attrs["Nx"]
    tic=tm.default_timer()
    dm = readsnap_simple(SnapDir, SnapNum, 1, cosmological=1)
    pdm = dm['p']; mdm = 1.0e10*dm['m']
    xdm = pdm[:,0]; ydm = pdm[:,1]; zdm = pdm[:,2]
                
    gas = readsnap_simple(SnapDir, SnapNum, 0, cosmological=1)
    pgas = gas['p']; mg = 1.0e10*gas['m']
    xg = pgas[:,0]; yg = pgas[:,1]; zg = pgas[:,2]
                            
    star = readsnap_simple(SnapDir, SnapNum, 4, cosmological=1)
    pstar = star['p']; ms = 1.0e10*star['m']
    xs = pstar[:,0]; ys = pstar[:,1]; zs = pstar[:,2]
    toc=tm.default_timer()
    if loud: print("All particles loaded in %.2f sec." % (toc-tic))
    
    tic=tm.default_timer()
    ndx = len(xdm); ngx = len(xg); nsx = len(xs)
    x = np.zeros(ndx+ngx+nsx); x[0:ndx]=xdm;x[ndx:(ndx+ngx)]=xg;x[(ndx+ngx):]=xs
    y = np.zeros(ndx+ngx+nsx); y[0:ndx]=ydm;y[ndx:(ndx+ngx)]=yg;y[(ndx+ngx):]=ys
    z = np.zeros(ndx+ngx+nsx); z[0:ndx]=zdm;z[ndx:(ndx+ngx)]=zg;z[(ndx+ngx):]=zs
    m = np.zeros(ndx+ngx+nsx); m[0:ndx]=mdm;m[ndx:(ndx+ngx)]=mg;m[(ndx+ngx):]=ms
    
    #x = np.concatenate((xdm, xg, xs))
    #y = np.concatenate((ydm, yg, ys))
    #z = np.concatenate((zdm, zg, zs))
        
    #m = np.concatenate((mdm, mg, ms))
    toc=tm.default_timer()
    if loud: print("Vectors concatenated in %.2f sec." % (toc-tic))
    tic=tm.default_timer()
    x -= Xsc[0]; y -= Xsc[1]; z -= Xsc[2]
    
    maxr=gridsize/np.sqrt(2) # max coordinate that x,y,z could have to fall into the omega_dyn region.
    ok=((np.abs(x)<maxr)&(np.abs(y)<maxr)&(np.abs(z)<maxr))
    # project the coordinates (rotations are unitary, so only need to calc R, not new X,Y,Z)
    R = (x[ok]**2+y[ok]**2+z[ok]**2)**0.5
    m = m[ok]
    toc=tm.default_timer()
    if loud: print("Particles centered & projected in %.2f sec." % (toc-tic))
    omega_dyn = np.zeros((Nx,Nx)) # dynamical angular frequence, units: 1/Gyr
    
    # dynamical times
    # we do this calculation in three parts, adding the mass of each
    # component (stars, gas, DM) in separately, then finally doing
    # the omega = sqrt(G*Mint)/r^1.5 calc.
    # Stellar component:
    tic=tm.default_timer()
    ok = (R)<(gridsize/np.sqrt(2))
    G = 4.488e-6  # G in units kpc^3/Gyr^2/M_sol
    r_in = R[ok]
    m_in = m[ok]
    order = np.argsort(r_in)
    r_sorted = r_in[order];m_in_cum = np.cumsum(m_in[order])
    toc=tm.default_timer()
    if loud: print("Radii found and ordered in %.2f sec." % (toc-tic))
    print("Particles in mapping: %d." % (m_in_cum.size))
    tic=tm.default_timer()
    for i in range(Nx*Nx):
        #if (not (i%Nx)): print("row i=", (i/(Nx**2)))
        ix = (np.floor(i/Nx)*cellsize + cellsize/2)-gridsize/2
        iy = ((i%Nx)*cellsize + cellsize/2)-gridsize/2
        rdyn = (ix**2 + iy**2)**0.5
        place = np.searchsorted(r_sorted,rdyn)
        Mint = m_in_cum[place-1] #cumulative mass, in Solar Masses
        omega_dyn[ int(np.floor(i/Nx)), int(i-Nx*np.floor(i/Nx)) ] = (G*Mint)**0.5/rdyn**1.5
    toc=tm.default_timer()
    if loud: print("Dynamical times mapped in %.2f sec." % (toc-tic))
    # Galaxy dynamical times

    file.create_dataset("omega_dyn", data=omega_dyn)
    
    file.close()
    return

##################################################################################################################################

## more robust length checker (will catch scalars)
def checklen(x):
    return len(np.array(x,ndmin=1));




##################################################################################################################################
def cubic_spline(x,h):
    prefactor = 1. / (np.pi * h**3.)
    #if len(x)==1:
    #    if x < 0.5:
    #        return prefactor * ( 1. - 6. * (x/h)**2. * (1. - (x/h)))
    #    elif x < 1.0:
    #        return prefactor * 2. * (1. - (x/h))**3.
    #    else:
    #        return 0.
    #else:
    r = x/h
    kernel = np.zeros(shape=x.shape)
    kernel[r<0.5] = prefactor * ( 1. - 6. * r[r<0.5]**2 * ( 1. - r[r<0.5]))
    kernel[(r>=0.5)&(r<1.)] = prefactor * 2. * (1. - r[(r>=0.5)&(r<1.)])**3
        
    return kernel

def SmoothMapper(SnapDir, MapFile, loud=0):
# Open the MapFile, read relavent data
    file = h5py.File(MapFile, 'r+')
    
    RunName = file.attrs["RunName"]
    SnapNum = file.attrs["SnapNum"]
    Xsc = file.attrs["StellarCenter"]
    (nx, ny, nz) = file.attrs["NormalVector"]
    Vcm = file.attrs["StellarCMVel"]
    gridsize = file.attrs["GridSize"]
    cellsize = file.attrs["CellSize"]
    Nx = file.attrs["Nx"]
    NMetal = file.attrs["NMetal"]
    boxsize = file.attrs["BoxSize"] # boxsize of parent snap, for meshoid routine
    
    # gas
    tic=tm.default_timer()
    gas = readsnap(SnapDir, SnapNum, 0, cosmological=1)
    toc=tm.default_timer()
    if loud: print(("Loaded gas particles in "+str(toc-tic)+" sec."))
    sys.stdout.flush()

    pg = gas['p'] # position
    mg = 1.0e10*gas['m'] # mass in solar mass
    Z_g = gas['z'][:,0:NMetal] # metallicity
    ne = gas['ne']*mg*(1-Z_g[:,0]-Z_g[:,1])*1.19e57 # N_free-electrons
    rho = 1e10 * gas['rho'] #*404.367*(1-Z_g[:,0]-Z_g[:,1]) # number density (xchma had it multiplied by 407.8)
    h_g = gas['h']  # smoothing lengths of gas particles, kpc
    cellVols = mg/rho # kpc^3


    sys.stdout.flush() 
    # Gas coords
    # shift the coordinates, trim out all those that can't possibly be mapped
    xg = pg[:,0]-Xsc[0]; yg = pg[:,1]-Xsc[1]; zg = pg[:,2]-Xsc[2]

    # Trim out all particles that can't possibly be mapped
    r2 =xg*xg+yg*yg+zg*zg
    ok = r2<(gridsize*gridsize*0.75)
    xg=xg[ok];yg=yg[ok];zg=zg[ok]
    
    mg=mg[ok];Z_g=Z_g[ok];ne=ne[ok]
    rho=rho[ok];h_g=h_g[ok]
    cellVols=cellVols[ok]
    
    
    
    # project the coordinates
    X_g = [xg*nx[0] + yg*nx[1] + zg*nx[2],\
           xg*ny[0] + yg*ny[1] + zg*ny[2],\
           xg*nz[0] + yg*nz[1] + zg*nz[2]]

    # -- Quantities to map 
    GasNe = np.zeros((Nx,Nx)) # free electron column density
    GasNe_smooth = np.zeros((Nx,Nx)) # free electron column density (NEW: smoothed over kernel!)
    
 
    # gas
    tics=tm.default_timer()
    gas_id = np.arange(len(mg))
    gas_in_mask = ((np.abs(X_g[0])<gridsize/2) & (np.abs(X_g[1])<gridsize/2) & (np.abs(X_g[2])<gridsize/2))
        
    #####
    
    #min_h_in_mask = np.min(h_g[gas_in_mask])
    
    #subpixel_upscaling = 32 #int(2**(int(np.log2(cellsize/min_h_in_mask))+1)) # will give us min number of subpixels such that NO particle gets through ...this is VERY inefficient, but careful.
    #subpixel_size = cellsize/subpixel_upscaling
    
    #subpixel_vec_x = np.arange(0.,gridsize,subpixel_size)+subpixel_size/2.-gridsize/2.
    #subpixel_vec_inds = np.array(range(len(subpixel_vec_x)))
    
    #Ne_subpixels = np.zeros((Nx*subpixel_upscaling,Nx*subpixel_upscaling))

    pixel_vec_x = np.arange(0.,gridsize,cellsize)+cellsize/2.-gridsize/2.
    pixel_vec_inds = np.array(range(len(pixel_vec_x)))
    
    
    time1=0
    time2=0
    time3=0
    time4=0
    
    count=0; gasnum=len(gas_id[gas_in_mask])
    if loud:
        chkpts=np.arange(0.05,1.01,0.05) #np.array([0.25,0.5,0.75,1.0])
        chknum=np.fix(chkpts*gasnum)
        print(("Gas particles in mapping region: "+str(gasnum)+", %3.f %% of total." % ( np.fix(100.*gasnum/len(pg)))))
        sys.stdout.flush()
    for i in gas_id[gas_in_mask].tolist():
        count+=1
        if loud:
            if np.any(chknum==count):
                print(("Gas (smoothed) calculation is "+str(int(100*float(count)/gasnum))+"% done ("+str(count)+" particles)."))
                sys.stdout.flush()
        # Which subcells do the particle belongs to?
        #if count>1000: continue
        #print(count)

        # which cell the particle belongs to?
        ix = int((X_g[0][i]+gridsize/2)/cellsize)
        iy = int((X_g[1][i]+gridsize/2)/cellsize)

        tic=tm.default_timer()
        GasNe[ix, iy] += ne[i] # old (non-smoothed)
        toc=tm.default_timer()
        time4+=toc-tic
        
        if (h_g[i] < cellsize/2.):
            # particle is 'unresolved'! yay! shortcut.
            GasNe_smooth[ix, iy] += ne[i]
            continue
        else:
            tic=tm.default_timer()
            # particle doesn't fit neatly
            
            x_vec_mask = np.abs(X_g[0][i]-pixel_vec_x)<h_g[i]
            y_vec_mask = np.abs(X_g[1][i]-pixel_vec_x)<h_g[i]

            
            x_small_inds = pixel_vec_inds[x_vec_mask]
            y_small_inds = pixel_vec_inds[y_vec_mask]
        
            nx_small = len(x_small_inds)
            ny_small = len(y_small_inds)
        
            pixel_grid_small_x = np.ones((nx_small,ny_small))*pixel_vec_x[x_vec_mask][:,None]
            pixel_grid_small_y = np.ones((nx_small,ny_small))*pixel_vec_x[y_vec_mask]
        
            r_grid_small = np.sqrt((X_g[0][i]-pixel_grid_small_x)**2+(X_g[1][i]-pixel_grid_small_y)**2) # how far is the particle from every subpixel center
            if np.all(r_grid_small/h_g[i]>1.):
                # Particle is small enough & placed such that it doesn't overlap w/ any pixel centers
                # ...just dump it in the nearest pixel center
                GasNe_smooth[ix, iy] += ne[i]
                continue
            
            toc=tm.default_timer()
            #print(r_grid_small/h_g[i])
            time1+=toc-tic

            tic=tm.default_timer()
            #if len(r_grid_small)==1: print(r_grid_small.shape)
            weights_grid_small = cubic_spline(r_grid_small, h_g[i]) # should give us a full weighted grid to blit the particles on, units: [1/h^3]
            toc=tm.default_timer()
            time2+=toc-tic

            tic=tm.default_timer()
            weights_sum = np.sum(weights_grid_small)    
            
            ne_sub = weights_grid_small * ne[i] / weights_sum
            GasNe_smooth[x_small_inds[0]:x_small_inds[-1]+1,y_small_inds[0]:y_small_inds[-1]+1] += ne_sub

            toc=tm.default_timer()
            time3+=toc-tic

        

    

    print(("rgrid in "+str(time1)+" sec."))
    print(("cubic_spline in "+str(time2)+" sec."))
    print(("subpixels in "+str(time3)+" sec."))
    print(("old pixels in "+str(time4)+" sec."))
    
    GasNe /= (3.086e21*cellsize)**2 # into e-/cm^2
    GasNe_smooth /= (3.086e21*cellsize)**2 # into e-/cm^2
    ### END CALCULATIONS
        
    tocs=tm.default_timer()
    if loud: print(("Gas (smoothed) calculations done in "+str(tocs-tics)+" sec."))

    file.create_dataset("GasNe", data=GasNe)
    file.create_dataset("GasNe_smooth", data=GasNe_smooth)
   
    file.close()
    return
