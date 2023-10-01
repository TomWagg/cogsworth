import numpy as np
import h5py
import os

def readsnap(sdir,snum,ptype,
             snapshot_name='snapshot',
             extension='.hdf5',
             h0=0,cosmological=0,skip_bh=0,four_char=0,
             header_only=0,loud=0):
    
    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};
    
    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    if(fname=='NULL'): return {'k':-1}
    if(loud==1): print('loading file : '+fname)

    ## open file and parse its header information
    nL = 0 # initial particle point to start at
    if(fname_ext=='.hdf5'):
      file = h5py.File(fname,'r') # Open hdf5 snapshot file
      header_master = file["Header"] # Load header dictionary (to parse below)
      header_toparse = header_master.attrs
    else:
        print("Cannot load non .hdf5 files.")
        return {'k':-1}
          
    npart = header_toparse["NumPart_ThisFile"]
    massarr = header_toparse["MassTable"]
    time = header_toparse["Time"]
    redshift = header_toparse["Redshift"]
    flag_sfr = header_toparse["Flag_Sfr"]
    flag_feedbacktp = header_toparse["Flag_Feedback"]
    npartTotal = header_toparse["NumPart_Total"]
    flag_cooling = header_toparse["Flag_Cooling"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    boxsize = header_toparse["BoxSize"]
    if "Omega0" in header_toparse: omega_matter = header_toparse["Omega0"]
    elif "Omega_Matter" in header_toparse: omega_matter = header_toparse["Omega_Matter"]
    if "OmegaLambda" in header_toparse: omega_lambda = header_toparse["OmegaLambda"]
    elif "Omega_Lambda" in header_toparse: omega_lambda = header_toparse["Omega_Lambda"]
    metals_key = ''
    if "Metals_Atomic_Number_Or_Key" in header_toparse: metals_key = header_toparse["Metals_Atomic_Number_Or_Key"]
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]
    print("npart_file: ",npart)
    print("npart_total:",npartTotal)
          
    hinv=1.
    if (h0==1):
        hinv=1./hubble
    ascale=1.
    if (cosmological==1):
        ascale=time
        hinv=1./hubble
    if (cosmological==0):
        time*=hinv
                                      
    boxsize*=hinv*ascale
    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};
    if (header_only==1): 
        file.close(); 
        return {'k':0,'time':time, 'boxsize':boxsize,
                'hubble':hubble,'npart':npart,
                'npartTotal':npartTotal, 'metals_key':metals_key};
                                                  
    # initialize variables to be read
    pos=np.zeros([npartTotal[ptype],3],dtype=np.float64)
    vel=np.copy(pos)
    ids=np.zeros([npartTotal[ptype]],dtype=int)
    mass=np.zeros([npartTotal[ptype]],dtype=np.float64)
    if (ptype==0):
        ugas=np.copy(mass)
        rho=np.copy(mass)
        hsml=np.copy(mass)
        #if (flag_cooling>0):
        nume=np.copy(mass)
        numh=np.copy(mass)
        #if (flag_sfr>0):
        sfr=np.copy(mass)
        fH2=np.copy(mass)
        metal=np.copy(mass)
        alpha=np.copy(mass)
        vorticity=np.copy(pos)
        bfield=np.copy(pos)
        divb=np.copy(mass)
    if (ptype==0 or ptype==4) and (flag_metals > 0):
        metal=np.zeros([npartTotal[ptype],flag_metals],dtype=np.float64)
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
        stellage=np.copy(mass)
    if (ptype==5) and (skip_bh==0):
        bhmass=np.copy(mass)
        bhmdot=np.copy(mass)
        bhmass_alpha=np.copy(mass)
                                      
    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            file = h5py.File(fname,'r') # Open hdf5 snapshot file

        input_struct = file
        npart = file["Header"].attrs["NumPart_ThisFile"]
        bname = "PartType"+str(ptype)+"/"

                          
                          
        # now do the actual reading
        if(npart[ptype]>0):
            nR=nL + npart[ptype]
            pos[nL:nR,:]=input_struct[bname+"Coordinates"]
            vel[nL:nR,:]=input_struct[bname+"Velocities"]
            ids[nL:nR]=input_struct[bname+"ParticleIDs"]
            mass[nL:nR]=massarr[ptype]
          
            if (massarr[ptype] <= 0.):
                mass[nL:nR]=input_struct[bname+"Masses"]
                  
            if (ptype==0):
                ugas[nL:nR]=input_struct[bname+"InternalEnergy"]
                rho[nL:nR]=input_struct[bname+"Density"]
                hsml[nL:nR]=input_struct[bname+"SmoothingLength"]
                if("MolecularHydrogenFraction" in input_struct[bname].keys()):
                    fH2[nL:nR]=input_struct[bname+"MolecularHydrogenFraction"]
                if("ArtificialViscosity" in input_struct[bname].keys()):
                    alpha[nL:nR]=input_struct[bname+"ArtificialViscosity"]
                if("Vorticity" in input_struct[bname].keys()):
                    vorticity[nL:nR,:]=input_struct[bname+"Vorticity"]
                if (flag_cooling > 0):
                    nume[nL:nR]=input_struct[bname+"ElectronAbundance"]
                    numh[nL:nR]=input_struct[bname+"NeutralHydrogenAbundance"]
                if (flag_sfr > 0):
                    sfr[nL:nR]=input_struct[bname+"StarFormationRate"]
                if("DivergenceOfMagneticField" in input_struct[bname].keys()):
                    divb[nL:nR]=input_struct[bname+"DivergenceOfMagneticField"]
                if("MagneticField" in input_struct[bname].keys()):
                    bfield[nL:nR,:]=input_struct[bname+"MagneticField"]
                                                                              
            if (ptype==0 or ptype==4) and (flag_metals > 0):
                metal_t=input_struct[bname+"Metallicity"]
                if (flag_metals > 1):
                    if (metal_t.shape[0] != npart[ptype]):
                        metal_t=np.transpose(metal_t)
                else:
                    metal_t=np.reshape(np.array(metal_t),(np.array(metal_t).size,1))
                metal[nL:nR,:]=metal_t
                    
            if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0):
                stellage[nL:nR]=input_struct[bname+"StellarFormationTime"]
                                                                                                                  
            if (ptype==5) and (skip_bh==0):
                if("BH_Mass" in input_struct[bname].keys()):
                    bhmass[nL:nR]=input_struct[bname+"BH_Mass"]
                if("BH_Mass_AlphaDisk" in input_struct[bname].keys()):
                    bhmass_alpha[nL:nR]=input_struct[bname+"BH_Mass_AlphaDisk"]
                if("BH_Mdot" in input_struct[bname].keys()):
                    bhmdot[nL:nR]=input_struct[bname+"BH_Mdot"]
            nL = nR # sets it for the next iteration
                
    ## correct to same ID as original gas particle for new stars, if bit-flip applied
    if ((np.min(ids)<0) | (np.max(ids)>1.e9)):
        bad = (ids < 0) | (ids > 1.e9)
        ids[bad] += (1 << 3)
                                                                                                                                                          
    # do the cosmological conversions on final vectors as needed
    pos *= hinv*ascale # snapshot units are comoving
    mass *= hinv
    vel *= np.sqrt(ascale) # remember gadget's weird velocity units!
    if (ptype==0):
        rho *= (hinv/((ascale*hinv)**3))
        hsml *= hinv*ascale
    if (ptype==4) and (flag_sfr>0) and (flag_stellarage>0) and (cosmological==0):
        stellage *= hinv
    if (ptype==5) and (skip_bh==0):
        bhmass *= hinv
                                  
    file.close();
    if (ptype==0):
        return {'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'u':ugas,'rho':rho,
                'h':hsml,'ne':nume,'nh':numh,'sfr':sfr,'z':metal,
                'ArtificialViscosity':alpha,'Velocity':vel,'Masses':mass,'Coordinates':pos,
                'Density':rho,'Hsml':hsml,'ParticleIDs':ids,'InternalEnergy':ugas,
                'Metallicity':metal,'SFR':sfr,'Vorticity':vorticity,
                'B':bfield,'divB':divb,'fh2':fH2};
    if (ptype==4):
        return {'k':1,'p':pos,'v':vel,'m':mass,'id':ids,'z':metal,'age':stellage}
    if (ptype==5) and (skip_bh==0):
        return {'k':1,'p':pos,'v':vel,'m':mass,'id':ids,
                'mbh':bhmass,'mdot':bhmdot,'BHMass_AlphaDisk':bhmass_alpha}
    return {'k':1,'p':pos,'v':vel,'m':mass,'id':ids}

##################################################################################################################################

def readsnap_simple(sdir,snum,ptype,
                    snapshot_name='snapshot',
                    extension='.hdf5',
                    h0=0,cosmological=0,skip_bh=0,four_char=0,
                    header_only=0,loud=0):
    
    if (ptype<0): return {'k':-1};
    if (ptype>5): return {'k':-1};
    
    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
        snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    if(fname=='NULL'): return {'k':-1}
    if(loud==1): print('loading file : '+fname)
                                                          
    ## open file and parse its header information
    nL = 0 # initial particle point to start at
    if(fname_ext=='.hdf5'):
        file = h5py.File(fname,'r') # Open hdf5 snapshot file
        header_master = file["Header"] # Load header dictionary (to parse below)
        header_toparse = header_master.attrs
    else:
        print("Cannot load non .hdf5 files.")
        return {'k':-1}
    npart = header_toparse["NumPart_ThisFile"]
    massarr = header_toparse["MassTable"]
    time = header_toparse["Time"]
    redshift = header_toparse["Redshift"]
    flag_sfr = header_toparse["Flag_Sfr"]
    flag_feedbacktp = header_toparse["Flag_Feedback"]
    npartTotal = header_toparse["NumPart_Total"]
    flag_cooling = header_toparse["Flag_Cooling"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    boxsize = header_toparse["BoxSize"]
    if "Omega0" in header_toparse: omega_matter = header_toparse["Omega0"]
    elif "Omega0" in header_toparse: omega_matter = header_toparse["Omega_Matter"]
    if "OmegaLambda" in header_toparse: omega_lambda = header_toparse["OmegaLambda"]
    elif "Omega_Lambda" in header_toparse: omega_lambda = header_toparse["Omega_Lambda"]
    
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]
    print("npart_file: ",npart)
    print("npart_total:",npartTotal)
                  
    hinv=1.
    if (h0==1):
        hinv=1./hubble
    ascale=1.
    if (cosmological==1):
        ascale=time
        hinv=1./hubble
    if (cosmological==0):
        time*=hinv
                                  
    boxsize*=hinv*ascale
    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};
    if (header_only==1): file.close(); return {'k':0,'time':time,
        'boxsize':boxsize,'hubble':hubble,'npart':npart,'npartTotal':npartTotal};
              
    # initialize variables to be read
    pos=np.zeros([npartTotal[ptype],3],dtype=np.float64)
    vel=np.copy(pos)
    #ids=np.zeros([npartTotal[ptype]],dtype=long)
    mass=np.zeros([npartTotal[ptype]],dtype=np.float64)
      
    if (ptype==0):
        rho=np.copy(mass)
        hsml=np.copy(mass)
    if (ptype==5) and (skip_bh==0): bhmass=np.copy(mass)
    # loop over the snapshot parts to get the different data pieces
    for i_file in range(numfiles):
        if (numfiles>1):
            file.close()
            fname = fname_base+'.'+str(i_file)+fname_ext
            file = h5py.File(fname,'r') # Open hdf5 snapshot file
                          
        input_struct = file
        npart = file["Header"].attrs["NumPart_ThisFile"]
        bname = "PartType"+str(ptype)+"/"
        
        # now do the actual reading
        if(npart[ptype]>0):
            nR=nL + npart[ptype]
            pos[nL:nR,:]=input_struct[bname+"Coordinates"]
            vel[nL:nR,:]=input_struct[bname+"Velocities"]
            #ids[nL:nR]=input_struct[bname+"ParticleIDs"]
            mass[nL:nR]=massarr[ptype]
            if (massarr[ptype] <= 0.): mass[nL:nR]=input_struct[bname+"Masses"]
            if (ptype==0):
                rho[nL:nR]=input_struct[bname+"Density"]
                hsml[nL:nR]=input_struct[bname+"SmoothingLength"]
            if (ptype==5) and (skip_bh==0):
                if("BH_Mass" in input_struct[bname].keys()):
                    bhmass[nL:nR]=input_struct[bname+"BH_Mass"]
            nL = nR # sets it for the next iteration
    ## correct to same ID as original gas particle for new stars, if bit-flip applied
    #if ((np.min(ids)<0) | (np.max(ids)>1.e9)):
    #    bad = (ids < 0) | (ids > 1.e9)
    #    ids[bad] += (1L << 31)
    # do the cosmological conversions on final vectors as needed
    pos *= hinv*ascale # snapshot units are comoving
    mass *= hinv
    vel *= np.sqrt(ascale) # remember gadget's weird velocity units!
    if (ptype==0):
        rho *= (hinv/((ascale*hinv)**3))
        hsml *= hinv*ascale
    if (ptype==5) and (skip_bh==0): bhmass *= hinv
    file.close();
    if (ptype==0):
        return {'k':1,'p':pos,'v':vel,'m':mass,'rho':rho,'h':hsml};
    if (ptype==5) and (skip_bh==0):
        return {'k':1,'p':pos,'v':vel,'m':mass,'mbh':bhmass}
    return {'k':1,'p':pos,'v':vel,'m':mass}
#############################################################################
def check_if_filename_exists(sdir,snum,snapshot_name='snapshot',extension='.hdf5',four_char=0):
    for extension_touse in [extension,'.bin','']:
        fname=sdir+'/'+snapshot_name+'_'
        ext='00'+str(snum);
        if (snum>=10): ext='0'+str(snum)
        if (snum>=100): ext=str(snum)
        if (four_char==1): ext='0'+ext
        if (snum>=1000): ext=str(snum)
        fname+=ext
        fname_base=fname
        
        s0=sdir.split("/"); snapdir_specific=s0[len(s0)-1];
        if(len(snapdir_specific)<=1): snapdir_specific=s0[len(s0)-2];
        
        ## try several common notations for the directory/filename structure
        fname=fname_base+extension_touse;
        if not os.path.exists(fname):
            ## is it a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+ext;
            fname=fname_base+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap(snapdir)' instead of 'snapshot'?
            fname_base=sdir+'/snap_'+snapdir_specific+'_'+ext;
            fname=fname_base+extension_touse;
        if not os.path.exists(fname):
            ## is the filename 'snap' instead of 'snapshot', AND its a multi-part file?
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is it in a snapshot sub-directory? (we assume this means multi-part files)
            fname_base=sdir+'/snapdir_'+ext+'/'+snapshot_name+'_'+ext;
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## is it in a snapshot sub-directory AND named 'snap' instead of 'snapshot'?
            fname_base=sdir+'/snapdir_'+ext+'/'+'snap_'+ext;
            fname=fname_base+'.0'+extension_touse;
        if not os.path.exists(fname):
            ## wow, still couldn't find it... ok, i'm going to give up!
            fname_found = 'NULL'
            fname_base_found = 'NULL'
            fname_ext = 'NULL'
            continue;
        fname_found = fname;
        fname_base_found = fname_base;
        fname_ext = extension_touse
        break; # filename does exist!
    return fname_found, fname_base_found, fname_ext;
##################################################################################################################################

def quick_lookback_time(z,h=0.71,Omega_M=0.27):
    ## exact solution for a flat universe
    a=1./(1.+z); x=Omega_M/(1.-Omega_M) / (a*a*a);
    t=(2./(3.*np.sqrt(1.-Omega_M))) * np.log( np.sqrt(x) / (-1. + np.sqrt(1.+x)) );
    t *= 13.777 * (0.71/h); ## in Gyr
    return t;
#############################################################################
def get_stellar_ages(ppp,ppp_head,cosmological=1):
    ### ppp = loaded 'brick' of new star particle quantities,
    ###        i.e. ppp=gadget.readsnap(directory,number,4,cosmological=1)
    ### ppp_head = loaded 'header' for the snapshot,
    ###        i.e. ppp_head=gadget_readsnap(directory,number,4,cosmological=1,header_only=1)
    if (ppp['k'] != 1): return -1;
    a_form=ppp['age'];
    a_now=ppp_head['time'];
    
    if (cosmological==1):
        z_form=1./a_form-1.; t_form=quick_lookback_time(z_form);
        z_now=1./a_now-1.; t_now=quick_lookback_time(z_now);
        ages = (t_now - t_form); # should be in gyr
    else:
        ages = a_now - a_form; # code units are gyr already!
    
    return ages;
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

def SimpleRunningAvg_n_Std(X_bar_N_minus_1, S2_N_minus_1, W_N_minus_1, x_N, w_N):
    # calculate the weighted (constant weights, not normalized) average and
    #   variance (return population stdev) of population, adding an element.
    #
    #   input:
    #    X_bar_N_minus_1 - weighted mean for first N-1 elements
    #    S2_N_minus_1 - weighted std deviation for first N-1 elements
    #    W_N_minus_1 - summation of weight for first N-1 elements
    #    x_N - the Nth element
    #    w_N - the weight of the Nth element
    #    if W_N_minus_1 <= 0.0: # for the first element, return first element and var = 0
    #       return x_N, 0.0
    W_N = W_N_minus_1 + w_N # new total weight
    X_bar_N = (W_N_minus_1*X_bar_N_minus_1+w_N*x_N)/W_N # calculate new mean.
    S2_N = (W_N_minus_1*S2_N_minus_1 + w_N*(x_N - X_bar_N_minus_1)*(x_N - X_bar_N))/W_N
    
    return X_bar_N, S2_N

def gas_mu(num_e):
    XH=0.76; # we track this with metal species now, could do better...
    yhelium=(1.-XH)/(4.*XH);
    return (1.+4.*yhelium)/(1.+yhelium+num_e);


def gas_temperature(u, num_e, keV=0):
    ## returns gas particles temperature in Kelvin
    
    g_gamma= 5.0/3.0
    g_minus_1= g_gamma-1.0
    PROTONMASS = 1.6726e-24
    BoltzMann_ergs= 1.3806e-16
    UnitMass_in_g= 1.989e43 # 1.0e10 solar masses
    UnitEnergy_in_cgs= 1.989e53
    # note gadget units of energy/mass = 1e10 ergs/g,
    # this comes into the formula below
    
    mu = gas_mu(num_e);
    MeanWeight= mu*PROTONMASS
    Temp= MeanWeight/BoltzMann_ergs * g_minus_1 * u * 1.e10
    
    # do we want units of keV?  (0.001 factor converts from eV to keV)
    if (keV==1):
        BoltzMann_keV = 8.617e-8;
        Temp *= BoltzMann_keV;
    
    return Temp

def InclContours(xData, yData, CtrPcts, weights=None, DatRange=-1, binNum=15):
    # Input:
    #    xData - data along x-axis to find percentile-inclusion contours of
    #    yData - data along y-axis to find percentile-inclusion contours of
    #    CtrPcts - percentiles to find contours of (e.g., 0.5, 0.7, 0.99)
    # Flags:
    #    DatRange - Range to find contours in format: [[xMin, xMax],[yMin, yMax]]
    #               defaults to min/max of supplied data in x/y dimensions.
    #    binNum   - number of bins to cut the data up into, defaults to 15
    # Returns:
    #    CtrCounts - counts to draw the given contours at.
    
    # If no DatRange supplied, default to x/yData min/max.
    if (DatRange==-1):
        DatRange = [[np.min(xData),np.max(xData)],[np.min(yData),np.max(yData)]]
    #s=np.array(CtrPcts).argsort()[::-1][:]
    #CtrPcts=CtrPcts[s]
    s=np.argsort(-np.array(CtrPcts)); CtrPcts=np.array(CtrPcts)[s]
    CtrCounts = np.zeros(len(CtrPcts)) # Initialize contour counts return vect.
    # Get our counts!
    (counts, xed, yed) = np.histogram2d(xData, yData, range=DatRange, bins=binNum, weights=weights)

    thresh=0
    for ii in range(len(CtrPcts)):
        CDFbad=1; CDFgoal=CtrPcts[ii]
        while CDFbad:
            if (np.sum(counts[counts>=thresh])/np.sum(counts))>CDFgoal:
                delt=np.abs((np.sum(counts[counts>=(thresh+1)])/np.sum(counts))-CDFgoal)-np.abs((np.sum(counts[counts>=(thresh)])/np.sum(counts))-CDFgoal) # check if +1 to thresh moves us further from goal..
                if (delt>0.): # if so, quit while we're ahead!
                    CDFbad=0;CtrCounts[ii]=thresh
                else: thresh+=1 # else, keep searchin'
            else: # we've shot past our goal (but we're closer than thresh-1). quit!
                CDFbad=0;CtrCounts[ii]=thresh
    return CtrCounts # send the pct-contour counts

