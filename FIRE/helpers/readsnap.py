import numpy as np
import h5py as h5py
import os.path
import scipy.interpolate as interpolate
import scipy.optimize as optimize
import math


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
        file = open(fname) # Open binary snapshot file
        header_toparse = load_gadget_binary_header(file)

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
    try:
        omega_matter = header_toparse["Omega0"]
        omega_lambda = header_toparse["OmegaLambda"]
    except:
        omega_matter = header_toparse["Omega_Matter"]
        omega_lambda = header_toparse["Omega_Lambda"]
    hubble = header_toparse["HubbleParam"]
    flag_stellarage = header_toparse["Flag_StellarAge"]
    flag_metals = header_toparse["Flag_Metals"]
    if loud == 1:
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
        'boxsize':boxsize,'hubble':hubble,'npart':npart,'npartTotal':npartTotal,'Omega0':omega_matter};

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
            if(fname_ext=='.hdf5'):
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            else:
                file = open(fname) # Open binary snapshot file
                header_toparse = load_gadget_binary_header(file)
                
        if (fname_ext=='.hdf5'):
            input_struct = file
            npart = file["Header"].attrs["NumPart_ThisFile"]
            bname = "PartType"+str(ptype)+"/"
        else:
            npart = header_toparse['NumPart_ThisFile']
            input_struct = load_gadget_binary_particledat(file, header_toparse, ptype, skip_bh=skip_bh)
            bname = ''
            
        
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
                if("MolecularHydrogenFraction" in list(input_struct[bname].keys())):
                    fH2[nL:nR]=input_struct[bname+"MolecularHydrogenFraction"]
                if("ArtificialViscosity" in list(input_struct[bname].keys())):
                    alpha[nL:nR]=input_struct[bname+"ArtificialViscosity"]
                if("Vorticity" in list(input_struct[bname].keys())):
                    vorticity[nL:nR,:]=input_struct[bname+"Vorticity"]
                if (flag_cooling > 0): 
                    nume[nL:nR]=input_struct[bname+"ElectronAbundance"]
                    numh[nL:nR]=input_struct[bname+"NeutralHydrogenAbundance"]
                if (flag_sfr > 0):
                    sfr[nL:nR]=input_struct[bname+"StarFormationRate"]
                if("DivergenceOfMagneticField" in list(input_struct[bname].keys())):
                    divb[nL:nR]=input_struct[bname+"DivergenceOfMagneticField"]
                if("MagneticField" in list(input_struct[bname].keys())):
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
                if("BH_Mass" in list(input_struct[bname].keys())):
                    bhmass[nL:nR]=input_struct[bname+"BH_Mass"]
                if("BH_Mass_AlphaDisk" in list(input_struct[bname].keys())):
                    bhmass_alpha[nL:nR]=input_struct[bname+"BH_Mass_AlphaDisk"]
                if("BH_Mdot" in list(input_struct[bname].keys())):
                    bhmdot[nL:nR]=input_struct[bname+"BH_Mdot"]
            nL = nR # sets it for the next iteration	

	## correct to same ID as original gas particle for new stars, if bit-flip applied
    if ((np.min(ids)<0) | (np.max(ids)>1.e9)):
        bad = (ids < 0) | (ids > 1.e9)
        ids[bad] += (1 << 31)

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
        file = open(fname) # Open binary snapshot file
        header_toparse = load_gadget_binary_header(file)
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
    try:
        omega_matter = header_toparse["Omega0"]
        omega_lambda = header_toparse["OmegaLambda"]
    except:
        omega_matter = header_toparse["Omega_Matter"]
        omega_lambda = header_toparse["Omega_Lambda"]
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
            if(fname_ext=='.hdf5'):
                file = h5py.File(fname,'r') # Open hdf5 snapshot file
            else:
                file = open(fname) # Open binary snapshot file
                header_toparse = load_gadget_binary_header(file)
    
        if (fname_ext=='.hdf5'):
            input_struct = file
            npart = file["Header"].attrs["NumPart_ThisFile"]
            bname = "PartType"+str(ptype)+"/"
        else:
            npart = header_toparse['NumPart_ThisFile']
            input_struct = load_gadget_binary_particledat(file, header_toparse, ptype, skip_bh=skip_bh)
            bname = ''
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
                if("BH_Mass" in list(input_struct[bname].keys())):
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

def readsnap_chimes(sdir,snum,ptype,
             snapshot_name='snapshot',
             extension='.hdf5',
             h0=0,cosmological=0,skip_bh=0,four_char=0,
             header_only=0,loud=0):
    
    if (ptype<0): return {'k':-1};
    if (ptype==1): return {'k':-1};
    if (ptype==2): return {'k':-1};
    if (ptype==3): return {'k':-1};
    if (ptype>5): return {'k':-1};
    
    fname,fname_base,fname_ext = check_if_filename_exists(sdir,snum,\
            snapshot_name=snapshot_name,extension=extension,four_char=four_char)
    if(fname=='NULL'): return {'k':-1}
    if(loud==1): print('loading file : '+fname)

    ## open file and parse its header information
    nL = 0 # initial particle point to start at

    file = h5py.File(fname,'r') # Open hdf5 snapshot file
    header_master = file["Header"] # Load header dictionary (to parse below)
    header_toparse = header_master.attrs
    

    npart = header_toparse["NumPart_ThisFile"]
    time = header_toparse["Time"]
    npartTotal = header_toparse["NumPart_Total"]
    numfiles = header_toparse["NumFilesPerSnapshot"]
    omega_matter = header_toparse["Omega0"]
    hubble = header_toparse["HubbleParam"]
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

    if (npartTotal[ptype]<=0): file.close(); return {'k':-1};
    if (header_only==1): file.close(); return {'k':0,'time':time,
        'hubble':hubble,'npart':npart,'npartTotal':npartTotal};

    # initialize variables to be read
    pos=np.zeros([npartTotal[ptype],3],dtype=np.float64)
    mass=np.zeros([npartTotal[ptype]],dtype=np.float64)
    if (ptype==0):
        vel=np.copy(pos)
        ugas=np.copy(mass)
        rho=np.copy(mass)
        hsml=np.copy(mass)
        nume=np.copy(mass)
        metal=np.copy(mass)
    if (ptype==0 or ptype==4) and (flag_metals > 0):
        metal=np.zeros([npartTotal[ptype],flag_metals],dtype=np.float64)
    if (ptype==4):
        stellage=np.copy(mass)


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
            mass[nL:nR]=input_struct[bname+"Masses"]
            
            if (ptype==0):
                vel[nL:nR,:]=input_struct[bname+"Velocities"]
                ugas[nL:nR]=input_struct[bname+"InternalEnergy"]
                rho[nL:nR]=input_struct[bname+"Density"]
                nume[nL:nR]=input_struct[bname+"ElectronAbundance"]
                hsml[nL:nR]=input_struct[bname+"SmoothingLength"]

            if (ptype==0 or ptype==4) and (flag_metals > 0):
                metal_t=input_struct[bname+"Metallicity"]
                if (flag_metals > 1):
                    if (metal_t.shape[0] != npart[ptype]):
                        metal_t=np.transpose(metal_t)
                else:
                    metal_t=np.reshape(np.array(metal_t),(np.array(metal_t).size,1))
                metal[nL:nR,:]=metal_t
            if (ptype==4):
                stellage[nL:nR]=input_struct[bname+"StellarFormationTime"]

            nL = nR # sets it for the next iteration

    # do the cosmological conversions on final vectors as needed
    pos *= hinv*ascale # snapshot units are comoving
    mass *= hinv
    if (ptype==0):
        vel *= np.sqrt(ascale) # remember gadget's weird velocity units!
        rho *= (hinv/((ascale*hinv)**3))
        hsml *= hinv*ascale
    if (ptype==4) and (cosmological==0):
        stellage *= hinv

    file.close();
    if (ptype==0):
        return {'k':1,'p':pos,'v':vel,'m':mass,'u':ugas,'rho':rho,
            'ne':nume,'z':metal,'Velocity':vel,'Masses':mass,
            'Coordinates':pos,'Density':rho,'InternalEnergy':ugas,
            'Metallicity':metal, 'h':hsml};
    if (ptype==4):
        return {'k':1,'p':pos,'m':mass,'z':metal,'age':stellage}
    return {'k':1,'p':pos,'m':mass}

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



def load_gadget_binary_header(f):
    ### Read header.
    import array
    # Skip 4-byte integer at beginning of header block.
    f.read(4)
    # Number of particles of each type. 6*unsigned integer.
    Npart = array.array('I')
    Npart.fromfile(f, 6)
    # Mass of each particle type. If set to 0 for a type which is present, 
    # individual particle masses from the 'mass' block are used instead.
    # 6*double.
    Massarr = array.array('d')
    Massarr.fromfile(f, 6)
    # Expansion factor (or time, if non-cosmological sims) of output. 1*double. 
    a = array.array('d')
    a.fromfile(f, 1)
    a = a[0]
    # Redshift of output. Should satisfy z=1/a-1. 1*double.
    z = array.array('d')
    z.fromfile(f, 1)
    z = float(z[0])
    # Flag for star formation. 1*int.
    FlagSfr = array.array('i')
    FlagSfr.fromfile(f, 1)
    # Flag for feedback. 1*int.
    FlagFeedback = array.array('i')
    FlagFeedback.fromfile(f, 1)
    # Total number of particles of each type in the simulation. 6*int.
    Nall = array.array('i')
    Nall.fromfile(f, 6)
    # Flag for cooling. 1*int.
    FlagCooling = array.array('i')
    FlagCooling.fromfile(f, 1)
    # Number of files in each snapshot. 1*int.
    NumFiles = array.array('i')
    NumFiles.fromfile(f, 1)
    # Box size (comoving kpc/h). 1*double.
    BoxSize = array.array('d')
    BoxSize.fromfile(f, 1)
    # Matter density at z=0 in units of the critical density. 1*double.
    Omega0 = array.array('d')
    Omega0.fromfile(f, 1)
    # Vacuum energy density at z=0 in units of the critical density. 1*double.
    OmegaLambda = array.array('d')
    OmegaLambda.fromfile(f, 1)
    # Hubble parameter h in units of 100 km s^-1 Mpc^-1. 1*double.
    h = array.array('d')
    h.fromfile(f, 1)
    h = float(h[0])
    # Creation times of stars. 1*int.
    FlagAge = array.array('i')
    FlagAge.fromfile(f, 1)
    # Flag for metallicity values. 1*int.
    FlagMetals = array.array('i')
    FlagMetals.fromfile(f, 1)

    # For simulations that use more than 2^32 particles, most significant word 
    # of 64-bit total particle numbers. Otherwise 0. 6*int.
    NallHW = array.array('i')
    NallHW.fromfile(f, 6)

    # Flag that initial conditions contain entropy instead of thermal energy
    # in the u block. 1*int.
    flag_entr_ics = array.array('i')
    flag_entr_ics.fromfile(f, 1)

    # Unused header space. Skip to particle positions.
    f.seek(4+256+4+4)

    return {'NumPart_ThisFile':Npart, 'MassTable':Massarr, 'Time':a, 'Redshift':z, \
    'Flag_Sfr':FlagSfr[0], 'Flag_Feedback':FlagFeedback[0], 'NumPart_Total':Nall, \
    'Flag_Cooling':FlagCooling[0], 'NumFilesPerSnapshot':NumFiles[0], 'BoxSize':BoxSize[0], \
    'Omega0':Omega0[0], 'OmegaLambda':OmegaLambda[0], 'HubbleParam':h, \
    'Flag_StellarAge':FlagAge[0], 'Flag_Metals':FlagMetals[0], 'Nall_HW':NallHW, \
    'Flag_EntrICs':flag_entr_ics[0]}


def load_gadget_binary_particledat(f, header, ptype, skip_bh=0):
    ## load old format=1 style gadget binary snapshot files (unformatted fortran binary)
    import array
    gas_u=0.; gas_rho=0.; gas_ne=0.; gas_nhi=0.; gas_hsml=0.; gas_SFR=0.; star_age=0.; 
    zmet=0.; bh_mass=0.; bh_mdot=0.; mm=0.;
    Npart = header['NumPart_ThisFile']
    Massarr = header['MassTable']
    NpartTot = np.sum(Npart)
    NpartCum = np.cumsum(Npart)
    n0 = NpartCum[ptype] - Npart[ptype]
    n1 = NpartCum[ptype]
    
    ### particles positions. 3*Npart*float.
    pos = array.array('f')
    pos.fromfile(f, 3*NpartTot)
    pos = np.reshape(pos, (NpartTot,3))
    f.read(4+4) # Read block size fields.

    ### particles velocities. 3*Npart*float.
    vel = array.array('f')
    vel.fromfile(f, 3*NpartTot)
    vel = np.reshape(vel, (NpartTot,3))
    f.read(4+4) # Read block size fields.

    ### Particle IDs. # (Npart[0]+...+Npart[5])*int
    id = array.array('i')
    id.fromfile(f, NpartTot)
    id = np.array(id)
    f.read(4+4) # Read block size fields.
        
    ### Variable particle masses. 
    Npart_MassCode = np.copy(np.array(Npart))
    Npart=np.array(Npart)
    Npart_MassCode[(Npart <= 0) | (np.array(Massarr,dtype='d') > 0.0)] = 0
    NwithMass = np.sum(Npart_MassCode)
    mass = array.array('f')
    mass.fromfile(f, NwithMass)
    f.read(4+4) # Read block size fields.
    if (Massarr[ptype]==0.0):
        Npart_MassCode_Tot = np.cumsum(Npart_MassCode)
        mm = mass[Npart_MassCode_Tot[ptype]-Npart_MassCode[ptype]:Npart_MassCode_Tot[ptype]]

    if ((ptype==0) | (ptype==4) | (ptype==5)):
        if (Npart[0]>0):
            ### Internal energy of gas particles ((km/s)^2).
            gas_u = array.array('f')
            gas_u.fromfile(f, Npart[0])
            f.read(4+4) # Read block size fields.
            ### Density for the gas paraticles (units?).
            gas_rho = array.array('f')
            gas_rho.fromfile(f, Npart[0])
            f.read(4+4) # Read block size fields.

            if (header['Flag_Cooling'] > 0):
                ### Electron number density for gas particles (fraction of n_H; can be >1).
                gas_ne = array.array('f')
                gas_ne.fromfile(f, Npart[0])
                f.read(4+4) # Read block size fields.
                ### Neutral hydrogen number density for gas particles (fraction of n_H).
                gas_nhi = array.array('f')
                gas_nhi.fromfile(f, Npart[0])
                f.read(4+4) # Read block size fields.

            ### Smoothing length (kpc/h). ###
            gas_hsml = array.array('f')
            gas_hsml.fromfile(f, Npart[0])
            f.read(4+4) # Read block size fields.

            if (header['Flag_Sfr'] > 0):
                ### Star formation rate (Msun/yr). ###
                gas_SFR = array.array('f')
                gas_SFR.fromfile(f, Npart[0])
                f.read(4+4) # Read block size fields.

        if (Npart[4]>0):
            if (header['Flag_Sfr'] > 0):
                if (header['Flag_StellarAge'] > 0):
                    ### Star formation time (in code units) or scale factor ###
                    star_age = array.array('f')
                    star_age.fromfile(f, Npart[4])
                    f.read(4+4) # Read block size fields.
        
        if (Npart[0]+Npart[4]>0):
            if (header['Flag_Metals'] > 0):
                ## Metallicity block (species tracked = Flag_Metals)
                if (Npart[0]>0):
                    gas_z = array.array('f')
                    gas_z.fromfile(f, header['Flag_Metals']*Npart[0])
                if (Npart[4]>0):
                    star_z = array.array('f')
                    star_z.fromfile(f, header['Flag_Metals']*Npart[4])
                f.read(4+4) # Read block size fields.
                if (ptype==0): zmet=np.reshape(gas_z,(-1,header['Flag_Metals']))
                if (ptype==4): zmet=np.reshape(star_z,(-1,header['Flag_Metals']))
        
        if (Npart[5]>0):
            if (skip_bh > 0):
                ## BH mass (same as code units, but this is the separately-tracked BH mass from particle mass)
                bh_mass = array.array('f')
                bh_mass.fromfile(f, Npart[5])
                f.read(4+4) # Read block size fields.
                ## BH accretion rate in snapshot
                bh_mdot = array.array('f')
                bh_mdot.fromfile(f, Npart[5])
                f.read(4+4) # Read block size fields.
    
    return {'Coordinates':pos[n0:n1,:], 'Velocities':vel[n0:n1,:], 'ParticleIDs':id[n0:n1], \
        'Masses':mm, 'Metallicity':zmet, 'StellarFormationTime':star_age, 'BH_Mass':bh_mass, \
        'BH_Mdot':bh_mdot, 'InternalEnergy':gas_u, 'Density':gas_rho, 'SmoothingLength':gas_hsml, \
        'ElectronAbundance':gas_ne, 'NeutralHydrogenAbundance':gas_nhi, 'StarFormationRate':gas_SFR}
