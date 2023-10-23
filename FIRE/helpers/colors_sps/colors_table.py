def colors_table( age_in_Gyr, metallicity_in_solar_units, 
    BAND_ID=0, SALPETER_IMF=0, CHABRIER_IMF=1, QUIET=0, CRUDE=0, 
    RETURN_NU_EFF=0, RETURN_LAMBDA_EFF=0, UNITS_SOLAR_IN_BAND=0 ):

    import numpy as np
    import math
    import scipy.ndimage.interpolation as interpolate
    import struct

    age_in_Gyr=np.array(age_in_Gyr,ndmin=1);
    metallicity_in_solar_units=np.array(metallicity_in_solar_units,ndmin=1);

    band=BAND_ID; # default=bolometric
    j = [  0,  6,  7,  8,  9, 10, 11, 12, 13,  1,   2,   3,   4,   5] # ordering I'm used to
    i = [  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13] # ordering of this
    band_standardordering = band
    band = j[band]
    if (band > 13): 
        print('BAND_ID must be < 13')
        return 0;
    
    b=['Bolometric', \
    'Sloan u','Sloan g','Sloan r','Sloan i','Sloan z', \
    'Johnsons U','Johnsons B', 'Johnsons V','Johnsons R','Johnsons I', \
    'Cousins J','Cousins H','Cousins K']
    if (QUIET==0): print('Calculating M/L in '+str(b[band])+' ('+str(band)+','+str(band_standardordering)+')')
    
    if (RETURN_NU_EFF==1) or (RETURN_LAMBDA_EFF==1):
        lam_eff=np.array([1.e-5, 3541., 4653., 6147., 7461., 8904., 3600., 4400., \
   		    5556., 6940., 8700., 12150., 16540., 21790.]);
        nu_eff = 2.998e18 / lam_eff;
        if (RETURN_NU_EFF==1): return nu_eff[band];
        if (RETURN_LAMBDA_EFF==1): return lam_eff[band];
    
    froot = '/mnt/home/morr/code/FIREMapper/colors_sps/'; # directory in which the data binaries are stored
    if (CHABRIER_IMF==1): fname=froot+'colors.chabrier.dat'
    if (SALPETER_IMF==1): fname=froot+'colors.salpeter.dat'

    lut = open(fname,'r');
    lut_dat = lut.read();
    Nl,Na,Nz = struct.unpack('3i',lut_dat[0:12])
    z_grid = np.array(struct.unpack(str(Nz)+'d',lut_dat[12:12+8*Nz]))
    age_grid = np.array(struct.unpack(str(Na)+'d',lut_dat[12+8*Nz:12+8*Nz+8*Na]))
    l_all_l = np.array(struct.unpack(str(Nl*Na*Nz)+'d',lut_dat[12+8*Nz+8*Na:12+8*Nz+8*Na+8*Nl*Na*Nz]))
    l_all = np.transpose(l_all_l.reshape(Nz,Na,Nl))
    lut.close()
    
    l_band = np.zeros((Na,Nz),dtype=np.float64);
    for iz in range(Nz): l_band[:,iz]=l_all[band,:,iz]
    
    # allow for extreme metallicities (extrapolate linearly past table)
    push_metals = 1;
    if (push_metals==1):
        Nz = Nz + 1;
        z_ext = [1000.0];
        z_grid = np.concatenate([z_grid,z_ext])
        lb1 = l_band[:,Nz-3]
        lb2 = l_band[:,Nz-2]
        lbx = np.zeros((Na,Nz),dtype=np.float64)
        lbx[:,0:Nz-1] = l_band
        lbx[:,Nz-1] = (lb2 - lb1) / (np.log10(z_grid[Nz-2]/z_grid[Nz-3])) * \
            np.log10(z_grid[Nz-1]/z_grid[Nz-2])
        l_band = lbx;

    # get the x-axis (age) locations of input points
    ia_pts=np.interp(np.log10(age_in_Gyr)+9.0,age_grid,np.arange(0,Na,1)); 
    # this returns the boundary values for points outside of them (no extrapolation)
    #f=interp.interp1d(age_grid,np.arange(0,Na,1),kind='linear'); 
    #ia_pts=f(np.log10(age_in_Gyr)+9.0);

    # get the y-axis (metallicity) locations of input points
    zsun = 0.02;
    iz_pts=np.interp(np.log10(metallicity_in_solar_units*zsun),np.log10(z_grid),np.arange(0,Nz,1)); 
    #f=interp.interp1d(np.log10(z_grid),np.arange(0,Nz,1),kind='linear'); 
    #iz_pts=f(np.log10(metallicity_in_solar_units*zsun));

    if (CRUDE==1):
        ia_pts=np.around(ia_pts).astype(int);
        iz_pts=np.around(iz_pts).astype(int);
        l_b=l_band[ia_pts,iz_pts];
    else:
        l_b = interpolate.map_coordinates(l_band, (ia_pts,iz_pts), order=1);
    l_b = 10.**l_b
	
    # output is currently L/M in L_sun_IN_THE_BAND_OF_INTEREST/M_sun, 
    # but we want our default to be L/M in units of L_bolometric/M_sun = 3.9e33/2.0e33, so 
    #   need to get rid fo the L_sun_IN_THE_BAND_OF_INTEREST/L_bolometric

    # AB system solar luminosities used for determining L_sun in absolute units for each of these
    N_BANDS=14
    mag_sun_ab = np.zeros(N_BANDS,dtype=float)
    mag_sun_ab[0] = 4.74;  
    l_bol_sun = 3.9e33; # bolometric solar in erg/s
    mag_sun_ab[1] = 6.34;  #U (BESSEL)
    mag_sun_ab[2] = 5.33;  #B (BESSEL)
    mag_sun_ab[3] = 4.81;  #V (BESSEL)
    mag_sun_ab[4] = 4.65;  #R (KPNO)
    mag_sun_ab[5] = 4.55;  #I (KPNO)
    mag_sun_ab[6] = 4.57;  #J (BESSEL)
    mag_sun_ab[7] = 4.71;  #H (BESSEL)
    mag_sun_ab[8] = 5.19;  #K (BESSEL)
    mag_sun_ab[9] = 6.75;  #SDSS u (unprimed AB)
    mag_sun_ab[10] = 5.33; #SDSS g (unprimed AB)
    mag_sun_ab[11] = 4.67; #SDSS r (unprimed AB)
    mag_sun_ab[12] = 4.48; #SDSS i (unprimed AB)
    mag_sun_ab[13] = 4.42; #SDSS z (unprimed AB)

    # Effective wavelengths of the bands (in Angstroms), to compute nuLnu<->Lnu
    # UBVRIJHK from http://cassfos02.ucsd.edu/physics/ph162/mags.html
    # SDSS ugriz from http://www.sdss.org/dr4/instruments/imager/index.html#filters
    lambda_eff = np.zeros(N_BANDS,dtype=float);
    lambda_eff[0] = 4243.93;  #bolometric, no nu
    lambda_eff[1] = 3600.0;  #U
    lambda_eff[2] = 4400.0;  #B
    lambda_eff[3] = 5556.0;  #V
    lambda_eff[4] = 6940.0;  #R
    lambda_eff[5] = 8700.0;  #I
    lambda_eff[6] = 12150.;  #J
    lambda_eff[7] = 16540.;  #H
    lambda_eff[8] = 21790.;  #K
    lambda_eff[9]  = 3551.;  #SDSS u
    lambda_eff[10] = 4686.;  #SDSS g
    lambda_eff[11] = 6165.;  #SDSS r
    lambda_eff[12] = 7481.;  #SDSS i
    lambda_eff[13] = 8931.;  #SDSS z
    c_light = 2.998e10; # speed of light in cm/s
    nu_eff  = c_light / (lambda_eff * 1.0e-8); # converts to nu_eff in Hz

    ten_pc   = 10.e0 * 3.086e18; # 10 pc in cm
    log_S_nu = -(mag_sun_ab + 48.6)/2.5; # zero point definition for ab magnitudes
    S_nu     = 10.**log_S_nu; # get the S_nu at 10 pc which defines M_AB
    lnu_sun_band = S_nu * (4.*math.pi*ten_pc*ten_pc); # multiply by distance modulus 
    nulnu_sun_band = lnu_sun_band * nu_eff; # multiply by nu_eff to get nu*L_nu
    l_bol_sun = nulnu_sun_band[0];

    if (UNITS_SOLAR_IN_BAND==0):
        l_b *= nulnu_sun_band[band_standardordering] / l_bol_sun; 

    return l_b;

