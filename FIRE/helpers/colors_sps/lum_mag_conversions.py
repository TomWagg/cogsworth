import numpy as np
import math

##
##
## function to take the input luminosity of a given band and return the 
##    corresponding magnitude, for the appropriate bands and units as 
##    listed below
##
## UNITS_... keywords tell it the units of L, whether solar L in the 
##    band, bolometric L_sun (nuLnu/3.9d33), cgs (nuLnu/[erg/s])
##		default : bolometric L_sun
##
## BAND_... keywords tell it the relevant band, of UBVRIJHKugriz (SDSS 
##	  un-primed filters), or /BOLOMETRIC, or BAND_NUMBER uses the number of the 
##    band below instead of making you explicitly name it
##		default : bolometric
##
## VEGA or AB keywords have it return either vega or ab magnitudes
##		default : VEGA for UBVRIJHK (Johnsons UBVRI, Cousins JHK), AB for ugriz  
##
## L_NU or NU_L_NU tell it whether the input is in specific luminosity L_NU (e.g. 
##   erg/s/Hz), or NU_L_NU (e.g. erg/s)
##		default : NU_L_NU
##
##
##
## Luminosity index legend
##  0 = bolometric luminosity
##  1 = Johnsons U
##  2 = Johnsons B
##  3 = Johnsons V
##  4 = Johnsons R
##  5 = Johnsons I
##  6 = Cousins J
##  7 = Cousins H
##  8 = Cousins K
##  9 = Sloan u
## 10 = Sloan g
## 11 = Sloan r
## 12 = Sloan i
## 13 = Sloan z
##

def luminosity_to_magnitude( L, \
	UNITS_SOLAR_BOL=0, UNITS_SOLAR_BAND=0, \
	UNITS_CGS=0, \
	NU_L_NU=0, L_NU=0, \
	BAND_U=0,BAND_B=0,BAND_V=0,BAND_R=0,BAND_I=0, \
	BAND_J=0,BAND_H=0,BAND_K=0,BAND_SDSS_u=0, \
	BAND_SDSS_g=0,BAND_SDSS_r=0,BAND_SDSS_i=0, \
	BAND_SDSS_z=0,BOLOMETRIC=0, BAND_NUMBER=0, \
	VEGA=0, AB=0 , \
	MAGNITUDE_TO_LUMINOSITY=0 ):

    N_BANDS = 14

    ## VEGA system
    ## from www.ucolick.org/~cnaw/sun.html
    ## following Fukugita et al. 1995, PASP, 105, 945
    mag_sun_vega = np.zeros(N_BANDS)
    mag_sun_vega[0] = 4.74;  ##bolometric from Allen's Astrophysical Quantities
    mag_sun_vega[1] = 5.56;  ##U [BESSEL]
    mag_sun_vega[2] = 5.45;  ##B [BESSEL]
    mag_sun_vega[3] = 4.80;  ##V [BESSEL]
    mag_sun_vega[4] = 4.46;  ##R [KPNO]
    mag_sun_vega[5] = 4.10;  ##I [KPNO]
    mag_sun_vega[6] = 3.66;  ##J [BESSEL]
    mag_sun_vega[7] = 3.32;  ##H [BESSEL]
    mag_sun_vega[8] = 3.28;  ##K [BESSEL]
    mag_sun_vega[9] = 5.82;  ##SDSS u [unprimed Vega]
    mag_sun_vega[10] = 5.44; ##SDSS g [unprimed Vega]
    mag_sun_vega[11] = 4.52; ##SDSS r [unprimed Vega]
    mag_sun_vega[12] = 4.11; ##SDSS i [unprimed Vega]
    mag_sun_vega[13] = 3.89; ##SDSS z [unprimed Vega]

    ## AB system
    mag_sun_ab = np.zeros(N_BANDS)
    mag_sun_ab[0] = 4.74;  
    mag_sun_ab[1] = 6.34;  ##U [BESSEL]
    mag_sun_ab[2] = 5.33;  ##B [BESSEL]
    mag_sun_ab[3] = 4.81;  ##V [BESSEL]
    mag_sun_ab[4] = 4.65;  ##R [KPNO]
    mag_sun_ab[5] = 4.55;  ##I [KPNO]
    mag_sun_ab[6] = 4.57;  ##J [BESSEL]
    mag_sun_ab[7] = 4.71;  ##H [BESSEL]
    mag_sun_ab[8] = 5.19;  ##K [BESSEL]
    mag_sun_ab[9] = 6.75;  ##SDSS u [unprimed AB]
    mag_sun_ab[10] = 5.33; ##SDSS g [unprimed AB]
    mag_sun_ab[11] = 4.67; ##SDSS r [unprimed AB]
    mag_sun_ab[12] = 4.48; ##SDSS i [unprimed AB]
    mag_sun_ab[13] = 4.42; ##SDSS z [unprimed AB]

    ## Effective wavelengths of the bands [in Angstroms], to compute nuLnu<->Lnu
    ## UBVRIJHK from http:##cassfos02.ucsd.edu/physics/ph162/mags.html
    ## SDSS ugriz from http:##www.sdss.org/dr4/instruments/imager/index.html#filters
    lambda_eff = np.zeros(N_BANDS)
    lambda_eff[0] = 1.0;  ##bolometric, no nu
    lambda_eff[1] = 3600.0;  ##U
    lambda_eff[2] = 4400.0;  ##B
    lambda_eff[3] = 5556.0;  ##V
    lambda_eff[4] = 6940.0;  ##R
    lambda_eff[5] = 8700.0;  ##I
    lambda_eff[6] = 12150.;  ##J
    lambda_eff[7] = 16540.;  ##H
    lambda_eff[8] = 21790.;  ##K
    lambda_eff[9]  = 3551.;  ##SDSS u
    lambda_eff[10] = 4686.;  ##SDSS g
    lambda_eff[11] = 6165.;  ##SDSS r
    lambda_eff[12] = 7481.;  ##SDSS i
    lambda_eff[13] = 8931.;  ##SDSS z

    l_bol_sun = 3.9e33; ## bolometric solar in erg/s
    c_light = 2.998e10; ## speed of light in cm/s
    nu_eff  = c_light/(lambda_eff * 1.0e-8); ## converts to nu_eff in Hz

    i_BAND = 0;  ## default to bolometric 
    if (1 == BAND_NUMBER) : i_BAND=BAND_NUMBER
    if (1 == BAND_U)  : i_BAND=1
    if (1 == BAND_B)  : i_BAND=2
    if (1 == BAND_V)  : i_BAND=3
    if (1 == BAND_R)  : i_BAND=4
    if (1 == BAND_I)  : i_BAND=5
    if (1 == BAND_J)  : i_BAND=6
    if (1 == BAND_H)  : i_BAND=7
    if (1 == BAND_K)  : i_BAND=8
    if (1 == BAND_SDSS_u) : i_BAND=9
    if (1 == BAND_SDSS_g) : i_BAND=10
    if (1 == BAND_SDSS_r) : i_BAND=11
    if (1 == BAND_SDSS_i) : i_BAND=12
    if (1 == BAND_SDSS_z) : i_BAND=13
    if (1 == BOLOMETRIC)  : i_BAND=0

    ## default to Vega for bolometric & UBVRIJHK, and AB for ugriz
    vega_key = 1
    if ((i_BAND > 8) & (i_BAND <= 13)) : vega_key = 0
    if (VEGA==1) : vega_key=1
    if (AB==1) : vega_key=0
    magnitude_zero_point = mag_sun_vega[i_BAND]
    if (vega_key == 0) : magnitude_zero_point = mag_sun_ab[i_BAND]

    ## use the AB magnitudes to convert to an actual L_nu of the sun in each band
    lnu_sun_band = np.zeros(N_BANDS)
    ten_pc   = 10.0 * 3.086e18; ## 10 pc in cm
    log_S_nu = -(mag_sun_ab + 48.6)/2.5;	## zero point definition for ab magnitudes
    S_nu     = 10.**log_S_nu;  ## get the S_nu at 10 pc which defines M_AB
    lnu_sun_band = S_nu * (4.0*math.pi*ten_pc*ten_pc);  ## multiply by distance modulus 
    nulnu_sun_band = lnu_sun_band * nu_eff;  ## multiply by nu_eff to get nu*L_nu
    ## correct the bolometric
    lnu_sun_band[0] = l_bol_sun;
    nulnu_sun_band[0] = l_bol_sun;

    ## check if we're reversing the routine to go magnitude to luminosity (instead of vice-versa)
    if (MAGNITUDE_TO_LUMINOSITY==1) : 
        L_of_M = nulnu_sun_band[i_BAND] * 10.**(-0.4 * (L- magnitude_zero_point)) # here is magnitude
        ## now convert to appropriate units
        if (L_NU==1): L_of_M /= nu_eff[i_BAND];
        if (UNITS_SOLAR_BOL==1): return L_of_M/l_bol_sun;
        if (UNITS_SOLAR_BAND==1): 
            if (L_NU==1): 
                return L_of_M/lnu_sun_band[i_BAND];
            else:
                return L_of_M/nulnu_sun_band[i_BAND];
        if (UNITS_CGS==1): return L_of_M;
        return L_of_M/l_bol_sun;

    ## alright, now have lnu of the sun in each band (the appropriate normalization
    ##   for either magnitude system), can compare with the input luminosity
    nulnu_given = L;
    if (1==NU_L_NU) : nulnu_given = L;
    if (1==L_NU)    : nulnu_given = nu_eff[i_BAND] * L;

    ## default to assume in units of solar bolometric (if nu*L_nu):  
    l_in_solar_in_band = nulnu_given * (l_bol_sun/nulnu_sun_band[i_BAND]);
    ## or L_nu(sun) in the band (if given L_nu):
    if (1==L_NU) : l_in_solar_in_band = L;

    ## convert to the appropriate units
    if (UNITS_SOLAR_BAND==1) : l_in_solar_in_band = nulnu_given; ## given in solar in band
    if (UNITS_SOLAR_BAND==1) and (L_NU==1) : l_in_solar_in_band = L;
    if (UNITS_SOLAR_BOL) : l_in_solar_in_band = nulnu_given * (l_bol_sun/nulnu_sun_band[i_BAND]);
    if (UNITS_CGS) : l_in_solar_in_band = nulnu_given / nulnu_sun_band[i_BAND];

    return magnitude_zero_point - 2.5*np.log10(l_in_solar_in_band);



## routine to return the solar absolute magnitude in each band that the colors code gives 
def get_solar_mags():
	s_UBVRIJHK = np.zeros(8)
	s_UBVRIJHK[0] = 5.66; #U
	s_UBVRIJHK[1] = 5.47; #B
	s_UBVRIJHK[2] = 4.82; #V
	s_UBVRIJHK[3] = 4.28; #R
	s_UBVRIJHK[4] = 3.94; #I
	s_UBVRIJHK[5] = 3.64; #J ?
	s_UBVRIJHK[6] = 3.44; #H ?
	s_UBVRIJHK[7] = 3.33; #K

	s_ugrizJHK = np.zeros(8)
	s_ugrizJHK[0] = 6.2789; #u
	s_ugrizJHK[1] = 4.9489; #g
	s_ugrizJHK[2] = 4.44964; #r
	s_ugrizJHK[3] = 4.34644; #i
	s_ugrizJHK[4] = 4.3592; #z
	s_ugrizJHK[5] = 3.64; #J ?
	s_ugrizJHK[6] = 3.44; #H ?
	s_ugrizJHK[7] = 3.33; #K

	solar_mags = np.zeros(14)
	solar_mags[0] = 4.74; #bolometric
	solar_mags[1] = s_UBVRIJHK[0]
	solar_mags[2] = s_UBVRIJHK[1]
	solar_mags[3] = s_UBVRIJHK[2]
	solar_mags[4] = s_UBVRIJHK[3]
	solar_mags[5] = s_UBVRIJHK[4]
	solar_mags[6] = s_UBVRIJHK[5]
	solar_mags[7] = s_UBVRIJHK[6]
	solar_mags[8] = s_UBVRIJHK[7]
	solar_mags[9] = s_ugrizJHK[0]
	solar_mags[10] = s_ugrizJHK[1]
	solar_mags[11] = s_ugrizJHK[2]
	solar_mags[12] = s_ugrizJHK[3]
	solar_mags[13] = s_ugrizJHK[4]

	return solar_mags
