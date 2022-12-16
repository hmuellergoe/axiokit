from reglyman.operators import LymanAlphaHydrogenDilatonUltraLight
from reglyman.density import Data_Generation, Cosmo_Translate, Biasing, LinearPower
from reglyman.util import cosmo

from regpy.discrs import UniformGrid

import numpy as np
import logging

from joblib import Parallel, delayed
import multiprocessing

import matplotlib.pyplot as plt

num_cores = multiprocessing.cpu_count()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)-40s :: %(message)s')

'''
The complete creation of synthetic data with dilatons involved following the description of
Hamaide, Mueller, Marsh, 2022
'''

'''We start from a user defined cosmologic model here'''
#Parameters for the creation of synthetic data.
parameters_cosmology={
    'h' : 0.68,
    'ombh2' : 0.02206,
    'omch2' : 0.12,
    'P_k_max' : 100,
    'n_s' : 0.96,
    'm_ncdm' : [],
    'A_s' : 2.1e-9
    }

'''Set cosmology to UserDefined'''
#Parameters for the creation of synthetic data.
parameters_box={	
	'cosmology' : 'UserDefined',
	'background_box' : np.array([100, 100]),
	'background_sampling' : np.array([100, 100]),
	'width' : 10,
	'nr_pix' : 400,
	'redshift' : 2.5,
	'seed' : 12
	}

'''Parameters of IGM'''
parameters_igm={
	'beta' : 0.2,
	'tilde_c' : 1.2*10**(-8),
	't_med' : 1
	}

'''Numerical parameters for internal computation of modified Voigt-profiles'''
parameters_integration={
    'astoc_tol' : 100,
    'a_bounds': [-1, 1],
    'a_tol': 100, 
    'workers' : 1,
    'rescale_units' : 10**(8),
    'cutoff_voigt' : 100
}

#Parameters for the inversion. Need not neccessarily be the same than for the forward computation.
#In an ideal case, beta and t_med would be known exactly and should match the choice of parameters that was used for the creation of synthetic data.
#However, in practice the equation of state of the IGM is only poorly constrained prior to inversion

parameters_inversion = {
        "redshift_back" : parameters_box['redshift'],
        "width" : parameters_box['width'],
        "sampling" : (100, 100),
        "beta" : parameters_igm['beta'],
        "t_med" : parameters_igm['t_med'],
        "tilde_c" : parameters_igm['tilde_c']
        }

#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
parameters_noise = {
        "snr" : 20,
        "sigma_0" : 0.005,
        "seed" : 4
}

'''Dilaton mass, coupling to photons and fraction of dark matter'''
#dilaton mass in GeV
dilaton_mass = 10**(-41)
#ga in GeV^(-1)
ga = 0 
#dilaton fraction on dark matter
fraction = 0.06

#Define cosmology model
cosmology_model = cosmo(h=parameters_cosmology["h"], Omega0_b = parameters_cosmology["ombh2"] / parameters_cosmology["h"]**2, 
      Omega0_cdm = parameters_cosmology["omch2"] / parameters_cosmology["h"]**2, m_ncdm = parameters_cosmology["m_ncdm"],
      P_k_max = parameters_cosmology["P_k_max"],n_s = parameters_cosmology["n_s"])

sigma8 = cosmology_model.sigma8 * np.sqrt( parameters_cosmology['A_s'] / cosmology_model.A_s )
cosmology_model = cosmology_model.match(sigma8=sigma8)

#Computes the Baryon overdensity and peculiar velocities
'''Specify transfer function, computed by axioCAMB'''
path_transfer = r"Transfer-ma=-32-axfrac=0_06.dat"
Generator_baryon = Data_Generation(parameters_box, parameters_igm,
                                 transfer="CAMB", path_transfer=path_transfer, cosmo=cosmology_model, column=3, species=2)
delta_baryon, vel_baryon, comoving=Generator_baryon.Compute_Density_Field()

#Translates the comoving distance to a Hubble distance and redshift (by the use of astropy)
Trans=Cosmo_Translate(comoving, parameters_box['cosmology'], cosmo=cosmology_model)
hubble_vel=Trans.Compute_Hubble_Velocity()
hubble_flow=Trans.Convert_To_Numpy(hubble_vel)
redshift=Trans.Compute_Redshift()
hubble=3*10**5*redshift/(1+parameters_box['redshift'])

dens_hydrogen=Generator_baryon.Find_Neutral_Hydrogen_Fraction(delta_baryon, redshift)

#Interpolates the computed densities and velocities on a uniform grid
dens_hydrogen, vel_baryon, hubble, comoving, redshift=Trans.Rebin_all(dens_hydrogen.copy(), vel_baryon.copy(), hubble, parameters_box['nr_pix'])

#For simplicity we assume vanishing peculiar velocities. Needs just to be commented out if peculiar velocities should be taken into account
vel_baryon = 0*vel_baryon
####################################################################################################################################################################
'''
Compute Ly-alpha forest flux from computed overdensities. More explained in the file synthetic_data/....
'''

#Calculate linear power spectrum of various components by biasing
Plin_b = Generator_baryon.Plin
Plin_ax = LinearPower(cosmology_model, parameters_inversion["redshift_back" ], transfer="CAMB", path_transfer=path_transfer, column=3)
Plin_ax.update_transfer(path_transfer, column=1)
biasing = Biasing(Plin_b, Plin_ax, (comoving[0], comoving[-1], len(comoving)))

#Which lines of sights to select from the box
array_x=5+5*np.linspace(0, 18, 19)
array_y=5+5*np.linspace(0, 18, 19)

#Number of lines of sights
Nr_LOS=array_x.shape[0]*array_y.shape[0]

#Define domain and codomain for forward operator
#Domain and codomain are defined on a uniform grid
coords=np.linspace(1, parameters_box['nr_pix'], parameters_box['nr_pix'])
domain=UniformGrid(coords, dtype=float)

coords=np.linspace(1, parameters_box['nr_pix'], parameters_box['nr_pix'])
codomain=UniformGrid(coords, dtype=float)

#parameters describing the forward operator
#->redshift_back: redshift of box
#->width: length of line of sight in [h^{-1}Mpc]
#->n_pix: (size domain, size codomain)
#->beta, t_med, tilde_c: three parameters describing the forward problem
parameters_forward = {
        "redshift_back" : parameters_box['redshift'],
        "width" : parameters_box['width'],
        "sampling" : (parameters_box['nr_pix'], parameters_box['nr_pix']),
        "beta" : parameters_igm['beta'],
        "t_med" : parameters_igm['t_med'],
        "tilde_c" : parameters_igm['tilde_c']
        }

#Project velocity to velocity in the direction along the line of sight
vel_all=vel_baryon[2, :, :, :]
vel_all*=(hubble[-1]-hubble[0])/(hubble_flow[-1]-hubble_flow[0])
hubble_spect=hubble
redshift_spect=redshift

#Allocate the exact density and exact data
exact_solution=np.zeros((Nr_LOS, parameters_box['nr_pix']))
exact_data=np.zeros((Nr_LOS, parameters_box['nr_pix']))

#Compute for every selected line of sight the exact Ly-alpha forest data
#LymanAlphaBar is the forward operator that computes the Ly-alpha forest flux from the baryonic overdensity
vel=np.zeros((Nr_LOS, parameters_box['nr_pix']))
counter=0

np.random.seed(1234)
delta_phase = 2*np.pi / 100
phases = delta_phase * np.arange(Nr_LOS) + np.random.uniform(0, 2*np.pi, 1)
  
rayleigh_val = np.ones(exact_solution.shape)
      
for i in array_x.astype(int):
    for j in array_y.astype(int):
        vel[counter, :]=vel_all[i, j, :]
        exact_solution[counter, :]=dens_hydrogen[i, j, :]
        counter += 1


indices = np.asarray(np.meshgrid(array_x, array_y), dtype=int).reshape((2, Nr_LOS))

def observe(counter, ga, dens):
    i, j = indices[:, counter]
    op = LymanAlphaHydrogenDilatonUltraLight(domain, parameters_forward, redshift, redshift_spect, hubble, hubble_spect, vel[counter, :], dilaton_mass, cosmology_model, parameters_integration, phase=phases[counter], ga=ga, fraction=fraction, biasing=biasing, rayleigh_val=rayleigh_val[counter])    
    #exact_data[counter, :]=op(exact_solution[counter, :])
    return op(dens[counter, :])

exact_data = np.asarray( Parallel(n_jobs=num_cores)(delayed(observe)(counter, ga=ga, dens=exact_solution) for counter in range(Nr_LOS)) )

###################################################################################################################################################################
'''
Add noise
'''

#Random numbers for noise contribution    
np.random.seed(parameters_noise["seed"])
random=np.random.randn(Nr_LOS, parameters_box['nr_pix'])

#Actually computes the noisy data
#Noise model; sigma_F^2=F^2/SNR^2+sigma_0^2
SNRatio=parameters_noise["snr"]
data=np.zeros((Nr_LOS, parameters_box['nr_pix']))
sigma=np.zeros((Nr_LOS, parameters_box['nr_pix']))
sigma_0=parameters_noise["sigma_0"]

#Add noise to data
for i in range(Nr_LOS):
    sigma[i, :]=np.sqrt((exact_data[i, :]/SNRatio)**2+sigma_0**2)
    noise = sigma[i, :]*random[i, :]
    data[i, :] = exact_data[i, :] + noise

###############################################################################
'''Predict data with varying ga'''

ga25_75 = np.asarray( Parallel(n_jobs=num_cores)(delayed(observe)(counter, ga=10**(-25.75), dens=exact_solution) for counter in range(Nr_LOS)) )
ga26 = np.asarray( Parallel(n_jobs=num_cores)(delayed(observe)(counter, ga=10**(-26), dens=exact_solution) for counter in range(Nr_LOS)) )
ga26_25 = np.asarray( Parallel(n_jobs=num_cores)(delayed(observe)(counter, ga=10**(-26.25), dens=exact_solution) for counter in range(Nr_LOS)) )
ga26_5 = np.asarray( Parallel(n_jobs=num_cores)(delayed(observe)(counter, ga=10**(-26.5), dens=exact_solution) for counter in range(Nr_LOS)) )
ga26_75 = np.asarray( Parallel(n_jobs=num_cores)(delayed(observe)(counter, ga=10**(-26.75), dens=exact_solution) for counter in range(Nr_LOS)) )
