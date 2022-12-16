from regpy.operators import Operator, Identity
from regpy import util
from scipy.special import voigt_profile
from scipy.integrate import nquad, quad_vec
from scipy import stats

from reglyman.kit import cosmology

import numpy as np

'''
Computes the Lyman-alpha forest from the neutral hydrogen density
Contains the Forward Operator, derivative and adjoint of Frechet-derivative
In:
->domain, codomain: domain and codomain of forward operator, a regpy.discrs object
->parameters: dictionary containing keywords describing the box and the physical state of the IGM
->redshift, redshift_spect: Redshift of pixels in domain and in codomain (need not to be the same)
->Hubble_vel: Hubble velocities of the pixels in domain and codomain (need not to be the same)
->vel_pec: peculiar velocities (if known)
->ga: axion-EM coupling constant (units eV^-1)
->compute_vel: estimate peculiar velocities along single lines of sight (not recommended)
->flux_data: The output of the computation is the flux (if no it is the optical depth)
->gamma: The argument is the density n or (if gamma=True) the parameter gamma=log(n)
'''


class LymanAlphaHydrogenDilaton(Operator):
    def __init__(self, domain, parameters, redshift, redshift_spect, Hubble_vel, Hubble_vel_spect, vel_pec, axion_mass, cosmo, parameters_integration, compute_ga=False, flux_data=True, gamma=False, rayleigh=1, seed=12, codomain=None, ga=None, fraction=None, biasing=None, rayleigh_val=None):
        codomain = codomain or domain

        self.redshift_background = parameters["redshift_back"]
        self.N_pix = parameters["sampling"]
        self.N_space = self.N_pix[0]
        self.N_spect = self.N_pix[1]

        self.seed = seed

        #H in km/(s*Mpc)
        H = 100*cosmo.efunc(self.redshift_background)

#       speed of light in km/s
        self.speed_of_light = cosmo.C

        #self.astoc_bounds = parameters_integration['astoc_bounds']
        self.astoc_tol = parameters_integration['astoc_tol']
        self.a_bounds = parameters_integration['a_bounds']
        self.a_tol = parameters_integration['a_tol']
        self.workers = parameters_integration['workers']
        self.rescale_units = parameters_integration['rescale_units']

        # Here we calculate the mean axion field amplitude along LOS.
        # HI bias at large scales taken as axion DM bias (Fig.5, J.Bauer arxiv.2003.09655). rho_crit and axion_mass is in GeV*rescale_units
        self.rho_crit = 8.0992*cosmo.h**2 * \
            10**(-47)*(1+self.redshift_background)**3/self.rescale_units**4
        self.fraction = fraction or 1
        self.Omega0_cdm = cosmo.Omega0_cdm
        self.axion_mass = axion_mass/self.rescale_units

        if biasing == None:
            if compute_ga:
                biasing = Identity(domain.summands[0])
            else:
                biasing = Identity(domain)
        self.biasing = biasing

        # integral range in h**(-1) Mpc
        # This unit matches with output of nbodykit simulation (where distance has units h**(-1)Mpc)
        integral_range = parameters["width"]

        # integral range in km/s
        # The computation is only exact for very small ranges, otherwise the variation of the Hubble constant needs to be taken into account
        # Note that Hubble constant is also corrected by h**(-1), so we get rid of the dimensionless Hubble constant
        integral_range *= H/(1+self.redshift_background)
        Delta_omega = integral_range/self.N_space

        # Parsec in units of meters
        Parsec = 3.0856775714409184*10**16
        MegaParsec = 10**6*Parsec

        # I_alpha in m**2 (Effective Ly-alpha cross section for resonant scattering)
        I_alpha = 4.45*10**(-22)

        # Compute prefactor, common to do here to avoid large float number computing later
        # Prefactor had units of km/s
        self.Prefactor = I_alpha*self.speed_of_light / \
            (H*cosmo.h)*MegaParsec*Delta_omega
        self.Prefactor_adjoint = self.Prefactor*self.N_space/self.N_spect

        # EOS of the IGM
        self.beta = parameters["beta"]
        self.alpha = 2-1.4*self.beta
        self.T_med = parameters["t_med"]

        # tildeC in units of 1/m**3
        self.tildeC = parameters["tilde_c"]

        self.redshift = redshift
        self.redshift_spect = redshift
        self.Hubble_vel = Hubble_vel
        self.Hubble_vel_spect = Hubble_vel_spect

        if compute_ga:
            self.ga = 0
        else:
            self.ga = ga*self.rescale_units

        self.vel_pec = vel_pec

        # Perform sanity checks
        if compute_ga:
            assert np.size(redshift)+1 == np.size(domain)
        else:
            assert np.size(redshift) == np.size(domain)
        assert redshift_spect.shape == codomain.shape
        assert np.size(Hubble_vel) == self.N_space
        assert np.size(Hubble_vel_spect) == self.N_spect
        assert self.vel_pec.shape == redshift.shape

        self.compute_ga = compute_ga
        self.flux_data = flux_data
        self.gamma = gamma
        self.rayleigh= rayleigh
        
        #only for rayleigh==2 neeeded
        if self.rayleigh == 2:
            self.rayleigh_val = rayleigh_val
            assert self.rayleigh_val.shape == redshift.shape

        self.density = np.empty(self.redshift.shape)
        self.density_baryon = np.empty(self.redshift.shape)
        self.flux = np.empty(self.N_spect)

        self.decay_rate = 4.6*10**8
        self.lambda0 = 1.216*10**(-10)
        self.damping_factor = self.decay_rate * self.lambda0 / (4 * np.pi)

        self.toadd = np.zeros((self.N_spect, self.N_spect))
        self.deriv_broadening = np.zeros((self.N_space, self.N_space))
        self.deriv_ga = np.zeros((self.N_space, self.N_space))

        super().__init__(
            domain=domain,
            codomain=codomain)

# Evaluate the forward operator
    def _eval(self, argument, differentiate, linearized=False, **kwargs):
        if self.compute_ga:
            x = argument[0:self.N_space]
            ga = argument[self.N_space]*self.rescale_units
        else:
            ga = self.ga
            x = argument

        vel_pec = self.vel_pec

        if self.gamma:
            density = np.exp(x)
        else:
            density = x
        density_baryon = self._find_baryonic_density(density)
        aDM = self._find_axion_amplitude(density_baryon)

        if linearized:
            tau = self._apply_kernel(density)
        else:
            self._compute_kernel_1(density_baryon, ga, vel_pec, aDM)
            tau = self._apply_kernel(density)

        if differentiate:
            self.density = density
            self.density_baryon = density_baryon
            self._compute_kernel_2(density_baryon, ga, vel_pec, aDM)
            if self.compute_ga:
                self._compute_kernel_3(density_baryon, ga, vel_pec, aDM)
        if self.flux_data:
            flux = np.exp(-tau)
            if differentiate:
                self.flux = flux
            toret = flux
        else:
            toret = tau
        return toret

# Computes the Frechet-derivative of the integral operator
    def _derivative(self, h_data, **kwargs):
        if self.compute_ga:
            h = h_data[0:self.N_space]
            h_ga = h_data[self.N_space]*self.rescale_units
        else:
            h = h_data

        density = self.density
        density_baryon = self.density_baryon

        if self.gamma:
            h = density*h

# First term in derivative matches the evaluation applied to the direction of descent
        d_tau_1 = self._apply_kernel(h)

# Second term of derivative (inner derivative of the integration kernel)
        bar_deriv = self._hydrogen_to_baryonic_deriv(
            density, density_baryon, h)
        d_tau_2 = self._apply_kernel_2(density_baryon, density*bar_deriv)
        if self.flux_data:
            toret = -self.flux*(d_tau_1+d_tau_2)
        else:
            toret = d_tau_1+d_tau_2

# If the ga is part of the vector of unknowns we also need to differentiate in the direction of the velocity
        if self.compute_ga:
            d_tau_3 = self._apply_kernel_3(density*h_ga)
            if self.flux_data:
                toret += -self.flux*d_tau_3
            else:
                toret += d_tau_3

        return toret

# Computes the adjoint of derivative
    def _adjoint(self, g, **kwargs):
        if self.flux_data:
            y = -self.flux*g
        else:
            y = g

        density = self.density
        density_baryon = self.density_baryon

        d_rho_1 = self._apply_kernel_adjoint(y)
        d_rho_2 = self._apply_kernel_2_adjoint(density_baryon, y)*density
        d_rho_2 = self._hydrogen_to_baryonic_adjoint(
            density, density_baryon, d_rho_2)

        if self.gamma:
            toret = density*(d_rho_1+d_rho_2)
        else:
            toret = d_rho_1+d_rho_2

        if self.compute_ga:
            d_rho_3 = np.sum(self._apply_kernel_3_adjoint(y)*density)
            toret = np.append(toret, d_rho_3*self.rescale_units)
        return toret

    def _integrand_a(self, a):
        #        return voigt_profile(self._x - 2*a*self._astoc*self._ga*self.speed_of_light, self._sigma, self.damping_factor)/(np.pi*np.sqrt(1-a**2))*2*self._astoc/self._aDM**2*np.exp(-(self._astoc/self._aDM)**2)
        return self._gaussian(self._x - 2*a*self._astoc*self._ga*self.speed_of_light, self._sigma)/(np.pi*np.sqrt(1-a**2))*2*self._astoc/self._aDM**2*np.exp(-(self._astoc/self._aDM)**2)

    def _integrand_a_no_rayleigh(self, a):
        return self._gaussian(self._x - 2*a*self._aDM*self._ga*self.speed_of_light, self._sigma)/(np.pi*np.sqrt(1-a**2))

    def _gaussian(self, x, std):
        return 1/np.sqrt(2*np.pi*std**2) * np.exp(-0.5*x**2/std**2)

    def _integrand_astoc(self, astoc):
        self._astoc = astoc
        return quad_vec(self._integrand_a, self.a_bounds[0], self.a_bounds[1], epsrel=self.a_tol, workers=self.workers)[0]

    # The modified Voigt profile needs to account for the complex behavior of the Lyman-alpha frequency in the presence of alpha-->alpha*(1+g_aF*a):
    def _V2(self, x, sigma, ga, aDM):
        astoc_upper_bound = 10 * np.max(aDM)
        self.cutoff_voigt = 10 * np.max(sigma)

        if self.cutoff_voigt > np.abs(x[0][-1]-x[0][0]):
            print('WARNING: Broadening exceeds length of LOS')

        if astoc_upper_bound * ga * self.speed_of_light > np.abs(x[0][-1]-x[0][0]):
            print('WARNING: Scalar dark matter broadening exceeds length of LOS')

        indices = np.abs(x) < self.cutoff_voigt

        self._x = x[indices]
        self._sigma = np.tile(sigma, (len(sigma), 1))[indices]
        self._ga = ga
        self._aDM = np.tile(aDM, (len(sigma), 1))[indices]
        
        if self.rayleigh==1:
            integrand = quad_vec(self._integrand_astoc, 0, astoc_upper_bound,
                                 epsrel=self.astoc_tol, workers=self.workers)[0]
        if self.rayleigh==2:
            self._aDM *= np.tile(self.rayleigh_val, (len(sigma), 1))[indices]
            integrand = quad_vec(self._integrand_a_no_rayleigh, self.a_bounds[0], self.a_bounds[1], epsrel=self.a_tol, workers=self.workers)[0]
        if self.rayleigh==3:
            integrand = quad_vec(self._integrand_a_no_rayleigh, self.a_bounds[0], self.a_bounds[1], epsrel=self.a_tol, workers=self.workers)[0]
        toret = np.zeros(np.size(x))
        toret[indices.flatten()] = integrand
        return toret.reshape(x.shape)

    def _apply_kernel(self, vector):
        toadd = self.toadd * self.Prefactor
        toret = np.trapz(toadd*vector)
        return toret

    def _compute_kernel_1(self, density, ga, vel_pec, aDM):
        argument = np.zeros((self.N_spect, self.N_space))
        broadening = 12.849*(self.T_med)**0.5*(density)**self.beta
        for i in range(self.N_spect):
            argument[i, :] = self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.toadd = self._V2(argument, broadening/np.sqrt(2), ga, aDM)

    def _apply_kernel_adjoint(self, vector):
        toadd = self.toadd.transpose() * self.Prefactor_adjoint
        toret = np.trapz(toadd*vector)
        return toret

    def _apply_kernel_2(self, density, vector):
        d_broadening = 12.849*(self.T_med)**0.5 * \
            (density)**(self.beta-1)*self.beta
        toadd = (self.deriv_broadening - self.toadd) / \
            10**(-10) * d_broadening * self.Prefactor
        toret = np.trapz(toadd*vector)
        return toret

    def _compute_kernel_2(self, density, ga, vel_pec, aDM):
        broadening = 12.849*(self.T_med)**0.5*(density)**self.beta
        argument = np.zeros((self.N_spect, self.N_space))
        d_broadening = 12.849*(self.T_med)**0.5 * \
            (density)**(self.beta-1)*self.beta
        normalized_direction = d_broadening/np.linalg.norm(d_broadening)
        for i in range(self.N_spect):
            argument[i, :] = self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_broadening = self._V2(
            argument, broadening/np.sqrt(2)+10**(-10)*normalized_direction, ga, aDM)

    def _apply_kernel_2_adjoint(self, density, vector):
        d_broadening = 12.849*(self.T_med)**0.5 * \
            (density)**(self.beta-1)*self.beta
        toadd = (self.deriv_broadening - self.toadd) / \
            10**(-10) * d_broadening * self.Prefactor_adjoint
        toret = np.trapz(toadd.transpose()*vector)
        return toret

    def _apply_kernel_3(self, vector):
        toadd = (self.deriv_ga - self.toadd)/10**(-8) * self.Prefactor
        toret = np.trapz(toadd*vector)
        return toret

    def _compute_kernel_3(self, density, ga, vel_pec, aDM):
        broadening = 12.849*(self.T_med)**0.5*(density)**self.beta
        argument = np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :] = self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_ga = self._V2(
            argument, broadening/np.sqrt(2), ga+10**(-8), aDM)

    def _apply_kernel_3_adjoint(self, vector):
        toadd = (self.deriv_ga - self.toadd)/10**(-8) * self.Prefactor_adjoint
        toret = np.trapz(toadd.transpose()*vector)
        return toret

# Computes the projection of the neutral hydrogen density to the overdensity of baryonic matter and its derivative and adjoint of derivative
# TODO: Enable the more complex form here
    def _find_baryonic_density(self, density):
        density_baryon = (
            density/(self.tildeC*(1+self.redshift)**6))**(1/(self.alpha))
        return density_baryon

    def _hydrogen_to_baryonic_deriv(self, density_hydrogen, density_baryon, h):
        return 1/self.alpha*density_baryon/density_hydrogen*h

    def _hydrogen_to_baryonic_adjoint(self, density_hydrogen, density_baryon, y):
        return 1/self.alpha*density_baryon/density_hydrogen*y

# Finds the axion field amplitude from the baryon density perturbation
    def _find_axion_amplitude(self, delta):
        return (2 * self.biasing(delta) * self.fraction * self.Omega0_cdm * self.rho_crit)**0.5 / self.axion_mass


###############################################################################

class LymanAlphaHydrogenDilatonUltraLight(Operator):
    def __init__(self, domain, parameters, redshift, redshift_spect, Hubble_vel, Hubble_vel_spect, vel_pec, axion_mass, cosmo, parameters_integration, phase=0, seed=12, compute_ga=False, flux_data=True, gamma=False, codomain=None, ga=None, fraction=None, biasing=None, rayleigh_val=None):
        codomain = codomain or domain

        self.redshift_background = parameters["redshift_back"]
        self.N_pix = parameters["sampling"]
        self.N_space = self.N_pix[0]
        self.N_spect = self.N_pix[1]

        self.seed = seed

        #H in km/(s*Mpc)
        H = 100*cosmo.efunc(self.redshift_background)

#       speed of light in km/s
        self.speed_of_light = cosmo.C

        #self.astoc_bounds = parameters_integration['astoc_bounds']
        self.astoc_tol = parameters_integration['astoc_tol']
        self.a_bounds = parameters_integration['a_bounds']
        self.a_tol = parameters_integration['a_tol']
        self.workers = parameters_integration['workers']
        self.rescale_units = parameters_integration['rescale_units']

        # Here we calculate the mean axion field amplitude along LOS.
        # HI bias at large scales taken as axion DM bias (Fig.5, J.Bauer arxiv.2003.09655). rho_crit and axion_mass is in GeV*rescale_units
        self.rho_crit = 8.0992*cosmo.h**2 * \
            10**(-47)*(1+self.redshift_background)**3/self.rescale_units**4
        self.fraction = fraction or 1
        self.Omega0_cdm = cosmo.Omega0_cdm
        self.axion_mass = axion_mass/self.rescale_units

        if biasing == None:
            if compute_ga:
                biasing = Identity(domain.summands[0])
            else:
                biasing = Identity(domain)
        self.biasing = biasing

        # integral range in h**(-1) Mpc
        # This unit matches with output of nbodykit simulation (where distance has units h**(-1)Mpc)
        integral_range = parameters["width"]

        # integral range in km/s
        # The computation is only exact for very small ranges, otherwise the variation of the Hubble constant needs to be taken into account
        # Note that Hubble constant is also corrected by h**(-1), so we get rid of the dimensionless Hubble constant
        integral_range *= H/(1+self.redshift_background)
        Delta_omega = integral_range/self.N_space

        # Parsec in units of meters
        Parsec = 3.0856775714409184*10**16
        MegaParsec = 10**6*Parsec

        # I_alpha in m**2 (Effective Ly-alpha cross section for resonant scattering)
        I_alpha = 4.45*10**(-22)

        # Compute prefactor, common to do here to avoid large float number computing later
        # Prefactor had units of km/s
        self.Prefactor = I_alpha*self.speed_of_light / \
            (H*cosmo.h)*MegaParsec*Delta_omega
        self.Prefactor_adjoint = self.Prefactor*self.N_space/self.N_spect

        self.phase = phase
        planck_mass = 1.221 * 10**19  # GeV
        planck_time = 5.391 * 10**(-44)  # s
        self.oscillation_period = planck_mass * \
            planck_time / (2 * np.pi * axion_mass)
        self.pixel_length = parameters["width"] / \
            self.N_space * cosmo.h * MegaParsec
        self.diff_phases = 1/self.oscillation_period * self.pixel_length / (1000 * self.speed_of_light)
        #self.oscillation_bounds = np.cos( self.phase + self.diff_phases * np.arange(self.N_space+1) )
        #self.oscillation_bounds_lower = self.oscillation_bounds[:-1]
        #self.oscillation_bounds_upper = self.oscillation_bounds[1:]
        self.oscillation_bounds = np.cos( self.phase + self.diff_phases * np.arange(self.N_space) )

        # EOS of the IGM
        self.beta = parameters["beta"]
        self.alpha = 2-1.4*self.beta
        self.T_med = parameters["t_med"]

        # tildeC in units of 1/m**3
        self.tildeC = parameters["tilde_c"]

        self.redshift = redshift
        self.redshift_spect = redshift
        self.Hubble_vel = Hubble_vel
        self.Hubble_vel_spect = Hubble_vel_spect
        self.rayleigh_val = rayleigh_val

        if compute_ga:
            self.ga = 0
        else:
            self.ga = ga*self.rescale_units

        self.vel_pec = vel_pec

        # Perform sanity checks
        if compute_ga:
            assert np.size(redshift)+1 == np.size(domain)
        else:
            assert np.size(redshift) == np.size(domain)
        assert redshift_spect.shape == codomain.shape
        assert np.size(Hubble_vel) == self.N_space
        assert np.size(Hubble_vel_spect) == self.N_spect
        assert self.vel_pec.shape == redshift.shape

        self.compute_ga = compute_ga
        self.flux_data = flux_data
        self.gamma = gamma

        self.density = np.empty(self.redshift.shape)
        self.density_baryon = np.empty(self.redshift.shape)
        self.flux = np.empty(self.N_spect)

        self.decay_rate = 4.6*10**8
        self.lambda0 = 1.216*10**(-10)
        self.damping_factor = self.decay_rate * self.lambda0 / (4 * np.pi)

        self.toadd = np.zeros((self.N_spect, self.N_spect))
        self.deriv_broadening = np.zeros((self.N_space, self.N_space))
        self.deriv_ga = np.zeros((self.N_space, self.N_space))

        super().__init__(
            domain=domain,
            codomain=codomain)

# Evaluate the forward operator
    def _eval(self, argument, differentiate, linearized=False, **kwargs):
        if self.compute_ga:
            x = argument[0:self.N_space]
            ga = argument[self.N_space]*self.rescale_units
        else:
            ga = self.ga
            x = argument

        vel_pec = self.vel_pec

        if self.gamma:
            density = np.exp(x)
        else:
            density = x
        density_baryon = self._find_baryonic_density(density)
        aDM = self._find_axion_amplitude(density_baryon)

        if linearized:
            tau = self._apply_kernel(density)
        else:
            self._compute_kernel_1(density_baryon, ga, vel_pec, aDM)
            tau = self._apply_kernel(density)

        if differentiate:
            self.density = density
            self.density_baryon = density_baryon
            self._compute_kernel_2(density_baryon, ga, vel_pec, aDM)
            if self.compute_ga:
                self._compute_kernel_3(density_baryon, ga, vel_pec, aDM)
        if self.flux_data:
            flux = np.exp(-tau)
            if differentiate:
                self.flux = flux
            toret = flux
        else:
            toret = tau
        return toret

# Computes the Frechet-derivative of the integral operator
    def _derivative(self, h_data, **kwargs):
        if self.compute_ga:
            h = h_data[0:self.N_space]
            h_ga = h_data[self.N_space]*self.rescale_units
        else:
            h = h_data

        density = self.density
        density_baryon = self.density_baryon

        if self.gamma:
            h = density*h

# First term in derivative matches the evaluation applied to the direction of descent
        d_tau_1 = self._apply_kernel(h)

# Second term of derivative (inner derivative of the integration kernel)
        bar_deriv = self._hydrogen_to_baryonic_deriv(
            density, density_baryon, h)
        d_tau_2 = self._apply_kernel_2(density_baryon, density*bar_deriv)
        if self.flux_data:
            toret = -self.flux*(d_tau_1+d_tau_2)
        else:
            toret = d_tau_1+d_tau_2

# If the ga is part of the vector of unknowns we also need to differentiate in the direction of the velocity
        if self.compute_ga:
            d_tau_3 = self._apply_kernel_3(density*h_ga)
            if self.flux_data:
                toret += -self.flux*d_tau_3
            else:
                toret += d_tau_3

        return toret

# Computes the adjoint of derivative
    def _adjoint(self, g, **kwargs):
        if self.flux_data:
            y = -self.flux*g
        else:
            y = g

        density = self.density
        density_baryon = self.density_baryon

        d_rho_1 = self._apply_kernel_adjoint(y)
        d_rho_2 = self._apply_kernel_2_adjoint(density_baryon, y)*density
        d_rho_2 = self._hydrogen_to_baryonic_adjoint(
            density, density_baryon, d_rho_2)

        if self.gamma:
            toret = density*(d_rho_1+d_rho_2)
        else:
            toret = d_rho_1+d_rho_2

        if self.compute_ga:
            d_rho_3 = np.sum(self._apply_kernel_3_adjoint(y)*density)
            toret = np.append(toret, d_rho_3*self.rescale_units)
        return toret

    def _gaussian(self, x, std):
        return 1/np.sqrt(2*np.pi*std**2) * np.exp(-0.5*x**2/std**2)

    def _integrand_astoc(self):
        return self._gaussian(self._x - 2*self._oscillation_bounds*self._aDM*self._ga*self.speed_of_light, self._sigma)

    # The modified Voigt profile needs to account for the complex behavior of the Lyman-alpha frequency in the presence of alpha-->alpha*(1+g_aF*a):
    def _V2(self, x, sigma, ga, aDM):
        astoc_upper_bound = 10 * np.max(aDM)
        self.cutoff_voigt = 10 * np.max(sigma)

        if self.cutoff_voigt > np.abs(x[0][-1]-x[0][0]):
            print('WARNING: Broadening exceeds length of LOS')

        if astoc_upper_bound * ga * self.speed_of_light > np.abs(x[0][-1]-x[0][0]):
            print('WARNING: Scalar dark matter broadening exceeds length of LOS')

        indices = np.abs(x) < self.cutoff_voigt
        #indices = np.abs(x) > -1

        self._x = x[indices]
        self._sigma = np.tile(sigma, (len(sigma), 1))[indices]
        self._ga = ga
        self._aDM = np.tile(aDM, (len(sigma), 1))[indices]

        self._aDM *= np.tile(self.rayleigh_val, (len(sigma), 1))[indices]
        
        self._oscillation_bounds = np.tile(self.oscillation_bounds, (len(sigma), 1))[indices]

        integrand = self._integrand_astoc()
        
        toret = np.zeros(np.size(x))
        toret[indices.flatten()] = integrand
        return toret.reshape(x.shape)

    def _apply_kernel(self, vector):
        toadd = self.toadd * self.Prefactor
        toret = np.trapz(toadd*vector)
        return toret

    def _compute_kernel_1(self, density, ga, vel_pec, aDM):
        argument = np.zeros((self.N_spect, self.N_space))
        broadening = 12.849*(self.T_med)**0.5*(density)**self.beta
        for i in range(self.N_spect):
            argument[i, :] = self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.toadd = self._V2(argument, broadening/np.sqrt(2), ga, aDM)

    def _apply_kernel_adjoint(self, vector):
        toadd = self.toadd.transpose() * self.Prefactor_adjoint
        toret = np.trapz(toadd*vector)
        return toret

    def _apply_kernel_2(self, density, vector):
        d_broadening = 12.849*(self.T_med)**0.5 * \
            (density)**(self.beta-1)*self.beta
        toadd = (self.deriv_broadening - self.toadd) / \
            10**(-10) * d_broadening * self.Prefactor
        toret = np.trapz(toadd*vector)
        return toret

    def _compute_kernel_2(self, density, ga, vel_pec, aDM):
        broadening = 12.849*(self.T_med)**0.5*(density)**self.beta
        argument = np.zeros((self.N_spect, self.N_space))
        d_broadening = 12.849*(self.T_med)**0.5 * \
            (density)**(self.beta-1)*self.beta
        normalized_direction = d_broadening/np.linalg.norm(d_broadening)
        for i in range(self.N_spect):
            argument[i, :] = self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_broadening = self._V2(
            argument, broadening/np.sqrt(2)+10**(-10)*normalized_direction, ga, aDM)

    def _apply_kernel_2_adjoint(self, density, vector):
        d_broadening = 12.849*(self.T_med)**0.5 * \
            (density)**(self.beta-1)*self.beta
        toadd = (self.deriv_broadening - self.toadd) / \
            10**(-10) * d_broadening * self.Prefactor_adjoint
        toret = np.trapz(toadd.transpose()*vector)
        return toret

    def _apply_kernel_3(self, vector):
        toadd = (self.deriv_ga - self.toadd)/10**(-8) * self.Prefactor
        toret = np.trapz(toadd*vector)
        return toret

    def _compute_kernel_3(self, density, ga, vel_pec, aDM):
        broadening = 12.849*(self.T_med)**0.5*(density)**self.beta
        argument = np.zeros((self.N_spect, self.N_space))
        for i in range(self.N_spect):
            argument[i, :] = self.Hubble_vel_spect[i]-self.Hubble_vel-vel_pec
        self.deriv_ga = self._V2(
            argument, broadening/np.sqrt(2), ga+10**(-8), aDM)

    def _apply_kernel_3_adjoint(self, vector):
        toadd = (self.deriv_ga - self.toadd)/10**(-8) * self.Prefactor_adjoint
        toret = np.trapz(toadd.transpose()*vector)
        return toret

# Computes the projection of the neutral hydrogen density to the overdensity of baryonic matter and its derivative and adjoint of derivative
# TODO: Enable the more complex form here
    def _find_baryonic_density(self, density):
        density_baryon = (
            density/(self.tildeC*(1+self.redshift)**6))**(1/(self.alpha))
        return density_baryon

    def _hydrogen_to_baryonic_deriv(self, density_hydrogen, density_baryon, h):
        return 1/self.alpha*density_baryon/density_hydrogen*h

    def _hydrogen_to_baryonic_adjoint(self, density_hydrogen, density_baryon, y):
        return 1/self.alpha*density_baryon/density_hydrogen*y

# Finds the axion field amplitude from the baryon density perturbation
    def _find_axion_amplitude(self, delta):
        return (2 * self.biasing(delta) * self.fraction * self.Omega0_cdm * self.rho_crit)**0.5 / self.axion_mass