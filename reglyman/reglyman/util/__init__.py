from reglyman.kit import cosmology
import numpy as np
from regpy.operators import Operator, MatrixMultiplication
from regpy.stoprules import StopRule
from regpy.hilbert import HilbertSpace
from regpy import util as regpy_utils
from scipy.ndimage import gaussian_filter  

import pathlib  

class EnergyNorm:
    def __init__(self, sigma):
        self.shape=sigma.shape
        self.sigma=sigma
        
    def norm(self, x):
        return np.linalg.norm(x/self.sigma)
    
class GaussianSmoothing(Operator):
    def __init__(self, domain, smoothing_scale):
        self.smoothing_scale=smoothing_scale
        super().__init__(domain, domain)
        
    def _eval(self, x, differentiate):
        return gaussian_filter(x, sigma=self.smoothing_scale)
    
    def _derivative(self, x):
        return gaussian_filter(x, sigma=self.smoothing_scale)
    
    def _adjoint(self, x):
        return gaussian_filter(x, sigma=self.smoothing_scale)

def camb2nbodykit(path, column=None):
    if column == None:
        column = 8
    current_path = str(pathlib.Path().absolute())
    array = np.loadtxt(current_path+r'/../reglyman/power_spectra/'+path, delimiter=',')
    return array[column]

def cosmo(h=0.67556, T0_cmb=2.7255, Omega0_b=0.022032/0.67556**2, Omega0_cdm=0.12038/0.67556**2, N_ur=None,
            m_ncdm=[0.06], P_k_max=10., P_z_max=100., gauge='synchronous', n_s=0.9667,
            nonlinear=False, verbose=False, **kwargs):
    
    return cosmology.Cosmology(h=h, T0_cmb=T0_cmb, Omega0_b=Omega0_b, Omega0_cdm=Omega0_cdm, N_ur=N_ur,
            m_ncdm=m_ncdm, P_k_max=P_k_max, P_z_max=P_z_max, gauge=gauge, n_s=n_s,
            nonlinear=nonlinear, verbose=verbose, **kwargs)

class Display(StopRule):
    def __init__(self, functional, string):
        super().__init__()
        self.functional = functional
        self.string = string
    
    def __repr__(self):
        return 'Display'

    def _stop(self, x, y=None):
        self.log.info(self.string + '--> {}'.format( self.functional(x) ))
        return False

class Generic(HilbertSpace):
    def __init__(self, matrix, inverse=None):
        self.op = MatrixMultiplication(matrix, inverse=inverse)
        super().__init__(self.op.domain)
    
    @regpy_utils.memoized_property
    def gram(self):
        return self.op