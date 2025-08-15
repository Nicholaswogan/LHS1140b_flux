import warnings
warnings.filterwarnings('ignore')

from gridutils import make_grid
import numpy as np
from threadpoolctl import threadpool_limits
_ = threadpool_limits(limits=1)

import utils

# Initialize the climate model
CLIMATE_MODEL = utils.AdiabatClimateRobust(
    'inputs/species_climate.yaml',
    'inputs/settings_climate.yaml',
    'inputs/GJ1132.txt'
)
CLIMATE_MODEL.verbose = False
CLIMATE_MODEL.P_top = 1e2
CLIMATE_MODEL.xtol_rc = 1e-7

def model(x):
    c = CLIMATE_MODEL

    P_i = x_to_Pi(x, c)

    # Compute climate
    converged = c.RCE_robust(P_i)

    result = make_result(x, c, converged)

    return result

def x_to_Pi(x, c):
    log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4 = x
    P_i = np.ones(len(c.species_names))*1e-10
    P_i[c.species_names.index('H2O')] = 270.0
    P_i[c.species_names.index('N2')] = 1.0 # always 1 bar
    P_i[c.species_names.index('CO2')] = 10.0**log10PCO2
    P_i[c.species_names.index('O2')] = 10.0**log10PO2
    P_i[c.species_names.index('CO')] = 10.0**log10PCO
    P_i[c.species_names.index('H2')] = 10.0**log10PH2
    P_i[c.species_names.index('CH4')] = 10.0**log10PCH4
    P_i *= 1.0e6 # convert to dynes/cm^2
    return P_i

def make_result(x, c, converged):
    # Save the P-z-T profile
    P = np.append(c.P_surf,c.P)
    z = np.append(0,c.z)
    T = np.append(c.T_surf,c.T)

    # Mixing ratios
    f_i = np.concatenate((np.array([c.f_i[0,:]]),c.f_i),axis=0)

    # Save results as 32 bit floats
    result = {}
    result['converged'] = np.array(converged)
    result['x'] = x.astype(np.float32)
    result['P'] = P.astype(np.float32)
    result['z'] = z.astype(np.float32)
    result['T'] = T.astype(np.float32)
    for i,sp in enumerate(c.species_names):
        result[sp] = f_i[:,i].astype(np.float32)

    return result

def get_gridvals():
    log10PCO2 = np.append(np.arange(-5,1.01,1),1.5)
    log10PO2 = np.append(np.arange(-7,1.01,2),1.5)
    log10PCO = np.append(np.arange(-7,1.01,2),1.5)
    log10PH2 = np.append(np.arange(-6,0.01,2),0.5)
    log10PCH4 = np.append(np.arange(-7,1.01,1),1.5)
    gridvals = (log10PCO2,log10PO2,log10PCO,log10PH2,log10PCH4)
    return gridvals

if __name__ == "__main__":
    # mpiexec -n X python climate_grid.py
    make_grid(
        model_func=model, 
        gridvals=get_gridvals(), 
        filename='results/climate_v1.h5', 
        progress_filename='results/climate_v1.log'
    )