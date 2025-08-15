import numpy as np
import requests
import tempfile
from astropy.io import fits
from photochem.utils import stars
from photochem.utils import climate
from photochem.utils import zahnle_rx_and_thermo_files
from astropy import constants
import planets

def create_climate_inputs():
    # Species file
    climate.species_file_for_climate(
        filename='inputs/species_climate.yaml', 
        species=['H2O','CO2','N2','H2','CH4','CO','O2'], 
        condensates=['H2O','CO2','CH4']
    )

    # Settings file
    climate.settings_file_for_climate(
        filename='inputs/settings_climate.yaml', 
        planet_mass=float(planets.LHS1140b.mass*constants.M_earth.cgs.value), 
        planet_radius=float(planets.LHS1140b.radius*constants.R_earth.cgs.value), 
        surface_albedo=0.1, 
        number_of_layers=50, 
        number_of_zenith_angles=4, 
        photon_scale_factor=1.0
    )

def create_zahnle_HNOC():
    "Creates a reactions file with H, N, O, C species."
    zahnle_rx_and_thermo_files(
        atoms_names=['H', 'N', 'O', 'C'],
        rxns_filename='inputs/zahnle_HNOC.yaml',
        thermo_filename=None
    )

def download_muscles_spectrum(url, outputfile, Teq=None, stellar_flux=None):

    response = requests.get(url)
    if response.status_code != 200:
        raise Exception('Failed to download '+url)
    
    # Read the download  
    with tempfile.TemporaryFile() as f:
        f.write(response.content)
        data = fits.getdata(f)

    # Get the spectrum
    wv = data['WAVELENGTH']/10 # convert from Angstroms to nm
    # (erg/cm2/s/Ang)*(1 W/1e7 erg)*(1e3 mW/1 W)*(1e4 cm^2/1 m^2)*(10 Ang/1 nm) = mW/m^2/nm
    F = data['FLUX']*(1/1e7)*(1e3/1)*(1e4/1)*(10/1) # convert from erg/cm2/s/Ang to mW/m^2/nm

    # Remove duplicated wavelengths
    wv, inds = np.unique(wv, return_index=True)
    F = F[inds]

    # Assert that spectrum goes to > 90 microns
    assert np.max(wv) > 90e3

    # Rescale to planet
    F = stars.scale_spectrum_to_planet(wv, F, Teq, stellar_flux)

    # Only consider needed resolution
    wv, F = stars.rebin_to_needed_resolution(wv, F)

    stars.save_photochem_spectrum(wv, F, outputfile, scale_to_planet=False)

    return wv, F

def create_stellar_fluxes():

    # GJ 1132
    url = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/muscles/v25/gj1132/hlsp_muscles_multi_multi_gj1132_broadband_v25_adapt-var-res-sed.fits'
    _ = download_muscles_spectrum(
        url, 
        'inputs/GJ1132.txt', 
        stellar_flux=planets.LHS1140b.stellar_flux
    )

    # GJ 699
    url = 'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HLSP/muscles/v25/gj699/hlsp_muscles_multi_multi_gj699_broadband_v25_adapt-var-res-sed.fits'
    _ = download_muscles_spectrum(
        url, 
        'inputs/GJ699.txt', 
        stellar_flux=planets.LHS1140b.stellar_flux
    )

def main():
    create_climate_inputs()
    create_zahnle_HNOC()
    create_stellar_fluxes()

if __name__ == '__main__':
    main()


