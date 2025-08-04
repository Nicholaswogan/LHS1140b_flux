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

def create_stellar_flux():
    _ = stars.muscles_spectrum(
        star_name='GJ1214',
        outputfile='inputs/stellar_flux.txt',
        stellar_flux=planets.LHS1140b.stellar_flux,
    )

def main():
    create_climate_inputs()
    create_zahnle_HNOC()
    create_stellar_flux()

if __name__ == '__main__':
    main()


