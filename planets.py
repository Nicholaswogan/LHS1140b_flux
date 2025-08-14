
class Star:
    radius : float # relative to the sun
    Teff : float # K
    metal : float # log10(M/H)
    kmag : float
    logg : float
    planets : dict # dictionary of planet objects

    def __init__(self, radius, Teff, metal, kmag, logg, planets):
        self.radius = radius
        self.Teff = Teff
        self.metal = metal
        self.kmag = kmag
        self.logg = logg
        self.planets = planets
        
class Planet:
    radius : float # in Earth radii
    mass : float # in Earth masses
    Teq : float # Equilibrium T in K
    transit_duration : float # in seconds
    a: float # semi-major axis in AU
    a: float # semi-major axis in AU
    stellar_flux: float # W/m^2
    
    def __init__(self, radius, mass, Teq, transit_duration, a, stellar_flux):
        self.radius = radius
        self.mass = mass
        self.Teq = Teq
        self.transit_duration = transit_duration
        self.a = a
        self.stellar_flux = stellar_flux

# All from Cadieux et al. (2024), except when otherwise noted

LHS1140b = Planet(
    radius=1.730,
    mass=5.60,
    Teq=226,
    transit_duration=2.055*60*60,
    a=0.0946,
    stellar_flux=0.43*1370.0
)

LHS1140 = Star(
    radius=0.2159,
    Teff=3096.0,
    metal=-0.15,
    kmag=8.8, # Exo.Mast
    logg=5.041,
    planets={'b':LHS1140b}
)








