import numpy as np
from tempfile import NamedTemporaryFile
from copy import deepcopy
import numba as nb
from numba import types
from scipy import integrate
from scipy import constants as const
from scipy import interpolate

from picaso import justdoit as jdi
from photochem.utils import stars
from photochem._clima import rebin, rebin_with_errors
from photochem.clima import AdiabatClimate, ClimaException
from photochem import EvoAtmosphere, PhotoException

class AdiabatClimateRobust(AdiabatClimate):

    def __init__(self, species_file, settings_file, flux_file, data_dir=None):

        super().__init__(
            species_file, 
            settings_file, 
            flux_file,
            data_dir
        )

        self.solve_for_T_trop = True # Enable solving for T_trop
        self.max_rc_iters = 30 # Lots of iterations
        self.P_top = 10.0 # 10 dynes/cm^2 top, or 1e-5 bars.
        self.convective_newton_step_size = 0.05
        self.RH = np.ones(len(self.species_names))*0.5 # 0.5 RH

    def surface_temperature_robust(self, P_i, T_guess_mid=None, T_perturbs=None):

        if T_guess_mid is None:
            T_guess_mid = self.rad.skin_temperature(0)*(1/2)**(-1/4) + 50
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 20.0, -20.0, 30.0, 50.0, 80.0, 100.0, 200.0, 300.0, 400.0, 600.0])

        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            try:
                self.T_trop = self.rad.skin_temperature(0.0)*1.2
                self.surface_temperature(P_i, T_surf_guess)
                converged = True
                break
            except ClimaException as e:
                converged = False
        
        return converged
    
    def adjust_convecting_pattern(self, convecting_with_below, remove_conv_param):

        convecting_with_below_copy = convecting_with_below.copy()

        k = 0
        num_to_switch = int(len(np.where(convecting_with_below_copy)[0])*remove_conv_param)
        for i in range(len(convecting_with_below_copy)):
            j = len(convecting_with_below_copy) - i - 1
            if k >= num_to_switch:
                break
            
            if convecting_with_below_copy[j]:
                convecting_with_below_copy[j] = False
                k += 1

        return convecting_with_below_copy
    
    def check_for_overconvection(self):
        "Checks if we are in an overconvecting regime"

        convecting_with_below = self.convecting_with_below
        
        # Compute the dT/dP
        T = np.append(self.T_surf, self.T)
        P = np.append(self.P_surf, self.P)
        log10P = np.log10(P)
        dT_dP = (T[1:] - T[:-1])/(log10P[1:] - log10P[:-1])
        
        # Find lowest convective zone
        ind = -1
        for i in range(len(convecting_with_below)):
            if convecting_with_below[i]:
                ind = i
                break 
        
        # Return if no convective zones
        if ind == -1:
            return False, None
        
        # Find top of first convective zone
        ind1 = -1
        for i in range(ind,len(convecting_with_below)):
            if not convecting_with_below[i]:
                ind1 = i
                break
                
        if ind1 == -1 and ind == 0:
            # Whole atmosphere is convective then we assume overconvective
            return True, None
        if ind1 == -1 and ind != 0:
            # Some convective zone above surface, then we assume it is OK.
            return False, None
        if ind1 == 0:
            # Weird case where the top of the convective zone is
            # the surface... seems impossible but we account for it anyway.
            return False, None
        
        # Now look at the top of the convective zone
        if dT_dP[ind1] < -10 and dT_dP[ind1-1] > 0:
            # If there is a "kink" in the P-T profile then this is overconvection
            return True, dT_dP[ind1]
        else:
            return False, None
    
    def RCE_simple_guess(self, P_i, remove_conv_params=None):

        if remove_conv_params is None:
            remove_conv_params = [0.5, 0.3, 0.2, 0.0]

        converged_simple = self.surface_temperature_robust(P_i)
        if not converged_simple:
            # If this fails, then we give up, returning no convergence
            return False

        # If simple climate model converged, then save the atmosphere
        T_surf_guess, T_guess, convecting_with_below_guess = self.T_surf, self.T, self.convecting_with_below

        remove_conv_param_save = []
        dT_dP_save = []
        for remove_conv_param in remove_conv_params:

            convecting_with_below_tmp = self.adjust_convecting_pattern(convecting_with_below_guess, remove_conv_param)
            
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below_tmp)
            except ClimaException:
                converged = False

            if converged:
                overconvecting, dT_dP = self.check_for_overconvection()
                if dT_dP is not None:
                    # Save reasonable failures
                    remove_conv_param_save.append(remove_conv_param)
                    dT_dP_save.append(dT_dP)
                if not overconvecting:
                    # Try other remove_conv_params to see if we
                    # can get a case without overconvection.
                    # Note that if the last one works, then 
                    # the model will report converged, even if
                    # overconvection is happening.
                    break

        if converged:
            if overconvecting and len(dT_dP_save) > 0:
                # If we made it here, then there are models that converged but all of
                # them were overconvecting. So, we are going to pick the best one
                # and stick with it.

                ind = np.argmax(dT_dP_save) # corresponds to the smallist "kink"

                if remove_conv_param == remove_conv_param_save[ind]:
                    # The best one is the last one, so no work needed
                    pass
                else:
                    # The best one is not the last one, so we must recompute
                    convecting_with_below_tmp = self.adjust_convecting_pattern(convecting_with_below_guess, remove_conv_param_save[ind])
                    try:
                        converged = self.RCE(P_i, T_surf_guess, T_guess, convecting_with_below_tmp)
                    except ClimaException:
                        converged = False

        return converged
        
    def RCE_isotherm_guess(self, P_i, T_guess_mid=None, T_perturbs=None):  

        if T_guess_mid is None:
            T_guess_mid = self.rad.skin_temperature(0)*(1/2)**(-1/4)
        
        if T_perturbs is None:
            T_perturbs = np.array([0.0, 10.0, -10.0, 20.0, 30.0, 50.0, 80.0])

        # First, we try a bunch of isothermal atmospheres.
        for i,T_perturb in enumerate(T_perturbs):
            T_surf_guess = T_guess_mid + T_perturb
            T_guess = np.ones(self.T.shape[0])*T_surf_guess
            try:
                converged = self.RCE(P_i, T_surf_guess, T_guess)
                if converged:
                    break
            except ClimaException:
                converged = False

        return converged
    
    def check_for_steam_dominated(self):
        ind = self.species_names.index('H2O')
        return np.all(self.f_i[0,ind] >= self.f_i[0,:])
    
    def RCE_robust(self, P_i, remove_conv_params=None, T_guess_mid=None, T_perturbs=None, convective_newton_step_size=0.05, max_rc_iters_convection=5):

        # Set convection parameters
        self.max_rc_iters_convection = max_rc_iters_convection
        self.convective_newton_step_size = convective_newton_step_size

        # First, we try a single isothermal case
        converged = self.RCE_isotherm_guess(P_i, T_guess_mid, np.array([0.0]))
        if converged:
            overconvecting, _ = self.check_for_overconvection()
            steam_dominated = self.check_for_steam_dominated()
            if not overconvecting and not steam_dominated:
                return converged

        # Try guess based on simple climate model
        converged = self.RCE_simple_guess(P_i, remove_conv_params)
        if converged:
            steam_dominated = self.check_for_steam_dominated()
            if not steam_dominated:
                return converged

        # Next try with isotherms
        converged = self.RCE_isotherm_guess(P_i, T_guess_mid, T_perturbs)
        if converged:
            steam_dominated = self.check_for_steam_dominated()
            if not steam_dominated:
                return converged
        
        # Finally, we make a compromise, and try for a steady-state climate that
        # has a single convecting layer at depth, and
        self.max_rc_iters_convection = -1
        self.convective_newton_step_size = 1e-100
        converged = self.RCE_simple_guess(P_i)
        self.max_rc_iters_convection = max_rc_iters_convection
        self.convective_newton_step_size = convective_newton_step_size

        # We will permit steam dominated here because it probably isn't so crazy

        return converged

class RobustData():
    
    def __init__(self):

        # Parameters for determining steady state
        self.atols = [1e-23, 1e-22, 1e-20, 1e-18]
        self.min_mix_reset = -1e-13
        self.TOA_pressure_avg = 1.0e-7*1e6 # mean TOA pressure (dynes/cm^2)
        self.max_dT_tol = 3 # The permitted difference between T in photochem and desired T
        self.max_dlog10edd_tol = 0.2 # The permitted difference between Kzz in photochem and desired Kzz
        self.freq_update_PTKzz = 1000 # step frequency to update PTKzz profile.
        self.freq_update_atol = 10_000
        self.max_total_step = 100_000 # Maximum total allowed steps before giving up
        self.min_step_conv = 300 # Min internal steps considered before convergence is allowed
        self.verbose = True # print information or not?
        self.freq_print = 100 # Frequency in which to print

        # Below for interpolation
        self.log10P_interp = None
        self.T_interp = None
        self.log10edd_interp = None
        self.P_desired = None
        self.T_desired = None
        self.Kzz_desired = None
        # information needed during robust stepping
        self.total_step_counter = None
        self.nerrors = None
        self.max_time = None
        self.robust_stepper_initialized = None
        # Surface pressures
        self.Pi = None

class EvoAtmosphereRobust(EvoAtmosphere):

    def __init__(self, mechanism_file, settings_file, flux_file, data_dir=None):

        with NamedTemporaryFile('w',suffix='.txt') as f:
            f.write(ATMOSPHERE_INIT)
            f.flush()
            super().__init__(
                mechanism_file, 
                settings_file, 
                flux_file,
                f.name,
                data_dir
            )

        self.rdat = RobustData()

        # Values in photochem to adjust
        self.var.verbose = 0
        self.var.upwind_molec_diff = True
        self.var.autodiff = True
        self.var.atol = 1.0e-23
        self.var.equilibrium_time = 1e15

        # Model state
        self.max_time_state = None

        for i in range(len(self.var.cond_params)):
            self.var.cond_params[i].smooth_factor = 1
            self.var.cond_params[i].k_evap = 0

    def set_surface_pressures(self, Pi):
        
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

    def initialize_to_PT(self, P, T, Kzz, mix):

        P, T, mix = deepcopy(P), deepcopy(T), deepcopy(mix)

        rdat = self.rdat

        # Ensure X sums to 1
        ftot = np.zeros(P.shape[0])
        for key in mix:
            ftot += mix[key]
        for key in mix:
            mix[key] = mix[key]/ftot

        # Compute mubar at all heights
        mu = {}
        for i,sp in enumerate(self.dat.species_names[:-2]):
            mu[sp] = self.dat.species_mass[i]
        mubar = np.zeros(P.shape[0])
        for key in mix:
            mubar += mix[key]*mu[key]

        # Altitude of P-T grid
        P1, T1, mubar1, z1 = compute_altitude_of_PT(P, T, mubar, self.dat.planet_radius, self.dat.planet_mass, rdat.TOA_pressure_avg)
        # If needed, extrapolate Kzz and mixing ratios
        if P1.shape[0] != Kzz.shape[0]:
            Kzz1 = np.append(Kzz,Kzz[-1])
            mix1 = {}
            for sp in mix:
                mix1[sp] = np.append(mix[sp],mix[sp][-1])
        else:
            Kzz1 = Kzz.copy()
            mix1 = mix

        rdat.log10P_interp = np.log10(P1.copy()[::-1])
        rdat.T_interp = T1.copy()[::-1]
        rdat.log10edd_interp = np.log10(Kzz1.copy()[::-1])
        
        # extrapolate to 1e6 bars
        T_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.T_interp, bounds_error=False, fill_value='extrapolate')(12)
        edd_tmp = interpolate.interp1d(rdat.log10P_interp, rdat.log10edd_interp, bounds_error=False, fill_value='extrapolate')(12)
        rdat.log10P_interp = np.append(rdat.log10P_interp, 12)
        rdat.T_interp = np.append(rdat.T_interp, T_tmp)
        rdat.log10edd_interp = np.append(rdat.log10edd_interp, edd_tmp)

        rdat.P_desired = P1.copy()
        rdat.T_desired = T1.copy()
        rdat.Kzz_desired = Kzz1.copy()

        # Calculate the photochemical grid
        ind_t = np.argmin(np.abs(P1 - rdat.TOA_pressure_avg))
        z_top = z1[ind_t]
        z_bottom = 0.0
        dz = (z_top - z_bottom)/self.var.nz
        z_p = np.empty(self.var.nz)
        z_p[0] = dz/2.0
        for i in range(1,self.var.nz):
            z_p[i] = z_p[i-1] + dz

        # Now, we interpolate all values to the photochemical grid
        P_p = 10.0**np.interp(z_p, z1, np.log10(P1))
        T_p = np.interp(z_p, z1, T1)
        Kzz_p = 10.0**np.interp(z_p, z1, np.log10(Kzz1))
        mix_p = {}
        for sp in mix1:
            mix_p[sp] = 10.0**np.interp(z_p, z1, np.log10(mix1[sp]))
        k_boltz = const.k*1e7
        den_p = P_p/(k_boltz*T_p)

        # Update photochemical model grid
        self.update_vertical_grid(TOA_alt=z_top) # this will update gravity for new planet radius
        self.set_temperature(T_p)
        self.var.edd = Kzz_p
        usol = np.ones(self.wrk.usol.shape)*1e-40
        species_names = self.dat.species_names[:(-2-self.dat.nsl)]
        for sp in mix_p:
            if sp in species_names:
                ind = species_names.index(sp)
                usol[ind,:] = mix_p[sp]*den_p
        self.wrk.usol = usol

        self.prep_atmosphere(self.wrk.usol)

    def initialize_to_PT_bcs(self, P, T, Kzz, mix, Pi):
        self.rdat.Pi = Pi
        self.set_surface_pressures(Pi)
        self.initialize_to_PT(P, T, Kzz, mix)

    def set_particle_radii(self, radii):
        particle_radius = self.var.particle_radius
        for key in radii:
            ind = self.dat.species_names.index(key)
            particle_radius[ind,:] = radii[key]
        self.var.particle_radius = particle_radius
        self.update_vertical_grid(TOA_alt=self.var.top_atmos)

    def initialize_robust_stepper(self, usol):
        """Initialized a robust integrator.

        Parameters
        ----------
        usol : ndarray[double,dim=2]
            Input number densities
        """
        rdat = self.rdat  
        rdat.total_step_counter = 0
        rdat.nerrors = 0
        rdat.max_time = 0
        self.max_time_state = None
        self.initialize_stepper(usol)
        rdat.robust_stepper_initialized = True

    def robust_step(self):
        """Takes a single robust integrator step

        Returns
        -------
        tuple
            The tuple contains two bools `give_up, reached_steady_state`. If give_up is True
            then the algorithm things it is time to give up on reaching a steady state. If
            reached_steady_state then the algorithm has reached a steady state within
            tolerance.
        """

        rdat = self.rdat

        if not rdat.robust_stepper_initialized:
            raise Exception('This routine can only be called after `initialize_robust_stepper`')

        give_up = False
        reached_steady_state = False

        for i in range(1):
            try:
                self.step()
                rdat.total_step_counter += 1
            except PhotoException as e:
                # If there is an error, lets reinitialize, but get rid of any
                # negative numbers
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Reset integrator if we get large magnitude negative numbers
            if not self.healthy_atmosphere():
                usol = np.clip(self.wrk.usol.copy(),a_min=1.0e-40,a_max=np.inf)
                self.initialize_stepper(usol)
                rdat.nerrors += 1

                if rdat.nerrors > 15:
                    give_up = True
                    break

            # Update the max time achieved
            if self.wrk.tn > rdat.max_time:
                rdat.max_time = self.wrk.tn
                self.max_time_state = self.model_state_to_dict() # save the model state

            # convergence checking
            converged = self.check_for_convergence()

            # Compute the max difference between the P-T profile in photochemical model
            # and the desired P-T profile
            T_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.T_interp)
            T_p = T_p.copy()[::-1]
            max_dT = np.max(np.abs(T_p - self.var.temperature))

            # Compute the max difference between the P-edd profile in photochemical model
            # and the desired P-edd profile
            log10edd_p = np.interp(np.log10(self.wrk.pressure_hydro.copy()[::-1]), rdat.log10P_interp, rdat.log10edd_interp)
            log10edd_p = log10edd_p.copy()[::-1]
            max_dlog10edd = np.max(np.abs(log10edd_p - np.log10(self.var.edd)))

            # TOA pressure
            TOA_pressure = self.wrk.pressure_hydro[-1]

            condition1 = converged and self.wrk.nsteps > rdat.min_step_conv or self.wrk.tn > self.var.equilibrium_time
            condition2 = max_dT < rdat.max_dT_tol and max_dlog10edd < rdat.max_dlog10edd_tol and rdat.TOA_pressure_avg/3 < TOA_pressure < rdat.TOA_pressure_avg*3

            if condition1 and condition2:
                if rdat.verbose:
                    print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                        (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                # success!
                reached_steady_state = True
                break

            if not (rdat.total_step_counter % rdat.freq_update_atol):
                ind = int(rdat.total_step_counter/rdat.freq_update_atol)
                ind1 = ind - len(rdat.atols)*int(ind/len(rdat.atols))
                self.var.atol = rdat.atols[ind1]
                if rdat.verbose:
                    print('new atol = %.1e'%(self.var.atol))
                self.initialize_stepper(self.wrk.usol)
                break

            if not (self.wrk.nsteps % rdat.freq_update_PTKzz) or (condition1 and not condition2):
                # After ~1000 steps, lets update P,T, edd and vertical grid, if possible.
                try:
                    self.set_press_temp_edd(rdat.P_desired,rdat.T_desired,rdat.Kzz_desired,hydro_pressure=True)
                except PhotoException:
                    pass
                try:
                    self.update_vertical_grid(TOA_pressure=rdat.TOA_pressure_avg)
                except PhotoException:
                    pass
                self.initialize_stepper(self.wrk.usol)

            if rdat.total_step_counter > rdat.max_total_step:
                give_up = True
                break

            if not (self.wrk.nsteps % rdat.freq_print) and rdat.verbose:
                print('nsteps = %i  longdy = %.1e  max_dT = %.1e  max_dlog10edd = %.1e  TOA_pressure = %.1e'% \
                    (rdat.total_step_counter, self.wrk.longdy, max_dT, max_dlog10edd, TOA_pressure/1e6))
                
        return give_up, reached_steady_state
    
    def find_steady_state(self):
        """Attempts to find a photochemical steady state.

        Returns
        -------
        bool
            If True, then the routine was successful.
        """    

        self.initialize_robust_stepper(self.wrk.usol)
        success = True
        while True:
            give_up, reached_steady_state = self.robust_step()
            if reached_steady_state:
                break
            if give_up:
                success = False
                break
        return success
    
    def healthy_atmosphere(self):
        return np.min(self.wrk.mix_history[:,:,0]) > self.rdat.min_mix_reset
    
    def find_steady_state_robust(self):

        # Change some rdat settings
        self.rdat.freq_update_atol = 100_000
        self.rdat.max_total_step = 10_000

        # First just try to get to steady-state with standard atol
        self.var.atol = 1.0e-23
        converged = self.find_steady_state()
        if converged:
            return converged

        # Convergence did not happen. Save the max time state.
        max_time = self.rdat.max_time
        max_time_state = deepcopy(self.max_time_state)

        # Lets try a couple different atols.
        for atol in [1.0e-18, 1.0e-15]:
            # Lets initialize to max time state
            self.initialize_from_dict(max_time_state)
            # Do some smaller number of steps
            self.rdat.max_total_step = 5_000
            self.var.atol = atol # set the atol
            converged = self.find_steady_state() # Integrate
            if converged:
                # If converged then lets return
                return converged

            # No convergence. We re-save max time state
            if self.rdat.max_time > max_time:
                max_time = self.rdat.max_time
                max_time_state = deepcopy(self.max_time_state)

        # No convergence, we reinitialize to max time state and return
        self.initialize_from_dict(max_time_state)

        return converged
        
    def model_state_to_dict(self):
        """Returns a dictionary containing all information needed to reinitialize the atmospheric
        state. This dictionary can be used as an input to "initialize_from_dict".
        """

        if self.rdat.log10P_interp is None:
            raise Exception('This routine can only be called after `initialize_to_PT_bcs`')

        out = {}
        out['rdat'] = deepcopy(self.rdat.__dict__)
        out['top_atmos'] = self.var.top_atmos
        out['temperature'] = self.var.temperature
        out['edd'] = self.var.edd
        out['usol'] = self.wrk.usol
        out['particle_radius'] = self.var.particle_radius

        # Other settings
        out['equilibrium_time'] = self.var.equilibrium_time
        out['verbose'] = self.var.verbose
        out['atol'] = self.var.atol
        out['autodiff'] = self.var.autodiff

        return out

    def initialize_from_dict(self, out):
        """Initializes the model from a dictionary created by the "model_state_to_dict" routine.
        """

        for key, value in out['rdat'].items():
            setattr(self.rdat, key, value)

        self.update_vertical_grid(TOA_alt=out['top_atmos'])
        self.set_temperature(out['temperature'])
        self.var.edd = out['edd']
        self.wrk.usol = out['usol']
        self.var.particle_radius = out['particle_radius']
        self.update_vertical_grid(TOA_alt=out['top_atmos'])

        # Other settings
        self.var.equilibrium_time = out['equilibrium_time']
        self.var.verbose = out['verbose']
        self.var.atol = out['atol']
        self.var.autodiff = out['autodiff']
        
        # Now set boundary conditions
        Pi = self.rdat.Pi
        for sp in Pi:
            self.set_lower_bc(sp, bc_type='press', press=Pi[sp])

        self.prep_atmosphere(self.wrk.usol)

@nb.experimental.jitclass()
class TempPressMubar:

    log10P : types.double[:] # type: ignore
    T : types.double[:] # type: ignore
    mubar : types.double[:] # type: ignore

    def __init__(self, P, T, mubar):
        self.log10P = np.log10(P)[::-1].copy()
        self.T = T[::-1].copy()
        self.mubar = mubar[::-1].copy()

    def temperature_mubar(self, P):
        T = np.interp(np.log10(P), self.log10P, self.T)
        mubar = np.interp(np.log10(P), self.log10P, self.mubar)
        return T, mubar

@nb.njit()
def gravity(radius, mass, z):
    G_grav = const.G
    grav = G_grav * (mass/1.0e3) / ((radius + z)/1.0e2)**2.0
    grav = grav*1.0e2 # convert to cgs
    return grav

@nb.njit()
def hydrostatic_equation(P, u, planet_radius, planet_mass, ptm):
    z = u[0]
    grav = gravity(planet_radius, planet_mass, z)
    T, mubar = ptm.temperature_mubar(P)
    k_boltz = const.Boltzmann*1e7
    dz_dP = -(k_boltz*T*const.Avogadro)/(mubar*grav*P)
    return np.array([dz_dP])

def compute_altitude_of_PT(P, T, mubar, planet_radius, planet_mass, P_top):
    ptm = TempPressMubar(P, T, mubar)
    args = (planet_radius, planet_mass, ptm)

    if P_top < P[-1]:
        # If P_top is lower P than P grid, then we extend it
        P_top_ = P_top
        P_ = np.append(P,P_top_)
        T_ = np.append(T,T[-1])
        mubar_ = np.append(mubar,mubar[-1])
    else:
        P_top_ = P[-1]
        P_ = P.copy()
        T_ = T.copy()
        mubar_ = mubar.copy()

    # Integrate to TOA
    out = integrate.solve_ivp(hydrostatic_equation, [P_[0], P_[-1]], np.array([0.0]), t_eval=P_, args=args, rtol=1e-6)
    assert out.success

    # Stitch together
    z_ = out.y[0]

    return P_, T_, mubar_, z_

ATMOSPHERE_INIT = \
"""alt      den        temp       eddy                       
0.0      1          1000       1e6              
1.0e3    1          1000       1e6         
"""

class Picaso():

    def __init__(self, filename_db, M_planet, R_planet, R_star, opannection_kwargs={}, star_kwargs={}):

        self.opa = jdi.opannection(filename_db=filename_db, **opannection_kwargs)
        self.case = jdi.inputs()
        self.case.phase_angle(0)
        self.case.gravity(mass=M_planet, mass_unit=jdi.u.Unit('M_earth'),
                     radius=R_planet, radius_unit=jdi.u.Unit('R_earth'))
        self.case.star(self.opa, radius=R_star, radius_unit=jdi.u.Unit('R_sun'), **star_kwargs)
        self.case.surface_reflect(np.ones(self.opa.wno.shape[0])*0.0,self.opa.wno)

    def set_custom_albedo(self, wv, albedo):
        self.case.surface_reflect(albedo[::-1].copy(), self.opa.wno, (1e4/wv[::-1]).copy())

    def clouds_reset(self):
        self.case.inputs['clouds'] = {
            'profile': None,
            'wavenumber': None,
            'scattering': {'g0': None, 'w0': None, 'opd': None}
        }

    def _spectrum(self, atm, calculation='thermal', atmosphere_kwargs={}, cloud_filename=None, log10Pcloudbottom=None, dlog10Pcloud=None, **kwargs):
        self.case.atmosphere(df=atm, verbose=False, **atmosphere_kwargs)
        self.case.approx(p_reference=np.max(atm['pressure'].to_numpy()))
        if cloud_filename is not None:
            self.case.clouds(filename=cloud_filename, delim_whitespace=True)
        if log10Pcloudbottom is not None or log10Pcloudbottom is not None:
            self.case.clouds(g0=[0.9], w0=[0.9], opd=[10], p=[log10Pcloudbottom], dp=[dlog10Pcloud])
        df = self.case.spectrum(self.opa, calculation=calculation, **kwargs)
        return df

    def rprs2(self, atm, R=100, wavl=None, atmosphere_kwargs={}, cloud_filename=None, log10Pcloudbottom=None, dlog10Pcloud=None, **kwargs):

        df = self._spectrum(
            atm, 'transmission', atmosphere_kwargs=atmosphere_kwargs, 
            cloud_filename=cloud_filename, log10Pcloudbottom=log10Pcloudbottom, dlog10Pcloud=dlog10Pcloud, 
            **kwargs
        )

        wv_h = 1e4/df['wavenumber'][::-1].copy()
        wavl_h = stars.make_bins(wv_h)
        rprs2_h = df['transit_depth'][::-1].copy()

        if wavl is None:
            wavl = stars.grid_at_resolution(np.min(wavl_h), np.max(wavl_h), R)

        rprs2 = rebin(wavl_h.copy(), rprs2_h.copy(), wavl.copy())

        return wavl, rprs2
    
    def create_exo_dict(self, R_planet, R_star, total_observing_time, eclipse_duration, kmag, starpath):
        from pandexo.engine import justdoit as pandexo_jdi

        exo_dict = pandexo_jdi.load_exo_dict()

        exo_dict['observation']['sat_level'] = 80
        exo_dict['observation']['sat_unit'] = '%'
        exo_dict['observation']['noccultations'] = 1
        exo_dict['observation']['R'] = None
        exo_dict['observation']['baseline_unit'] = 'total'
        exo_dict['observation']['baseline'] = total_observing_time
        exo_dict['observation']['noise_floor'] = 0

        exo_dict['star']['type'] = 'user'
        exo_dict['star']['mag'] = kmag
        exo_dict['star']['ref_wave'] = 2.22
        exo_dict['star']['starpath'] = starpath
        exo_dict['star']['w_unit'] = 'um'
        exo_dict['star']['f_unit'] = 'FLAM'
        exo_dict['star']['radius'] = R_star
        exo_dict['star']['r_unit'] = 'R_sun'

        exo_dict['planet']['type'] = 'constant'
        exo_dict['planet']['transit_duration'] = eclipse_duration
        exo_dict['planet']['td_unit'] = 's'
        exo_dict['planet']['radius'] = R_planet
        exo_dict['planet']['r_unit'] = 'R_earth'
        exo_dict['planet']['f_unit'] = 'rp^2/r*^2'

        return exo_dict
    
    def _run_pandexo(self, R_planet, R_star, total_observing_time, eclipse_duration, kmag, inst, starpath, verbose=False, **kwargs):
        from pandexo.engine import justdoit as pandexo_jdi

        exo_dict = self.create_exo_dict(R_planet, R_star, total_observing_time, eclipse_duration, kmag, starpath)

        # Run pandexo
        result = pandexo_jdi.run_pandexo(exo_dict, inst, verbose=verbose, **kwargs)

        return result

    def run_pandexo(self, R_planet, R_star, total_observing_time, eclipse_duration, kmag, inst, starpath, R=None, ntrans=1, verbose=False, **kwargs):

        # inst is just a string
        assert isinstance(inst, str)
        result = self._run_pandexo(R_planet, R_star, total_observing_time, eclipse_duration, kmag, [inst], starpath, verbose, **kwargs)

        spec = result['FinalSpectrum']
        wavl = stars.make_bins(spec['wave'])
        F = spec['spectrum']
        err = spec['error_w_floor']
        err = err/np.sqrt(ntrans)

        if R is not None:
            wavl_n = stars.grid_at_resolution(np.min(wavl), np.max(wavl), R)
            F_n, err_n = rebin_with_errors(wavl.copy(), F.copy(), err.copy(), wavl_n.copy())
            wavl = wavl_n
            F = F_n
            err = err_n

        return wavl, F, err
    