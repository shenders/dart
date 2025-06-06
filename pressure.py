import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve
from pipeinjector import piezo
import pyuda
client = pyuda.Client()

def gaussian_kernel(t_array, width):
    t_mid = (t_array[-1] + t_array[0]) / 2
    kernel = np.exp(-0.5 * ((t_array - t_mid) / width) ** 2)
    return kernel / np.sum(kernel)  # Normalize
# Function for smoothing Gamma (gas flow) with the custom kernel
def smooth_gas_flow(Gamma_raw, dt, kernel):
    """Smooth the raw gas flow using the Gaussian kernel."""
    return np.convolve(Gamma_raw, kernel, mode='same')
def safe_fraction(partial, total, threshold=1e-10):
    if total > threshold:
        species_fraction = partial / total
    else:
        species_fraction = np.zeros_like(partial)    
    return species_fraction 
# Constants
kB       = 1.38e-23         # Boltzmann constant [J/K] 
T_pump   = 300.0            # Temperature at pump [K]
V_machine= 29.77             # Effective volume 26.6 m^3
V_PFR    = 0.16
V_HFS    = 3.57
V_LFS    = 17.5
V_div    = 2.66
V_sdiv   = 1.51
A        = kB * T_pump      # Pre-factor in pressure equation

class pressure:
    def __init__(self,shot,plasma_conf=None,plasma_conftime=None,closure_time=0.4,subdiv_time=0.5,
                 plasma=True,cryo=False,turbo=True,drsep=None,volume=None,voltime=None,
                 drseptime=None,gasvolts=False,valve='all',gas_matrix=None,plasma_fracs=[0.1,0.12,0.77,0.01,0.0]):
        self.loaded = False
        self.turbo = turbo
        self.species_list = ['D','N']
        self.nspecies   = len(self.species_list)
        self.loaded = False
        self.turbo  = turbo
        self.cryo   = cryo       
        self.plasmaplot  = plasma
        if plasma:
            diverted     = [0   ,0.0   ,0.000,0.000 ,1.0   ,1.0   ,1.0   ,1.0   ,1.0   ,1.0   ,1.0   ,0.0]
        else:
            diverted     = [0   ,0.0   ,0.0  ,0.0   ,0.0   ,0.0   ,0.0   ,0.0  ,0.0  ,0.0  ,0.0  ,0.0]            
        time_plasma      = [-0.1,0.015 ,0.02 ,0.06  ,0.11  ,0.12  ,0.2   ,0.4  ,0.5  ,0.6  ,1.01 ,2.0] 
        if plasma_conf is not None:            
            self.plasma_conf  = interp1d(plasma_conftime ,plasma_conf,bounds_error=False, fill_value=0.0)
        else:
            if plasma:
                plasma_conf  = [0   ,0.0   ,0.002,0.002 ,0.002 ,0.0025,0.0023,0.003 ,0.00165 ,0.0013 ,0.0008,0.0]
            else:
                plasma_conf  = [0   ,0.0   ,0.0  ,0.0   ,0.0   ,0.0   ,0.0   ,0.0  ,0.0  ,0.0  ,0.0  ,0.0]                            
            self.plasma_conf  = interp1d(time_plasma ,plasma_conf,bounds_error=False, fill_value=0.0)
        self.diverted  = interp1d(time_plasma ,diverted,bounds_error=False, fill_value=0.0)
        if drsep is not None:
            self.drsep = interp1d(drseptime,drsep,bounds_error=False, fill_value=0.0)
        else:
            self.drsep = interp1d([-0.1,0.1,10.0],[0.002,0.002,0.002],bounds_error=False, fill_value=0.0)
        if volume is not None:
            self.V_plasma = interp1d(voltime,volume,bounds_error=False, fill_value=0.0)
        else:
            self.V_plasma = interp1d([-0.1,0.1,10.0],[9.0,9.0,9.0],bounds_error=False, fill_value=0.0)

        self.setup_arrays()
        self.setup_times(closure_time=closure_time,subdiv_time=subdiv_time,plasma_fracs=plasma_fracs)
        self.setup_pumping()
        self.setup_influx(shot,gasvolts=gasvolts,valve=valve,gas_matrix=gas_matrix)
        self.track_particles()

    def setup_times(self,closure_time=0.4,subdiv_time=0.5,plasma_fracs=[0.1,0.12,0.77,0.01,0.0]):
        # Setup conductances between reservoirs
        self.k_leak_hfs_lfs    = 1.0/0.04
        self.k_leak_hfs_div    = 1.0/0.30
        self.k_leak_div_hfs    = 1.0/0.1
        self.k_leak_lfs_div    = 1.0/0.1
        self.k_leak_div_lfs    = 1.0/0.30
        self.k_leak_pfr_div    = 1.0/0.04
        self.k_leak_hfs_pfr    = 1.0/0.04
        self.k_leak_div_lfs_plasma    = 1.0/closure_time
        self.k_leak_div_sub    = 1.0/subdiv_time
        self.k_leak_pipe_main  = 1/0.003
        # Setup ballistic streaming boost factor
        self.ballistic_boost   = 6.0
        # Setup fuelling efficiencies
        self.hfs_fuelling      = 0.8
        self.lfs_fuelling      = 0.5
        self.pfr_fuelling      = 0.3
        # Setup plasma recycling fractions for diverted plasma
        
        self.plasma_div_frac   = plasma_fracs[0]
        self.plasma_lfs_frac   = plasma_fracs[1]
        self.plasma_hfs_frac   = plasma_fracs[2]
        self.plasma_pfr_frac   = plasma_fracs[3]
        self.plasma_wall_frac  = plasma_fracs[4]
        # Setup plasma recycling fractions for limited plasma
        self.limiter_div_frac   = 0.0
        self.limiter_lfs_frac   = 0.25
        self.limiter_hfs_frac   = 0.65
        self.limiter_pfr_frac   = 0.0
        self.limiter_wall_frac  = 0.1
    def setup_pumping(self):
        self.f_wall_hit = 25.0
        if self.turbo:
            self.recycling = 0.98955
        else:
            self.recycling = 0.0
        if self.cryo:
            self.lower_S_subdiv = 10.0
            self.upper_S_subdiv = 0.0
        else:
            self.lower_S_subdiv = 0.0
            self.upper_S_subdiv = 0.0
    def calc_Ndot(self,trace,shot,pipe_length,plenum_pressure_bar, calc_piezo,valve):
        data = client.get(trace, shot)
        time = np.array(data.time.data)
        if valve == '':
            mul = 1e21
        else:
            mul = 1.0
        Ndot = calc_piezo.simulate_gas_flow_with_pipe_delay(np.array(data.data)*mul,plenum_pressure_bar,
                                                            pipe_length,6.0,1e-7,time[1]-time[0],valve)
        return time,Ndot
    def setup_influx(self,shot,gasvolts=False,valve=None,gas_matrix=None):
        calc_piezo = piezo()
        
        if gasvolts:
            hfs_valves = {
                'hfs_mid_u02': [1.0 if valve in ['all', 'hfs_mid_u02'] else 0.0,0.3],
                'hfs_mid_u08': [1.0 if valve in ['all', 'hfs_mid_u08'] else 0.0,0.3],
                'hfs_mid_l02': [1.0 if valve in ['all', 'hfs_mid_l02'] else 0.0,0.3],
                'hfs_mid_l08': [1.0 if valve in ['all', 'hfs_mid_l08'] else 0.0,0.3]
            }
            lfsv_valves ={ 
                'lfsv_bot_l03': [1.0 if valve in ['all', 'lfsv_bot_l03'] else 0.0,0.3],
                'lfsv_bot_l09': [1.0 if valve in ['all', 'lfsv_bot_l09'] else 0.0,0.3],
                'lfsv_top_u05': [1.0 if valve in ['all', 'lfsv_top_u05'] else 0.0,0.3],
                'lfsv_top_u11': [1.0 if valve in ['all', 'lfsv_top_u11'] else 0.0,0.3]
            }
            lfsd_bot_valves = {
                'lfsd_bot_l0102': [1.0 if valve in ['all', 'lfsd_bot_l0102'] else 0.0,0.6],
                'lfsd_bot_l0506': [1.0 if valve in ['all', 'lfsd_bot_l0506'] else 0.0,0.6]
            }
            lfsd_top_valves = {
                'lfsd_top_u0102': [1.0 if valve in ['all', 'lfsd_top_u0102'] else 0.0,0.6],
                'lfsd_top_u0506': [1.0 if valve in ['all', 'lfsd_top_u0506'] else 0.0,0.6]
            }
            lfss_bot_valves = {
                'lfss_bot_l0405': [1.0 if valve in ['all', 'lfss_bot_l0405'] else 0.0,0.6]
            }
            lfss_top_valves = {
                'lfss_top_u0405': [1.0 if valve in ['all', 'lfss_top_u0405'] else 0.0,0.3]
            }
            pfr_top_valves = {
                'pfr_top_t01': [1.0 if valve in ['all', 'pfr_top_t01'] else 0.0,0.3],
                'pfr_top_t05': [1.0 if valve in ['all', 'pfr_top_t05'] else 0.0,0.3]               
            }
            pfr_bot_valves = {
                'pfr_bot_b01': [1.0 if valve in ['all', 'pfr_bot_b01'] else 0.0,0.3],
                'pfr_bot_b05': [1.0 if valve in ['all', 'pfr_bot_b05'] else 0.0,0.3]               
            }
            
            Ndot_HFS, Ndot_LFS, Ndot_UDV, Ndot_LDV, Ndot_UDVS, Ndot_LDVS, Ndot_LPFR, Ndot_UPFR = 0, 0, 0, 0, 0, 0, 0, 0
            plenum_lfs,plenum_ldv,plenum_ldvs,plenum_hfs,plenum_pfr=1.5,1.5,1.5,1.5,1.5
            for key, (mul,pipe_length) in hfs_valves.items():
                time_HFS, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_hfs, calc_piezo, key)
                Ndot_HFS += mul * Ndot
            for key, (mul,pipe_length) in lfsv_valves.items():
                time_LFS, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_lfs, calc_piezo, key)
                Ndot_LFS += mul * Ndot
            for key, (mul,pipe_length) in lfsd_bot_valves.items():
                time_LDV, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_ldv, calc_piezo, key)
                Ndot_LDV += mul * Ndot
            for key, (mul,pipe_length) in lfsd_top_valves.items():
                time_UDV, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_ldv, calc_piezo, key)
                Ndot_UDV += mul * Ndot
            for key, (mul,pipe_length) in lfss_bot_valves.items():
                time_LDVS, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_ldvs, calc_piezo, key)
                Ndot_LDVS += mul * Ndot
            for key, (mul,pipe_length) in lfss_top_valves.items():
                time_UDVS, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_ldvs, calc_piezo, key)
                Ndot_UDVS += mul * Ndot
            for key, (mul,pipe_length) in pfr_top_valves.items():
                time_LPFR, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_ldvs, calc_piezo, key)
                Ndot_LPFR += mul * Ndot
            for key, (mul,pipe_length) in pfr_bot_valves.items():
                time_UPFR, Ndot = self.calc_Ndot('/xdc/gas/f/'+key, shot, pipe_length, plenum_ldvs, calc_piezo, key)
                Ndot_UPFR += mul * Ndot 
        else:
            time_HFS,Ndot_HFS = self.calc_Ndot('/xdc/flow/s/hfs_mid_flow', shot, 0.3, 1.5, calc_piezo,'')        

            time_LFS,Ndot_lfsv_bot = self.calc_Ndot('/xdc/flow/s/lfsv_bot_flow', shot, 0.3, 1.5, calc_piezo,'')        
            time_LFS,Ndot_lfsv_top = self.calc_Ndot('/xdc/flow/s/lfsv_top_flow', shot, 0.3, 1.5, calc_piezo,'')        
            Ndot_LFS = (Ndot_lfsv_bot + Ndot_lfsv_top)

            time_LDV,Ndot_LDV = self.calc_Ndot('/xdc/flow/s/lfsd_bot_flow', shot, 0.6, 1.5, calc_piezo,'')
            time_UDV,Ndot_UDV = self.calc_Ndot('/xdc/flow/s/lfsd_top_flow', shot, 0.6, 1.5, calc_piezo,'')

            time_LDVS,Ndot_LDVS = self.calc_Ndot('/xdc/flow/s/lfss_bot_flow', shot, 0.6, 1.5, calc_piezo,'')
            time_UDVS,Ndot_UDVS = self.calc_Ndot('/xdc/flow/s/lfss_top_flow', shot, 0.6, 1.5, calc_piezo,'')

            time_UPFR,Ndot_pfrt01 = self.calc_Ndot('/xdc/flow/s/pfr_top_t01', shot, 0.6, 1.5, calc_piezo,'')
            time_UPFR,Ndot_pfrt05 = self.calc_Ndot('/xdc/flow/s/pfr_top_t05', shot, 0.6, 1.5, calc_piezo,'')
            Ndot_UPFR = (Ndot_pfrt01 + Ndot_pfrt05)
            
            time_LPFR,Ndot_pfrb01 = self.calc_Ndot('/xdc/flow/s/pfr_bot_b01', shot, 0.6, 1.5, calc_piezo,'')
            time_LPFR,Ndot_pfrb05 = self.calc_Ndot('/xdc/flow/s/pfr_bot_b05', shot, 0.6, 1.5, calc_piezo,'')
            Ndot_LPFR = (Ndot_pfrb01 + Ndot_pfrb05)

           
        hfs_Gamma_interp        = interp1d(time_HFS, Ndot_HFS, bounds_error=False, fill_value=0.0)
        lfs_Gamma_interp        = interp1d(time_LFS, Ndot_LFS, bounds_error=False, fill_value=0.0)
        udv_Gamma_interp        = interp1d(time_UDV, Ndot_UDV, bounds_error=False, fill_value=0.0)
        ldv_Gamma_interp        = interp1d(time_LDV, Ndot_LDV, bounds_error=False, fill_value=0.0)
        udvs_Gamma_interp       = interp1d(time_UDVS, Ndot_UDVS, bounds_error=False, fill_value=0.0)
        ldvs_Gamma_interp       = interp1d(time_LDVS, Ndot_LDVS, bounds_error=False, fill_value=0.0)
        upfr_Gamma_interp       = interp1d(time_UPFR, Ndot_UPFR, bounds_error=False, fill_value=0.0)
        lpfr_Gamma_interp       = interp1d(time_LPFR, Ndot_LPFR, bounds_error=False, fill_value=0.0)
        if gas_matrix is None:
            gas_matrix              = {'HFS':0,
                                       'LFS':0,
                                       'UDV':0,
                                       'UDVS':0,
                                       'LDV':0,
                                       'LDVS':0,
                                       'LPFR':0,
                                       'UPFR':0}
        self.injected['HFS'][gas_matrix['HFS'],:]= hfs_Gamma_interp(self.t_array)                       
        self.injected['LFS'][gas_matrix['LFS'],:]= lfs_Gamma_interp(self.t_array)               
        self.injected['UDV'][gas_matrix['UDV'],:]= udv_Gamma_interp(self.t_array)             
        self.injected['UDVS'][gas_matrix['UDVS'],:]= udvs_Gamma_interp(self.t_array)             
        self.injected['LDV'][gas_matrix['LDV'],:]= ldv_Gamma_interp(self.t_array)               
        self.injected['LDVS'][gas_matrix['LDVS'],:]= ldvs_Gamma_interp(self.t_array)               
        self.injected['UPFR'][gas_matrix['UPFR'],:]= upfr_Gamma_interp(self.t_array)             
        self.injected['LPFR'][gas_matrix['LPFR'],:]= lpfr_Gamma_interp(self.t_array)               

    def setup_arrays(self):
        # Time settings
        t_start                 = -0.08
        t_end                   = 1.0
        # Initialise arrays
        self.dt                 = 0.0005
        self.t_array            = np.arange(t_start, t_end, self.dt)
        self.hfs_main           = np.zeros((len(self.t_array),self.nspecies))
        self.lfs_main           = np.zeros((len(self.t_array),self.nspecies))
        self.plasma             = np.zeros((len(self.t_array),self.nspecies))
        self.div                = np.zeros((len(self.t_array),self.nspecies))
        self.subdiv             = np.zeros((len(self.t_array),self.nspecies))
        self.subuppdiv_pressure = np.zeros((len(self.t_array),self.nspecies))
        self.sublowdiv_pressure = np.zeros((len(self.t_array),self.nspecies))
        self.lowdiv_pressure    = np.zeros((len(self.t_array),self.nspecies))
        self.uppdiv_pressure    = np.zeros((len(self.t_array),self.nspecies))
        self.main_pressure      = np.zeros((len(self.t_array),self.nspecies))
        self.total_particles    = np.zeros((len(self.t_array),self.nspecies))
        self.pumped_particles   = np.zeros((len(self.t_array),self.nspecies))
        self.injected_particles = np.zeros((len(self.t_array),self.nspecies))
        self.electron_density   = np.zeros((len(self.t_array),self.nspecies))
        self.hfs_vessel_influx  = np.zeros((len(self.t_array),self.nspecies))
        self.lfs_vessel_influx  = np.zeros((len(self.t_array),self.nspecies))
        self.udv_vessel_influx  = np.zeros((len(self.t_array),self.nspecies))
        self.ldv_vessel_influx  = np.zeros((len(self.t_array),self.nspecies))
        self.udvs_vessel_influx = np.zeros((len(self.t_array),self.nspecies))
        self.ldvs_vessel_influx = np.zeros((len(self.t_array),self.nspecies))
        self.upfr_vessel_influx = np.zeros((len(self.t_array),self.nspecies))
        self.lpfr_vessel_influx = np.zeros((len(self.t_array),self.nspecies))
        self.injected           = {'HFS':np.zeros((self.nspecies,len(self.t_array))),
                                   'LFS':np.zeros((self.nspecies,len(self.t_array))),
                                   'UDVS':np.zeros((self.nspecies,len(self.t_array))),
                                   'LDVS':np.zeros((self.nspecies,len(self.t_array))),
                                   'UPFR':np.zeros((self.nspecies,len(self.t_array))),
                                   'LPFR':np.zeros((self.nspecies,len(self.t_array))),
                                   'UDV':np.zeros((self.nspecies,len(self.t_array))),
                                   'LDV':np.zeros((self.nspecies,len(self.t_array)))}

    def evolve_reservoirs(self,dt,reservoirs,res1,res2,V_res1,V_res2,k_leak,k_leak2):
        n_total_res1 = np.sum(reservoirs[res1])
        n_total_res2 = np.sum(reservoirs[res2])
        delta        = n_total_res1 /V_res1 - n_total_res2/V_res2
        boost_factor = 1.0 + self.ballistic_boost * np.tanh(abs(delta) / 1e18)
        if delta > 0:
            leak = k_leak * boost_factor * delta * dt
            leak = min(leak,n_total_res1)
            if n_total_res1 == 0.0:
                species_fraction = np.array([0.0,0.0])
            else:
                species_fraction   = reservoirs[res1] / n_total_res1 
        else:
            leak = k_leak2 * boost_factor * delta * dt
            leak = max(leak,-n_total_res2)
            if n_total_res2 == 0.0:
                species_fraction = np.array([0.0,0.0])
            else:
                species_fraction   = reservoirs[res2] / n_total_res2 
        species_flux       = species_fraction * leak
        reservoirs[res1]  -= species_flux
        reservoirs[res2]  += species_flux
        return reservoirs
    
    def track_particles(self):
        upper_balance               = [0.0   , 0.1   ,0.2     ,0.3   ,0.4    ,0.57 , 0.6   ,0.7  , 0.8   , 0.9  , 1.0 ]
        drsep_balance               = [-0.07 , -0.02 , -0.015 ,-0.01 ,-0.005 ,0.002 , 0.005 ,0.01 , 0.015 , 0.02 , 0.07]
        updown_balance              = interp1d(drsep_balance, upper_balance, bounds_error=False, fill_value=0.0)
        reservoirs                  = {'HFS':np.zeros(self.nspecies),
                                       'LFS':np.zeros(self.nspecies),
                                       'UDV':np.zeros(self.nspecies),
                                       'USD':np.zeros(self.nspecies),
                                       'LDV':np.zeros(self.nspecies),
                                       'LSD':np.zeros(self.nspecies),
                                       'UPR':np.zeros(self.nspecies),
                                       'LPR':np.zeros(self.nspecies),
                                       'WALL':np.zeros(self.nspecies),
                                       'PLASMA':np.zeros(self.nspecies)}                                       
        injected                    = 0.0
        pumped                      = 0.0
        dt                          = self.dt
        for i, t in enumerate(self.t_array):
            # First pump out particles in subdivertors through cryopump
            if self.cryo:
                n_total_usd  = np.sum(reservoirs['USD'])
                n_total_lsd  = np.sum(reservoirs['LSD'])
                pump_upper = (self.upper_S_subdiv/V_sdiv) * n_total_usd * dt  # [particles removed]    
                pump_lower = (self.lower_S_subdiv/V_sdiv) * n_total_lsd * dt  # [particles removed]    
                if n_total_usd == 0.0:
                    species_fraction = np.array([0.0,0.0])
                else:
                    species_fraction   = reservoirs['USD'] / n_total_usd  
                species_flux     = species_fraction * pump_upper
                reservoirs['USD'] -= species_flux
                if n_total_lsd == 0.0:
                    species_fraction = np.array([0.0,0.0])
                else:
                    species_fraction   = reservoirs['LSD'] / n_total_lsd 
                species_flux     = species_fraction * pump_lower
                reservoirs['LSD'] -= species_flux

            # Diffuse particles from upper divertor into upper subdivertor, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'UDV','USD',V_div,V_sdiv,self.k_leak_div_sub,self.k_leak_div_sub)

            # Diffuse particles from lower divertor into lower subdivertor, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LDV','LSD',V_div,V_sdiv,self.k_leak_div_sub,self.k_leak_div_sub)

            # Diffuse particles from LFS into the upper divertor. Different conductance for forward and reverse
            if self.plasma_conf(t) == 0:
                k_leak_div_lfs = self.k_leak_div_lfs
            else:
                k_leak_div_lfs = self.k_leak_div_lfs_plasma
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LFS','UDV',V_LFS,V_div,self.k_leak_lfs_div,k_leak_div_lfs)

            # Diffuse particles from LFS into the lower divertor. Different conductance for forward and reverse
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LFS','LDV',V_LFS,V_div,self.k_leak_lfs_div,k_leak_div_lfs)
            # If plasma does not exist, diffusive transport between HFS and LFS, and HFS and divertor
            if self.plasma_conf(t) == 0:
                # Diffuse particles from HFS into the upper pfr
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'HFS','UPR',V_HFS,V_PFR,self.k_leak_hfs_pfr,self.k_leak_hfs_pfr)
                # Diffuse particles from HFS into the lower pfr
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'HFS','LPR',V_HFS,V_PFR,self.k_leak_hfs_pfr,self.k_leak_hfs_pfr)
                # Diffuse particles from upper pfr into the upper divertor
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'UPR','UDV',V_PFR,V_div,self.k_leak_pfr_div,self.k_leak_pfr_div)
                # Diffuse particles from lower pfr into the lower divertor                
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'LPR','LDV',V_PFR,V_div,self.k_leak_pfr_div,self.k_leak_pfr_div)
                # Diffuse particles from HFS into the LFS. Assume same conductance in both directions 
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'HFS','LFS',V_HFS,V_LFS,self.k_leak_hfs_lfs,self.k_leak_hfs_lfs)
            else:
                # Recycle from plasma into divertors
                if self.diverted(t) ==0:
                    reservoirs['WALL']+= self.limiter_wall_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LFS'] += self.limiter_lfs_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['HFS'] += self.limiter_hfs_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['UDV'] += self.limiter_div_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LDV'] += self.limiter_div_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['UPR'] += self.limiter_pfr_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LPR'] += self.limiter_pfr_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                else:                                       
                    reservoirs['UDV'] += self.plasma_div_frac * updown_balance(self.drsep(t)) * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LDV'] += self.plasma_div_frac * (1.0 - updown_balance(self.drsep(t))) * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LFS'] += self.plasma_lfs_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['HFS'] += self.plasma_hfs_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['UPR'] += self.plasma_pfr_frac * updown_balance(self.drsep(t)) * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LPR'] += self.plasma_pfr_frac * (1.0 - updown_balance(self.drsep(t))) * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['WALL']+= self.plasma_wall_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)                                          
                reservoirs['PLASMA']  -= reservoirs['PLASMA']* dt / self.plasma_conf(t)
                # Fuel the plasma from HFS
                reservoirs['PLASMA'] += self.pfr_fuelling * reservoirs['UPR'] + self.pfr_fuelling * reservoirs['LPR'] + self.hfs_fuelling * reservoirs['HFS'] + self.lfs_fuelling * reservoirs['LFS']
                reservoirs['HFS']    -= self.hfs_fuelling * reservoirs['HFS']
                reservoirs['LFS']    -= self.lfs_fuelling * reservoirs['LFS']
                reservoirs['LPR']    -= self.pfr_fuelling * reservoirs['LPR']
                reservoirs['UPR']    -= self.pfr_fuelling * reservoirs['UPR']
            # Finally, set recycling to mimic turbo pumps
            if self.recycling >0:
                reservoirs['LSD'] -= reservoirs['LSD'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['USD'] -= reservoirs['USD'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LDV'] -= reservoirs['LDV'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['UDV'] -= reservoirs['UDV'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LPR'] -= reservoirs['LPR'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['UPR'] -= reservoirs['UPR'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LFS'] -= reservoirs['LFS'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['HFS'] -= reservoirs['HFS'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))

            # Inject particles into each reservoir
            self.hfs_vessel_influx[i,:]  = self.injected['HFS'][:,i]
            self.lfs_vessel_influx[i,:]  = self.injected['LFS'][:,i]
            self.udv_vessel_influx[i,:]  = self.injected['UDV'][:,i]
            self.udvs_vessel_influx[i,:] = self.injected['UDVS'][:,i]
            self.ldv_vessel_influx[i,:]  = self.injected['LDV'][:,i]
            self.ldvs_vessel_influx[i,:] = self.injected['LDVS'][:,i]
            self.upfr_vessel_influx[i,:] = self.injected['UPFR'][:,i]
            self.lpfr_vessel_influx[i,:] = self.injected['LPFR'][:,i]
            reservoirs['HFS'] += self.hfs_vessel_influx[i,:] * dt 
            reservoirs['LFS'] += self.lfs_vessel_influx[i,:] * dt          
            reservoirs['LPR'] += self.lpfr_vessel_influx[i,:] * dt                
            reservoirs['UPR'] += self.upfr_vessel_influx[i,:] * dt                
            reservoirs['UDV'] += self.udv_vessel_influx[i,:] * dt                
            reservoirs['UDV'] += self.udvs_vessel_influx[i,:]* dt                  
            reservoirs['LDV'] += self.ldv_vessel_influx[i,:] * dt                 
            reservoirs['LDV'] += self.ldvs_vessel_influx[i,:]* dt             

            # Compute D2 reservoirs
            self.div[i,:]      = reservoirs['LDV'] 
            self.lfs_main[i,:] = reservoirs['LFS']
            self.hfs_main[i,:] = reservoirs['HFS']  
            self.plasma[i,:]   = reservoirs['PLASMA']  
            self.subdiv[i,:]   = reservoirs['LSD']  
            # Compute D2 Pressures
            self.main_pressure[i,:]      = (A/V_LFS)  * reservoirs['LFS']  
            self.lowdiv_pressure[i,:]    = (A/V_div)  * reservoirs['LDV'] 
            self.sublowdiv_pressure[i,:] = (A/V_sdiv) * reservoirs['LSD'] 
            self.uppdiv_pressure[i,:]    = (A/V_div)  * reservoirs['UDV'] 
            self.subuppdiv_pressure[i,:] = (A/V_sdiv) * reservoirs['USD']  
            # Compute average electron density
            self.electron_density[i,:] = self.plasma[i,:] * 2 / self.V_plasma(t)

    def display(self,time=None,p0_sublowdiv=None,p0_subuppdiv=None,calfac=1.0,p0_main=None,shot=None,tdens=None, dens=None):
        t_start = np.min(self.t_array)
        t_end   = np.max(self.t_array)
        if shot is not None:
            shotstr = f": {shot}"
        else:
            shotstr = ""
        # Plotting

        if self.plasmaplot:
            
            plt.figure(figsize=(10, 6))
            plt.gcf().canvas.manager.set_window_title(f"Gas Reservoir and Divertor Leakage Model"+shotstr)
            plt.subplot(2, 3, 1)
            if (self.hfs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.hfs_vessel_influx[:, 0] / 1e21, label='D2 HFS')
            if (self.lfs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.lfs_vessel_influx[:,0] / 1e21, label='D2 LFS')
            if (self.udv_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.udv_vessel_influx[:,0] / 1e21, label='D2 UDV')
            if (self.ldv_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.ldv_vessel_influx[:,0] / 1e21, label='D2 LDV')
            if (self.udvs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.udvs_vessel_influx[:,0] / 1e21, label='D2 UDVS')
            if (self.ldvs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.ldvs_vessel_influx[:,0] / 1e21, label='D2 LDVS')
            if (self.lpfr_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.lpfr_vessel_influx[:,1] / 1e21, label='D2 LPFR')
            if (self.upfr_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.upfr_vessel_influx[:,1] / 1e21, label='D2 UPFR')
            if (self.hfs_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.hfs_vessel_influx[:, 1] / 1e21, label='N2 HFS')
            if (self.lfs_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.lfs_vessel_influx[:,1] / 1e21, label='N2 LFS')
            if (self.udv_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.udv_vessel_influx[:,1] / 1e21, label='N2 UDV')
            if (self.ldv_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.ldv_vessel_influx[:,1] / 1e21, label='N2 LDV')
            if (self.lpfr_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.lpfr_vessel_influx[:,1] / 1e21, label='N2 LPFR')
            if (self.upfr_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.upfr_vessel_influx[:,1] / 1e21, label='N2 UPFR')
            plt.xlim([t_start,t_end])
            plt.ylabel("1e21 #/s")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 3, 2)
            plt.plot(self.t_array,self.lfs_main,label='LFS')
            plt.plot(self.t_array,self.hfs_main,label='HFS')
            plt.plot(self.t_array,self.div,label='Divertor')
            plt.plot(self.t_array,self.subdiv,label='Subdivertor')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Particles per reservoir")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            plt.subplot(2, 3, 3)
            plt.plot(self.t_array, self.electron_density[:,0] / 1e19, label='Average electron density')
            if tdens is not None and dens is not None:
                plt.plot(tdens, dens / 1e19, label='Average interferometer')
                
            plt.xlim([t_start,t_end])
            plt.ylabel(r"1e19 m$^{-3}$")
            plt.legend()
            plt.ylim([0,10])
            plt.grid(True)

            plt.subplot(2, 3, 4)
            if time is not None and p0_sublowdiv is not None:
                plt.plot(time,p0_sublowdiv,label='Meas. HL11 FIG')
            plt.plot(self.t_array, self.sublowdiv_pressure[:,0], label='Predicted FIG')
            plt.plot(self.t_array, self.lowdiv_pressure[:,0], label='Predicted divertor',linestyle='--')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.ylim([0,3])
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 3, 5)
            if time is not None and p0_subuppdiv is not None:
                plt.plot(time,p0_subuppdiv,label='Meas. HU08 FIG')
            plt.plot(self.t_array, self.subuppdiv_pressure[:,0], label='Predicted FIG')
            plt.plot(self.t_array, self.uppdiv_pressure[:,0], label='Predicted divertor',linestyle='--')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.ylim([0,3])
            plt.legend()
            plt.grid(True)
            plt.subplot(2, 3, 6)
            if time is not None and p0_main is not None:
                plt.plot(time,p0_main*1e3,label='Meas. HM12 FIG')
            plt.plot(self.t_array, self.main_pressure[:,0]*1e3, label='Predicted FIG')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [mPa]")
            plt.ylim([0,15])
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("mastu_plasma.png", dpi=300,transparent=True)
            plt.show()
        else:
            plt.figure(figsize=(8, 6))
            plt.gcf().canvas.manager.set_window_title(f"Gas Reservoir and Divertor Leakage Model"+shotstr)
            plt.subplot(2, 2, 1)
            if (self.hfs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.hfs_vessel_influx[:, 0] / 1e21, label='D2 HFS')
            if (self.lfs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.lfs_vessel_influx[:,0] / 1e21, label='D2 LFS')
            if (self.udv_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.udv_vessel_influx[:,0] / 1e21, label='D2 UDV')
            if (self.ldv_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.ldv_vessel_influx[:,0] / 1e21, label='D2 LDV')
            if (self.udvs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.udvs_vessel_influx[:,0] / 1e21, label='D2 UDVS')
            if (self.ldvs_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.ldvs_vessel_influx[:,0] / 1e21, label='D2 LDVS')
            if (self.lpfr_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.lpfr_vessel_influx[:,1] / 1e21, label='D2 LPFR')
            if (self.upfr_vessel_influx[:, 0] > 0).any():
                plt.plot(self.t_array, self.upfr_vessel_influx[:,1] / 1e21, label='D2 UPFR')
            if (self.hfs_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.hfs_vessel_influx[:, 1] / 1e21, label='N2 HFS')
            if (self.lfs_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.lfs_vessel_influx[:,1] / 1e21, label='N2 LFS')
            if (self.udv_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.udv_vessel_influx[:,1] / 1e21, label='N2 UDV')
            if (self.ldv_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.ldv_vessel_influx[:,1] / 1e21, label='N2 LDV')
            if (self.lpfr_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.lpfr_vessel_influx[:,1] / 1e21, label='N2 LPFR')
            if (self.upfr_vessel_influx[:, 1] > 0).any():
                plt.plot(self.t_array, self.upfr_vessel_influx[:,1] / 1e21, label='N2 UPFR')
            plt.xlim([t_start,t_end])
            plt.ylabel("1e21 #/s")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 2)
            if time is not None and p0_subuppdiv is not None:
                plt.plot(time,p0_subuppdiv,label='Meas. HU08 FIG')
            plt.plot(self.t_array, self.subuppdiv_pressure[:,0], label='Predicted HU08 FIG')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            if time is not None and p0_sublowdiv is not None:
                plt.plot(time,p0_sublowdiv,label='Meas. HL11 FIG')
            plt.plot(self.t_array, self.sublowdiv_pressure[:,0], label='Predicted HL11 FIG')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            if time is not None and p0_main is not None:
                plt.plot(time,p0_main,label='Meas. HM12 FIG')
            plt.plot(self.t_array, self.main_pressure[:,0], label='Predicted FIG')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig("mastu_gas.png", dpi=300,transparent=True)
            plt.show()

        particle_balance = False
        if particle_balance:
            plt.figure(figsize=(6, 5))
            plt.gcf().canvas.manager.set_window_title(f"Particle balance"+shotstr)
            plt.plot(self.t_array,self.injected_particles,label='Injected particles')
            plt.plot(self.t_array,self.total_particles,label='Total particles')
            plt.xlabel("Time [s]")
            plt.legend()
            plt.grid(True)           
            plt.show()

if __name__ == "__main__":
    plasma = True
    reservoir_plot=False    
    valve  = 'lfsv_bot_l03'
    calfac = 1.0
    if plasma:
    # Plasma pulse
        shot = 51787
        status = client.get('/epm/equilibriumStatusInteger', shot).data
        drsep = client.get('/epm/output/separatrixGeometry/drsepOut', shot).data[status==1]
        drseptime = client.get('/epm/time', shot).data[status==1]
        data  = client.get('/ane/density',shot)
        tdens = np.array(data.time.data)
        dens  = np.array(data.data)/5
        turbo = True
        cryo = False
        gasvolts  = False
        valve = 'all'
        plasma_conftime  = [-0.1,0.015 ,0.02 ,0.06  ,0.11  ,0.12  ,0.2   ,0.4  ,0.5  ,0.6  ,1.01 ,2.0] 
        plasma_conf      = [0   ,0.0   ,0.002,0.002 ,0.002 ,0.0018,0.0015,0.002 ,0.0018 ,0.0019 ,0.0008,0.0]

    else:
        if valve == 'hfs_mid_u02':
            shot = 50012
        if valve == 'hfs_mid_u08':
            shot = 49938
        if valve == 'hfs_mid_l08':
            shot = 49504
        if valve == 'hfs_mid_l02':
            shot = 49494
        if valve == 'lfsv_bot_l03':
            shot = 51642
        if valve == 'lfsv_bot_l09':
            shot = 49628
        if valve == 'lfsv_top_u011':
            shot = 49619
        if valve == 'lfsd_top_u0506':
            shot = 49786
        if valve == 'lfsd_top_u0102':
            shot = 49778
        if valve == 'lfsd_bot_l0506':
            shot = 49802
        if valve == 'lfss_bot_l0405':
            shot = 49750
        if valve == 'lfss_top_u0405':
            shot = 49748
        if valve == 'pfr_top_t01':
            shot = 49575
        if valve == 'pfr_top_t05':
            shot = 49583
        if valve == 'pfr_bot_b01':
            shot = 49590
        if valve == 'pfr_bot_b05':
            shot = 49594
        drsep     = None
        drseptime = None
        gasvolts  = True
        tdens     = None
        dens      = None
        turbo     = False
        cryo      = False
        plasma_conf=None
        plasma_conftime=None
    FIG_p0_lowdiv = client.get('/aga/HL11',shot)
    FIG_p0_uppdiv = client.get('/aga/HU08',shot)
    FIG_p0_main   = client.get('/aga/HM12',shot)
    p0_sublowdiv  = np.array(FIG_p0_lowdiv.data)
    p0_subuppdiv  = np.array(FIG_p0_uppdiv.data)
    p0_main       = np.array(FIG_p0_main.data)
    time_press    = np.array(FIG_p0_lowdiv.time.data)
    if reservoir_plot:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.95, bottom=0.15,right=0.96)
        fs = 12

        axes.fill([0.27,0.49,0.7,0.7,0.49,0.27,0.27],[-1.3,-1.5,-1.5,1.5,1.5,1.3,-1.3],label='HFS',alpha=0.6,edgecolor='black')
        axes.fill([0.49,0.7, 0.87,0.49],[-1.5,-1.5,-1.87,-1.5],label='LPFR',alpha=0.6,edgecolor='black')
        axes.fill([0.49,0.7, 0.87,0.49],[1.5,1.5,1.87,1.5],label='UPFR',alpha=0.6,edgecolor='black')
        axes.fill([0.7,0.87,2.0,2.0,0.87,0.7,0.7],[-1.5,-1.5,-0.25,0.25,1.5,1.5,-1.5],label='LFS',alpha=0.6,edgecolor='black')
        axes.fill([0.7,0.87,0.87,1.75,1.75,1.35,1.1,0.87,0.7],[-1.5,-1.5,-1.58,-1.58,-1.7,-2.1,-2.1,-1.87,-1.5],label='Lower divertor',alpha=0.6,edgecolor='black')
        axes.fill([0.7,0.87,0.87,1.75,1.75,1.35,1.1,0.87,0.7],[1.5,1.5,1.58,1.58,1.7,2.1,2.1,1.87,1.5],label='Upper divertor',alpha=0.6,edgecolor='black')
        axes.fill([1.75,2.0,2.0,1.75,1.75],[1.58,1.58,2.1,2.1,1.58],alpha=0.6,label='Upper sub-divertor',edgecolor='black')
        axes.fill([1.75,2.0,2.0,1.75,1.75],[-1.58,-1.58,-2.1,-2.1,-1.58],alpha=0.6,label='Lower sub-divertor',edgecolor='black')

        axes.fill([1.98,2.2,2.2,1.98,1.98],[-0.02,-0.02,0.02,0.02,-0.02],alpha=0.6,edgecolor='black',color='black')
        axes.fill([1.98,2.2,2.2,1.98,1.98],[-1.9,-1.9,-1.86,-1.86,-1.9],alpha=0.6,edgecolor='black',color='black')
        axes.fill([1.98,2.2,2.2,1.98,1.98],[1.9,1.9,1.86,1.86,1.9],alpha=0.6,edgecolor='black',color='black')
        axes.legend(fontsize=fs,loc='upper right')
        axes.set_xlabel('R / m',fontsize=fs)
        axes.set_ylabel('Z / m',fontsize=fs)
        axes.set_xlim([0,3.0])
        axes.set_ylim([-2.2,2.2])
        axes.set_aspect(aspect=1.0)

        axes.text(2.02,0.05,'Main \nFIG',fontsize=fs)
        axes.text(2.02,-1.82,'Sub-divertor \nFIG',fontsize=fs)
        
        plt.tight_layout()
        plt.savefig("mast_u_regions_highres.png", dpi=300,transparent=True)
        plt.show()
        exit()        
    run  = pressure(shot,cryo=cryo,turbo=turbo,plasma=plasma,drsep=drsep,
                    plasma_conf=plasma_conf,plasma_conftime=plasma_conftime,
                    drseptime=drseptime,valve=valve,gasvolts=gasvolts,closure_time=0.8)
    run.display(time=time_press,p0_sublowdiv=p0_sublowdiv,p0_subuppdiv=p0_subuppdiv,
                p0_main=p0_main,tdens=tdens,dens=dens,shot=shot,calfac=calfac)




