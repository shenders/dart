import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import convolve
from pipeinjector import piezo
from scipy.optimize import minimize
import pyuda
client = pyuda.Client()

# Constants
kB       = 1.38e-23         # Boltzmann constant [J/K] 
T_pump   = 300.0            # Temperature at pump [K]
V_machine= 50.5             # Effective volume ~50.0 m^3
V_PFR    = 0.26
V_HFS    = 0.56
V_LFS    = 9.88
V_COR    = 8.8
V_SOL    = 3.35
V_SLF    = 4.51
V_div    = 2.66
V_sdiv   = 3.03
V_ssdv   = 2.49
A        = kB * T_pump      # Pre-factor in pressure equation

def fit_confinement_time(tdens, dens, shot, initial_knots, initial_tau_guess, **kwargs):
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    popup = tk.Toplevel(root)
    popup.title("Fitting Progress")
    popup.geometry("300x50")
    label = tk.Label(popup, text="Starting fit...")
    label.pack(padx=20, pady=10)
    popup.update()
    finite_mask = ~np.isnan(dens) & ~np.isinf(dens)
    dens = dens[finite_mask]
    tdens = tdens[finite_mask]
    model = pressure(shot, plasma=True,dt=0.001, 
                     plasma_conf=initial_tau_guess, plasma_conftime=initial_knots,track=False, **kwargs)
    def loss(tau_values,model):
        model.plasma_conf = interp1d(initial_knots ,tau_values,bounds_error=False, fill_value=0.0)
        model.track_particles()
        model_density_interp = interp1d(model.t_array, model.electron_density[:,0],kind='linear',fill_value='extrapolate')
        model_density = model_density_interp(tdens)
        chi2n = np.sum((model_density/1e19 - dens/1e19)**2)/(len(dens)-len(tau_values))
        label.config(text=f"Normalised chi^2={chi2n:.2f}")
        popup.update()
        return np.mean((model_density/1e19 - dens/1e19)**2)
    bounds = [(1e-6, 1000.0) for _ in initial_tau_guess]
    res = minimize(loss, initial_tau_guess, args=(model,),method='Nelder-Mead', tol=1e-3,options={'disp': True, 'maxiter': 100,'adaptive':True})
    label = tk.Label(popup, text="Complete...")
    popup.destroy()
    root.destroy()
    return res

class pressure:
    def __init__(self,shot,plasma_conf=None,plasma_conftime=None,closure_time=0.4,subdiv_time=0.3,
                 plasma=True,cryo=False,nbipump=True,nbi_S=5.0,lower_S_subdiv=20.0,upper_S_subdiv=0.0,turbo=True,drsep=None,
                 volume=None,voltime=None,dt=0.0003,track=True,f_wall_hit=25.0,
                 recycling=0.98955,drseptime=None,gasvolts=False,valve='all',gas_matrix=None,
                 inputgas=False,gastraces=None):
        self.dt = dt
        self.loaded = False
        self.turbo = turbo
        self.nbipump = nbipump
        self.nbi_S = nbi_S
        self.species_list = ['D','N']
        self.nspecies   = len(self.species_list)
        self.loaded = False
        self.turbo  = turbo
        self.recycling = recycling
        self.f_wall_hit = f_wall_hit
        self.cryo   = cryo
        self.lower_S_subdiv = lower_S_subdiv
        self.upper_S_subdiv = upper_S_subdiv
        self.gastraces = gastraces
        self.plasmaplot  = plasma
        time_plasma      = [-0.1,0.015 ,0.02 ,0.06  ,0.1,0.11  ,0.12  ,0.2   ,0.4  ,0.5  ,0.6  ,1.01 ,1.1] 
        if plasma_conf is not None:            
            self.plasma_conf  = interp1d(plasma_conftime ,plasma_conf,bounds_error=False, fill_value=0.0)
        else:
            if plasma:
                plasma_conf  = [0   ,0.0   ,0.002,0.002 ,0.002,0.002 ,0.0025,0.0023,0.003 ,0.00165 ,0.0013 ,0.0008,0.0]
            else:
                plasma_conf  = [0   ,0.0   ,0.0  ,0.0   ,0.0,0.0   ,0.0   ,0.0   ,0.0  ,0.0  ,0.0  ,0.0  ,0.0]                            
            self.plasma_conf  = interp1d(time_plasma ,plasma_conf,bounds_error=False, fill_value=0.0)
        if drsep is not None:
            self.drsep = interp1d(drseptime,drsep,bounds_error=False, fill_value=0.0)
        else:
            self.drsep = interp1d([-0.1,0.1,10.0],[0.002,0.002,0.002],bounds_error=False, fill_value=0.0)
        if volume is not None:
            self.V_plasma = interp1d(voltime,volume,bounds_error=False, fill_value=0.0)
        else:
            self.V_plasma = interp1d([-0.1,0.1,10.0],[9.0,9.0,9.0],bounds_error=False, fill_value=0.0)

        self.setup_arrays()
        self.setup_times(closure_time=closure_time,subdiv_time=subdiv_time)
        self.setup_pumping()
        self.setup_influx(shot,gasvolts=gasvolts,valve=valve,gas_matrix=gas_matrix,inputgas=inputgas)
        if track:
            self.track_particles()

    def setup_times(self,closure_time=0.4,subdiv_time=0.5):
        # Setup conductances between reservoirs
        # Diffusion from HFS
        self.k_leak_hfs_cor    = 1.0/0.02
        self.k_leak_hfs_pfr    = 1.0/0.02

        # Diffusion from core
        self.k_leak_core_sol   = 1.0/0.02
        self.k_leak_core_pfr   = 1.0/0.02

        # Diffusion from SOL
        self.k_leak_sol_lfs    = 1.0/0.04
        self.k_leak_sol_div    = 1.0/0.04

        # Diffusion from LFS
        self.k_leak_lfs_slf    = 1.0/0.04

        # Diffusion from PFR
        self.k_leak_pfr_div    = 1.0/0.02

        # Diffusion from DIV
        self.k_leak_div_sol_plasma = 1.0/closure_time
        self.k_leak_div_sub    = 1.0/subdiv_time
        self.k_leak_sub_ssub   = 1.0/0.04

        # Puffing timescale
        self.k_leak_pipe_main  = 1/0.003

        # Setup ballistic streaming boost factor
        self.ballistic_boost   = 6.0

        # Setup fuelling efficiencies
        self.hfs_fuelling      = 0.8
        self.sol_fuelling      = 0.6
        self.pfr_fuelling      = 0.1
        
        # Setup plasma recycling fractions for limited plasma
        self.limiter_div_frac  = 0.0
        self.limiter_lfs_frac  = 0.1
        self.limiter_hfs_frac  = 0.8
        self.limiter_pfr_frac  = 0.0
        self.limiter_wall_frac = 0.1

        # Set fraction of SOL particle flux lost to baffle
        self.wall              = 0.1

        # Set the number of prompt recycling events vs. longer outgassing
        self.div_Rprompt       = 0.7
        self.div_Rslow         = 0.3
        self.div_Rlost         = 1.0 - self.div_Rprompt - self.div_Rslow
        self.main_Rprompt      = 0.3
        self.main_Rslow        = 0.6
        self.main_Rlost        = 1.0 - self.main_Rprompt - self.main_Rslow

        # Setup plasma timescale for outgassing
        self.tau_div_outgas    = 0.2   
        self.tau_main_outgas   = 0.5
    def setup_pumping(self):
        if not self.turbo:
            self.recycling = 0.0
        if not self.cryo:
            self.lower_S_subdiv = 0.0
            self.upper_S_subdiv = 0.0
        if not self.nbipump:
            self.nbi_S = 0.0
    def calc_Ndot(self,trace,shot,pipe_length,plenum_pressure_bar, calc_piezo,valve,mul=None):
        data = client.get(trace, shot)
        time = np.array(data.time.data)
        if mul is None:
            if valve == '':
                mul = 1e21
            else:
                mul = 1.0
        Ndot = calc_piezo.simulate_gas_flow_with_pipe_delay(np.array(data.data)*mul,plenum_pressure_bar,
                                                            pipe_length,6.0,1e-7,time[1]-time[0],valve)
        return time,Ndot
    def setup_influx(self,shot,gasvolts=False,valve=None,gas_matrix=None,userinput=None,inputgas=False):
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
            self.gastraces={
                'flow':{
                    'HFS':Ndot_HFS,
                    'LFS':Ndot_LFS,
                    'UDV':Ndot_UDV,
                    'LDV':Ndot_LDV,
                    'UDVS':Ndot_UDVS,
                    'LDVS':Ndot_LDVS,
                    'UPFR':Ndot_UPFR,
                    'LPFR':Ndot_LPFR,
                    },
                'time':{
                    'HFS':time_HFS,
                    'LFS':time_LFS,
                    'UDV':time_UDV,
                    'LDV':time_LDV,
                    'UDVS':time_UDVS,
                    'LDVS':time_LDVS,
                    'UPFR':time_UPFR,
                    'LPFR':time_LPFR,
                    }
                }
        else:
            if not inputgas or self.gastraces is None:          
                time_HFS,Ndot_HFS = self.calc_Ndot('/xdc/flow/s/hfs_mid_flow', shot, 0.3, 1.5, calc_piezo,'')        
                time_LFS,Ndot_lfsv_bot = self.calc_Ndot('/xdc/flow/s/lfsv_bot_flow', shot, 0.3, 1.5, calc_piezo,'')        
                time_LFS,Ndot_lfsv_top = self.calc_Ndot('/xdc/flow/s/lfsv_top_flow', shot, 0.3, 1.5, calc_piezo,'')        
                Ndot_LFS = (Ndot_lfsv_bot + Ndot_lfsv_top)

                time_LDV,Ndot_LDV = self.calc_Ndot('/xdc/flow/s/lfsd_bot_flow', shot, 0.6, 1.5, calc_piezo,'')
                time_UDV,Ndot_UDV = self.calc_Ndot('/xdc/flow/s/lfsd_top_flow', shot, 0.6, 1.5, calc_piezo,'')

                time_LDVS,Ndot_LDVS = self.calc_Ndot('/xdc/flow/s/lfss_bot_flow', shot, 0.6, 1.5, calc_piezo,'')
                time_UDVS,Ndot_UDVS = self.calc_Ndot('/xdc/flow/s/lfss_top_flow', shot, 0.6, 1.5, calc_piezo,'')

                time_UPFR,Ndot_pfrt = self.calc_Ndot('/xdc/flow/s/pfr_top_flow', shot, 0.6, 1.5, calc_piezo,'')
                Ndot_UPFR = (Ndot_pfrt )
               
                time_LPFR,Ndot_pfrb = self.calc_Ndot('/xdc/flow/s/pfr_bot_flow', shot, 0.6, 1.5, calc_piezo,'')
                Ndot_LPFR = (Ndot_pfrb)
                self.gastraces={
                    'flow':{
                        'HFS':Ndot_HFS,
                        'LFS':Ndot_LFS,
                        'UDV':Ndot_UDV,
                        'LDV':Ndot_LDV,
                        'UDVS':Ndot_UDVS,
                        'LDVS':Ndot_LDVS,
                        'UPFR':Ndot_UPFR,
                        'LPFR':Ndot_LPFR,
                        },
                    'time':{
                        'HFS':time_HFS,
                        'LFS':time_LFS,
                        'UDV':time_UDV,
                        'LDV':time_LDV,
                        'UDVS':time_UDVS,
                        'LDVS':time_LDVS,
                        'UPFR':time_UPFR,
                        'LPFR':time_LPFR,
                        }
                    }
                        

           
        hfs_Gamma_interp        = interp1d(self.gastraces['time']['HFS'], self.gastraces['flow']['HFS'], bounds_error=False, fill_value=0.0)
        lfs_Gamma_interp        = interp1d(self.gastraces['time']['LFS'], self.gastraces['flow']['LFS'], bounds_error=False, fill_value=0.0)
        udv_Gamma_interp        = interp1d(self.gastraces['time']['UDV'], self.gastraces['flow']['UDV'], bounds_error=False, fill_value=0.0)
        ldv_Gamma_interp        = interp1d(self.gastraces['time']['LDV'], self.gastraces['flow']['LDV'], bounds_error=False, fill_value=0.0)
        udvs_Gamma_interp       = interp1d(self.gastraces['time']['UDVS'], self.gastraces['flow']['UDVS'], bounds_error=False, fill_value=0.0)
        ldvs_Gamma_interp       = interp1d(self.gastraces['time']['LDVS'], self.gastraces['flow']['LDVS'], bounds_error=False, fill_value=0.0)
        upfr_Gamma_interp       = interp1d(self.gastraces['time']['UPFR'], self.gastraces['flow']['UPFR'], bounds_error=False, fill_value=0.0)
        lpfr_Gamma_interp       = interp1d(self.gastraces['time']['LPFR'], self.gastraces['flow']['LPFR'], bounds_error=False, fill_value=0.0)
        if gas_matrix is None:
            gas_matrix              = {'HFS':0,
                                       'LFS':0,
                                       'UDV':0,
                                       'UDVS':0,
                                       'LDV':0,
                                       'LDVS':0,
                                       'LPFR':0,
                                       'UPFR':0}
        d2_cal_fac = 0.35    
        self.injected['HFS'][gas_matrix['HFS'],:]= hfs_Gamma_interp(self.t_array)/d2_cal_fac                       
        self.injected['LFS'][gas_matrix['LFS'],:]= lfs_Gamma_interp(self.t_array)/d2_cal_fac               
        self.injected['UDV'][gas_matrix['UDV'],:]= udv_Gamma_interp(self.t_array)/d2_cal_fac              
        self.injected['UDVS'][gas_matrix['UDVS'],:]= udvs_Gamma_interp(self.t_array)/d2_cal_fac              
        self.injected['LDV'][gas_matrix['LDV'],:]= ldv_Gamma_interp(self.t_array)/d2_cal_fac                
        self.injected['LDVS'][gas_matrix['LDVS'],:]= ldvs_Gamma_interp(self.t_array)/d2_cal_fac                
        self.injected['UPFR'][gas_matrix['UPFR'],:]= upfr_Gamma_interp(self.t_array)/d2_cal_fac              
        self.injected['LPFR'][gas_matrix['LPFR'],:]= lpfr_Gamma_interp(self.t_array)/d2_cal_fac                

    def setup_arrays(self):
        # Time settings
        t_start                 = -0.08
        t_end                   = 1.2
        # Initialise arrays
        self.t_array            = np.arange(t_start, t_end, self.dt)
        self.hfs_main           = np.zeros((len(self.t_array),self.nspecies))
        self.lfs_main           = np.zeros((len(self.t_array),self.nspecies))
        self.sol_main           = np.zeros((len(self.t_array),self.nspecies))
        self.llfs_main          = np.zeros((len(self.t_array),self.nspecies))
        self.ulfs_main          = np.zeros((len(self.t_array),self.nspecies))
        self.plasma             = np.zeros((len(self.t_array),self.nspecies))
        self.ldiv               = np.zeros((len(self.t_array),self.nspecies))
        self.udiv               = np.zeros((len(self.t_array),self.nspecies))
        self.lpfr               = np.zeros((len(self.t_array),self.nspecies))
        self.upfr               = np.zeros((len(self.t_array),self.nspecies))
        self.subldiv            = np.zeros((len(self.t_array),self.nspecies))
        self.subudiv            = np.zeros((len(self.t_array),self.nspecies))
        self.ssubldiv           = np.zeros((len(self.t_array),self.nspecies))
        self.ssubudiv           = np.zeros((len(self.t_array),self.nspecies))
        self.subuppdiv_pressure = np.zeros((len(self.t_array),self.nspecies))
        self.sublowdiv_pressure = np.zeros((len(self.t_array),self.nspecies))
        self.lowdiv_pressure    = np.zeros((len(self.t_array),self.nspecies))
        self.uppdiv_pressure    = np.zeros((len(self.t_array),self.nspecies))
        self.main_pressure      = np.zeros((len(self.t_array),self.nspecies))
        self.total_particles    = np.zeros((len(self.t_array),self.nspecies))
        self.pumped_particles   = np.zeros((len(self.t_array),self.nspecies))
        self.injected_particles = np.zeros((len(self.t_array),self.nspecies))
        self.electron_density   = np.zeros((len(self.t_array),self.nspecies))
        self.plasma_influx      = np.zeros((len(self.t_array),self.nspecies))
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
    def updown_balance(self,drsep,flfs=0.9,flot=0.8,fhot=0.5,lq=7e-3,offset=-2e-3):
        drsep = drsep-offset
        fdisc = 1 - np.exp(-(np.abs(drsep)) / lq)
        plfs = flfs 
        phfs = (1 - flfs) 
        plit = fdisc * plfs
        plet = (1 - fdisc) * plfs
        phit = fdisc * phfs
        phet = (1 - fdisc) * phfs
        ppou = 0.5 * plet + flot * plit + fhot * phit
        ppin = 0.5 * phet + (1 - flot) * plit + (1 - fhot) * phit
        if drsep <= 0:
            lo = ppou
            li = ppin
            uo = 0.5 * plet
            ui = 0.5 * phet
        else:
            lo = 0.5 * plet
            li = 0.5 * phet 
            uo = ppou
            ui = ppin
        return lo, li, uo, ui
    def track_particles(self):
        reservoirs                  = {'HFS':np.zeros(self.nspecies),
                                       'LFS':np.zeros(self.nspecies),
                                       'SOL':np.zeros(self.nspecies),
                                       'ULF':np.zeros(self.nspecies),
                                       'LLF':np.zeros(self.nspecies),
                                       'UDV':np.zeros(self.nspecies),
                                       'USD':np.zeros(self.nspecies),
                                       'USS':np.zeros(self.nspecies),
                                       'LDV':np.zeros(self.nspecies),
                                       'LSD':np.zeros(self.nspecies),
                                       'LSS':np.zeros(self.nspecies),
                                       'UPR':np.zeros(self.nspecies),
                                       'LPR':np.zeros(self.nspecies),
                                       'LDV_WALL':np.zeros(self.nspecies),
                                       'UDV_WALL':np.zeros(self.nspecies),
                                       'LFS_WALL':np.zeros(self.nspecies),
                                       'HFS_WALL':np.zeros(self.nspecies),
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
            # Next pump out particles through NBI ducts
            if self.nbipump:
                n_total_lfs  = np.sum(reservoirs['LFS'])
                pump_lfs     = (self.nbi_S/V_LFS) * n_total_lfs * dt  # [particles removed]    
                if n_total_lfs == 0.0:
                    species_fraction = np.array([0.0,0.0])
                else:
                    species_fraction   = reservoirs['LFS'] / n_total_lfs  
                species_flux     = species_fraction * pump_lfs
                reservoirs['LFS'] -= species_flux
            #======================================================================================================================
            # Diffuse particles from sub-divertors to divertors
            #======================================================================================================================

            # upper sub-divertor into upper sub-subdivertor, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'USD','USS',V_sdiv,V_ssdv,self.k_leak_sub_ssub,self.k_leak_sub_ssub)

            # lower sub-divertor into lower sub-subdivertor, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LSD','LSS',V_sdiv,V_ssdv,self.k_leak_sub_ssub,self.k_leak_sub_ssub)

            # upper divertor into upper subdivertor, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'UDV','USD',V_div,V_sdiv,self.k_leak_div_sub,self.k_leak_div_sub)

            # lower divertor into lower subdivertor, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LDV','LSD',V_div,V_sdiv,self.k_leak_div_sub,self.k_leak_div_sub)
            #======================================================================================================================

            #======================================================================================================================
            # Diffuse particles from divertor into main chamber
            #======================================================================================================================

            # upper divertor to SOL, or vice-versa. Assume same conductance each way but different if plasma exists due to plugging

            if self.plasma_conf(t) < 5e-5:
                k_leak_sol_div = self.k_leak_sol_div
            else:
                k_leak_sol_div = self.k_leak_div_sol_plasma
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LFS','UDV',V_LFS,V_div,k_leak_sol_div,k_leak_sol_div)

            # lower divertor to SOL, or vice-versa. Assume same conductance each way but different if plasma exists due to plugging
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LFS','LDV',V_LFS,V_div,k_leak_sol_div,k_leak_sol_div)


            # LFS into the upper LFS, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LFS','ULF',V_LFS,V_SLF,self.k_leak_lfs_slf,self.k_leak_lfs_slf)

            # LFS into the lower LFS, or vice-versa. Assume same conductance each way
            reservoirs = self.evolve_reservoirs(dt,reservoirs,'LFS','LLF',V_LFS,V_SLF,self.k_leak_lfs_slf,self.k_leak_lfs_slf)

            #======================================================================================================================
            # If plasma does not exist, diffusive transport between HFS, core, pfr, etc.
            #======================================================================================================================
            if self.plasma_conf(t) < 5e-5:
                # SOL to LFS, or vice-versa. Assume same conductance each way 
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'SOL','LFS',V_SOL,V_LFS,self.k_leak_sol_lfs,self.k_leak_sol_lfs)
                # HFS into the upper pfr
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'HFS','UPR',V_HFS,V_PFR,self.k_leak_hfs_pfr,self.k_leak_hfs_pfr)
                # HFS into the lower pfr
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'HFS','LPR',V_HFS,V_PFR,self.k_leak_hfs_pfr,self.k_leak_hfs_pfr)
                # HFS into the core region
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'HFS','PLASMA',V_HFS,V_COR,self.k_leak_hfs_cor,self.k_leak_hfs_cor)
                # core region into SOL
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'PLASMA','SOL',V_COR,V_SOL,self.k_leak_core_sol,self.k_leak_core_sol)
                # core region into upperPFR
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'PLASMA','UPR',V_COR,V_PFR,self.k_leak_core_pfr,self.k_leak_core_pfr)
                # core region into lower PFR
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'PLASMA','LPR',V_COR,V_PFR,self.k_leak_core_pfr,self.k_leak_core_pfr)
                # upper pfr into the upper divertor
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'UPR','UDV',V_PFR,V_div,self.k_leak_pfr_div,self.k_leak_pfr_div)
                # lower pfr into the lower divertor                
                reservoirs = self.evolve_reservoirs(dt,reservoirs,'LPR','LDV',V_PFR,V_div,self.k_leak_pfr_div,self.k_leak_pfr_div)
            else:
                # Fuel the plasma
                reservoirs['PLASMA'] += self.pfr_fuelling * reservoirs['UPR'] + self.pfr_fuelling * reservoirs['LPR'] + self.hfs_fuelling * reservoirs['HFS'] + self.sol_fuelling * reservoirs['SOL']                                                                    
                self.plasma_influx[i,:]  = (self.pfr_fuelling * reservoirs['UPR'] + self.pfr_fuelling * reservoirs['LPR'] + self.hfs_fuelling * reservoirs['HFS'] + self.sol_fuelling * reservoirs['SOL'])/dt
                
                # Fuel the SOL
                reservoirs['SOL'] += 0.5 * reservoirs['LFS']
                # Recycle from plasma into divertors
                if t < 0.1:
                    reservoirs['WALL']+= self.limiter_wall_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['SOL'] += self.limiter_lfs_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['HFS'] += self.limiter_hfs_frac * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['UDV'] += self.limiter_div_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LDV'] += self.limiter_div_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['UPR'] += self.limiter_pfr_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LPR'] += self.limiter_pfr_frac/2.0 * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                else:                                       
                    # Calculate directional flows of particle flux to each divertor
                    if t > 0.5:
                        offset = 2e-3
                    else:
                        offset = -2e-3
                    lo, li, uo, ui = self.updown_balance(self.drsep(t))            
                    # Set the fraction of particles entering the PFR from the inner divertor flux
                    fPFR      = 0.1
                    # Setup the relevant fractions for prompt and slow recycling into each reservoir
                    ldiv      = lo * (1.0-self.wall) * self.div_Rprompt
                    udiv      = uo * (1.0-self.wall) * self.div_Rprompt
                    div_lost  = (uo + lo) * (1.0-self.wall) * self.div_Rlost
                    main_lost = ((uo+lo)*self.wall+(li+ui)*(1-fPFR)) * self.main_Rlost

                    lpr       = fPFR * li * self.main_Rprompt
                    upr       = fPFR * ui * self.main_Rprompt
                    lfs       = self.wall * (uo + lo) * self.main_Rprompt
                    hfs       = (1-fPFR)  * (li + ui) * self.main_Rprompt                    
                    ldiv_slow = lo * (1.0-self.wall) * self.div_Rslow
                    udiv_slow = uo * (1.0-self.wall) * self.div_Rslow                    
                    lfs_slow  = self.wall * (uo + lo) * self.main_Rslow
                    hfs_slow  = (1-fPFR)  * (li + ui) * self.main_Rslow                    
                    # Fuel the reservoirs from immediate recycling of plasma
                    reservoirs['UDV'] += udiv * reservoirs['PLASMA'] * dt / self.plasma_conf(t) 
                    reservoirs['LDV'] += ldiv * reservoirs['PLASMA'] * dt / self.plasma_conf(t) 
                    reservoirs['LFS'] += lfs * reservoirs['PLASMA'] * dt / self.plasma_conf(t) 
                    reservoirs['HFS'] += hfs * reservoirs['PLASMA'] * dt / self.plasma_conf(t) 
                    reservoirs['UPR'] += upr * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LPR'] += lpr * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    # Setup the relevant fractions for particles implanted into the wall that will outgas
                    reservoirs['UDV_WALL'] += udiv_slow * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LDV_WALL'] += ldiv_slow * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['LFS_WALL'] += lfs_slow * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    reservoirs['HFS_WALL'] += hfs_slow * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                    # Store the amount of particles permanently trapped in the wall
                    reservoirs['WALL']+= (main_lost+div_lost) * reservoirs['PLASMA'] * dt / self.plasma_conf(t)
                reservoirs['PLASMA']   -= reservoirs['PLASMA']* dt / self.plasma_conf(t)
    
                # Fuel the plasma 
                reservoirs['HFS']    -= self.hfs_fuelling * reservoirs['HFS']
                reservoirs['SOL']    -= self.sol_fuelling * reservoirs['SOL']
                reservoirs['LPR']    -= self.pfr_fuelling * reservoirs['LPR']
                reservoirs['UPR']    -= self.pfr_fuelling * reservoirs['UPR']
                reservoirs['LFS']    -= 0.5 * reservoirs['LFS']
            # Handle outgassing regardless of whether plasma exists
            # Setup the relevant fractions for particles implanted into the wall that will outgas
            reservoirs['UDV_WALL'] -= reservoirs['UDV_WALL'] * dt/self.tau_div_outgas
            reservoirs['LDV_WALL'] -= reservoirs['LDV_WALL'] * dt/self.tau_div_outgas
            reservoirs['LFS_WALL'] -= reservoirs['LFS_WALL'] * dt/self.tau_main_outgas
            reservoirs['HFS_WALL'] -= reservoirs['HFS_WALL'] * dt/self.tau_main_outgas
            reservoirs['UDV'] += reservoirs['UDV_WALL'] * dt/self.tau_div_outgas
            reservoirs['LDV'] += reservoirs['LDV_WALL'] * dt/self.tau_div_outgas
            reservoirs['LFS'] += reservoirs['LFS_WALL'] * dt/self.tau_main_outgas
            reservoirs['HFS'] += reservoirs['HFS_WALL'] * dt/self.tau_main_outgas
    

            # Finally, set recycling to mimic turbo pumps
            if self.recycling >0:
                reservoirs['LSD'] -= reservoirs['LSD'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['USD'] -= reservoirs['USD'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LSS'] -= reservoirs['LSS'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['USS'] -= reservoirs['USS'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LDV'] -= reservoirs['LDV'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['UDV'] -= reservoirs['UDV'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LPR'] -= reservoirs['LPR'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['UPR'] -= reservoirs['UPR'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LLF'] -= reservoirs['LLF'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['ULF'] -= reservoirs['ULF'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['SOL'] -= reservoirs['SOL'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['LFS'] -= reservoirs['LFS'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                reservoirs['HFS'] -= reservoirs['HFS'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                if self.plasma_conf(t) < 5e-5:
                    reservoirs['PLASMA'] -= reservoirs['PLASMA'] * (1-np.exp(-self.f_wall_hit * (1.0-self.recycling) * dt))
                    
            # Inject particles into the relevant reservoirs
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
            # Store all reservoir pressures
            self.ldiv[i,:]      = (A/V_div)  *reservoirs['LDV'] 
            self.udiv[i,:]      = (A/V_div)  *reservoirs['UDV'] 
            self.lpfr[i,:]      = (A/V_PFR)  *reservoirs['LPR'] 
            self.upfr[i,:]      = (A/V_PFR)  *reservoirs['UPR'] 
            self.lfs_main[i,:]  = (A/V_LFS)  *reservoirs['LFS']
            self.sol_main[i,:]  = (A/V_LFS)  *reservoirs['SOL']
            self.ulfs_main[i,:] = (A/V_SLF)  *reservoirs['ULF']
            self.llfs_main[i,:] = (A/V_SLF)  *reservoirs['LLF']
            self.hfs_main[i,:]  = (A/V_HFS)  *reservoirs['HFS']  
            self.plasma[i,:]    = reservoirs['PLASMA']  
            self.subldiv[i,:]   = (A/V_sdiv)  *reservoirs['LSD']  
            self.subudiv[i,:]   = (A/V_sdiv)  *reservoirs['USD']  
            self.ssubldiv[i,:]  = (A/V_ssdv)  *reservoirs['LSS']  
            self.ssubudiv[i,:]  = (A/V_ssdv)  *reservoirs['USS']  

            # Store specific pressures for FIG comparisons
            self.main_pressure[i,:]      = (A/V_LFS)  * reservoirs['LFS']  
            self.lowdiv_pressure[i,:]    = (A/V_div)  * reservoirs['LDV'] 
            self.sublowdiv_pressure[i,:] = (A/V_sdiv) * reservoirs['LSD'] 
            self.uppdiv_pressure[i,:]    = (A/V_div)  * reservoirs['UDV'] 
            self.subuppdiv_pressure[i,:] = (A/V_sdiv) * reservoirs['USD']  
            # Compute average electron density
            self.electron_density[i,:] = self.plasma[i,:] * 2 / self.V_plasma(t)

    def display(self,time=None,p0_sublowdiv=None,p0_subuppdiv=None,calfac=1.0,tg1=0.0,p0_main=None,shot=None,tdens=None, dens=None):
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
            plt.ylim([0,100])
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
            from scipy.integrate import simps
            integral = simps(self.hfs_vessel_influx[:, 0] / 1e21, self.t_array)
            print(f"Injected particles: {integral}")
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
            plt.axhline(y=tg1,linestyle='--')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 3)
            if time is not None and p0_sublowdiv is not None:
                plt.plot(time,p0_sublowdiv,label='Meas. HL11 FIG')
            plt.plot(self.t_array, self.sublowdiv_pressure[:,0], label='Predicted HL11 FIG')
            plt.axhline(y=tg1,linestyle='--')
            plt.xlim([t_start,t_end])
            plt.xlabel("Time [s]")
            plt.ylabel("Pressure [Pa]")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 2, 4)
            if time is not None and p0_main is not None:
                plt.plot(time,p0_main,label='Meas. HM12 FIG')
            plt.plot(self.t_array, self.main_pressure[:,0], label='Predicted FIG')
            plt.axhline(y=tg1,linestyle='--')
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
    valve  = 'hfs_mid_u02'
    calfac = 1.0
    tg1 = 0.0
    f_wall_hit = 25.0
    if plasma:
    # Plasma pulse
        shot = 50894
        status = client.get('/epm/equilibriumStatusInteger', shot).data
        drsep = client.get('/epm/output/separatrixGeometry/drsepOut', shot).data[status==1]
        drseptime = client.get('/epm/time', shot).data[status==1]
        #data  = client.get('/ane/density',shot)
        data  = client.get('/esm/density/nebar',shot)
        tdens = np.array(data.time.data)
        dens  = np.array(data.data)
        turbo = True
        cryo = True
        gasvolts  = False
        valve = 'all'
        initial_knots     = [0.015 , 0.200, 0.3200, 0.7, 0.9,1.1,1.2,1.3]
        initial_tau_guess = [0.0002, 0.004, 0.0055, 0.009,0.0092,0.002,0.005,0.0]
        volume              = 50.0
        turbo_pumpspeed     = 7.8+5.0 # 7.8 turbopumps and 2.5 per beam
        recycling           = 0.98955
        f_wall_hit          = turbo_pumpspeed/(volume * (1.0-recycling))
        plasma_fracs        = [0.1,0.12,1.0-0.01-0.1-0.12-0.06,0.01,0.06]
        closure_time        = 5.0
        div2sub             = 0.6
        #fit_result = fit_confinement_time(tdens, dens, shot,
        #                               initial_knots, initial_tau_guess,
        #                               drsep=drsep, drseptime=drseptime,
        #                               turbo=turbo, cryo=cryo, valve=valve,
        #                               gasvolts=gasvolts)

    # Use the best-fit result to re-run the model and display
        plasma_conf = initial_tau_guess# fit_result.x
        plasma_conftime = initial_knots
        #print("best fit calculated:",fit_result.x)
    else:
        if valve == 'hfs_mid_u02':
            shot = 50012
            tg1  = 2.42e-4
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
        d2_gas_correction_factor = 0.35
        mbar2pa   = 100.0
        tg1       = tg1 * mbar2pa / d2_gas_correction_factor
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
    run = pressure(shot, plasma_conf=plasma_conf, plasma_conftime=plasma_conftime,closure_time=closure_time,
                   cryo=cryo, turbo=turbo, plasma=plasma, drsep=drsep,plasma_fracs=plasma_fracs,subdiv_time=div2sub,
                   drseptime=drseptime, valve=valve, gasvolts=gasvolts,f_wall_hit=f_wall_hit)
    run.display(tdens=tdens, dens=dens, time=time_press,
                p0_sublowdiv=p0_sublowdiv, p0_subuppdiv=p0_subuppdiv, p0_main=p0_main, tg1=tg1)




