import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.cm as cm
import string
import math
from matplotlib.patches import Rectangle

# Plasma constants
mu0   = 1.2566e-6
kB    = 1.38e-23
mD    = 1.67e-27 * 2
ec    = 1.602e-19
labels= iter(string.ascii_lowercase)
def white_axes_background(ax):
    ax.add_patch(Rectangle((0,0), 1, 1,
                           transform=ax.transAxes,
                           facecolor='white', edgecolor='none',
                           zorder=-1))

class dart:
    def __init__(self,jetto=None):
        self.loaded=False        
        self.time = np.array([])
        self.Ip   = np.array([])
        self.nsep = np.array([])
        self.Psep = np.array([])
        self.Pfus = np.array([])
        self.Paux = np.array([])
        self.Prad = np.array([])
        self.B0time = np.array([])
        self.kptime = np.array([])
        self.amtime = np.array([])
        self.R0time = np.array([])
        self.imp    = 'Ar'
        self.solm   = 1.0
        self.hfsflow= None
        self.lfsflow= None
        self.uppdivflow= None
        self.lowdivflow= None
        if jetto is not None:
            self.alias = jetto
            self.read_jetto()
        
    def read_jetto(self):
        import netCDF4
        if isinstance(self.alias, list):
            alias_jetto = self.alias
            
        elif isinstance(self.alias, str):
            alias_jetto = [self.alias]
        for alias in alias_jetto:
            file = '/common/step-simdb/simulations/aliases/'+alias+'/timetraces.CDF'
            f = netCDF4.Dataset(file)
            self.time=np.concatenate([self.time,np.array(f['TIME'][:])])
            self.Ip=np.concatenate([self.Ip,np.array(f['CUR'][:])])
            self.nsep=np.concatenate([self.nsep,np.array(f['NEBO'][:])])
            self.Psep=np.concatenate([self.Psep,np.array(f['PNT2'][:])])
            self.Pfus=np.concatenate([self.Pfus,np.array(f['PFUS'][:])])
            self.Paux=np.concatenate([self.Paux,np.array(f['PAUX'][:])])
            self.Prad=np.concatenate([self.Prad,np.array(f['PRAD'][:])])
            self.B0time=np.concatenate([self.B0time,np.array(f['BGEO'][:])])
            self.kptime=np.concatenate([self.kptime,np.array(f['K95'][:])])
            self.amtime=np.concatenate([self.amtime,np.array(f['AMIN'][:])])
            self.R0time=np.concatenate([self.R0time,np.array(f['RGEO'][:])])
        # Plasma current
        Ip_vals   = np.array([2.0e6 , 20.0e6 , 21.0e6 ])
        alft_vals = np.radians(np.array([0.5   , 4.0    , 4.0]))
        alft = interp1d(Ip_vals, alft_vals, kind='linear', fill_value='extrapolate')
        self.alft = alft(self.Ip)
        self.qdet0 = np.full(len(self.time), 1.0)

        # Set engineering values
        self.B0=np.mean(np.array(f['BGEO'][:]))
        self.kp=np.mean(np.array(f['K95'][:]))
        self.am=np.mean(np.array(f['AMIN'][:]))
        self.R0=np.mean(np.array(f['RGEO'][:]))
        self.Spump  = 20.0
        self.Twall  = 580.0
        self.tilt   = 0.0 # np.radians(1.5) can be inserted for any global tilting
        self.fdiv   = 0.4
        self.Rt     = self.R0 + self.am
        self.imp    = 'Ar'
        self.dp     = 2.0
        self.b      = 2.5
        self.set_impurity(self.imp)

    def calc_inputs(self):        
        self.set_impurity(self.imp)
        self.kc = np.sqrt((1.0 + self.kp**2) / 2.0)
        self.Ploss = self.Pfus/5.0 + self.Paux
        self.Ru = self.R0+self.am
        self.Bpol()
        self.qcylin()
        self.nGW()
        self.conn_length()
        self.lambdaq()
        self.lq = self.lq_mean
        self.lq_upper = self.lq+self.lq_error/3.0
        self.lq_lower = np.clip(self.lq-self.lq_error/3.0,np.min(self.lq_14),100.0)
        
    def Bpol(self):
    # Poloidal magnetic field
        self.Bp = 4.0 * np.pi * 0.1 * (self.Ip/1e6) / (2*np.pi*self.am*(self.kp)**(0.5))

    def qcylin(self):
    # Cylindrical q-factor
        self.qcyl = self.am * self.B0 * self.kc / self.R0 / self.Bp

    def nGW(self):
    # Greenwald Density fraction
        self.nGW = 1e20 *(self.Ip/1e6) / np.pi / self.am**2 

    def conn_length(self,facSXD=2.0):
    # Connection length approximation (assume double for SXD)
        
        if isinstance(self.Rt, float) or np.isscalar(self.Rt):
            self.lc = self.kp * np.pi * self.am * self.B0 / self.Bp
            if self.Rt > 1.2*self.R0:
                self.lc = facSXD * self.lc
        else:
            self.lc = np.zeros(len(self.time))
            for i,t in enumerate(self.time):
                if self.Rt[i] > 1.2*self.R0[i]:
                    self.lc[i] = facSXD * self.kp[i] * np.pi * self.am[i] * self.B0[i] / self.Bp[i]
                else:
                    self.lc[i] = self.kp[i] * np.pi * self.am[i] * self.B0[i] / self.Bp[i]
        self.lx = self.lc/2.0
    def power_sharing(self,flfs=0.9,flot=0.8,fhot=0.5):
        offset = 0 # Offset from drsep=0 mm due to drifts
        drsep = self.drsep-offset
        fdisc = 1 - np.exp(-(np.abs(drsep)) / (self.lq/2.0))
        plfs = flfs 
        phfs = (1 - flfs) 
        plit = fdisc * plfs
        plet = (1 - fdisc) * plfs
        phit = fdisc * phfs
        phet = (1 - fdisc) * phfs
        ppou = 0.5 * plet + flot * plit + fhot * phit
        ppin = 0.5 * phet + (1 - flot) * plit + (1 - fhot) * phit
        id0 = np.where(drsep <= 0)[0]
        id1 = np.where(drsep > 0)[0]
        lo  = np.zeros_like(drsep)
        li  = np.zeros_like(drsep)
        uo  = np.zeros_like(drsep)
        ui  = np.zeros_like(drsep)
        lo[id0] = ppou[id0]
        li[id0] = ppin[id0]
        uo[id0] = 0.5 * plet[id0]  
        ui[id0] = 0.5 * phet[id0]  
        lo[id1] = 0.5 * plet[id1]  
        li[id1] = 0.5 * phet[id1]  
        uo[id1] = ppou[id1]
        ui[id1] = ppin[id1]
        self.fdiv = lo

    def runGUIDE(self):
        from pressure import pressure, fit_confinement_time
        if self.Spump > 0:
            cryo = True
        else:
            cryo = False
        if self.nbi_S > 0:
            nbipump = True
        else:
            nbipump = False
        if self.recycling > 0:
            turbo = True
        else:
            turbo = False
        if self.fitconf:
            print("Fitting confinement time...")
            print("Initial values:",self.conftime,self.conf)
            fit_result = fit_confinement_time(self.time, self.dens, self.shot,
                                              self.conftime, self.conf,                                             
                                              cryo=cryo,nbipump=nbipump,turbo=turbo,closure_time=self.closure,
                                              drsep=self.drsep,drseptime=self.time,
                                              gas_matrix=self.gas_matrix,subdiv_time=self.div2sub,
                                              lower_S_subdiv=self.Spump,nbi_S=self.nbi_S,f_wall_hit=self.collision,recycling=self.recycling,
                                              inputgas=self.inputgas,gastraces=self.gastraces)
            print("Fitted values:",fit_result.x)
            self.conftime = self.conftime
            self.conf     = fit_result.x
            pconftime     = interp1d(self.conftime,self.conf,bounds_error=False, fill_value=0.0)
            self.plasma_conf = pconftime(self.time)
        p0        = pressure(self.shot,plasma_conf=self.plasma_conf,plasma_conftime=self.time,lower_S_subdiv=self.Spump,f_wall_hit=self.collision,recycling=self.recycling,turbo=turbo,cryo=cryo,closure_time=self.closure,
                             nbipump=nbipump,nbi_S=self.nbi_S,drsep=self.drsep,drseptime=self.time,gas_matrix=self.gas_matrix,subdiv_time=self.div2sub,
                             inputgas=self.inputgas,gastraces=self.gastraces)
        self.reservoir_output = p0
        useratio1 = True
        if useratio1:
            ratio = np.divide((p0.lowdiv_pressure[:,1]/7.0),(p0.lowdiv_pressure[:,0]+p0.lowdiv_pressure[:,0]/7.0),out=np.full_like(p0.lowdiv_pressure[:,1],np.nan),
                              where=p0.lowdiv_pressure[:,0] != 0)
        else:
            ratio = np.divide(p0.lowdiv_pressure[:,1],p0.lowdiv_pressure[:,0],out=np.full_like(p0.lowdiv_pressure[:,1],np.nan),
                              where=p0.lowdiv_pressure[:,0] != 0)
        valid = ~np.isnan(ratio)
        
        ldivpres = interp1d(p0.t_array,p0.lowdiv_pressure[:,0],kind='linear',fill_value='extrapolate')
        udivpres = interp1d(p0.t_array,p0.uppdiv_pressure[:,0],kind='linear',fill_value='extrapolate')
        concz    = interp1d(p0.t_array[valid],ratio[valid],kind='linear',fill_value='extrapolate')
        FIG_N2_fac = 0.35 # Factor to account for FIG calibration of N pressure
        subdivpres = interp1d(p0.t_array,p0.sublowdiv_pressure[:,0]+FIG_N2_fac*p0.sublowdiv_pressure[:,1],kind='linear',fill_value='extrapolate')
        usubdivpres = interp1d(p0.t_array,p0.subuppdiv_pressure[:,0]+FIG_N2_fac*p0.subuppdiv_pressure[:,1],kind='linear',fill_value='extrapolate')
        midpress = interp1d(p0.t_array,p0.main_pressure[:,0]+FIG_N2_fac*p0.main_pressure[:,1],kind='linear',fill_value='extrapolate')
        self.gastraces = p0.gastraces
        self.cz = concz(self.time)
        self.imp = 'N'
        self.p0 = ldivpres(self.time)
        self.p0midpred = midpress(self.time)
        self.p0uppred = usubdivpres(self.time)
        self.p0uppred2 = udivpres(self.time)
        self.avrp0 = (ldivpres(self.time)+udivpres(self.time))
        pred_dens = interp1d(p0.t_array,p0.electron_density[:,0],kind='linear',fill_value='extrapolate')
        self.pred_dens = pred_dens(self.time)                     
        self.p0sub = subdivpres(self.time)
        self.dp = 1.0
        self.set_impurity(self.imp)
        self.zeff= self.Zeff(self.cz)
        self.kc = np.sqrt((1.0 + self.kp**2) / 2.0)
        self.Ploss = self.Pfus/5.0 + self.Paux
        self.Ru = self.R0+self.am
        self.Bpol()
        self.Bp = self.Bp
        self.qcylin()
        self.conn_length(facSXD=1.5)
        # Eich multi-machine empirical regression #14 and #9
        # Eich multi-machine empirical regression #14 and #9
        lmb_func = interp1d([-100.0e-3,-5.0e-3,0.0,5.0e-3,100.0e-3],[1.0,1.0,2.0,1.0,1.0],fill_value='extrapolate')
        self.lq = 0.63e-3 * self.Bp**(-1.19) * 1.8
        self.power_sharing()
#        fpow  = self.fdiv_rad(self.cz,self.p0)
        self.fwall = 1.0-1.0/np.exp(1)
        fpow       = self.fdiv * self.fwall
        self.qpar  = self.pqpar(self.Psep,fpow, self.lq, self.Bp)
        self.qd    = 3.0 * ((self.Psep/1e6) * fpow/self.Rt) * (2e-3/self.lq) * (12.0/self.lx)**(0.043) * (self.p0 * (300.0/self.Twall) * (1.0 + self.fz * self.cz))**(-1.0)
        self.td    = 10**((np.log10(self.qd) + 0.11) / 0.54)
        self.nsep  = self.pnsep(self.qpar, self.zeff, self.alft, self.lc, self.avrp0)
        self.Tsep  = self.ptsep(self.zeff,self.qpar,self.lc)
        fx         = self.Rt/self.Ru
        self.nsep2 = self.pnsep2(self.qpar,self.td,self.Tsep,self.avrp0,fx)
        # Compare to experimental separatrix conditions
        self.nsep_exp = np.zeros(len(self.time))
        for i,t in enumerate(self.time):
            t_inter = interp1d(self.te_fit[i,:],self.r_fit,kind='linear',fill_value='extrapolate')
            r_sep   = t_inter(self.Tsep[i])
            n_inter = interp1d(self.r_fit,self.ne_fit[i,:],kind='linear',fill_value='extrapolate')
            self.nsep_exp[i] = n_inter(r_sep)
            
        self.tilt= 0.0
        self.qperp = self.pqperp(self.td,self.Tsep,self.nsep, self.alft)
        if self.writefile:
            outfile = self.dartfile
            np.savez(self.dartfile, time=self.time, qdet=self.qd, pdiv=self.p0,psubdiv=self.p0sub,qperp=self.qperp,tdiv=self.td)
            print(f"Data written to {outfile}")                
    def plot_GUIDE(self,canvas=None):
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 18
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.Psep/self.R0/1e6,label=r'P$_{sep}$/R$_0$',loc='upper right',xlim=[0,self.tend],col=def_col[0],ytitle='MW/m,MA')
        self.plot_output(axes,0,0,self.time,self.Ip/1e6,ylim=[0,6],label=f'Plasma current',loc='upper right',xlim=[0,self.tend],col=def_col[2],ytitle='MW/m,MA')
        if (self.cz > 0).any():
            self.plot_output(axes,1,0,self.time,self.cz,alpha=0.6,label=self.imp+' conc.',loc='upper right',xlim=[0,self.tend],col=def_col[1],ytitle='')
        self.plot_output(axes,1,0,self.time,self.p0,ylim=[0,0.6],label=r'DART lower div.',loc='upper right',xlim=[0,self.tend],col=def_col[0],ytitle='Pa')
        self.plot_output(axes,2,0,self.time,self.IRt2,ylim=[0,3],label=r'IR peak q$_{\perp}$',col=def_col[0],xlim=[0,self.tend],xtitle='Time [s]',ytitle='MWm$^{-2}$')
        self.plot_output(axes,2,0,self.time,self.IRt5,ylim=[0,1.5],label=r'',col=def_col[0],xtitle='Time [s]',xlim=[0,self.tend],ytitle='MWm$^{-2}$') 
        self.plot_output(axes,2,0,self.time,self.qperp,ylim=[0,3],label=r'DART',col=def_col[1],xlim=[0,self.tend],xtitle='Time [s]',ytitle='MWm$^{-2}$') 
        try:
##            import pandas as pd
##            df=pd.read_csv(f'Tesep/{self.shot}_sep_data.csv')
##            self.plot_output(axes,0,1,df['time (s)'], df['Tesep (eV)'],label='T$_{e,u}$ exp.',col=def_col[0],xlim=[0,self.tend],ls='',psym='o',xtitle='Time [s]',ytitle=r'eV')
            from database import mast_database
            data = mast_database(self.shot)
            self.plot_output(axes,0,1,data.time['Tesep'],data.data['Tesep'],label='T$_{e,u}$ exp.',col=def_col[0],xlim=[0,self.tend],xtitle='Time [s]',ytitle=r'eV')
            axes[0,1].fill_between(data.time['Tesep'], data.data['Tesep'] - data.std['Tesep'], data.data['Tesep'] + data.std['Tesep'], color=def_col[0], alpha=0.2)
        except:
            print("No exp. Te,sep available")
        self.plot_output(axes,0,1,self.time,self.Tsep,ylim=[0,60],label=r'DART',col=def_col[1],xlim=[0,self.tend],xtitle='Time [s]',ytitle=r'eV')
        try:
            self.plot_output(axes,1,1,data.time['nesep'],data.data['nesep']/1e19,label='n$_{e,u}$ exp.',xlim=[0,self.tend],col=def_col[0],xtitle='Time [s]')
            axes[1,1].fill_between(data.time['nesep'], (data.data['nesep'] - data.std['nesep'])/1e19, (data.data['nesep'] + data.std['nesep'])/1e19, color=def_col[0], alpha=0.2)
            #self.plot_output(axes,1,1,df['time (s)'], df['nesep (m-3)']/1e19,label='n$_{e,u}$ exp.',xlim=[0,self.tend],ls='',psym='o',col=def_col[0],xtitle='Time [s]',ytitle=r'eV')
        except:
            print("No exp. ne,sep available")
        self.plot_output(axes,1,1,self.time,self.nsep/1e19,ylim=[0,2.0],label=r'DART',xlim=[0,self.tend],col=def_col[1],xtitle='Time [s]',ytitle=r'1e19 m$^{-3}$')
        self.plot_output(axes,2,1,self.time,self.qd,ylim=[0,10],col=def_col[1],xtitle='Time [s]',xlim=[0,self.tend],label=r'DART q$_{det}$',loc='upper right',ytitle=r'')
        self.plot_output(axes,2,1,[-100,100],[1,1],ylim=[0,8],col=def_col[0],ls='--',xlim=[0,self.tend],xtitle='Time [s]',label=r'Detachment',loc='upper right',ytitle=r'')
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()
    def plot_GUIDEuseful(self,canvas=None):
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        nrows=2
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 18
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.lq*1e3,ylim=[0,15],label=r'1.8x $\lambda_{q,Eich \#14}$',xlim=[0,self.tend],col=def_col[1],xtitle='Time [s]',ytitle='mm')
        try:
            import pandas as pd
            df=pd.read_csv(f'Tesep/{self.shot}_sep_data.csv')
            self.plot_output(axes,0,0,df['time (s)'], df['lambda_q (m)']*1e3,label='lq$_{u}$ exp.',xlim=[0,self.tend],col=def_col[0],ls='',psym='o',xtitle='Time [s]',ytitle=r'mm')
        except:
            print("No experimental lq available")
        self.plot_output(axes,1,0,self.time,self.B0,ylim=[0,1.2],nrows=nrows,label=r'B$_{0}$',xlim=[0,self.tend],col=def_col[2],loc='upper right',xtitle='Time [s]',ytitle=r'T')
        self.plot_output(axes,1,0,self.time,self.Bp,ylim=[0,1.2],nrows=nrows,label=r'B$_{p}$',xlim=[0,self.tend],col=def_col[3],loc='upper right',xtitle='Time [s]',ytitle=r'T')
        self.plot_output(axes,1,0,self.time,self.fdiv,ylim=[0,1.2],nrows=nrows,label=r'f$_{div}$',xlim=[0,self.tend],col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'T')
        self.plot_output(axes,0,1,self.time,np.degrees(self.alft),nrows=nrows,ylim=[0,10],xlim=[0,self.tend],label=r'Target grazing angle',loc='upper right',col=def_col[0],xtitle='Time [s]',ytitle=r'Degrees')
        self.plot_output(axes,1,1,self.time,self.lc,ylim=[0,20],nrows=nrows,label=r'Connection length',xlim=[0,self.tend],col=def_col[0],loc='upper right',xtitle='Time [s]',ytitle=r'm,eV')
        self.plot_output(axes,1,1,self.time,self.td,ylim=[0,20],nrows=nrows,label=r'T$_{e,div}$',xlim=[0,self.tend],col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'm,eV')
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()
    def plot_GUIDEcondensed(self,canvas=None):
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        nrows=2

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        fig.patch.set_facecolor('none')     # transparent outside the axes
        self.fs = 18
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.dens/1e19,ylim=[0,9],nrows=nrows,xlim=[0,self.tend],label=r'Line averaged density',col=def_col[0],loc='upper right',xtitle='Time [s]',ytitle=r'm')
        self.plot_output(axes,0,0,self.time,self.pred_dens/1e19,ylim=[0,9],nrows=nrows,xlim=[0,self.tend],label=r'DART density',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'm')
        self.plot_output(axes,1,0,self.time,self.p0mid*1000.0,ylim=[0,6.0],nrows=nrows,xlim=[0,self.tend],label=r'Midplane FIG',col=def_col[0],loc='upper right',xtitle='Time [s]',ytitle=r'mPa')
        self.plot_output(axes,1,0,self.time,self.p0midpred*1000.0,ylim=[0,6.0],nrows=nrows,xlim=[0,self.tend],label=r'DART midplane (FIG)',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'mPa')
        self.plot_output(axes,0,1,self.time,self.p0up,ylim=[0,1.5],label=r'Upper FIG',nrows=nrows,xlim=[0,self.tend],col=def_col[0],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,0,1,self.time,self.p0uppred,ylim=[0,1.5],label=r'DART sub-div. (FIG)',nrows=nrows,xlim=[0,self.tend],col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,0,1,self.time,self.p0uppred2,ylim=[0,1.5],label=r'DART div.',xlim=[0,self.tend],ls='--',alpha=0.8,nrows=nrows,col=def_col[2],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,1,self.time,self.press,ylim=[0,1.5],label=r'Lower FIG',xlim=[0,self.tend],nrows=nrows,loc='upper right',col=def_col[0],ytitle='Pa')
        self.plot_output(axes,1,1,self.time,self.p0sub,ylim=[0,1.5],label=r'DART sub-div. (FIG)',xlim=[0,self.tend],loc='upper right',col=def_col[1],ytitle='Pa',nrows=nrows)
        self.plot_output(axes,1,1,self.time,self.p0,ylim=[0,1.5],label=r'DART div.',ls='--',xlim=[0,self.tend],alpha=0.8,nrows=nrows,loc='upper right',col=def_col[2],ytitle='Pa',xtitle='Time [s]')
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()
      
    def plot_GUIDEextra(self,canvas=None):
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        nrows=2

        fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 16
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.reservoir_output.t_array,self.reservoir_output.plasma_influx[:,0],xlim=[0,5],ylim=[0,1e22],nrows=nrows,label=r'HFS',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
##        self.plot_output(axes,0,0,self.reservoir_output.t_array,self.reservoir_output.hfs_main[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'HFS',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
##        self.plot_output(axes,0,0,self.reservoir_output.t_array,self.reservoir_output.lpfr[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'LPFR',col=def_col[2],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
##        self.plot_output(axes,0,0,self.reservoir_output.t_array,self.reservoir_output.upfr[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'UPFR',col=def_col[3],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,0,1,self.time,self.p0mid,ylim=[0,1.0],nrows=nrows,xlim=[0,5],label=r'FIG HM12',col=def_col[0],loc='upper right',xtitle='Time [s]',ytitle=r'mPa')
        self.plot_output(axes,0,1,self.reservoir_output.t_array,self.reservoir_output.lfs_main[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'LFS',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,0,1,self.reservoir_output.t_array,self.reservoir_output.llfs_main[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'LLFS',col=def_col[2],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,0,1,self.reservoir_output.t_array,self.reservoir_output.ulfs_main[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'ULFS',col=def_col[3],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,0,self.time,self.p0up,xlim=[0,5],ylim=[0,1],label=r'FIG HU08',nrows=nrows,col=def_col[0],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,0,self.reservoir_output.t_array,self.reservoir_output.udiv[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'UDIV',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,0,self.reservoir_output.t_array,self.reservoir_output.subudiv[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'SUDIV',col=def_col[2],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,0,self.reservoir_output.t_array,self.reservoir_output.ssubudiv[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'SSUDIV',col=def_col[3],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,1,self.time,self.press,xlim=[0,5],ylim=[0,1.0],label=r'FIG HL11',nrows=nrows,loc='upper right',col=def_col[0],ytitle='Pa')
        self.plot_output(axes,1,1,self.reservoir_output.t_array,self.reservoir_output.ldiv[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'LDIV',col=def_col[1],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,1,self.reservoir_output.t_array,self.reservoir_output.subldiv[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'SLDIV',col=def_col[2],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        self.plot_output(axes,1,1,self.reservoir_output.t_array,self.reservoir_output.ssubldiv[:,0],xlim=[0,5],ylim=[0,1],nrows=nrows,label=r'SSLDIV',col=def_col[3],loc='upper right',xtitle='Time [s]',ytitle=r'Pa')
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()
    def run(self,progress_bar=None,root=None):
        if self.time is None:
            exit('Please provide inputs')
        self.calc_inputs()

        # Initialize arrays
        shape = len(self.time)

        arrays = ['cz_lower', 'qdet_lower', 'p0_lower', 'ts_lower', 'td_lower','qp_lower', 'qt_lower', 'ns_lower', 'lqat_lower',
                  'cz_upper', 'qdet_upper', 'p0_upper', 'ts_upper', 'td_upper','qp_upper', 'qt_upper', 'ns_upper', 'lqat_upper']
        for arr in arrays:
            setattr(self, arr, np.zeros(shape))
        # Predict every time step
        for i,t in enumerate(self.time):
            xx1 = self.predict_exhaust(self.Psep[i], self.lq_lower[i], self.lc[i],self.lx[i],self.Bp[i], self.alft[i], self.nsep[i],self.qdet0[i],self.qcyl[i])
            xx2 = self.predict_exhaust(self.Psep[i], self.lq_upper[i], self.lc[i],self.lx[i],self.Bp[i], self.alft[i], self.nsep[i],self.qdet0[i],self.qcyl[i])
            for key in xx1:
                getattr(self, key+'_lower')[i] = xx1[key]
                getattr(self, key+'_upper')[i] = xx2[key]
            if progress_bar is not None:
                prog = t / np.max(self.time) * 100.0
                progress_bar["value"] = prog
                root.update_idletasks()  # Refresh GUI

    def lambdaq(self,cz=0.035):
    # Near SOL power width 
        # Eich multi-machine empirical regression #14 and #9
        self.lq_14 = 0.63e-3 * self.Bp**(-1.19)
        self.lq_09 = 0.70e-3 * self.B0**(-0.77)*self.qcyl**(1.05)*(self.Psep/1e6)**(0.09)
        # Thornton MAST-U scaling
        self.lq_ST  = 1.84e-3 * self.Bp**(-0.68)*(self.Psep/1e6)**(0.18)
        # HD model with impurity
        Zeff    = self.Zeff(cz)
        nz      = cz * self.nsep
        ni      = self.nsep - nz*self.Zavr
        zbar    = self.nsep/(nz+ni)
        abar    = (nz*self.amass+ni*2.0)/(nz+ni)
        self.lq_HDz = (5671.*self.Psep**(1./8)*(1.0+self.kp**2)**(5./8.)* 
                       self.am**(17./8.)*self.B0**(1/4.0)/(self.Ip**(9.0/8)*self.R0)*  
                       (2*abar/(1+zbar))**(7.0/16.0)*((Zeff+4)/5.0)**(1.0/8.0))
        # HD model with Zeff=1.0
        abar = 2.0
        zbar = 1.0
        self.lq_HD = (5671.*self.Psep**(1./8)*(1.0+self.kp**2)**(5./8.)* 
                      self.am**(17./8.)*self.B0**(1/4.0)/(self.Ip**(9.0/8)*self.R0)*  
                     (4.0/(2.0))**(7.0/16.0)*((1.0+4)/5.0)**(1.0/8.0))
        
        # Geometric mean
        lq_values     = [self.lq_14, self.lq_09, self.lq_HDz, self.lq_HD, self.lq_ST]
        self.lq_mean   = np.zeros_like(self.lq_ST)
        self.lq_error  = np.zeros_like(self.lq_ST)
        for i,t in enumerate(self.time):
            # Get the values for this time step for all lq_ arrays
            lq_at_timestep = [lq[i] for lq in lq_values]
            # Calculate the geometric mean and standard deviation
            self.lq_mean[i] = np.exp(np.mean(np.log(lq_at_timestep)))
            # Calculate the geometric standard deviation for this time step
            gsd = np.exp(np.sqrt(np.mean((np.log(lq_at_timestep) - np.log(self.lq_mean[i]))**2)))
            # Estimate uncertainty/error as the geometric standard deviation
            self.lq_error[i] = self.lq_mean[i] * (gsd - 1)  # Use deviation from the mean to calculate error
        self.lq_mean = self.lq_mean * self.solm
        self.lq_error = self.lq_error * self.solm
    def set_impurity(self,imp='Ar'):
        if imp.lower() == 'ar':            
            self.Zavr = 17
            self.fz=90.0
            self.max_conc=0.045
            self.amass = 40.0
        if imp.lower() == 'ne':            
            self.Zavr = 9
            self.fz=45.0
            self.max_conc=0.1  
            self.amass = 20.0
        if imp.lower() == 'n':             
            self.Zavr = 5
            self.fz=18.0
            self.max_conc=0.2  
            self.amass = 14.0
        if imp.lower() == 'd':             
            self.Zavr = 1
            self.fz=1.0
            self.max_conc=0.5  
            self.amass = 2.0
        if imp.lower() == 'none':             
            self.Zavr = 1
            self.fz=0.0
            self.max_conc=0.0  
            self.amass = 1.0
    def Zeff(self,cz):
        return 1.0 + self.Zavr * (self.Zavr-1) * cz

    def vary_parm(self,x,y1,y2,start,mid1,mid2,end):
        result = np.copy(y1)
        start_idx = np.searchsorted(x, start)
        mid1_idx = np.searchsorted(x, mid1)
        mid2_idx = np.searchsorted(x, mid2)
        end_idx = np.searchsorted(x, end)
        transition1 = np.linspace(y1[start_idx], y2[start_idx], mid1_idx - start_idx)
        transition2 = np.linspace(y2[mid2_idx], y1[end_idx], end_idx - mid2_idx)
        result[start_idx:mid1_idx] = transition1
        result[mid1_idx:mid2_idx] = y2[mid1_idx:mid2_idx]
        result[mid2_idx:end_idx] = transition2
        return result

    def fdiv_rad(self,cz,p0):
        # Scaling to account for upstream Ar radiation
        if self.imp.lower() == 'ar': 
            return self.fdiv - (1 + self.fz * cz) * p0 * 0.8e-3
        else:
            return self.fdiv
    def predict_exhaust(self,psep, lq, lc, lx, bp, alf, nsep, qdet0,qcyl):
    # Set initial guess
        if qdet0 <=1:
            cz   = 0.03   # Ar concentration
            p0   = 10.0   # Divertor pressure
        else:
            cz = 0.0
            p0=10
        zeff = self.Zeff(cz)
    # Iterate solution until convergence
        itx  = 0
        acc  = 1000.0
        while itx < 100 and acc > 1e-6:
            fpow  = self.fdiv_rad(cz,p0)
            qpar  = self.pqpar(psep,fpow, lq, bp)
            p00   = self.log_vector(0.01, 40.0, 1000)
            ns1   = self.pnsep(qpar, zeff, alf, lc, p00)
            p0_inter = interp1d(ns1, p00,kind='linear',fill_value='extrapolate')
            p0    = p0_inter(nsep)
            czz   = self.log_vector(0.001,20.0,1000)/100.0
            fpow1 = self.fdiv_rad(czz,p0)
            qdet1 = self.qd(psep/1e6,fpow1,lq*1e3,lx,p0,czz)
            if qdet1[0] < 1.0:
                acc = 0.2
                cz=0.0
            else:
                qdet_interp = interp1d(qdet1,czz,kind='linear',fill_value='extrapolate')
                acc   = np.abs(qdet_interp(qdet0)-cz)
                cz    = qdet_interp(qdet0)
            if cz < 0.0:
                cz = 0.0
            
            if cz > self.max_conc:
                cz = self.max_conc
            zeff  = self.Zeff(cz)
            itx   = itx+1.0
            fpow  = self.fdiv_rad(cz,p0)
            qdet  = self.qd(psep/1e6,fpow,lq*1e3,lx,p0,cz)
            qpar  = self.pqpar(psep,fpow,lq,bp)
            ts    = self.ptsep(zeff,qpar,lc)
            td    = 10**((np.log10(qdet) + 0.11) / 0.54)
            qperp = self.pqperp(td,ts,nsep,alf)
        # Eich generalised scaling
        rhos   = np.sqrt(mD*ts*ec)/(ec*bp)
        alphat = 3.0e-18 * qcyl**2 * self.R0 * nsep * zeff / ts**2
        lq_at  = 2.1*(2.0/7.0)*rhos*(1.0+2.1*alphat**(1.7))
        return {'cz': cz, 'p0': p0, 'lqat':lq_at,'qdet': qdet, 'ts': ts,
                'qt':qperp,'qp': qpar, 'ns': nsep,'td':td}

    def log_vector(self,low, high, num, linear=False):
        if linear:
            return np.linspace(low, high, num)
        else:
            return np.logspace(np.log10(low), np.log10(high), num)
        
    def gamma(self):
    # Function to define gamma
        return 7.0

    def kappa(self):
    # Function to define electron conductivity
        loglamb = 12.0
        kap = (3.0 / 4.0 / np.sqrt(2 * np.pi) / np.sqrt(9.1093837E-31) / loglamb *
               (4.0 * np.pi * 8.854E-12 * 1.28280e-33 / 1.602E-19 / 1.6e-19)**2)
        return kap

    def kappaz(self,zfsep):
    # Function for calculating finite-Z correction to kappa
        z  = np.linspace(1.0, 16.0, 100)
        kz = (3.9 + 2.3 / z) / (0.31 + 1.2 / z + 0.41 / z**2) / z
        kzinterp = interp1d(z,kz,kind='linear',fill_value='extrapolate')
        return kzinterp(zfsep)

    def press_to_flux(self,Tgas=300):
    # Function to convert a pressure to flux
        md2       = 4.0 * 1.67e-27
        R         = 8.314
        return 2.0 * np.sqrt(8.0*kB*Tgas/np.pi /md2)/4 /(kB * Tgas)

    def dflow(self,p0):
        return 2 * p0 * self.Spump / (kB * self.Twall)

    def qd(self,Psep, fdiv, lq, lx, p0, cz):
    # Function for qdet calculations
        return 3. * (Psep * fdiv / self.Rt) * (2.0 / lq) *  (12.0 / lx)**0.043 / (self.dp * (300.0/self.Twall) * p0 * (1.0 + self.fz * cz))

    def pqpar(self,psep,fpow, lq, bp):
    # Function for parallel energy flux density
        return psep * fpow * self.R0 * self.B0 / (2.0 * np.pi * lq * self.Ru * self.Ru * bp)

    def pnsep(self,qpar, zeff, alf, l_c, p0):
    # Function to upstream electron density
        frad_fmom = 0.8
        ion_elec  = 1.5
        b_press   = 1.7
        DP        = self.dp
        p_exp     = 0.31
        fac = (frad_fmom * ion_elec * (self.gamma())**(-0.5) / 1.6e-19 *
               (2.0 * self.kappa() / 7.0)**(2.0 / 7.0) * np.sqrt(1.67e-27) *
               np.sqrt(self.press_to_flux(Tgas=self.Twall) * DP / b_press))
        return fac * (self.kappaz(zeff) / l_c)**(2.0 / 7.0) * np.maximum(np.sin(alf), 0.01)**(-0.5) * qpar**(3.0 / 14.0) * p0**p_exp

    def pnsep2(self,qpar,tdiv,Tu,p0,fx,epsilon=0.0):
        nt   = (1/fx) * qpar * (mD/2.0)**(0.5) * ec**(-1.5)  / (self.gamma()*tdiv**1.5 + epsilon*tdiv**(0.5))
        rhoTt,rhont,rhoTu,rhonu = 2.5, 2.0, 2.5,1.8  
        fac       = 2.0 * rhoTt * rhont / (rhoTu * rhonu)
        const1    = 0.06
        const2    = 0.35
        A = const1 * tdiv**(const2) # Best fit to describe frad/fmom
        return (fac  * tdiv * nt / Tu) * A

    def ptsep(self,zeff, qpar, lc):
    # Function for upstream electron temperature
        fac = (7.0 / 2.0 / self.kappa())**(2.0 / 7.0)
        return fac * (self.kappaz(zeff))**(-2.0 / 7.0) * (qpar * lc)**(2.0 / 7.0)

    def pqperp(self,tdiv,tsep,nsep,alf):
        fmom = np.tanh(0.16 * tdiv**1.42)
        fac  = 0.33 * (1.0/1.672e-27)**(0.5)*(1.6e-19)**(3/2.)/1e6
        return fac * tsep * nsep * fmom * (7.0 * tdiv**0.5 + 13.6 * tdiv**(-0.5)) * np.sin(alf+self.tilt)

    def display(self,canvas=None):
        # Create figure and axes
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 16
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.Psep/self.R0/1e6,ylim=[0,30],label=r'P$_{sep}$/R$_0$',loc='upper right',col=def_col[0],ytitle='MW/m,MA')
        self.plot_output(axes,0,0,self.time,self.Ip/1e6,ylim=[0,np.nanmax([np.max(self.Ip)/1e6,np.max(self.Psep/self.R0)/1e6])*1.7],label=f'Plasma current',loc='upper right',col=def_col[1],ytitle='MW/m,MA')
        self.plot_output(axes,1,0,self.time,self.nsep/1e19,ylim=[0,np.nanmax(self.nsep)/1e19*1.5],col=def_col[0],label=f'Separatrix density',loc='upper right',ytitle=r'10$^{19}$ m$^{-3}$')
        self.plot_output(axes,2,0,self.time,self.lq_lower*1000,z=self.lq_upper*1000,alpha=0.7,ylim=[0.1,100],ylog=True,label=r'$\lambda_{q,Sim. input}$',col=def_col[0],xtitle='Time [s]')
        self.plot_output(axes,2,0,self.time,self.lq_14*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich \#14}$',col=def_col[1],xtitle='Time [s]')
        self.plot_output(axes,2,0,self.time,self.lq_09*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich \#9}$',col=def_col[2],xtitle='Time [s]')
        self.plot_output(axes,2,0,self.time,self.lqat_lower*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich (inc. \alpha_T)}$',col=def_col[4],xtitle='Time [s]')
        self.plot_output(axes,2,0,self.time,self.lq_HDz*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,HD (Z_{eff}=10)}$',col=def_col[3],xtitle='Time [s]')
        self.plot_output(axes,2,0,self.time,self.lq_ST*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Thornton \#2}$',col=def_col[5],xtitle='Time [s]',ytitle='mm')
        self.plot_output(axes,0,1,self.time,self.qp_lower/1e9,alpha=0.6,z=self.qp_upper/1e9,ylim=[0,np.nanmax(self.qp_lower)/1e9*2.0],label=f'Unmitigated parallel\nenergy flux density',loc='upper right',col=def_col[0],ytitle=r'GWm$^{-2}$')
       
        self.plot_output(axes,1,1,self.time,self.cz_lower*100,z=self.cz_upper*100,alpha=0.6,label=self.imp+' conc.',loc='upper right',ylim=[0,25],col=def_col[0],ytitle='%, Pa, 10$^{22}$ #/s')
        self.plot_output(axes,1,1,self.time,self.p0_lower,z=self.p0_upper,ylim=[0,15],label='Divertor press.',loc='upper right',col=def_col[1])
        self.dpuff_lower = self.dflow(self.p0_lower)
        self.dpuff_upper = self.dflow(self.p0_upper)
        self.plot_output(axes,1,1,self.time,self.dpuff_lower/1e22,z=self.dpuff_upper/1e22,label='DT flow rate',loc='upper right',ylim=[0,np.min([np.max([np.nanmax(self.cz_lower)*100,np.nanmax(self.p0_upper),np.nanmax(self.dpuff_lower)/1e22])*2.0,25])],col=def_col[2],ytitle='%, Pa, 10$^{22}$ #/s')
        self.plot_output(axes,2,1,self.time,self.qt_lower,alpha=0.6,z=self.qt_upper,ylim=[0,30],col=def_col[0],xtitle='Time [s]',label=f'Target heat load',loc='center left',ytitle=r'MWm$^{-2}$')
        self.plot_output(axes,2,1,self.time,self.td_lower,z=self.td_upper,ylim=[0,np.min([np.max([np.nanmax(self.td_upper),np.nanmax(self.qt_lower)])*1.7,30.0])],col=def_col[1],xtitle='Time [s]',label=f'Target temp.',loc='center left',ytitle=r'MWm$^{-2}$, eV')
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()

    def display_condensed(self,canvas=None):
        # Create figure and axes
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        nrows=2
        fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 16
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.Psep/self.R0/1e6,ylim=[0,30],label=r'P$_{sep}$/R$_0$',loc='upper right',nrows=nrows,col=def_col[0],ytitle='MW/m,MA')
        self.plot_output(axes,0,0,self.time,self.Ip/1e6,ylim=[0,np.max([np.nanmax(self.Ip)/1e6,np.nanmax(self.Psep/self.R0)/1e6])*2.0],nrows=nrows,label=f'Plasma current',loc='upper right',col=def_col[1],ytitle='MW/m,MA')
        self.plot_output(axes,0,0,self.time,self.nsep/1e19,nrows=nrows,ylim=[0,np.max([np.nanmax(self.nsep)/1e19,np.nanmax(self.Ip)/1e6,np.nanmax(self.Psep/self.R0)/1e6])*1.7],col=def_col[2],label=f'Separatrix density',loc='upper right',ytitle=r'10$^{19}$ m$^{-3}$')
        self.plot_output(axes,1,0,self.time,self.lq_lower*1000,z=self.lq_upper*1000,alpha=0.7,ylim=[0.1,100],nrows=nrows,ylog=True,label=r'$\lambda_{q,Sim. input}$',col=def_col[0],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_14*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich \#14}$',nrows=nrows,col=def_col[1],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_09*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich \#9}$',nrows=nrows,col=def_col[2],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lqat_lower*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich (inc. \alpha_T)}$',nrows=nrows,col=def_col[4],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_HDz*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,HD (Z_{eff}=10)}$',nrows=nrows,col=def_col[3],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_ST*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Thornton \#2}$',nrows=nrows,col=def_col[5],xtitle='Time [s]',ytitle='mm')
       
        self.plot_output(axes,0,1,self.time,self.cz_lower*100,z=self.cz_upper*100,alpha=0.6,label=self.imp+' conc.',loc='upper right',nrows=nrows,ylim=[0,25],col=def_col[0],ytitle='%, Pa, 10$^{22}$ #/s')
        self.plot_output(axes,0,1,self.time,self.p0_lower,z=self.p0_upper,ylim=[0,15],label='Divertor press.',loc='upper right',nrows=nrows,col=def_col[1])
        self.dpuff_lower = self.dflow(self.p0_lower)
        self.dpuff_upper = self.dflow(self.p0_upper)
        self.plot_output(axes,0,1,self.time,self.dpuff_lower/1e22,z=self.dpuff_upper/1e22,label='DT flow rate',nrows=nrows,loc='upper right',ylim=[0,np.max([np.nanmax(self.cz_lower)*100,np.nanmax(self.p0_upper),np.nanmax(self.dpuff_lower)/1e22])*2.0],col=def_col[2],ytitle='%, Pa, 10$^{22}$ #/s')
        self.plot_output(axes,1,1,self.time,self.qt_lower,alpha=0.6,z=self.qt_upper,ylim=[0,30],col=def_col[0],nrows=nrows,xtitle='Time [s]',label=f'Target heat load',loc='center left',ytitle=r'MWm$^{-2}$')
        self.plot_output(axes,1,1,self.time,self.td_lower,z=self.td_upper,ylim=[0,np.max([np.nanmax(self.td_upper),np.nanmax(self.qt_lower)])*1.7],col=def_col[1],xtitle='Time [s]',nrows=nrows,label=f'Target temp.',loc='upper right',ytitle=r'MWm$^{-2}$, eV')
        if canvas is None:
            plt.show()
        else:
            #plt.savefig('/home/shenders/Images/DART.png',dpi=300,transparent=True)
            canvas.figure = fig
            canvas.draw()
    def display_talk(self,canvas=None):
        # Create figure and axes
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        nrows=3
        fig, axes = plt.subplots(nrows=nrows, ncols=1, sharex=True, figsize=(6, 8))
        fig.subplots_adjust(hspace=0.1,left=0.15,top=0.96, bottom=0.1,right=0.97)
        self.fs = 16
        def_col = ['black','#E41A1C' ,'#0072B2', '#E69F00', '#009E73', '#CC79A7','#666666']
        self.plot_output(axes,0,0,self.time,self.Psep/self.R0/1e6,ylim=[0,50],label=r'P$_{sep}$/R$_0$',loc='upper right',ncol=1,nrows=nrows,col=def_col[0],ytitle='MW/m, MA, 10$^{18}$ m$^{-3}$')
        self.plot_output(axes,0,0,self.time,self.Ip/1e6,ylim=[0,50],nrows=nrows,ncol=1,label=f'Plasma current',loc='upper right',col=def_col[1],ytitle='MW/m, MA, 10$^{18}$ m$^{-3}$')
        self.plot_output(axes,0,0,self.time,self.nsep/1e18,nrows=nrows,ylim=[0,50],ncol=1,col=def_col[2],label=f'Separatrix density',loc='upper center',ytitle=r'MW/m, MA, 10$^{18}$ m$^{-3}$')
        self.plot_output(axes,1,0,self.time,self.lq_lower*1000,z=self.lq_upper*1000,alpha=0.7,ylim=[0.1,100],nrows=nrows,ylog=True,loc='upper center',label=r'$\lambda_{q,Sim. input}$',ncol=1,col=def_col[0],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_14*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich \#14}$',nrows=nrows,ncol=1,loc='upper center',col=def_col[1],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_09*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich \#9}$',nrows=nrows,ncol=1,loc='upper center',col=def_col[2],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lqat_lower*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Eich (inc. \alpha_T)}$',nrows=nrows,loc='upper center',ncol=1,col=def_col[4],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_HDz*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,HD (Z_{eff}=10)}$',nrows=nrows,ncol=1,loc='upper center',col=def_col[3],xtitle='Time [s]')
        self.plot_output(axes,1,0,self.time,self.lq_ST*1000,ylim=[0.1,800],ylog=True,label=r'$\lambda_{q,Thornton \#2}$',nrows=nrows,ncol=1,loc='upper center',col=def_col[5],xtitle='Time [s]',ytitle='mm')
       
        self.plot_output(axes,2,0,self.time,self.cz_lower*100,z=self.cz_upper*100,alpha=0.6,label=self.imp+' conc.',loc='upper center',nrows=nrows,ncol=1,ylim=[0,10],col=def_col[0],ytitle='%, Pa, MWm$^{-2}$')
        self.plot_output(axes,2,0,self.time,self.p0_lower,z=self.p0_upper,ylim=[0,10],label='Divertor press.',loc='upper center',nrows=nrows,ncol=1,col=def_col[1])
        self.plot_output(axes,2,0,self.time,self.qt_lower,alpha=0.6,z=self.qt_upper,ylim=[0,10],col=def_col[2],nrows=nrows,xtitle='Time [s]',ncol=1,label=f'Target heat load',loc='upper center',ytitle=r'%, Pa, MWm$^{-2}$')
        if canvas is None:
            plt.show()
        else:
            try:
                plt.savefig('/home/shenders/Images/DART1.png',dpi=300,transparent=True)
            except:
                pass
            canvas.figure = fig
            canvas.draw()
    def display_useful(self,canvas=None):
        # Create figure and axes
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        nrows =2
        fig, axes = plt.subplots(nrows=nrows, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 16
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.lc,ylim=[0,np.nanmax(self.lc)*1.2],col=def_col[0],nrows=nrows,label=f'Connection length',loc='upper right',ytitle=r'm')
        self.plot_output(axes,1,0,self.time,self.Bp,label='Poloidal field',xtitle='Time [s]',loc='upper right',nrows=nrows,ylim=[0,np.nanmax(self.Bp)*2.0],col=def_col[2],ytitle='T')
        self.plot_output(axes,0,1,self.time,self.qcyl,ylim=[0,np.nanmax(self.qcyl)*1.2],col=def_col[0],xtitle='Time [s]',nrows=nrows,label=f'qcyl',loc='upper right',ytitle=r'')
        self.plot_output(axes,1,1,self.time,self.ts_lower,alpha=0.6,z=self.ts_upper,xtitle='Time [s]',nrows=nrows,ylim=[0,np.nanmax(self.ts_lower)*1.5],label=f'Separatrix temperature',loc='upper right',col=def_col[0],ytitle=r'eV')
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()

    def plot_setup(self,axes,i,j,ytitle=None,xtitle=None,xlim=[0,10],xlog=False,ylog=False,ylim=None,nrows=3,ncol=2):
        bbox_props = dict(boxstyle='square',  facecolor=(0.97, 0.97, 0.97))
        if ncol == 1:
            ax = axes[i]
        else:
            ax = axes[i,j]
        if nrows == 3 and ncol > 1:
            labels=np.array([['a','b','c'],['d','e','f']])
            ax.text(0.03, 0.95, f'({labels[j,i]})', transform=ax.transAxes, fontsize=self.fs, va='top', ha='left')
        if nrows == 2 and ncol > 1:
            labels=np.array([['a','b'],['c','d']])
            ax.text(0.03, 0.95, f'({labels[j,i]})', transform=ax.transAxes, fontsize=self.fs, va='top', ha='left')
        if ncol == 1:
            labels=np.array(['a','b','c'])
            ax.text(0.03, 0.95, f'({labels[i]})', transform=ax.transAxes, fontsize=self.fs, va='top', ha='left')
        ax.tick_params(axis='both', right=True, which='both',top=True, direction='in', length=4, width=0.5, bottom=True, labelbottom=False,labelsize=self.fs)
        ax.tick_params(axis='both', right=True, which='minor',length=2)
        ax.minorticks_on()
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')
        ax.spines['top'].set_color('black')
        ax.spines['right'].set_color('black')
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.fs)
        ax.set_xlabel(ax.get_xlabel(), fontsize=self.fs)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_ylabel(ytitle, fontsize=self.fs)
        ax.set_facecolor('white')           # white plot area
        ax.patch.set_alpha(1.0)
        white_axes_background(ax)
##        if nrows == 3:
##            if i == 0 and j == 0:
##                ax.set_title('Inputs', fontsize=self.fs)
##            if i == 0 and j == 1:
##                ax.set_title('Outputs', fontsize=self.fs)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')

        if i == nrows-1:
            ax.tick_params(axis='x', which='both', bottom=True,
                           labelbottom=True,labelsize=self.fs) 
            ax.set_xlabel(xtitle)

    def plot_output(self,axes,i,j,x,y,z=None,psym=None,mfc=None,mec='black',ls='-',
                    xtitle=None,ytitle=None,ylim=None,col=None,xlim=None,nlcol=1,
                    label=None,alf=1.0,xlog=False,ylog=False,ncol=2,nrows=3,loc=None,alpha=1.0):
        if xlim is None:
            xlim = [0.0,np.max(self.time)]
        self.plot_setup(axes,i,j,ytitle=ytitle,xtitle=xtitle,xlim=xlim,ylim=ylim,xlog=xlog,ylog=ylog,nrows=nrows,ncol=ncol)
        if ncol == 1:
            ax = axes[i]
        else:
            ax = axes[i,j]
        if z is not None:
            ax.fill_between(x,y,z,alpha=alpha*0.5,color=col,label=label)
        else:
            ax.plot(x, y, linestyle=ls,marker=psym,markerfacecolor=mfc,
                    markeredgecolor=mec,color=col,label=label,alpha=alpha)
        if label is not None:
            
            ax.legend(fontsize=self.fs-3,ncol=nlcol,loc=loc)
if __name__ == "__main__":
    # Standard manual input of waveforms
    #run = dart(machine='step',configuration='double-null')
    # Example case loading JETTO output file timetraces.CDF
    alias = 'feriks/jetto/step/88888/jun2623/seq-1'
    run = dart(jetto=alias)
    run.run()
    run.display()
