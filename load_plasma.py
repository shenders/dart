import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from skimage import measure
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
def fit_mtanh(R,h,w,a,b,R0):
    sigma = 0.25 * w
    z = (R0 - R) / sigma
    G = h - b + (a * sigma) * z
    iL = 1 + np.exp(-z)
    return (G / iL) + b

def line_int_2d(x, y, arr):
    
    ans = 0.0
    
    for i in np.arange(len(x)-1):

        dxsq=(x[i]-x[i+1])**2 # x component of distance
        dysq=(y[i]-y[i+1])**2# y component of distance
        ans=ans+0.5*(arr[i]+arr[i+1])*np.sqrt(dxsq+dysq)
  
    return ans

class plasma:
    def __init__(self,shot=49261,device=None, progress_bar = None,root=None):
        if device.lower() == 'mast' or device.lower() == 'mast-u':
            self.mastu(shot, progress_bar = progress_bar,root=root)
    def mastu(self,shot, progress_bar = None,root=None):
        if progress_bar is not None:
            progress_bar["value"] = 5.0
            root.update_idletasks()  # Refresh GUI
        import pyuda
        client = pyuda.Client()
        try:
            # Set IP time base as common time base
            src       = 'epm'
            data_Ip   = client.get('/amc/plasma_current', shot)
            self.time = np.array(data_Ip.time.data)
            self.Ip   = np.array(data_Ip.data)*1e3
            idt       = np.where(self.time < 1.0)
            self.time = self.time[idt]
            self.Ip   = self.Ip[idt]
        except Exception as e:
            self.time = np.linspace(0,1,100)
            self.Ip   = np.zeros(len(self.time))
            print("Plasma current loading error:",e)
        
        if progress_bar is not None:
            progress_bar["value"] = 10.0
            root.update_idletasks()  # Refresh GUI
        try:
            # Load in line integrated density
            data_dens  = self.common_time(client,'/esm/density/nebar',shot)
            self.dens  = data_dens(self.time)
        except Exception as e:
            self.dens  = np.zeros(len(self.time))
            print("Interferometer loading error:",e)
        if progress_bar is not None:
            progress_bar["value"] = 12.0
            root.update_idletasks()  # Refresh GUI
        try:
            # Load in gas flow traces
            data_gas       = self.common_time(client,'/xdc/flow/s/fuelling_req_flow',shot)
            data_hfs_gas   = self.common_time(client,'/xdc/flow/s/hfs_mid_flow',shot)
            data_lfsb_gas  = self.common_time(client,'/xdc/flow/s/lfsv_bot_flow',shot)
            data_lfst_gas  = self.common_time(client,'/xdc/flow/s/lfsv_top_flow',shot)
            data_divl_gas  = self.common_time(client,'/xdc/flow/s/lfsd_bot_flow',shot)
            data_divu_gas  = self.common_time(client,'/xdc/flow/s/lfsd_top_flow',shot)
            data_imp       = self.common_time(client,'/xdc/flow/s/impurity_1_req_flow',shot)
            self.gas       = data_gas(self.time)*1e21
            self.imp       = data_imp(self.time)*1e21
            self.hfsgas    = data_hfs_gas(self.time)*1e21
            self.lfsgas    = data_lfsb_gas(self.time)*1e21+data_lfst_gas(self.time)*1e21
            self.ldvgas    = data_divl_gas(self.time)*1e21
            self.udvgas    = data_divu_gas(self.time)*1e21
        except Exception as e:
            self.gas       = np.zeros(len(self.time))
            self.imp       = np.zeros(len(self.time))
            self.hfsgas    = np.zeros(len(self.time))
            self.lfsgas    = np.zeros(len(self.time))
            self.udvgas    = np.zeros(len(self.time))
            print("XDC gas waveform loading error:",e)
        if progress_bar is not None:
            progress_bar["value"] = 14.0
            root.update_idletasks()  # Refresh GUI
        try:
            # Load in FIG pressures
            data_p0        = self.common_time(client,'/aga/HL11',shot)
            data_p0mid     = self.common_time(client,'/aga/HM12',shot)
            data_p0up      = self.common_time(client,'/aga/HU08',shot)
            self.p0        = data_p0(self.time)
            self.p0mid     = data_p0mid(self.time)
            self.p0up      = data_p0up(self.time)
        except Exception as e:
            self.p0        = np.zeros(len(self.time))
            self.p0mid     = np.zeros(len(self.time))
            self.p0up      = np.zeros(len(self.time))
            print("FIG loading error:",e)
        if progress_bar is not None:
            progress_bar["value"] = 16.0
            root.update_idletasks()  # Refresh GUI

        try:
            # Load in radiation
            if shot == 49397 or shot == 49400:
                data_rad   = self.common_time(client,'/abm/core/prad',49392)
            else:
                data_rad   = self.common_time(client,'/abm/core/prad',shot)
            self.rad    = data_rad(self.time)
        except Exception as e:
            self.rad    = np.zeros(len(self.time))
            print("Bolometry loading error:",e)

        if progress_bar is not None:
            progress_bar["value"] = 18.0
            root.update_idletasks()  # Refresh GUI
        
        try:
            # Load in neutral beam power
            data_nbi    = self.common_time(client,'/anb/sum/power',shot)
            self.Paux   = data_nbi(self.time)*1e6
        except Exception as e:
            self.Paux   = np.zeros(len(self.time))
            print("NBI loading error:",e)

        try:
            # Load in IR peak values
            data_IRt2 = self.common_time(client,'/ait/t2lt3l_std/heatflux_peak_value',shot)
            self.IRt2 = data_IRt2(self.time)
        except Exception as e:
            print("IR tile 2 loading error:",e)
            self.IRt2 = np.zeros(len(self.time))

        try:
            data_IRt5 = self.common_time(client,'/aiv/t5l_std/heatflux_peak_value',shot)
            self.IRt5 = data_IRt5(self.time)
        except Exception as e:
            print("IR tile 5 loading error:",e)
            self.IRt5 = np.zeros(len(self.time))
            
        if progress_bar is not None:
            progress_bar["value"] = 20.0
            root.update_idletasks()  # Refresh GUI
        
        try:
            # Calculate Psep using equilibrium to calculate stored energy and ohmic power
            status      = client.get('/'+src+'/equilibriumStatusInteger', shot).data
            efittime    = client.get('/'+src+'/time', shot).data[status == 1]
            psi         = np.transpose(client.get('/'+src+'/output/profiles2D/poloidalFlux', shot).data,(0, 2, 1))
            psi_n       = np.transpose(client.get('/'+src+'/output/profiles2D/psiNorm', shot).data,(0, 2, 1))
            psi         = psi[status == 1,:,:]
            psi_n       = psi_n[status == 1,:,:]
            if progress_bar is not None:
                progress_bar["value"] = 22.0
                root.update_idletasks()  # Refresh GUI
            r           = client.get('/'+src+'/output/profiles2D/r', shot).data
            z           = client.get('/'+src+'/output/profiles2D/z', shot).data
            axisr       = client.get('/'+src+'/output/globalParameters/magneticAxis/R', shot).data[status == 1]
            axisz       = client.get('/'+src+'/output/globalParameters/magneticAxis/Z', shot).data[status == 1]
            bpol        = np.transpose(client.get('/'+src+'/output/profiles2D/Bpol', shot).data, (0, 2, 1))
            bpol        = bpol[status == 1,:,:]
            if progress_bar is not None:
                progress_bar["value"] = 29.0
                root.update_idletasks()  # Refresh GUI
            w           = client.get('/'+src+'/output/globalParameters/plasmaEnergy', shot).data[status == 1]
            smooth_dt   = 0.015
            window_size = np.int(smooth_dt / np.median(np.gradient(efittime)))   
            if window_size % 2 == 0:
                window_size = window_size + 1   
            wsm         = savgol_filter(w, window_size, window_size-1)    
            wdot        = np.gradient(wsm) / np.gradient(efittime)
            data_wdot   = interp1d(efittime,wdot,kind='linear',fill_value='extrapolate')
            data_pohm   = interp1d(efittime,self.calc_pohm(efittime,psi,psi_n,r,z,axisr,axisz,bpol),kind='linear',fill_value='extrapolate')
            if progress_bar is not None:
                progress_bar["value"] = 36.0
                root.update_idletasks()  # Refresh GUI
            # Calculate plasma geometry parameters
            rtar        = client.get('/'+src+'/output/separatrixGeometry/strikepointR', shot).data[status == 1,1]
            amin        = client.get('/'+src+'/output/separatrixGeometry/minorRadius', shot).data[status == 1]
            kappa       = client.get('/'+src+'/output/separatrixGeometry/elongation', shot).data[status == 1]
            bt          = client.get('/'+src+'/output/globalParameters/bvacRgeom', shot).data[status == 1]
            drsep       = client.get('/epm/output/separatrixGeometry/drsepOut', shot).data[status==1]
            volume      = client.get('/epm/output/globalParameters/plasmavolume', shot).data[status==1]
            if progress_bar is not None:
                progress_bar["value"] = 43.0
                root.update_idletasks()  # Refresh GUI
            data_kp     = interp1d(efittime,np.array(kappa),kind='linear',fill_value='extrapolate')
            data_rt     = interp1d(efittime,np.array(rtar),kind='linear',fill_value='extrapolate')
            data_r0     = interp1d(efittime,np.array(axisr),kind='linear',fill_value='extrapolate')
            data_b0     = interp1d(efittime,-np.array(bt),kind='linear',fill_value='extrapolate')
            data_am     = interp1d(efittime,np.array(amin),kind='linear',fill_value='extrapolate')
            data_dr     = interp1d(efittime,np.array(drsep),kind='linear',fill_value='extrapolate')
            data_vl     = interp1d(efittime,np.array(volume),kind='linear',fill_value='extrapolate')
            self.am     = data_am(self.time)
            self.kp     = data_kp(self.time)
            self.R0     = data_r0(self.time)
            self.B0     = data_b0(self.time)
            self.drsep  = data_dr(self.time)
            self.Rt     = data_rt(self.time)
            self.vol    = data_vl(self.time)
        except Exception as e:
            self.am     = np.zeros(len(self.time))
            self.kp     = np.zeros(len(self.time))
            self.R0     = np.zeros(len(self.time))
            self.B0     = np.zeros(len(self.time))
            self.drsep  = np.zeros(len(self.time))
            self.Rt     = np.zeros(len(self.time))
            self.vol    = np.zeros(len(self.time))
            print("EFIT loading error:",e)
        
        if progress_bar is not None:
            progress_bar["value"] = 50.0
            root.update_idletasks()  # Refresh GUI
        try:
            self.Psep   = data_nbi(self.time)*1e6+data_pohm(self.time)-data_rad(self.time)-data_wdot(self.time)
            self.Pohm   = data_pohm(self.time)
            self.frad   = data_rad(self.time)/(data_nbi(self.time)*1e6+data_pohm(self.time))
        except Exception as e:
            self.Psep   = np.zeros(len(self.time))
            self.Pohm   = np.zeros(len(self.time))
            self.frad   = np.zeros(len(self.time))
            print("Psep loading error:",e)

        if progress_bar is not None:
            progress_bar["value"] = 55.0
            root.update_idletasks()  # Refresh GUI
            
        try:
            # Calculate fitted electron density and temperature pedestal profiles
            fittime     = client.get('/apf/core/mtanh/lfs/time', shot).data
            rarr        = np.arange(1.25,1.5,0.005)
            te_fit      = np.zeros((len(fittime),len(rarr)))
            ne_fit      = np.zeros((len(fittime),len(rarr)))
            tb          = client.get('/apf/core/mtanh/lfs/t_e/background_level', shot).data
            th          = client.get('/apf/core/mtanh/lfs/t_e/pedestal_height', shot).data
            tr0         = client.get('/apf/core/mtanh/lfs/t_e/pedestal_location', shot).data
            ta          = client.get('/apf/core/mtanh/lfs/t_e/pedestal_top_gradient', shot).data
            tw          = client.get('/apf/core/mtanh/lfs/t_e/pedestal_width', shot).data
            nb          = client.get('/apf/core/mtanh/lfs/n_e/background_level', shot).data
            nh          = client.get('/apf/core/mtanh/lfs/n_e/pedestal_height', shot).data
            nr0         = client.get('/apf/core/mtanh/lfs/n_e/pedestal_location', shot).data
            na          = client.get('/apf/core/mtanh/lfs/n_e/pedestal_top_gradient', shot).data
            nw          = client.get('/apf/core/mtanh/lfs/n_e/pedestal_width', shot).data
            for i,t in enumerate(fittime):
                te_fit[i,:] = fit_mtanh(rarr,th[i],tw[i],ta[i],tb[i],tr0[i])
                ne_fit[i,:] = fit_mtanh(rarr,nh[i],nw[i],na[i],nb[i],nr0[i])
            self.te_fit = np.zeros((len(self.time),len(rarr)))
            self.ne_fit = np.zeros((len(self.time),len(rarr)))
            for i,r in enumerate(rarr):
                te_fit_int       = interp1d(fittime,te_fit[:,i],kind='linear',fill_value='extrapolate')
                ne_fit_int       = interp1d(fittime,ne_fit[:,i],kind='linear',fill_value='extrapolate')
                self.te_fit[:,i] = te_fit_int(self.time)
                self.ne_fit[:,i] = ne_fit_int(self.time)
            self.r_fit  = rarr
        except Exception as e:
            self.r_fit  = np.zeros(len(self.time))
            self.te_fit = np.zeros((len(self.time),len(self.r_fit)))
            self.ne_fit = np.zeros((len(self.time),len(self.r_fit)))
            print("mtanh TS loading error:",e)

        if progress_bar is not None:
            progress_bar["value"] = 60.0
            root.update_idletasks()  # Refresh GUI
        
        try:
            # Load in target grazing angle
            targ_angle = client.get('/esm/fluxexp/lower/cos_total_angle_target', shot)
            tt         = targ_angle.time.data
            data       = np.array(targ_angle.data)
            targ_rr    = np.array(client.get('/esm/fluxexp/lower/R_target', shot).data)
            target_angle = np.zeros(len(tt))
            for i,t in enumerate(tt):
                r_target = data_rt(t)
                angs     = interp1d(targ_rr[i,:],90.0-np.arccos(data[i,:])*180.0/np.pi,kind='linear',fill_value='extrapolate')
                target_angle[i] = angs(r_target)
            data_alft = interp1d(tt,target_angle,kind='linear',fill_value='extrapolate')
            self.alft   = data_alft(self.time)
        except Exception as e:
            self.alft  = np.zeros(len(self.time))
            print("LP loading error:",e)

        if progress_bar is not None:
            progress_bar["value"] = 65.0
            root.update_idletasks()  # Refresh GUI

    def common_time(self,client,trace,shot):
        data  = client.get(trace, shot)
        return interp1d(np.array(data.time.data),np.array(data.data),kind='linear',fill_value='extrapolate')

    def calc_pohm(self,time,psi,psi_n,r,z,axisr,axisz,bpol):
        mu0    = np.pi * 4.0E-7
        dpsidi = np.gradient(psi, axis=0)
        dtdi   = np.gradient(time)
        
        dpsidt = dpsidi * 0.0
        for i in np.arange(len(time)):
            dpsidt[i,:,:] = dpsidi[i,:,:] / dtdi[i]
        
        vpoint = -2.0 * np.pi * dpsidt
        r_interp = interp1d(np.linspace(0, len(r)-1, len(r)), r)
        z_interp = interp1d(np.linspace(0, len(z)-1, len(z)), z)

        pohm  = time*0.0

        for i in np.arange(len(time)):

            contours = measure.find_contours(psi_n[i,:,:],0.995)
            
            if len(contours) > 0:

                contour_r = []
                contour_z = []
                mean_contour_dist = np.zeros(len(contours))

                for contour in contours:    
                    contour_r.append(r_interp(contour[:, 1]))
                    contour_z.append(z_interp(contour[:, 0]))

                for j in np.arange(len(contours)):
                    mean_contour_dist[j] = np.mean((contour_r[j] - axisr[i])**2 + 
                                                   (contour_z[j] - axisz[i])**2)
                    
                contour_indx = np.argmin(np.abs(mean_contour_dist))

                bpol_interp = RectBivariateSpline(r, z, np.transpose(bpol[i, :, :]))
                psi_interp = RectBivariateSpline(r, z, np.transpose(psi[i, :, :]))
                vpoint_interp = RectBivariateSpline(r, z, np.transpose(vpoint[i, :, :]))

                lcfs_r = contour_r[contour_indx]
                lcfs_z = contour_z[contour_indx]
                lcfs_bpol = contour_r[contour_indx]*0.0
                lcfs_psi = contour_r[contour_indx]*0.0
                lcfs_vpoint = contour_r[contour_indx]*0.0

                for k in np.arange(len(lcfs_bpol)):
                    lcfs_bpol[k] = bpol_interp(lcfs_r[k], lcfs_z[k])
                    lcfs_psi[k] = psi_interp(lcfs_r[k], lcfs_z[k])
                    lcfs_vpoint[k] = vpoint_interp(lcfs_r[k], lcfs_z[k])

                pohm[i] = line_int_2d(lcfs_r, lcfs_z, lcfs_bpol * lcfs_vpoint) / mu0
        return pohm
    def display(self,canvas=None):
        # Create figure and axes
        plt.rcParams['font.family'] = 'serif'  # Choose font family
        plt.rcParams['font.serif'] = ['Arial']  # Specify font
        fig, axes = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(11, 8))
        fig.subplots_adjust(hspace=0.09,left=0.1,top=0.96, bottom=0.1,right=0.96)
        self.fs = 16
        def_col = ['black','#E41A1C' ,'#0072B2', '#D95F02', '#4DAF4A', '#377EB8', '#A65628']
        self.plot_output(axes,0,0,self.time,self.Ip/1e6,ylim=[0,np.max(self.Ip)/1e6],label=f'Plasma current',loc='upper right',ncol=1,col=def_col[1],ytitle='MW/m,MA')
        self.plot_output(axes,1,0,self.time,self.p0,ylim=[0,1.5],col=def_col[0],label=f'Pressure [Pa]',loc='upper right',ncol=1,ytitle=r'10$^{19}$ m$^{-3}$')
        self.plot_output(axes,2,0,self.time,self.Psep/1e6,ylim=[0,4],label=r'Psep',col=def_col[0],xtitle='Time [s]')
        self.plot_output(axes,0,1,self.time,self.hfsgas/1e21,ylim=[0,5],label='Fuelling gas',loc='upper right',ncol=1,col=def_col[0],ytitle=r'1e21 #/s')       
        self.plot_output(axes,0,1,self.time,self.lfsgas/1e21,ylim=[0,5],label='Imp gas',loc='upper right',ncol=1,col=def_col[0],ytitle='1e21 #/s')
        self.plot_output(axes,1,1,self.time,self.rad/1e6,ylim=[0,4],col=def_col[1],xtitle='Time [s]',label=f'Radiation',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,1,1,self.time,self.Paux/1e6,ylim=[0,4],col=def_col[1],xtitle='Time [s]',label=f'NBI',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,1,1,self.time,self.Pohm/1e6,ylim=[0,4],col=def_col[1],xtitle='Time [s]',label=f'Pohm',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,2,1,self.time,self.R0,ylim=[0,3],col=def_col[0],xtitle='Time [s]',label=f'R0',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,2,1,self.time,self.am,ylim=[0,3],col=def_col[1],xtitle='Time [s]',label=f'am',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,2,1,self.time,self.B0,ylim=[0,3],col=def_col[2],xtitle='Time [s]',label=f'B0',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,2,1,self.time,self.kp,ylim=[0,3],col=def_col[1],xtitle='Time [s]',label=f'kappa',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        self.plot_output(axes,2,1,self.time,self.alft,ylim=[0,10],col=def_col[4],xtitle='Time [s]',label=f'alft',loc='upper right',ncol=1,ytitle=r'MWm$^{-2}$, eV')
        #print(f"Rt = {self.Rt} \n R0 = {self.R0} \n am = {self.am} \n kappa = {self.kp} \n B0 = {self.B0} \n ")
        if canvas is None:
            plt.show()
        else:
            canvas.figure = fig
            canvas.draw()
    def plot_setup(self,axes,i,j,ytitle=None,xtitle=None,xlim=[0,10],xlog=False,ylog=False,ylim=None,nrows=3):
        bbox_props = dict(boxstyle='square',  facecolor=(0.97, 0.97, 0.97))
        ax = axes[i,j]
        if nrows == 3:
            labels=np.array([['a','b','c'],['d','e','f']])
        if nrows == 2:
            labels=np.array([['a','b'],['c','d']])
        ax.text(0.03, 0.95, f'({labels[j,i]})', transform=ax.transAxes, fontsize=self.fs, va='top', ha='left')
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
        if nrows == 3:
            if i == 0 and j == 0:
                ax.set_title('Inputs', fontsize=self.fs)
            if i == 0 and j == 1:
                ax.set_title('Outputs', fontsize=self.fs)
        if xlog:
            ax.set_xscale('log')
        if ylog:
            ax.set_yscale('log')
        if i == nrows-1:
            ax.tick_params(axis='x', which='both', bottom=True,
                           labelbottom=True,labelsize=self.fs) 
            ax.set_xlabel(xtitle)

    def plot_output(self,axes,i,j,x,y,z=None,psym=None,mfc=None,mec='black',ls='-',
                    xtitle=None,ytitle=None,ylim=None,col=None,xlim=None,
                    label=None,alf=1.0,xlog=False,ylog=False,ncol=2,nrows=3,loc=None,alpha=0.5):
        xlim = [0.0,np.max(self.time)]
        self.plot_setup(axes,i,j,ytitle=ytitle,xtitle=xtitle,xlim=xlim,ylim=ylim,xlog=xlog,ylog=ylog,nrows=nrows)
        ax = axes[i,j]
        if z is not None:
            ax.fill_between(x,y,z,alpha=alpha,color=col,label=label)
        else:
            ax.plot(x, y, linestyle=ls,marker=psym,markerfacecolor=mfc,
                    markeredgecolor=mec,color=col,label=label)
        if label is not None:
            
            ax.legend(fontsize=self.fs-3,ncol=ncol,loc=loc)        
if __name__ == "__main__":
    run = plasma(shot=51514,device='MAST-U')
    run.display()
