from mastu_exhaust_analysis.read_efit import read_uda
from mastu_exhaust_analysis.fluxsurface_tracer import trace_flux_surface
import matplotlib.pyplot as plt
import numpy as np
from pyEquilibrium.equilibrium import equilibrium
def lookup_connection_length(shot,show=False):
    # Read the EFIT++ equilibrium
    efit_data = read_uda(shot=shot, calc_bfield=True)
    connection_L_upper=efit_data['t']*np.nan
    connection_L_lower=efit_data['t']*np.nan
    connection_L_drsep=efit_data['t']*np.nan
    connection_X_drsep=efit_data['t']*np.nan
    for ii in range(len(efit_data['t'])):
        start_r = efit_data['rmidplaneOut'][ii]+5e-3 #1 mm out from EFIT determined Rsep
        start_z = efit_data['z_axis'][ii]
        #start_r = efit_data['upper_xpoint_r'][ii]+5e-3 #1 mm out from EFIT determined Rsep
        #start_z = efit_data['upper_xpoint_z'][ii]
        cut_surface=True #splits flux surface in two and then trace up and down.
        drsep_out = efit_data['dr_sep_out'][ii]
 
        if np.isfinite(start_r):
            surface=trace_flux_surface(efit_data, ii, start_r, start_z, cut_surface=cut_surface)
 
            # Store the connection length in a separate array
            if len(surface)==2:
                connection_L_upper[ii]=surface[0].pardist[-1]
                connection_L_lower[ii]=surface[1].pardist[-1]
            elif len(surface)==1:
                if np.max(surface[0].z)>0:
                    connection_L_upper[ii]=surface[0].pardist[-1]
                else:
                    connection_L_lower[ii]=surface[0].pardist[-1]
 
            if drsep_out<=0:
                connection_L_drsep[ii]=connection_L_lower[ii]
            else:
                connection_L_drsep[ii]=connection_L_upper[ii]
            if show and efit_data['t'][ii] > 0.5:
                # Plot the equilibrium flux
                ax = plt.subplot()
                ax.contour(efit_data['r'], efit_data['z'], efit_data['psi'][ii,:,:], levels=50)
 
                # Plot the flux surfaces
                ax.plot(surface[0].r, surface[0].z, 'r')
                #ax.plot(surface[1].r, surface[1].z, 'r')
                # Plot the wall, which is stored in the flux surface object
                ax.plot(surface[0].wall.xy[0], surface[0].wall.xy[1], 'k')
                ax.set_aspect(1.0)
                print(connection_L_drsep[ii])
                plt.show()
    lc =  -efit_data["RBphi"] * 0.45 * np.pi * 2.1 / 0.15
    plt.plot(efit_data["t"],connection_L_drsep)
    plt.plot(efit_data["t"],lc,'--')
    plt.plot(efit_data["t"],lc*1.5,'--')
    return connection_L_drsep
lookup_connection_length(51368,show=False)
lookup_connection_length(51369,show=False)
plt.show()
