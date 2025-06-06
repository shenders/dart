from dart import dart
from scipy.interpolate import interp1d
import numpy as np


# If using JETTO, set alias to not None
##alias = 'feriks/jetto/step/88888/jun2623/seq-1'
alias = None
#Initialise DART
dart      = dart(jetto=alias)

if alias is None:
    # Setup inputs user defined
    # Input time base
    tarr      = np.array([0.0   , 2.45   , 2.5    ,  7.5    , 7.55   , 10.0   ])
    # Plasma current
    Ip_vals   = np.array([2.0e6 , 20.0e6 , 21.0e6 ,  21e6   , 20.0e6 , 2.0e6  ])
    # Fusion power
    Pfus_vals = np.array([0.0   , 0.0    , 1.7e9  ,  1.7e9  , 0.5e9  , 0.0    ])
    # Auxiliary power
    Paux_vals = np.array([14.0e6, 100.0e6, 150.0e6,  150.0e6, 150.0e6, 14.0e6 ])
    # Core radiation fraction
    frad_vals = np.array([0.3   , 0.3    , 0.7   ,  0.7   , 0.5    , 0.3    ])
    # Separatrix density
    nsep_vals = np.array([1.0e19, 3.0e19 , 4.2e19 ,  4.2e19 , 4.2e19 , 1.0e19 ])
    # Target grazing angle
    alft_vals = np.radians(np.array([0.5   , 4.0    , 4.0    ,  4.0    , 4.0    , 0.5    ]))
    # Requested detachment qualifier value
    qdet_vals = np.array([1.0   , 1.0    , 1.0    ,  1.0    , 1.0    , 1.0    ])
    Ip, Pfus, Paux, frad, nsep, alft, qdet0 = [interp1d(tarr, vals, kind='linear', fill_value='extrapolate') 
                                               for vals in [Ip_vals, Pfus_vals, Paux_vals, frad_vals, nsep_vals, alft_vals, qdet_vals]]
    # Interpolate onto finer time grid
    dart.time  = dart.log_vector(0.0, 10, num=100, linear=True)
    dart.Ip, dart.Pfus, dart.Paux, dart.frad, dart.nsep, dart.alft, dart.qdet0 = [f(dart.time) for f in [Ip, Pfus, Paux, frad, nsep, alft, qdet0]]
    # Total tilting of the divertor target
    dart.tilt = np.radians(1.5)
    # Input Power crossing separatrix 
    dart.Ploss = dart.Pfus/5.0 + dart.Paux
    dart.Psep  = dart.Ploss * (1-dart.frad)
    # Geometric radius, magnetic field at Rgeo, minor radius, and elongation
    dart.R0, dart.B0, dart.am, dart.kp = 3.6, 3.2, 2.0, 2.98
    # Set ratio of sub to divertor pressure
    dart.dp = 2.0
# SOL power fraction to the divertor
dart.fdiv = 0.4
# Target radius, pump speed, and wall temperature
dart.Rt, dart.Spump, dart.Twall = 5.6, 24.0, 580.0

# Run the calculation
dart.run()

# Plot the results
dart.display()
