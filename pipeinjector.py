import math
import numpy as np
# --- Global Constants for Molecular Flow Calculations ---
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
UNIVERSAL_GAS_CONSTANT = 8.314     # J/(mol·K)
MOLAR_MASS_D2_KG_PER_MOL = 0.004028 # Molar mass of D2 in kg/mol
class piezo:
    def __init__(self):
        self.loaded = False
    def calculate_deuterium_flow_rate(self,voltage_V, plenum_pressure_bar,valve):
        """
        Calculates the particle flow rate of Deuterium (D2) through a piezo valve
        into a vessel, assuming sonic (choked) flow conditions.

        The function uses a simplified model for the piezo valve's opening and
        standard choked flow equations for ideal gases.

        Args:
            voltage_V (float): The voltage applied to the piezo actuator in Volts.
                               (e.g., 0 to 150 V)
            plenum_pressure_bar (float): The absolute pressure in the plenum
                                         (upstream of the valve) in bar.

        Returns:
            float: The particle flow rate of D2 in D2 molecules per second.
                   Returns 0.0 if inputs are invalid or flow area is zero.
        """

        # --- 1. Define Fixed Parameters and Assumptions ---
        # These values would ideally come from the specific piezo valve's datasheet
        # and the properties of Deuterium.

        # Piezo actuator characteristics (example values)
        MAX_PIEZO_VOLTAGE_V = 150.0  # Maximum voltage for the piezo actuator
        MAX_PIEZO_DISPLACEMENT_UM = 370.0  # Max displacement (for FPA-0500C max displacement ranges between 370 - 440 micrometers)
        VALVE_SEAT_DIAMETER_MM = 1.0  # Diameter of the valve seat/orifice in mm

        # Piezo actuator characteristics based on exp. calibrations
        factor = MAX_PIEZO_DISPLACEMENT_UM/12.807
        grad = 0.1207 * factor
        offset = -5.022 * factor
        corr = 1.0
        if valve == 'hfs_mid_u02':
            corr = 1.0
            factor = MAX_PIEZO_DISPLACEMENT_UM/13.576
            grad = 0.1298 * factor
            offset=-5.894 * factor
        if valve == 'hfs_mid_u08':
            corr = 1.0
            factor = MAX_PIEZO_DISPLACEMENT_UM/12.807
            grad = 0.1331 * factor
            offset = -7.158 * factor
        if valve == 'hfs_mid_l02':
            corr = 1.5
            factor = MAX_PIEZO_DISPLACEMENT_UM/16.081
            grad = 0.1388 * factor
            offset = -4.739 * factor
        if valve == 'hfs_mid_l08':
            corr = 1.5
            factor = MAX_PIEZO_DISPLACEMENT_UM/14.698
            grad = 0.1323 * factor
            offset = -5.147 * factor
        if valve == 'lfsv_bot_l03':
            corr = 1.5
            factor = MAX_PIEZO_DISPLACEMENT_UM/19.56
            grad = 0.1647 * factor
            offset = -5.153 * factor
        if valve == 'lfsv_bot_l09':
            corr = 1.5
            factor = MAX_PIEZO_DISPLACEMENT_UM/14.947
            grad = 0.1314 * factor
            offset = -4.763 * factor
        if valve == 'lfsv_top_u011':
            corr = 2.0
            factor = MAX_PIEZO_DISPLACEMENT_UM/13.903
            grad = 0.1295 * factor
            offset = -5.522 * factor
        if valve == 'lfss_top_u0405':
            corr = 1.3
            factor = MAX_PIEZO_DISPLACEMENT_UM/16.57
            grad = 0.136 * factor
            offset = -3.975 * factor
        if valve == 'lfss_bot_l0405':
            corr = 0.7
            factor = MAX_PIEZO_DISPLACEMENT_UM/11.707
            grad = 0.1089 * factor
            offset = -4.63 * factor
        if valve == 'lfsd_top_u0102':
            corr = 2.0
            factor = MAX_PIEZO_DISPLACEMENT_UM/23.797
            grad = 0.2267 * factor
            offset = -10.208 * factor
        if valve == 'lfsd_bot_l0506':
            corr = 3.2
            factor = MAX_PIEZO_DISPLACEMENT_UM/22.304
            grad = 0.236 * factor
            offset = -13.14 * factor
        if valve == 'lfsd_top_u0506':
            corr = 1.45
            factor = MAX_PIEZO_DISPLACEMENT_UM/19.673
            grad = 0.189 * factor
            offset = -8.677 * factor
        if valve == 'pfr_top_t01':
            factor = MAX_PIEZO_DISPLACEMENT_UM/15.763
            grad = 0.133 * factor
            offset = -4.118 * factor
        if valve == 'pfr_top_t05':
            factor = MAX_PIEZO_DISPLACEMENT_UM/12.62
            grad = 0.122 * factor
            offset = -5.725 * factor
        if valve == 'pfr_bot_b01':
            factor = MAX_PIEZO_DISPLACEMENT_UM/13.241
            grad = 0.1176 * factor
            offset = -4.399 * factor
        if valve == 'pfr_bot_b05':
            factor = MAX_PIEZO_DISPLACEMENT_UM/13.241
            grad = 0.1176 * factor
            offset = -4.399 * factor

        # Deuterium (D2) gas properties
        DEUTERIUM_K = 1.38  # Specific heat ratio (adiabatic index) for D2
        DEUTERIUM_R = 2064.0  # Specific gas constant for D2 in J/(kg·K) - This is for the piezo calculation
        UPSTREAM_TEMPERATURE_K = 293.15  # Upstream gas temperature in Kelvin (20 °C)
        MASS_D2_MOLECULE_KG = 6.688e-27 # Mass of one D2 molecule in kg

        # Flow characteristics
        DISCHARGE_COEFFICIENT = 0.8  # Cd, accounts for real flow effects (0.6 to 1.0)

        # --- 2. Input Validation ---
        if voltage_V < 0 or plenum_pressure_bar <= 0:
            return 0.0

        # --- 3. Calculate Piezo Displacement (Gap Height) ---
        # Assume a linear relationship between voltage and displacement,
        # capped at the maximum displacement.
        # Note, the variable corr is defined to give best agreement to FIG
        # based on a fixed piezo displacement
        piezo_displacement_um = (voltage_V * grad + offset) * corr
        piezo_displacement_um = max(0.0, piezo_displacement_um)
        piezo_displacement_um = min(piezo_displacement_um, MAX_PIEZO_DISPLACEMENT_UM)
        # Convert displacement to meters for area calculation
        gap_height_m = piezo_displacement_um * 1e-6

        # If the gap is effectively zero, there's no flow
        if gap_height_m <= 0:
            return 0.0

        # --- 4. Calculate Effective Flow Area (A) ---
        # Assuming an annular gap where flow occurs around a central seat.
        # Area = pi * (seat diameter) * (gap height)
        valve_seat_diameter_m = VALVE_SEAT_DIAMETER_MM * 1e-3
        effective_flow_area_m2 = math.pi * valve_seat_diameter_m * gap_height_m

        # --- 5. Convert Plenum Pressure to Pascals ---
        plenum_pressure_Pa = plenum_pressure_bar * 1e5

        # --- 6. Calculate the Constant Term for Choked Flow Formula ---
        # This term depends only on gas properties and temperature
        # (2 / (k+1)) ^ ((k+1) / (k-1))
        choked_flow_factor_exponent = (DEUTERIUM_K + 1) / (DEUTERIUM_K - 1)
        choked_flow_factor_base = 2 / (DEUTERIUM_K + 1)
        choked_flow_factor_term = choked_flow_factor_base ** choked_flow_factor_exponent

        # Square root term: sqrt(k / (R * T_u) * choked_flow_factor_term)
        sqrt_term = math.sqrt(
            (DEUTERIUM_K / (DEUTERIUM_R * UPSTREAM_TEMPERATURE_K)) * choked_flow_factor_term
        )

        # --- 7. Calculate Mass Flow Rate (m_dot) using Choked Flow Formula ---
        # m_dot = Cd * A * P_u * sqrt_term
        mass_flow_rate_kg_s = (
            DISCHARGE_COEFFICIENT * effective_flow_area_m2 * plenum_pressure_Pa * sqrt_term
        )

        # --- 8. Convert Mass Flow Rate to Particle Flow Rate ---
        particle_flow_rate_per_second = mass_flow_rate_kg_s / MASS_D2_MOLECULE_KG

        return particle_flow_rate_per_second

    def calculate_molecular_conductance_pipe(self,pipe_diameter_m, pipe_length_m, temperature_K):
        """
        Calculates the molecular flow conductance of a long cylindrical pipe.

        Args:
            pipe_diameter_m (float): Diameter of the pipe in meters.
            pipe_length_m (float): Length of the pipe in meters.
            temperature_K (float): Temperature of the gas in Kelvin.

        Returns:
            float: Conductance in Pa·m³/s (or mbar·L/s if pressures are mbar and volumes L).
                   This is the pV conductance.
        """
        if pipe_diameter_m <= 0 or pipe_length_m <= 0 or temperature_K <= 0:
            return 0.0

        # Molecular flow conductance formula for a long cylindrical pipe
        # C_pV = (pi * D^3) / (12 * L) * sqrt(2 * pi * R_universal * T / M_molar)
        # R_universal is in J/(mol·K)
        # M_molar is in kg/mol
        # This will give conductance in m^3/s * Pa = Pa·m³/s

        conductance = (math.pi * (pipe_diameter_m ** 3)) / (12 * pipe_length_m) * \
                      math.sqrt((2 * math.pi * UNIVERSAL_GAS_CONSTANT * temperature_K) / MOLAR_MASS_D2_KG_PER_MOL)
        return conductance

    def simulate_gas_flow_with_pipe_delay(self,voltages_over_time,plenum_pressure_bar,pipe_length_m,pipe_diameter_mm,
                                          vessel_pressure_mbar,time_step_s,valve):
        """
        Simulates the particle flow rate out of a pipe into a vessel,
        accounting for the time delay introduced by the pipe's volume and molecular conductance.

        Args:
            voltages_over_time (List[float]): A list of voltages applied to the piezo
                                              at each time step.
            plenum_pressure_bar (float): The constant absolute pressure in the plenum (upstream).
            pipe_length_m (float): The length of the pipe in meters.
            pipe_diameter_mm (float): The diameter of the pipe in millimeters.
            vessel_pressure_mbar (float): The constant absolute pressure in the vessel in mbar.
            time_step_s (float): The duration of each simulation time step in seconds.

        Returns:
            List[float]: An array of particle flow rates (D2/second) out of the pipe
                         into the vessel at each time step.
        """

        # --- Define Constants (re-using from calculate_deuterium_flow_rate or global) ---
        UPSTREAM_TEMPERATURE_K = 293.15  # Assumed temperature of gas in pipe
        MASS_D2_MOLECULE_KG = 6.688e-27 # Mass of one D2 molecule in kg

        # --- Convert pipe diameter to meters ---
        pipe_diameter_m = pipe_diameter_mm * 1e-3

        # --- Calculate Pipe Volume ---
        pipe_volume_m3 = math.pi * (pipe_diameter_m / 2)**2 * pipe_length_m

        # --- Calculate Pipe Molecular Conductance ---
        # This conductance is for pV flow in Pa·m³/s
        pipe_conductance_Pa_m3_s = self.calculate_molecular_conductance_pipe(
            pipe_diameter_m, pipe_length_m, UPSTREAM_TEMPERATURE_K
        )

        # --- Initial State of the Pipe ---
        # Assume pipe initially at vessel pressure
        current_pipe_pressure_Pa = vessel_pressure_mbar * 1e-3 * 1e5 # Convert mbar to Pa

        # --- Simulation Loop ---
        outflow_rates_D2_per_s = []

        for voltage in voltages_over_time:
            # 1. Calculate inflow from piezo valve (particles/s)
            if voltage < 200:
                inflow_piezo_D2_per_s = self.calculate_deuterium_flow_rate(voltage, plenum_pressure_bar,valve)
            else:
                inflow_piezo_D2_per_s = voltage
            # 2. Convert piezo inflow to pV flow (Pa·m³/s)
            # Q_pV = N * k_B * T
            inflow_piezo_pV_Pa_m3_s = inflow_piezo_D2_per_s * BOLTZMANN_CONSTANT * UPSTREAM_TEMPERATURE_K

            # 3. Calculate outflow from pipe (pV flow in Pa·m³/s)
            # Q_out_pV = C_pipe * (P_pipe - P_vessel)
            vessel_pressure_Pa = vessel_pressure_mbar * 1e-3 * 1e5 # Convert mbar to Pa
            outflow_pipe_pV_Pa_m3_s = pipe_conductance_Pa_m3_s * \
                                      (current_pipe_pressure_Pa - vessel_pressure_Pa)

            # Ensure outflow is not negative (i.e., no backflow from vessel if pipe pressure is too low)
            outflow_pipe_pV_Pa_m3_s = max(0.0, outflow_pipe_pV_Pa_m3_s)

            # 4. Update pipe pressure based on net pV flow (Euler integration)
            # dP/dt = (Q_in_pV - Q_out_pV) / V_pipe
            # Delta P = (Q_in_pV - Q_out_pV) / V_pipe * Delta t
            net_pV_flow_Pa_m3_s = inflow_piezo_pV_Pa_m3_s - outflow_pipe_pV_Pa_m3_s
            delta_pressure_Pa = (net_pV_flow_Pa_m3_s / pipe_volume_m3) * time_step_s
            current_pipe_pressure_Pa += delta_pressure_Pa

            # Ensure pipe pressure doesn't drop below vessel pressure (or absolute zero)
            current_pipe_pressure_Pa = max(vessel_pressure_Pa, current_pipe_pressure_Pa)
            current_pipe_pressure_Pa = max(0.0, current_pipe_pressure_Pa) # Absolute minimum

            # 5. Convert outflow from pipe to particle flow rate (D2/second)
            # N_out = Q_out_pV / (k_B * T)
            final_outflow_D2_per_s = outflow_pipe_pV_Pa_m3_s / (BOLTZMANN_CONSTANT * UPSTREAM_TEMPERATURE_K)

            outflow_rates_D2_per_s.append(final_outflow_D2_per_s)
        
        return np.array(outflow_rates_D2_per_s)

if __name__ == "__main__":

    # --- Example Usage for Pipe Flow Simulation ---

    # Define simulation parameters
    sim_plenum_pressure_bar = 1.5
    sim_pipe_length_m = 0.5  # 50 cm long pipe
    sim_pipe_diameter_mm = 6.0 # 6 mm diameter pipe
    sim_vessel_pressure_mbar = 1e-7 # Very low vacuum
    sim_time_step_s = 0.01 # 10 ms time step

    calc = piezo()

    # Simulate a pulse voltage input (e.g., valve opens for a short period)
    # 0.1s at 100V, then back to 0V for 0.4s
    pulse_voltages = [100.0] * int(0.1 / sim_time_step_s) + [0.0] * int(0.4 / sim_time_step_s)
    sim_outflow_pulse = calc.simulate_gas_flow_with_pipe_delay(
        pulse_voltages,
        sim_plenum_pressure_bar,
        sim_pipe_length_m,
        sim_pipe_diameter_mm,
        sim_vessel_pressure_mbar,
        sim_time_step_s
    )
    import matplotlib.pyplot as plt
    import numpy as np
    plt.plot(np.linspace(0,1.4,len(pulse_voltages)),sim_outflow_pulse)
    plt.show()
