from typing import List
import numpy as np
import pandas as pd
import time
import concurrent.futures
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from src.setup_dataclass import PresetValues, PropulsionSpecs
from src.internal_dataclass import PhysicalConstants, MissionParameters, AircraftAnalysisResults, PlaneState, PhaseType, MissionConfig, Aircraft
from src.propulsion import thrust_analysis, determine_max_thrust, thrust_reverse_solve, SoC2Vol
from src.vsp_analysis import  loadAnalysisResults

# use climb_ratio as vtol_level

WP1 = [0,  25,  0]   # Waypoint 1
WP2 = [0, 250,  0]   # Waypoint 2 & Transition
WP3 = [400, 800, 0]  # Waypoint 3
WP4 = [-400,800, 0]  # Waypoint 4

## Constant values
g = PhysicalConstants.g
rho = PhysicalConstants.rho

class MissionAnalyzer():
    def __init__(self, 
                 analResult:AircraftAnalysisResults, 
                 missionParam:MissionParameters, 
                 presetValues:PresetValues,
                 propulsionSpecs : PropulsionSpecs,
                 dt:float=0.1):

        self.analResult = self._convert_units(analResult, presetValues)
        self.aircraft = self.analResult.aircraft
        self.missionParam = missionParam
        self.presetValues = presetValues
        self.propulsionSpecs = propulsionSpecs
        self.dt = dt
        self.m_fuel = max(self.missionParam.m_takeoff - self.analResult.m_empty,0) 

        self.convert_propellerCSV_to_ndarray(self.missionParam.propeller_data_path)
        self.convert_batteryCSV_to_ndarray(self.propulsionSpecs.battery_data_path)
        self.clearState()
        self.setAuxVals()

    def _convert_units(self, results: AircraftAnalysisResults, presetValues:PresetValues) -> AircraftAnalysisResults:
        # Create new aircraft instance with converted units
        new_aircraft = Aircraft(
            # Mass conversions (g to kg)
            m_fuselage=results.aircraft.m_fuselage / 1000,
            wing_area_blocked_by_fuselage= results.aircraft.wing_area_blocked_by_fuselage / 1e6, 
            # Density conversions (g/mm³ to kg/m³)
            wing_density=results.aircraft.wing_density * 1e9,
            
            # Length conversions (mm to m)
            mainwing_span=results.aircraft.mainwing_span / 1000,
            
            # These are ratios, no conversion needed
            mainwing_AR=results.aircraft.mainwing_AR,
            mainwing_taper=results.aircraft.mainwing_taper,
            mainwing_twist=results.aircraft.mainwing_twist,
            mainwing_sweepback=results.aircraft.mainwing_sweepback,
            mainwing_dihedral=results.aircraft.mainwing_dihedral,
            mainwing_incidence=results.aircraft.mainwing_incidence,
            
            # Lists of ratios/angles, no conversion needed
            flap_start=results.aircraft.flap_start,
            flap_end=results.aircraft.flap_end,
            flap_angle=results.aircraft.flap_angle,
            flap_c_ratio=results.aircraft.flap_c_ratio,
            
            # Ratios and angles, no conversion needed
            horizontal_volume_ratio=results.aircraft.horizontal_volume_ratio,
            horizontal_area_ratio=results.aircraft.horizontal_area_ratio,
            horizontal_AR=results.aircraft.horizontal_AR,
            horizontal_taper=results.aircraft.horizontal_taper,
            horizontal_ThickChord=results.aircraft.horizontal_ThickChord,
            vertical_volume_ratio=results.aircraft.vertical_volume_ratio,
            vertical_taper=results.aircraft.vertical_taper,
            vertical_ThickChord=results.aircraft.vertical_ThickChord,
            mainwing_airfoil_datapath=results.aircraft.mainwing_airfoil_datapath,
            horizontal_airfoil_datapath=results.aircraft.horizontal_airfoil_datapath,
            vertical_airfoil_datapath=results.aircraft.vertical_airfoil_datapath
            
        )
        
        # Create new analysis results with converted units
        return AircraftAnalysisResults(
            aircraft=new_aircraft,
            alpha_list=results.alpha_list,
            
            # Mass conversions (g to kg)
            m_empty=results.m_empty / 1000,
            m_boom=results.m_boom / 1000,
            m_wing=results.m_wing / 1000,
            
            # Length conversions (mm to m)
            span=results.span / 1000,
            
            # These are ratios, no conversion needed
            AR=results.AR,
            taper=results.taper,
            twist=results.twist,
            
            # Area conversion (mm² to m²)
            Sref=results.Sref / 1e6,
            
            # Length conversions (mm to m)
            Lw=results.Lw / 1000,
            Lh=results.Lh / 1000,
            
            # These are dimensionless coefficients, no conversion needed
            CL=results.CL,
            # CL_max=results.CL_max,
            CD_wing=results.CD_wing,
            CD_fuse=results.CD_fuse,
            CD_total=results.CD_total,
            
            # Angles, no conversion needed
            AOA_stall=results.AOA_stall,
            AOA_takeoff_max=results.AOA_takeoff_max,
            AOA_climb_max=results.AOA_climb_max,
            AOA_turn_max=results.AOA_turn_max,
            
            # These are dimensionless coefficients, no conversion needed
            CL_flap_max=results.CL_flap_max,
            CL_flap_zero=results.CL_flap_zero,
            CD_flap_max=results.CD_flap_max,
            CD_flap_zero=results.CD_flap_zero,

            max_load=presetValues.max_load
        )

    def convert_propellerCSV_to_ndarray(self, csvPath):

        propeller_df = pd.read_csv(csvPath)
        propeller_df.dropna(how='any',inplace=True)
        propeller_df = propeller_df.sort_values(by=['RPM', 'V(speed) (m/s)']).reset_index(drop=True)

        rpm_array = propeller_df['RPM'].to_numpy()
        v_speed_array = propeller_df['V(speed) (m/s)'].to_numpy()
        torque_array = propeller_df['Torque (N-m)'].to_numpy()
        thrust_array = propeller_df['Thrust (kg)'].to_numpy()
        self.propeller_array = np.column_stack((rpm_array, v_speed_array, torque_array, thrust_array))
    
        return
    
    def convert_batteryCSV_to_ndarray(self, csvPath):

        df = pd.read_csv(csvPath,skiprows=[1]) 
        time_array = df['Time'].to_numpy()
        voltage_array = df['Voltage'].to_numpy()
        current_array = df['Current'].to_numpy()
        dt_array = np.diff(time_array, prepend=time_array[0])
        cumulative_Wh = np.cumsum(voltage_array*current_array*dt_array) * self.propulsionSpecs.n_cell / 3600
        SoC_array = 100 - (cumulative_Wh / self.propulsionSpecs.battery_Wh)*100
        mask = SoC_array >= 0
        time_array = time_array[mask]
        voltage_array = voltage_array[mask]
        current_array = current_array[mask]
        SoC_array = SoC_array[mask]
        battery_array = np.column_stack((time_array, voltage_array, current_array, SoC_array))
        self.battery_array = battery_array[battery_array[:, 3].argsort()]
        return
    
    def clearState(self):
        self.state = PlaneState()
        self.stateLog = []
    
    def setAuxVals(self) -> None:
        
        self.weight = self.missionParam.m_takeoff * g
        
        self.v_takeoff = (np.sqrt((2*self.weight) / (rho*self.analResult.Sref*self.analResult.CL_flap_max)))

        # Create focused alpha range from -10 to 10 degrees
        alpha_extended = np.linspace(-5, 15, 2000)  # 0.01 degree resolution
    
        CL_interp1d = interp1d(self.analResult.alpha_list, self.analResult.CL, kind="linear", fill_value="extrapolate")
        CD_interp1d = interp1d(self.analResult.alpha_list, self.analResult.CD_total, kind="quadratic", fill_value="extrapolate")
        # Create lookup tables
        CL_table = CL_interp1d(alpha_extended)
        CD_table = CD_interp1d(alpha_extended)
        
        self._cl_cache = {}
        self._cd_cache = {}

        # Create lambda functions for faster lookup
        self._cl_func_original = lambda alpha: np.interp(alpha, alpha_extended, CL_table)
        self._cd_func_original = lambda alpha: np.interp(alpha, alpha_extended, CD_table)
        self.alpha_func = lambda CL: np.interp(CL, CL_table, alpha_extended)
        return

    def CL_func(self,alpha):
        key = int(alpha*1000+0.5)  # Reduce precision for better cache hits
        if key not in self._cl_cache:
            self._cl_cache[key] = self._cl_func_original(alpha)

        #print(self._cl_cache[key] - self._cl_func_original(alpha))
        return self._cl_cache[key]
        #return np.interp(alpha, alpha_extended, CL_table)

    def CD_func(self,alpha):
        key = int(alpha*1000+0.5)  # Reduce precision for better cache hits
        if key not in self._cd_cache:
            self._cd_cache[key] = self._cd_func_original(alpha)

        #print(self._cl_cache[key] - self._cl_func_original(alpha))
        return self._cd_cache[key]
        #return np.interp(alpha, alpha_extended, CL_table)

    
    def run_mission(self, missionPlan: List[MissionConfig],clearState = True) -> int:

        flag = 0
        M3_time_limit = 300 - self.presetValues.x1_time_margin 
        if(clearState): self.clearState()

        for phase in missionPlan:
            try:
                match phase.phaseType:
                    case PhaseType.TAKEOFF:
                        flag = self.vertical_takeoff_simulation(phase.numargs[0])
                        # print(f"takeoff = {flag}")
                    case PhaseType.HOVER:
                        flag = self.hover_simulation(phase.numargs[0]) 
                        # print(f"climb = {flag}")  
                    case PhaseType.TRANSITION:
                        flag = self.transition_simulation(phase.waypoint_position)
                    case PhaseType.LEVEL_FLIGHT:
                        flag = self.waypoint_level_flight_simulation(phase.waypoint_position)
                        # print(f"level flight = {flag}")
                    case PhaseType.TURN:
                        flag = self.waypoint_turn_simulation(phase.waypoint_position, phase.direction)
                        # print(f"turn = {flag}")
                    case PhaseType.VTOL_LEVEL_FLIGHT:
                        self.hover_waypoint_simulation(phase.waypoint_position)
                        flag = self.vtol_waypoint_level_flight_simulation(phase.waypoint_position)
                    case PhaseType.HOVER_TURN:
                        flag = self.hover_waypoint_simulation(phase.waypoint_position)
                    case PhaseType.BACK_TRANSITION:
                        flag = self.back_transition_simulation()
                    case _: 
                        raise ValueError("Didn't provide a correct PhaseType!")
                if (self.state.time > M3_time_limit or self.state.battery_voltage < self.presetValues.min_battery_voltage):
                    return -2
                self.state.phase += 1
                
                if flag==-1: 
                    return -1
                
            except Exception as e:
                print(e)
                return -1        
    
        return 0

    def run_mission2(self) -> float:

        result = 0
        

        mission2 = [
            MissionConfig(PhaseType.TAKEOFF, [30]),
            MissionConfig(PhaseType.HOVER,   [3]),

            MissionConfig(PhaseType.VTOL_LEVEL_FLIGHT, waypoint_position=WP1),
            MissionConfig(PhaseType.TRANSITION,        waypoint_position=WP2),
            MissionConfig(PhaseType.LEVEL_FLIGHT,      waypoint_position=WP2),

            MissionConfig(PhaseType.TURN,   waypoint_position=WP3, direction="CW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT,      waypoint_position=WP3),

            MissionConfig(PhaseType.TURN,   waypoint_position=WP4, direction="CW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT,      waypoint_position=WP4),

            MissionConfig(PhaseType.TURN,   waypoint_position=WP2, direction="CW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT,      waypoint_position=WP2),

            #MissionConfig(PhaseType.BACK_TRANSITION),
            #MissionConfig(PhaseType.HOVER, [180]),
        ]
        result = self.run_mission(mission2)  
        
        first_state = self.stateLog[0]
        first_state.mission = 2 
        last_state = self.stateLog[-1]
        last_state.N_laps = 3   
        last_z_pos = last_state.position[2] 
        last_battery_voltage = last_state.battery_voltage 
        if(result == -1 or last_z_pos < 20 or last_battery_voltage < self.presetValues.min_battery_voltage): return -1,-1
        
        return self.m_fuel, self.state.phase

    def run_mission3(self) -> float:
        result = 0
        mission3 = [
                MissionConfig(PhaseType.TAKEOFF, [30]),
                ]

        # Run initial mission sequence
        result = self.run_mission(mission3)
        first_state = self.stateLog[0]
        first_state.mission = 3
        if(result == -1): 
            return -1

        # Store starting index for each lap to handle truncation if needed
        self.state.N_laps = 1
        time_limit = 300 - self.presetValues.x1_time_margin  

        # Define lap2 phases
        lap2 = [
            MissionConfig(PhaseType.LEVEL_FLIGHT, [-152], "left"),
            MissionConfig(PhaseType.TURN, [180], "CW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "right"),
            MissionConfig(PhaseType.TURN, [360], "CCW"),
            MissionConfig(PhaseType.LEVEL_FLIGHT, [152], "right"),
            MissionConfig(PhaseType.TURN, [180], "CW"),
            #MissionConfig(PhaseType.LEVEL_FLIGHT, [0], "left"),
        ]

        while self.state.N_laps < 2:
            lap_start_index = len(self.stateLog)
            self.state.N_laps += 1
            
            result = self.run_mission(lap2,clearState=False)
            if(result == -1): return -1
            if(result == -2):
                self.state.N_laps -= 1
                return self.state.N_laps, self.state.phase, self.state.time
            
            # Check if we've exceeded time limit or voltage limit
            if (self.state.time > time_limit or 
                self.state.battery_voltage < self.presetValues.min_battery_voltage):
                
                # Truncate the results and finish
                self.stateLog = self.stateLog[:lap_start_index]
                self.state.N_laps -= 1
                break
        
        return self.state.N_laps, self.state.phase, self.state.time
        
    def calculate_level_alpha(self, v):
        #  Function that calculates the AOA required for level flight using the velocity vector and thrust
        return self.calculate_level_alpha_fast(v)
        speed = fast_norm(v)
        def equation(alpha:float):

            CL = float(self.CL_func(alpha)[0])
            L,_ = self.calculate_Lift_and_Loadfactor(CL,float(speed))
            return float(L-self.weight)

        alpha_solution = fsolve(equation, 5, xtol=1e-4, maxfev=1000)

        fast_sol = self.calculate_level_alpha_fast(v)
        print(fast_sol - alpha_solution[0])
        return alpha_solution[0]
    
    # Use the fact that CL is quadratic(increasing) and do binary search instead
    def calculate_level_alpha_fast(self,v):
        speed = fast_norm(v)
        # Pre-calculate shared values
        dynamic_pressure = 0.5 * PhysicalConstants.rho * speed**2 * self.analResult.Sref
        weight = self.missionParam.m_takeoff * PhysicalConstants.g
        
        # Binary search instead of fsolve
        alpha_min, alpha_max = -3, 13
        tolerance = 1e-4
        
        while (alpha_max - alpha_min) > tolerance:
            alpha = (alpha_min + alpha_max) / 2
            CL = float(self.CL_func(alpha))
            L = dynamic_pressure * CL
            
            if L > weight:
                alpha_max = alpha
            else:
                alpha_min = alpha
                
        return (alpha_min + alpha_max) / 2

    def calculate_Lift_and_Loadfactor(self, CL, speed:float=-1):
        if(speed == -1): speed = fast_norm(self.state.velocity)
        L = 0.5 * rho * speed**2 * self.analResult.Sref * CL
        return L, L/self.weight 
    
    def isBelowFlapTransition(self):
        return self.state.position[2] < self.presetValues.h_flap_transition  
    
    def updateBatteryState(self,SoC):
        capacity = self.propulsionSpecs.battery_Wh * SoC / 100
        capacity -= self.state.motor_input_power * self.dt / 3600 
        self.state.battery_SoC = capacity/self.propulsionSpecs.battery_Wh * 100  
        voltage_per_cell = SoC2Vol(self.state.battery_SoC,self.battery_array)
        self.state.battery_voltage = self.propulsionSpecs.n_cell * voltage_per_cell
        return
    
    
    def vertical_takeoff_simulation(self, h_target):
        self.dt = 0.1
        max_steps = int(30 / self.dt)
        mass = self.missionParam.m_takeoff
        Weight = self.weight
        self.current_motor_count = 4 
        # PID gains for altitude approach
        Kp = 1.0
        Kd = 0.05

        # reset state
        self.state.velocity = np.array([0.0, 0.0, 0.0])
        self.state.position = np.array([0.0, 0.0, 0.0])
        self.state.time = 0.0
        self.state.battery_voltage = 4.2 * self.propulsionSpecs.n_cell
        self.state.battery_SoC = 100.0

        for _ in range(max_steps):
            z = self.state.position[2]
            v_z = self.state.velocity[2]

            # phase 1: aggressive climb until 80% of h_target
            if z < 0.3 * h_target:
                throttle = self.presetValues.throttle_takeoff
                _, _, amps, power, thrust_per_motor = thrust_analysis(
                    throttle,
                    fast_norm(self.state.velocity),
                    self.state.battery_voltage,
                    self.propulsionSpecs,
                    self.propeller_array,
                    0
                )
                T_total =  4 * thrust_per_motor * g
                a_z = (T_total - Weight) / mass

            else:
                # phase 2: PID-controlled approach to h_target
                e_z = h_target - z
                v_des = Kp * e_z
                a_des = Kd * (v_des - v_z)
                # total thrust needed = weight + m * a_des
                T_total = Weight + mass * a_des
                # per-motor thrust in kgf
                thrust_per_motor = (T_total / g) / 4

                _, _, amps, power, throttle = thrust_reverse_solve(
                    thrust_per_motor,
                    fast_norm(self.state.velocity),
                    self.state.battery_voltage,
                    self.propulsionSpecs.Kv,
                    self.propulsionSpecs.R,
                    self.propeller_array
                )
                a_z = (T_total - Weight) / mass

            # update state
            self.state.throttle = throttle
            self.state.Amps = amps
            self.state.motor_input_power = power * self.current_motor_count
            self.state.acceleration = np.array([0.0, 0.0, a_z])
            self.state.velocity += self.state.acceleration * self.dt
            self.state.position += self.state.velocity * self.dt
            self.state.time += self.dt

            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

            # phase 3: when at or above h_target and nearly zero vertical speed, stop
            if z >= h_target and abs(v_z) < 0.1:
                break

                
    def hover_simulation(self, t_target):
        
        self.dt = 0.1
        step = 0
        max_steps = int(t_target / self.dt)
        self.state.acceleration = np.array([0.0, 0.0, 0.0])
        self.state.velocity = np.array([0.0, 0.0, 0.0])
        self.current_motor_count = 4 
        
        for step in range(max_steps):
            self.state.time += self.dt
            speed = fast_norm(self.state.velocity)
            desired_thrust_per_motor = self.weight / (4*g)
            
            _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(
                desired_thrust_per_motor, 
                speed,
                self.state.battery_voltage, 
                self.propulsionSpecs.Kv, 
                self.propulsionSpecs.R, 
                self.propeller_array
                )
            
            self.state.thrust = self.weight / g
            T_takeoff = self.state.thrust * g
            self.state.motor_input_power = self.state.motor_input_power * self.current_motor_count
            
            self.state.acceleration = calculate_acceleration_vertical_takeoff(
                self.state.velocity,
                self.missionParam.m_takeoff,
                self.weight,
                T_takeoff
            )
            
            self.state.velocity += self.state.acceleration * self.dt
            self.state.position += self.state.velocity * self.dt
            
            self.updateBatteryState(self.state.battery_SoC)
            self.logState()
    
    def transition_simulation(self, waypoint_position):
        """
        Smoothly accelerate from hover toward the next waypoint so that
        waypoint_level_flight_simulation can compute a nonzero heading.
        Here we apply a small kinematic accel, but consume battery
        exactly like in hover_simulation.
        """
        self.dt = 0.1
        max_steps = int(5 / self.dt)   # 5초간
        self.current_motor_count = 2 

        # 현재 XY와 목표 XY
        current_xy = self.state.position[:2].copy()
        target_xy  = np.array(waypoint_position[:2])
        delta_xy   = target_xy - current_xy
        dist       = np.linalg.norm(delta_xy)
        if dist < 1e-6:
            return

        u_xy = delta_xy / dist
        accel = np.array([u_xy[0]*3, u_xy[1]*3, 0.0])

        for _ in range(max_steps):
            # 1) kinematics
            self.state.time += self.dt
            self.state.acceleration = accel
            self.state.velocity     += accel * self.dt
            self.state.position     += self.state.velocity * self.dt

            # 2) thrust 계산 (hovering과 동일)
            # 목표 추력 per motor in kgf
            kgf_per_motor = (self.weight / g) / 4
            speed = fast_norm(self.state.velocity)
            _, _, amps, power, throttle = thrust_reverse_solve(
                kgf_per_motor,
                speed,
                self.state.battery_voltage,
                self.propulsionSpecs.Kv,
                self.propulsionSpecs.R,
                self.propeller_array
            )

            # 상태 업데이트
            self.state.Amps            = amps
            self.state.motor_input_power = power * self.current_motor_count
            self.state.throttle        = throttle
            # thrust 기록은 hover_simulation 방식 그대로
            self.state.thrust         = self.weight / g

            # 3) 배터리 소모 & 로그
            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

    def vtol_waypoint_level_flight_simulation(self, waypoint_position):
        self.dt = 0.1
        max_steps = int(180 / self.dt)
        self.current_motor_count = 4 

        target_xy = np.array(waypoint_position[:2])
        mass = self.missionParam.m_takeoff
        W = mass * g
        Sref = self.analResult.Sref

        for _ in range(max_steps):
            pos_xy = self.state.position[:2]
            delta_xy = target_xy - pos_xy
            dist = np.linalg.norm(delta_xy)
            if dist < 3:
                return 0

            vel_xy = self.state.velocity[:2]
            speed_xy = np.linalg.norm(vel_xy)
            if speed_xy < 1e-6:
                print("Error: zero velocity; cannot determine heading. (VTOL)")
                return -1
            cross = vel_xy[0] * delta_xy[1] - vel_xy[1] * delta_xy[0]
            dot   = vel_xy.dot(delta_xy)
            if abs(cross) > 10 or dot <= 0:
                print(f"Error: heading {vel_xy} not aligned with waypoint {target_xy}.")
                return -1

            u_xy = delta_xy / dist
            speed = fast_norm(self.state.velocity)
            q     = 0.5 * rho * speed**2

            alpha = -5.0
            theta = -math.radians(alpha)
            self.state.AOA = alpha

            CL = float(self.CL_func(alpha))
            CD = float(self.CD_func(alpha))

            L = q * Sref * CL
            D = q * Sref * CD

            T_total = (W - L) / math.cos(theta)
            if T_total < 0:
                T_total = 0.0

            thrust_per_motor_kgf = (T_total / g) / 4

            _, _, amps, power, throttle = thrust_reverse_solve(
                thrust_per_motor_kgf,
                speed,
                self.state.battery_voltage,
                self.propulsionSpecs.Kv,
                self.propulsionSpecs.R,
                self.propeller_array
            )
            self.state.Amps            = amps
            self.state.motor_input_power = power * self.current_motor_count
            self.state.throttle        = throttle
            self.state.thrust          = T_total / g
            self.state.lift            = L

            F_forward = T_total * math.sin(theta) - D
            a_forward = F_forward / mass
            accel     = np.array([u_xy[0]*a_forward, u_xy[1]*a_forward, 0.0])

            self.state.time         += self.dt
            self.state.acceleration  = accel
            self.state.velocity     += accel * self.dt
            self.state.position     += self.state.velocity * self.dt

            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

        return 0




    def hover_waypoint_simulation(self, waypoint_position):
        """
        Smoothly accelerate from hover toward the next waypoint so that
        waypoint_level_flight_simulation can compute a nonzero heading.
        Here we apply a small kinematic accel, but consume battery
        exactly like in hover_simulation.
        """
        self.dt = 0.1
        max_steps = int(3 / self.dt)   # 3초간
        self.current_motor_count = 4 

        # 현재 XY와 목표 XY
        current_xy = self.state.position[:2].copy()
        target_xy  = np.array(waypoint_position[:2])
        delta_xy   = target_xy - current_xy
        dist       = np.linalg.norm(delta_xy)
        if dist < 1e-6:
            return

        u_xy = delta_xy / dist
        accel = np.array([u_xy[0]*0.1, u_xy[1]*0.1, 0.0])

        for _ in range(max_steps):
            # 1) kinematics
            self.state.time += self.dt
            self.state.acceleration = accel
            self.state.velocity     += accel * self.dt
            self.state.position     += self.state.velocity * self.dt

            # 2) thrust 계산 (hovering과 동일)
            # 목표 추력 per motor in kgf
            kgf_per_motor = (self.weight / g) / 4
            speed = fast_norm(self.state.velocity)
            _, _, amps, power, throttle = thrust_reverse_solve(
                kgf_per_motor,
                speed,
                self.state.battery_voltage,
                self.propulsionSpecs.Kv,
                self.propulsionSpecs.R,
                self.propeller_array
            )

            # 상태 업데이트
            self.state.Amps            = amps
            self.state.motor_input_power = power * self.current_motor_count
            self.state.throttle        = throttle
            # thrust 기록은 hover_simulation 방식 그대로
            self.state.thrust         = self.weight  

            # 3) 배터리 소모 & 로그
            self.updateBatteryState(self.state.battery_SoC)
            self.logState()
        
    def waypoint_level_flight_simulation(self, waypoint_position):
        self.dt = 0.1
        max_steps = int(180 / self.dt)
        self.current_motor_count = 2 
        pos_xy = self.state.position[:2].copy()
        vel_xy = self.state.velocity[:2].copy()
        speed_xy = np.linalg.norm(vel_xy)
        if speed_xy < 1e-6:
            print("Error: zero velocity; cannot determine heading. (FIXED)")
            return -1
        target_xy = np.array(waypoint_position[:2])
        delta_xy = target_xy - pos_xy
        cross = vel_xy[0]*delta_xy[1] - vel_xy[1]*delta_xy[0]
        dot = vel_xy.dot(delta_xy)
        if abs(cross) > 5 or dot <= 0:
            print(f"Error: heading {vel_xy} not aligned with waypoint {target_xy}.")
            return -1
        u_xy = vel_xy / speed_xy
        self.state.velocity = np.array([u_xy[0]*speed_xy, u_xy[1]*speed_xy, 0.0])
        for _ in range(max_steps):
            # 1) 새로 위치·방향 계산
            pos_xy   = self.state.position[:2]
            delta_xy = target_xy - pos_xy
            dist     = np.linalg.norm(delta_xy)
            if dist < 5:          # 5 m 이내면 도착
                return 0

            u_xy = delta_xy / dist            #   <── 루프 안으로 이동
            speed = fast_norm(self.state.velocity)

            # 2) α 계산, 추진계 호출
            alpha = self.calculate_level_alpha(self.state.velocity)
            self.state.AOA = alpha
            _, _, amps, power, t_per_motor = thrust_analysis(
                self.missionParam.level_thrust_ratio,
                speed,
                self.state.battery_voltage,
                self.propulsionSpecs,
                self.propeller_array,
                0
            )
            self.state.Amps = amps
            self.state.motor_input_power = power * self.current_motor_count
            self.state.throttle = self.missionParam.level_thrust_ratio

            # 3) 힘 / 가속
            T_total = t_per_motor * self.presetValues.number_of_motor * g
            CD      = float(self.CD_func(alpha))
            D       = 0.5 * rho * speed**2 * self.analResult.Sref * CD
            a_mag   = (T_total*np.cos(np.radians(alpha)) - D) / self.missionParam.m_takeoff
            acc_vec = np.array([u_xy[0]*a_mag, u_xy[1]*a_mag, 0.0])
            self.state.acceleration = acc_vec

            # 4) 적분
            self.state.time      += self.dt
            self.state.velocity  += acc_vec * self.dt
            self.state.position  += self.state.velocity * self.dt

            # 5) 배터리·로그
            self.updateBatteryState(self.state.battery_SoC)
            self.logState()
        else:
            print("Warning: waypoint not reached within time limit.")
            return -1



    def waypoint_turn_simulation(self, waypoint_position, direction):
        """
        Turn along a circle until your heading (XY) aligns with the line to waypoint_position.
        direction: 'CW' or 'CCW'
        """
        # 0) setup
        self.dt = 0.001
        max_steps = int(180 / self.dt)      # limit 3 
        self.current_motor_count = 2 
        target_xy = np.array(waypoint_position[:2])

        # 1) initialize
        speed = fast_norm(self.state.velocity)
        initial_angle = np.arctan2(self.state.velocity[1], self.state.velocity[0])
        current_angle = initial_angle

        dyn_q_base = 0.5 * rho * self.analResult.Sref
        weight     = self.weight
        max_load   = self.missionParam.max_load_factor

        for step in range(max_steps):
            # 1a) compute lift & turn parameters
            speed = fast_norm(self.state.velocity)
            dynamic_pressure = dyn_q_base * speed**2

            CL_turn = min(
                float(self.CL_func(self.analResult.AOA_turn_max)),
                (max_load * weight) / dynamic_pressure
            )

            alpha = float(self.alpha_func(CL_turn))
            self.state.AOA = alpha   # AOA 기록

            L = dynamic_pressure * CL_turn
            if weight / L >= 1:
                print("Error: too heavy to turn")
                return -1

            phi = np.arccos(min(weight / L, 0.99))
            phi = np.clip(phi, -np.radians(10), np.radians(10))

            R     = (self.missionParam.m_takeoff * speed**2) / (L * np.sin(phi))
            omega = speed / R

            D = float(self.CD_func(alpha)) * dynamic_pressure

            T_per_motor = (
                determine_max_thrust(
                    speed,
                    self.state.battery_voltage,
                    self.propulsionSpecs,
                    self.propeller_array,
                    0
                ) * self.missionParam.turn_thrust_ratio
            )
            T_total = T_per_motor * self.presetValues.number_of_motor * g

            # 1b) update turn center & angle
            sin_c, cos_c = np.sin(current_angle), np.cos(current_angle)
            if direction == "CCW":
                center_x = self.state.position[0] - R * sin_c
                center_y = self.state.position[1] + R * cos_c
                current_angle += omega * self.dt
            else:  # CW
                center_x = self.state.position[0] + R * sin_c
                center_y = self.state.position[1] - R * cos_c
                current_angle -= omega * self.dt

            # 1c) compute acceleration vector and integrate
            sin_n, cos_n = np.sin(current_angle), np.cos(current_angle)
            tangent = np.array([ cos_n,  sin_n, 0.0])
            normal  = np.array([-sin_n,  cos_n, 0.0])
            if direction == "CW":
                normal *= -1

            a_tan = (T_total - D) / self.missionParam.m_takeoff
            a_cen = (L * np.sin(phi))      / self.missionParam.m_takeoff

            acc = a_tan * tangent + a_cen * normal
            self.state.acceleration = acc

            # integrate velocity & position
            self.state.velocity += acc * self.dt
            self.state.position += self.state.velocity * self.dt

            # → **시간 업데이트를 반드시** 해 줍니다.
            self.state.time += self.dt

            # 1d) throttle & amps
            _, _, self.state.Amps, self.state.motor_input_power, self.state.throttle = \
                thrust_reverse_solve(
                    T_per_motor,
                    speed,
                    self.state.battery_voltage,
                    self.propulsionSpecs.Kv,
                    self.propulsionSpecs.R,
                    self.propeller_array
                )

            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

            # 2) check alignment
            pos_xy = self.state.position[:2]
            vel_xy = self.state.velocity[:2]
            delta  = target_xy - pos_xy
            cross  = vel_xy[0]*delta[1] - vel_xy[1]*delta[0]
            dot    = np.dot(vel_xy, delta)
            if abs(cross) < 5 and dot > 0:
                # heading aligned to waypoint line
                break
        else:
            print("Warning: waypoint alignment not reached within time limit.")
            return -1

        return 0



    def back_transition_simulation(self):
        """
        고정익 순항 상태에서 속도를 줄여 호버 상태(거의 0 m/s)로 전환하는 단계
        반환값 : 0(성공) / -1(예외)
        """
        # ------------------------------------------------------------------
        self.dt      = 0.1
        t_max        = 50.0
        steps        = int(t_max / self.dt)

        mass        = self.missionParam.m_takeoff
        W           = self.weight                     # [N]
        Sref        = self.analResult.Sref
        self.current_motor_count = 4                  # VTOL 모터 4개

        alpha_deg   = -10.0                           # 프로펠러 10° 뒤로 tilt
        theta       =  math.radians(alpha_deg)        # ← ‘-’ 제거!  (θ = -10°)
        T_total     = W / math.cos(theta)             # [N]
        thrust_per_motor_kgf = (T_total / g) / 4      # [kgf]

        STOP_SPEED  = 0.10                            # m/s 이하면 정지로 간주
        self.state.AOA = alpha_deg

        # ------------------------------------------------------------------
        for _ in range(steps):
            speed = fast_norm(self.state.velocity)
            if speed <= STOP_SPEED:
                break

            # 1) 스로틀 역계산
            _, _, amps, power, throttle = thrust_reverse_solve(
                thrust_per_motor_kgf,
                speed,
                self.state.battery_voltage,
                self.propulsionSpecs.Kv,
                self.propulsionSpecs.R,
                self.propeller_array
            )
            self.state.Amps              = amps
            self.state.motor_input_power = power * self.current_motor_count
            self.state.throttle          = throttle

            # 2) 힘 계산
            CD   = float(self.CD_func(alpha_deg))
            D    = 0.5 * rho * speed**2 * Sref * CD          # 항력 [N]
            T_h  = T_total * math.sin(theta)                 # 수평추력(음수)
            F_x  = T_h - D                                   # (대부분 음)  ← 감속

            # 3) 가속도 (속도 방향 반대)
            a_mag = F_x / mass                               # 음수
            u_xy  = self.state.velocity[:2] / speed
            acc   = np.array([a_mag * u_xy[0],
                              a_mag * u_xy[1],
                              0.0])
            self.state.acceleration = acc

            # 4) 적분
            self.state.time     += self.dt
            self.state.velocity += acc * self.dt
            self.state.position += self.state.velocity * self.dt

            # 5) 배터리·로그
            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

        return 0









####### Simulations used in AIAA DBF #######

    def takeoff_simulation(self):
    
        self.dt= 0.1
        step=0
        max_steps = int(15 / self.dt) # 15 sec simulation
        self.state.velocity = np.array([0.0, 0.0, 0.0])
        self.state.position = np.array([0.0, 0.0, 0.0])
        self.state.time = 0.0
        self.state.battery_voltage = 4.2 * self.propulsionSpecs.n_cell 
        self.state.battery_SoC = 100.0
   
        for step in range(max_steps): 
            # Ground roll until 0.9 times takeoff speed
            if fast_norm(self.state.velocity) < 0.9 * self.v_takeoff :
                
                self.state.time += self.dt
                
                self.state.throttle = self.presetValues.throttle_takeoff
                _, _, self.state.Amps, self.state.motor_input_power, thrust_per_motor = thrust_analysis(
                                self.state.throttle,
                                fast_norm(self.state.velocity),
                                self.state.battery_voltage,
                                self.propulsionSpecs,
                                self.propeller_array,
                                0 #graphFlag
                )
                self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor 
                T_takeoff = self.state.thrust * g
                
                
                self.state.acceleration = calculate_acceleration_groundroll(
                        self.state.velocity,
                        self.missionParam.m_takeoff,
                        self.weight,
                        self.analResult.Sref,
                        self.analResult.CD_flap_zero, self.analResult.CL_flap_zero,
                        T_takeoff
                        )

                self.state.velocity -= self.state.acceleration * self.dt
                self.state.position += self.state.velocity * self.dt
                
                _, loadfactor = self.calculate_Lift_and_Loadfactor(self.analResult.CL_flap_zero)
                self.state.loadfactor = loadfactor

                self.state.AOA = 0
                self.state.climb_pitch_angle =np.nan
                self.state.bank_angle = np.nan

                self.updateBatteryState(self.state.battery_SoC)
                self.logState()
            
            # Ground rotation until takeoff speed    
            elif 0.9 * self.v_takeoff <= fast_norm(self.state.velocity) <= self.v_takeoff:
                self.state.time += self.dt

                self.state.throttle = self.presetValues.throttle_takeoff
                _, _, self.state.Amps, self.state.motor_input_power, thrust_per_motor = thrust_analysis(
                                self.presetValues.throttle_takeoff,
                                fast_norm(self.state.velocity),
                                self.state.battery_voltage,
                                self.propulsionSpecs,
                                self.propeller_array,
                                0 #graphFlag
                )
                self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor 
                T_takeoff = self.state.thrust * g
                
                self.state.acceleration = calculate_acceleration_groundrotation(
                        self.state.velocity,
                        self.missionParam.m_takeoff,
                        self.weight,
                        self.analResult.Sref,
                        self.analResult.CD_flap_max, self.analResult.CL_flap_max,
                        T_takeoff
                        )
                self.state.velocity -= self.state.acceleration * self.dt
                self.state.position += self.state.velocity * self.dt
                
            
                
                _, loadfactor = self.calculate_Lift_and_Loadfactor(self.analResult.CL_flap_max)
                self.state.loadfactor = loadfactor

                self.state.AOA=10
                self.state.climb_pitch_angle=np.nan
                self.state.bank_angle = np.nan

                self.updateBatteryState(self.state.battery_SoC)
                self.logState()
            else:
                break
            
            if(step == max_steps-1) : return -1  

    def climb_simulation(self, h_target, x_max_distance, direction):
        """
        Args:
            h_target (float): Desired altitude to climb at the maximum climb AOA (m)
            x_max_distance (float): Restricted x-coordinate for climb (m)
            direction (string): The direction of movement. Must be either 'left' or 'right'.
        """
     
        
        if self.state.position[2] > h_target: return
        self.dt = 0.1
        step=0
        max_steps = int(60 / self.dt)  # Max 60 seconds simulation
        break_flag = False
        alpha_w_deg = 0 
        
        for step in range(max_steps):
            self.state.time += self.dt

            # Calculate climb angle
            gamma_rad = np.atan2(self.state.velocity[2], abs(self.state.velocity[0]))

            if direction == 'right':
                # set AOA at climb (if altitude is below target altitude, set AOA to AOA_climb. if altitude exceed target altitude, decrease AOA gradually to -5 degree)
                if(self.state.position[2] < self.presetValues.h_flap_transition and 
                   self.state.position[0] < x_max_distance):
                    alpha_w_deg = self.analResult.AOA_takeoff_max
                elif(self.presetValues.h_flap_transition <= self.state.position[2] < h_target and 
                     self.state.position[0] < x_max_distance):
                    load_factor = self.calculate_Lift_and_Loadfactor(float(self.CL_func(self.analResult.AOA_climb_max)))[1]
            
                    if (load_factor < self.missionParam.max_load_factor and 
                        gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = self.analResult.AOA_climb_max
                    elif (load_factor >= self.missionParam.max_load_factor and 
                          gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = float(self.alpha_func((2 * self.weight * self.missionParam.max_load_factor)/
                                                          (rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref)))
                    else:
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                else:
                    break_flag = True
                    if gamma_rad > np.radians(self.presetValues.max_climb_angle):
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
                    else:
                        alpha_w_deg -= 1
                        alpha_w_deg = max(alpha_w_deg, -5)
            
            elif direction == 'left':
                # set AOA at climb (if altitude is below target altitude, set AOA to AOA_climb. if altitude exceed target altitude, decrease AOA gradually to -5 degree)
                if(self.state.position[2] < self.presetValues.h_flap_transition and 
                   self.state.position[0] > x_max_distance):
                    alpha_w_deg = self.analResult.AOA_takeoff_max
                elif(self.presetValues.h_flap_transition <= self.state.position[2] < h_target and 
                     self.state.position[0] > x_max_distance):
                    load_factor = self.calculate_Lift_and_Loadfactor(float(self.CL_func(self.analResult.AOA_climb_max)))[1]
            
                    if (load_factor < self.missionParam.max_load_factor and 
                        gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = self.analResult.AOA_climb_max
                    elif (load_factor >= self.missionParam.max_load_factor and 
                          gamma_rad < np.radians(self.presetValues.max_climb_angle)):
                        alpha_w_deg = float(self.alpha_func((2 * self.weight * self.missionParam.max_load_factor)/
                                                          (rho * fast_norm(self.state.velocity)**2 * self.analResult.Sref)))
                    else:
                        alpha_w_deg -= 3
                        alpha_w_deg = max(alpha_w_deg, -5)
                else:
                    break_flag = True
                    if gamma_rad > np.radians(self.presetValues.max_climb_angle):
                        alpha_w_deg -= 2
                        alpha_w_deg = max(alpha_w_deg, -5)
                    else:
                        alpha_w_deg -= 2
                        alpha_w_deg = max(alpha_w_deg, -5)            
                    
                    
            if (self.isBelowFlapTransition()):
                CL = self.analResult.CL_flap_max
            else:
                CL = float(self.CL_func(alpha_w_deg))
                
            speed = fast_norm(self.state.velocity)  

            T_climb_max_per_motor = determine_max_thrust(speed,
                                            self.state.battery_voltage,
                                            self.propulsionSpecs,
                                            self.propeller_array,
                                            0#graphFlag
            ) #kg
            thrust_per_motor = T_climb_max_per_motor * self.missionParam.climb_thrust_ratio #kg    

            if speed >= self.missionParam.max_speed * 0.95:

                D = 0.5 * rho * speed**2 * self.analResult.Sref * self.CD_func(alpha_w_deg)
                T_desired = (D + self.weight * np.sin(gamma_rad)) / np.cos(np.deg2rad(alpha_w_deg))
                thrust_per_motor_desired = T_desired / (2*g)
                thrust_per_motor = min(thrust_per_motor, thrust_per_motor_desired)


            _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
            
            T_climb = thrust_per_motor * self.presetValues.number_of_motor * g # total N
            
            self.state.acceleration = RK4_step(self.state.velocity,self.dt,
                         lambda v: calculate_acceleration_climb(v, 
                                                                self.missionParam.m_takeoff,
                                                                self.weight,
                                                                self.analResult.Sref,
                                                                self.CL_func,
                                                                self.CD_func,
                                                                self.analResult.CL_flap_max,
                                                                self.analResult.CD_flap_max,
                                                                alpha_w_deg,
                                                                gamma_rad,
                                                                T_climb,
                                                                not self.isBelowFlapTransition()
                                                                ))
            self.state.velocity[2] += self.state.acceleration[2]*self.dt
            if direction == 'right':
                self.state.velocity[0] += self.state.acceleration[0]*self.dt
            else:
                self.state.velocity[0] -= self.state.acceleration[0]*self.dt
            
            self.state.position[0] += self.state.velocity[0]* self.dt
            self.state.position[2] += self.state.velocity[2]* self.dt

            _, loadfactor = self.calculate_Lift_and_Loadfactor(CL)
            
            self.state.loadfactor = loadfactor

            self.state.throttle = self.missionParam.climb_thrust_ratio
             
            self.state.AOA = alpha_w_deg
            self.state.climb_pitch_angle =alpha_w_deg + np.degrees(gamma_rad)
            self.state.bank_angle = np.nan


            self.updateBatteryState(self.state.battery_SoC)
            self.logState()

            # break when climb angle goes to zero
            if break_flag == 1 and gamma_rad < 0:
                # print(f"cruise altitude is {z_pos:.2f} m.")
                break
            
            if step==max_steps-1 : return -1 
    
    def level_flight_simulation(self, x_final, direction):
     
        #print("\nRunning Level Flight Simulation...")
        # print(max_steps)
        step = 0
        self.dt = 0.1
        max_steps = int(180/self.dt) # max 3 minuites
        # Initialize vectors
        self.state.velocity[2] = 0  # Zero vertical velocity
        speed = fast_norm(self.state.velocity)

        if direction == 'right':
            self.state.velocity = np.array([speed, 0, 0])  # Align with x-axis
        elif direction=='left':
            self.state.velocity = np.array([-speed, 0, 0])
        
        cruise_flag = 0
        
        for step in range(max_steps):
            
            self.state.time += self.dt
            speed = fast_norm(self.state.velocity)
            
            # Calculate alpha_w first
            alpha_w_deg = self.calculate_level_alpha(self.state.velocity)
                
            # Speed limiting while maintaining direction
            if speed >= self.missionParam.max_speed - 0.005:  # Original speed limit
                cruise_flag = 1

            if cruise_flag == 1:
                self.state.velocity = self.state.velocity * (self.missionParam.max_speed / speed)
                T_cruise = 0.5 * rho * self.missionParam.max_speed**2 \
                                * self.analResult.Sref * float(self.CD_func(alpha_w_deg))
                T_cruise_max = determine_max_thrust(speed,
                                               self.state.battery_voltage,
                                               self.propulsionSpecs,
                                               self.propeller_array,
                                               0#graphFlag
                ) #kg
                T_cruise_max = T_cruise_max * self.presetValues.number_of_motor * g
                T_cruise = min(T_cruise, T_cruise_max )
                self.state.thrust = T_cruise / g #kg
            
                alpha_w_deg = self.calculate_level_alpha(self.state.velocity)
                _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(
                    self.state.thrust/self.presetValues.number_of_motor,
                    speed,self.state.battery_voltage,
                    self.propulsionSpecs.Kv,
                    self.propulsionSpecs.R,
                    self.propeller_array)

                self.updateBatteryState(self.state.battery_SoC)
    
                self.state.acceleration = RK4_step(self.state.velocity,self.dt,
                             lambda v: calculate_acceleration_level(v,self.missionParam.m_takeoff, 
                                                                    self.analResult.Sref,
                                                                    self.CD_func, alpha_w_deg,
                                                                    T_cruise))
                if abs(self.state.acceleration[0]) > 0.1 : cruise_flag = 0
            else:
                
                T_level_max_per_motor = determine_max_thrust(
                                                speed,
                                                self.state.battery_voltage,
                                                self.propulsionSpecs,
                                                self.propeller_array,
                                                0#graphFlag
                ) #kg
                thrust_per_motor = T_level_max_per_motor * self.missionParam.level_thrust_ratio #kg
                self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor #kg
                _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
                
                T_climb = self.state.thrust * g # total N
                self.updateBatteryState(self.state.battery_SoC)

                self.state.acceleration =  RK4_step(self.state.velocity,self.dt,
                             lambda v: calculate_acceleration_level(v,self.missionParam.m_takeoff, 
                                                                    self.analResult.Sref,
                                                                    self.CD_func, alpha_w_deg,
                                                                    T_climb))

                
            # Update Acc, Vel, position
            if direction == 'right': 
                self.state.velocity += self.state.acceleration * self.dt
            elif direction == 'left': 
                self.state.velocity -= self.state.acceleration * self.dt
            
            self.state.position[0] += self.state.velocity[0] * self.dt
            self.state.position[1] += self.state.velocity[1] * self.dt
            
            # Calculate and store results

            _,load_factor = self.calculate_Lift_and_Loadfactor(float(self.CL_func(alpha_w_deg)))
            
            self.state.loadfactor = load_factor 
            self.state.AOA = alpha_w_deg
            self.state.bank_angle = np.nan
            self.state.climb_pitch_angle = np.nan
            self.logState()
            
            # Check if we've reached target x position
            if direction == 'right':
                if self.state.position[0] >= x_final:
                    break
            elif direction == 'left':
                if self.state.position[0] <= x_final:
                    break
            
            if step==max_steps-1 : return -1        

    def turn_simulation(self, target_angle_deg, direction):
        """
        Args:
            target_angle_degree (float): Required angle of coordinate level turn (degree)
            direction (string): The direction of movement. Must be either 'CW' or 'CCW'.
        """     
        
        speed = fast_norm(self.state.velocity) 
        self.dt = 0.1  
        step = 0
        max_steps = int(180/self.dt) 
        # Initialize turn tracking
        target_angle_rad = np.radians(target_angle_deg)
        turned_angle_rad = 0

        # Get initial heading and setup turn center
        initial_angle_rad = np.atan2(self.state.velocity[1], self.state.velocity[0])
        current_angle_rad = initial_angle_rad

        # Pre-calculate constants
        dynamic_pressure_base = 0.5 * rho * self.analResult.Sref
        max_speed = self.missionParam.max_speed
        max_load = self.missionParam.max_load_factor
        weight = self.weight

        for step in range(max_steps):
            # print(step)
            if abs(turned_angle_rad) < abs(target_angle_rad):
                self.state.time += self.dt

                if speed < max_speed - 0.005: # numerical error
                        # Pre-calculate shared terms
                        dynamic_pressure = dynamic_pressure_base * speed * speed
                        
                        CL = min(float(self.CL_func(self.analResult.AOA_turn_max)), 
                                float((max_load * weight)/(dynamic_pressure)))

                        alpha_turn = float(self.alpha_func(CL))
                        L = dynamic_pressure * CL
                        if weight / L >=1: 
                            # print("too heavy")
                            return -1
                        phi_rad = np.acos(min(weight/L,0.99))
                        
                        a_centripetal = (L * np.sin(phi_rad)) / self.missionParam.m_takeoff
                        R = (self.missionParam.m_takeoff * speed**2)/(L * np.sin(phi_rad))
                        omega = speed / R

                        self.state.loadfactor = 1 / np.cos(phi_rad)

                        CD = float(self.CD_func(alpha_turn))
                        D = CD * dynamic_pressure
                    
                        T_turn_max_per_motor = determine_max_thrust(
                                                        speed,
                                                        self.state.battery_voltage,
                                                        self.propulsionSpecs,
                                                        self.propeller_array,
                                                        0#graphFlag
                        ) #kg
                        thrust_per_motor = T_turn_max_per_motor * self.missionParam.turn_thrust_ratio #kg
                        self.state.thrust = self.presetValues.number_of_motor * thrust_per_motor #kg
                        _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
                        
                        T_turn = self.state.thrust * g # total N              
                        a_tangential = (T_turn - D) / self.missionParam.m_takeoff
                        
                        speed += a_tangential * self.dt
                        self.updateBatteryState(self.state.battery_SoC)

                else:
                        speed = max_speed
                        dynamic_pressure = dynamic_pressure_base * speed * speed
                    
                        CL = min(float(self.CL_func(self.analResult.AOA_turn_max)), 
                            float((max_load * weight)/(dynamic_pressure)))
                            
                        alpha_turn = float(self.alpha_func(CL))
                        L = dynamic_pressure * CL
                        if weight / L >=1: 
                            #print("too heavy")
                            return -1
                        phi_rad = np.acos(min(weight/L,0.99))

                        a_centripetal = (L * np.sin(phi_rad)) / self.missionParam.m_takeoff
                        R = (self.missionParam.m_takeoff * speed**2)/(L * np.sin(phi_rad))
                        omega = speed / R

                        self.state.loadfactor = 1 / np.cos(phi_rad)

                        CD = float(self.CD_func(alpha_turn))
                        D = CD * dynamic_pressure
                    
                        T_turn_max_per_motor = determine_max_thrust(
                                                        speed,
                                                        self.state.battery_voltage,
                                                        self.propulsionSpecs,
                                                        self.propeller_array,
                                                        0#graphFlag
                        ) #kg
                        
                        T = min(D, T_turn_max_per_motor*self.presetValues.number_of_motor*self.missionParam.turn_thrust_ratio*g)
                        self.state.thrust = T/g
                        thrust_per_motor = self.state.thrust / self.presetValues.number_of_motor
                        _,_,self.state.Amps,self.state.motor_input_power,self.state.throttle = thrust_reverse_solve(thrust_per_motor, speed,self.state.battery_voltage, self.propulsionSpecs.Kv, self.propulsionSpecs.R, self.propeller_array)
                        
                        a_tangential = (T - D) / self.missionParam.m_takeoff
                        speed += a_tangential * self.dt

                        self.updateBatteryState(self.state.battery_SoC)

                # Calculate turn center
                sin_current = np.sin(current_angle_rad)
                cos_current = np.cos(current_angle_rad)
                
                if direction == "CCW":
                    center_x = self.state.position[0] - R * sin_current
                    center_y = self.state.position[1] + R * cos_current
                    current_angle_rad += omega * self.dt
                    turned_angle_rad += omega * self.dt
                else:  # CW
                    center_x = self.state.position[0] + R * sin_current
                    center_y = self.state.position[1] - R * cos_current
                    current_angle_rad -= omega * self.dt
                    turned_angle_rad -= omega * self.dt

                # Update position
                sin_new = np.sin(current_angle_rad)
                cos_new = np.cos(current_angle_rad)
                
                if direction == "CCW":
                    self.state.position[0] = center_x + R * sin_new
                    self.state.position[1] = center_y - R * cos_new
                else:  # CW
                    self.state.position[0] = center_x - R * sin_new
                    self.state.position[1] = center_y + R * cos_new

                # Update velocity direction
                self.state.velocity = np.array([
                    speed * cos_new,
                    speed * sin_new,
                    0
                ])

                self.state.acceleration = np.array([
                    a_tangential * cos_new - a_centripetal * sin_new,
                    a_tangential * sin_new + a_centripetal * cos_new,
                    0
                ])

                self.state.AOA = alpha_turn
                self.state.bank_angle = np.degrees(phi_rad)
                self.state.climb_pitch_angle = np.nan

                self.logState() 
            else:
                break
            if step==max_steps-1 :
                # print("declined")
                return -1
        
    
    def logState(self) -> None:
        # Append current state as a copy
        self.stateLog.append(PlaneState(
            mission=self.state.mission,
            N_laps=self.state.N_laps,
            position=self.state.position.copy(),
            velocity=self.state.velocity.copy(), 
            acceleration=self.state.acceleration.copy(),
            time=self.state.time,
            throttle=self.state.throttle,
            thrust=self.state.thrust,
            loadfactor=self.state.loadfactor,
            AOA=self.state.AOA,
            climb_pitch_angle=self.state.climb_pitch_angle,
            bank_angle=self.state.bank_angle,
            phase=self.state.phase,
            battery_SoC=self.state.battery_SoC,
            battery_voltage=self.state.battery_voltage,
            Amps=self.state.Amps
        ))
    
## end of class    
#########################################################


def RK4_step(v, dt, func):
    """ Given v and a = f(v), solve for (v(t+dt)-v(dt))/dt or approximately a(t+dt/2)"""

    dt2 = dt/2
    a1 = func(v)
    a2 = func(v + a1 * dt2)
    a3 = func(v + a2 * dt2)
    a4 = func(v + a3 * dt)
    return (a1 + 2*(a2 + a3) + a4) * (1/6)

def fast_norm(v):
    """Faster alternative to np.linalg.norm for 3D vectors"""
    return np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

def calculate_acceleration_vertical_takeoff(v, m, Weight, T_takeoff):
    speed = fast_norm(v)
    a_z = (T_takeoff - Weight) / m
    return np.array([0, 0, a_z])

def calculate_acceleration_groundroll(v, m, Weight,
                                      Sref,
                                      CD_zero_flap,CL_zero_flap,
                                      T_takeoff)->np.ndarray:
    # Function that calculates the acceleration of an aircraft during ground roll
    speed = fast_norm(v)
    D = 0.5 * rho * speed**2 * Sref * CD_zero_flap
    L = 0.5 * rho * speed**2 * Sref * CL_zero_flap
    a_x = (T_takeoff - D - 0.03*(Weight-L)) / m              # calculate x direction acceleration 
    return np.array([a_x, 0, 0])

def calculate_acceleration_groundrotation(v, m, Weight,
                                          Sref,
                                          CD_max_flap,CL_max_flap,
                                          T_takeoff)->np.ndarray:
    # Function that calculate the acceleration of the aircraft during rotation for takeoff
    speed = fast_norm(v)
    D = 0.5 * rho * speed**2 * Sref * CD_max_flap
    L = 0.5 * rho * speed**2 * Sref * CL_max_flap
    a_x = (T_takeoff - D - 0.03*(Weight-L)) / m            # calculate x direction acceleration 
    return np.array([a_x, 0, 0])

def calculate_acceleration_level(v, m, Sref, CD_func, alpha_deg, T):
    # Function that calculates the acceleration during level flight
    speed = fast_norm(v)
    CD = float(CD_func(alpha_deg))
    D = 0.5 * rho * speed**2 * Sref * CD
    a_x = (T * np.cos(np.radians(alpha_deg)) - D) / m
    return np.array([a_x, 0, 0])

def calculate_acceleration_climb(v, m, Weight, 
                                 Sref, 
                                 CL_func, CD_func, 
                                 CL_max_flap, CD_max_flap, 
                                 alpha_deg, gamma_rad, 
                                 T_climb, 
                                 over_flap_transition)->np.ndarray:
    # gamma rad : climb angle
    # over_flap_transition: checks if plane is over the flap transition (boolean)
    # Function that calculates the acceleration during climb
    CL=0
    CD=0
    speed = fast_norm(v)
    if (over_flap_transition):
        CL = float(CL_func(alpha_deg))
        CD = float(CD_func(alpha_deg))
    else:
        CL = CL_max_flap
        CD = CD_max_flap
    theta_deg = np.degrees(gamma_rad) + alpha_deg
    theta_rad = np.radians(theta_deg)
    
    D = 0.5 * rho * speed**2 * Sref * CD
    L = 0.5 * rho * speed**2 * Sref * CL

    a_x = (T_climb * np.cos(theta_rad) - L * np.sin(gamma_rad) - D * np.cos(gamma_rad) )/ m
    a_z = (T_climb * np.sin(theta_rad) + L * np.cos(gamma_rad) - D * np.sin(gamma_rad) - Weight )/ m
    
    return np.array([a_x,0,a_z])

def get_state_df(stateLog):
    # Convert numpy arrays to lists for proper DataFrame conversion
    states_dict = []
    for state in stateLog:
        state_dict = {
            'mission' : state.mission,
            "N_laps" : state.N_laps,
            'position': np.array([state.position[0],state.position[1],state.position[2]]),
            'velocity': np.array([state.velocity[0],state.velocity[1],state.velocity[2]]),
            'acceleration': np.array([state.acceleration[0],state.acceleration[1],state.acceleration[2]]),
            'time': state.time,
            'throttle': state.throttle,
            'thrust' : state.thrust,
            'loadfactor': state.loadfactor,
            'AOA': state.AOA,
            'climb_pitch_angle': state.climb_pitch_angle,
            'bank_angle': state.bank_angle,
            'phase': state.phase,
            'battery_SoC': state.battery_SoC,
            'battery_voltage': state.battery_voltage,
            'Amps' : state.Amps,
            'motor_input_power': state.motor_input_power
        }
        states_dict.append(state_dict)
    return pd.DataFrame(states_dict)

def visualize_mission(stateLog):
    """Generate all visualization plots for the mission in a single window"""
    stateLog = get_state_df(stateLog)

    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(4, 4)
    
    # Get phases and colors
    phases = stateLog['phase'].unique()
    # colors = plt.cm.rainbow(np.random.rand(len(phases)))
    color_list = ['red', 'green', 'blue', 'orange', 'black']
    colors = [color_list[i % len(color_list)] for i in range(len(phases))]
    
    # --- 1) 3D Trajectory 전용 Figure ---
    fig3d = plt.figure(figsize=(6,6), dpi=100)
    ax3d = fig3d.add_subplot(111, projection='3d')
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax3d.plot(
            stateLog[mask]['position'].apply(lambda x:x[0]),
            stateLog[mask]['position'].apply(lambda x:x[1]),
            stateLog[mask]['position'].apply(lambda x:x[2]),
            color=color, label=f'Phase {phase}'
        )
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Trajectory')
    ax3d.legend(loc='upper left', fontsize=8)
    # 등비로 축 맞추기
    x_lims = ax3d.get_xlim3d(); y_lims = ax3d.get_ylim3d(); z_lims = ax3d.get_zlim3d()
    max_range = max(x_lims[1]-x_lims[0], y_lims[1]-y_lims[0])
    x_c, y_c = np.mean(x_lims), np.mean(y_lims)
    ax3d.set_xlim3d(x_c-max_range/2, x_c+max_range/2)
    ax3d.set_ylim3d(y_c-max_range/2, y_c+max_range/2)
    ax3d.set_zlim3d(0, z_lims[1]*1.2)
    plt.tight_layout()
    plt.show()


    # --- 2) 나머지 6개의 2D 플롯을 6×1 배열로 ---
    fig2, axs = plt.subplots(5, 1,
                        figsize=(8, 16),  # 세로를 충분히
                        dpi=100,
                        sharex=False)      # 공통 x축 쓰면 축 레이블 하나로 줄일 수 있습니다

    # # (1) Top-Down View
    # ax = axs[0]
    # for phase, color in zip(phases, colors):
    #     mask = stateLog['phase'] == phase
    #     ax.plot(
    #         stateLog[mask]['position'].apply(lambda x:x[0]),
    #         stateLog[mask]['position'].apply(lambda x:x[1]),
    #         color=color
    #     )
    # ax.set_title('Top-Down')
    # ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
    # ax.grid(True); ax.set_aspect('equal')

    ax = axs[0]
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax.plot(stateLog[mask]['time'], stateLog[mask]['AOA'], color=color)
    ax.set_ylabel('AOA (°)')
    ax.grid(True)

    # (2) Speed
    ax = axs[1]
    speeds = np.sqrt(stateLog['velocity'].apply(lambda v: v[0]**2+v[1]**2+v[2]**2))
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax.plot(stateLog[mask]['time'], speeds[mask], color=color)
    ax.set_ylabel('Speed (m/s)')
    ax.grid(True)

    # (3) Throttle
    ax = axs[2]
    ax.plot(stateLog['time'], stateLog['throttle']*100, 'r-')
    ax.set_ylabel('Throttle (%)')
    ax.grid(True)

    # (4) SoC & Voltage
    ax = axs[3]
    ax.plot(stateLog['time'], stateLog['battery_SoC'], 'b-', label='SoC')
    ax.set_ylabel('SoC (%)', color='b')
    ax.tick_params(labelcolor='b')
    ax2 = ax.twinx()
    ax2.plot(stateLog['time'], stateLog['battery_voltage'], 'r-', label='Volt')
    ax2.set_ylabel('Voltage (V)', color='r')
    ax2.tick_params(labelcolor='r')
    ax.grid(True)

    # (5) Current (마지막 축)
    ax = axs[4]
    ax.plot(stateLog['time'], stateLog['Amps'], 'r-')
    ax.set_ylabel('Current (A)')
    ax.set_xlabel('Time (s)')
    ax.grid(True)
    
    fig2.suptitle(f"Total flight time : {stateLog['time'].iloc[-1]:.2f}s", y=0.98)
    fig2.subplots_adjust(
    top=0.93,    # suptitle과 서브플롯 사이 간격
    bottom=0.05, # 맨 아래 서브플롯과 바닥 사이
    left=0.1,    # 왼쪽 여백
    right=0.95,  # 오른쪽 여백
    hspace=0.4   # 서브플롯 간 수직 간격
)
    plt.tight_layout()
    plt.show()


    # --- 3) 기존 주석 처리된 나머지 그래프들 (필요 시 활성화) ---
    """
    # Graph2 : Side view colored by phase
    fig_side = plt.figure(figsize=(6,4), dpi=100)
    ax_side = fig_side.add_subplot(111)
    for phase, color in zip(phases, colors):
        mask = stateLog['phase'] == phase
        ax_side.plot(
            stateLog[mask]['position'].apply(lambda x: x[0]),
            stateLog[mask]['position'].apply(lambda x: x[2]),
            color=color, label=f'Phase {phase}'
        )
    ax_side.set_xlabel('X Position (m)')
    ax_side.set_ylabel('Altitude (m)')
    ax_side.set_title('Side View')
    ax_side.grid(True)
    ax_side.set_aspect('equal')
    plt.tight_layout()
    plt.show()


    # Graph5 : Bank, Pitch angle
    fig_bp = plt.figure(figsize=(6,4), dpi=100)
    ax_bp = fig_bp.add_subplot(111)
    ax_bp.plot(stateLog['time'], stateLog['bank_angle'], label='Bank Angle', color='blue')
    ax_bp.plot(stateLog['time'], stateLog['climb_pitch_angle'], label='Climb Pitch Angle', color='red')
    ax_bp.set_xlabel('Time (s)'); ax_bp.set_ylabel('Angle (°)')
    ax_bp.set_title('Bank & Pitch Angles'); ax_bp.grid(True)
    ax_bp.legend(); plt.tight_layout(); plt.show()


    # Graph8 : Thrust
    fig_th = plt.figure(figsize=(6,4), dpi=100)
    ax_th = fig_th.add_subplot(111)
    ax_th.plot(stateLog['time'], stateLog['thrust'], 'r-')
    ax_th.set_xlabel('Time (s)'); ax_th.set_ylabel('Thrust (kg)')
    ax_th.set_title('Thrust'); ax_th.grid(True)
    plt.tight_layout(); plt.show()


    # Graph9 : Load factor
    fig_lf = plt.figure(figsize=(6,4), dpi=100)
    ax_lf = fig_lf.add_subplot(111)
    for phase, color in zip(phases, colors):
        mask = stateLog['phase']==phase
        ax_lf.plot(stateLog[mask]['time'], stateLog[mask]['loadfactor'], color=color)
    ax_lf.set_xlabel('Time (s)'); ax_lf.set_ylabel('Load Factor')
    ax_lf.set_title('Load Factor by Phase'); ax_lf.grid(True)
    plt.tight_layout(); plt.show()


    # Graph12 : Mission Phase Step
    fig_ph = plt.figure(figsize=(6,2), dpi=100)
    ax_ph = fig_ph.add_subplot(111)
    ax_ph.step(stateLog['time'], stateLog['phase'], where='post', color='purple')
    ax_ph.set_xlabel('Time (s)'); ax_ph.set_ylabel('Phase')
    ax_ph.set_title('Mission Phases'); ax_ph.set_yticks(phases); ax_ph.grid(True)
    plt.tight_layout(); plt.show()
    """


if __name__=="__main__":

    a=loadAnalysisResults("'700773271413233544'")
    
    param = MissionParameters(
        m_takeoff= 10,
        max_speed= 40,                       # Fixed
        max_load_factor = 4.0,               # Fixed
        climb_thrust_ratio = 0.9,
        level_thrust_ratio = 0.5,
        turn_thrust_ratio = 0.5,
                      
        propeller_data_path = "data/propDataCSV/PER3_8x6E.csv", 

    )
    
    presetValues = PresetValues(
        m_x1 = 200,                         # g
        x1_time_margin = 150,                # sec
        
        throttle_takeoff = 0.6,             # 0~1
        max_climb_angle = 40,                 #deg
        max_load = 30,                      # kg
        h_flap_transition = 5,              # m
        
        number_of_motor = 2,            
        min_battery_voltage = 21.8,         # V 
        score_weight_ratio = 0.5            # mission2/3 score weight ratio (0~1)
        )
        
    propulsionSpecs = PropulsionSpecs(
        M2_propeller_data_path = "data/propDataCSV/PER3_8x6E.csv",
        M3_propeller_data_path = "data/propDataCSV/PER3_8x6E.csv",
        battery_data_path = "data/batteryDataCSV/Maxamps_2250mAh_6S.csv",
        Kv = 109.91,
        R = 0.062,
        number_of_battery = 2,
        n_cell = 6,
        battery_Wh = 49.95,
        max_current = 60,
        max_power = 1332    
    )
        
    missionAnalyzer = MissionAnalyzer(a, param, presetValues, propulsionSpecs)
    print(missionAnalyzer.run_mission3())
    visualize_mission(missionAnalyzer.stateLog)
 
