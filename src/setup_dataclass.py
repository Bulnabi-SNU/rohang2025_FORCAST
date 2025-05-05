from dataclasses import dataclass

@dataclass
class PresetValues:
    m_x1: float
    x1_time_margin: float
    
    throttle_takeoff : float
    max_climb_angle : float
    max_load : float
    
    h_flap_transition : float
    
    number_of_motor : int
    
    min_battery_voltage: float
    
    score_weight_ratio: float=1

@dataclass
class PropulsionSpecs:
    M2_propeller_data_path: str
    M3_propeller_data_path : str
    battery_data_path : str
    Kv : float
    R : float
    number_of_battery : int
    n_cell : int
    battery_Wh : float
    max_current : float
    max_power : float
    
@dataclass
class AerodynamicSetup:
    
    alpha_start : float
    alpha_end : float
    alpha_step : float
    fuselage_cross_section_area : float
    fuselage_Cd_datapath : str
    AOA_stall : str
    AOA_takeoff_max : str
    AOA_climb_max : str
    AOA_turn_max : str    
    
    
@dataclass
class AircraftParamConstraints:
    # wing parameter ranges
    span_max: float
    span_min: float
    span_interval: float
    AR_max: float
    AR_min: float
    AR_interval: float
    taper_max: float
    taper_min: float
    taper_interval: float
    twist_max: float
    twist_min: float
    twist_interval: float
    airfoil_list : list


@dataclass
class MissionParamConstraints:

    MTOW_min : float
    MTOW_max : float
    MTOW_analysis_interval: float

    M2_max_speed_min : float
    M2_max_speed_max : float
    M3_max_speed_min : float
    M3_max_speed_max : float
    max_speed_analysis_interval : float

    #Constraints for calculating mission2
    M2_climb_thrust_ratio_min : float
    M2_climb_thrust_ratio_max : float
    M2_turn_thrust_ratio_min : float
    M2_turn_thrust_ratio_max : float
    M2_level_thrust_ratio_min : float
    M2_level_thrust_ratio_max : float
    M2_thrust_analysis_interval : float
    
    #Constraints for calculating mission3  
    M3_climb_thrust_ratio_min : float
    M3_climb_thrust_ratio_max : float
    M3_turn_thrust_ratio_min : float
    M3_turn_thrust_ratio_max : float
    M3_level_thrust_ratio_min : float
    M3_level_thrust_ratio_max : float
    M3_thrust_analysis_interval : float
    
    wing_loading_min : float
    wing_loading_max : float