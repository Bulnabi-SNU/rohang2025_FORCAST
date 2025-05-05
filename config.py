from src.internal_dataclass import *
from src.setup_dataclass import *

def get_config():
    presetValues = PresetValues(
        m_x1 = 200,                             # g
        x1_time_margin = 10,                    # sec
        
        throttle_takeoff = 0.55,                 # 0~1
        max_climb_angle = 40,                   # deg
        max_load = 30,                          # kg
        h_flap_transition = 5,                  # m
        
        number_of_motor = 2,               
        min_battery_voltage = 21.8,             # V 
        score_weight_ratio = 1               # mission2/3 score weight ratio (0~1)
        )
    
    propulsionSpecs = PropulsionSpecs(
        M2_propeller_data_path = "data/propDataCSV/PER3_14x55MR.csv",
        M3_propeller_data_path = "data/propDataCSV/PER3_8x6E.csv",
        battery_data_path = "data/batteryDataCSV/Maxamps_2250mAh_6S.csv",
        Kv = 55,                            # (rad/s) / V
        R = 0.09,                              # Ohm
        number_of_battery = 1,
        n_cell = 6,
        battery_Wh = 2200,                     # Wh
        max_current = 115,                       # A
        max_power = 2553                        # W
    )
    
    aircraftParamConstraints = AircraftParamConstraints (
  
        span_min = 1800.0,                      # mm
        span_max = 2800.0,                   
        span_interval = 100.0,
    
        AR_min = 4.0,                       
        AR_max = 7.0,
        AR_interval = 0.5,
        
        taper_min = 0.50,
        taper_max = 0.90,                      
        taper_interval = 0.05,
        
        twist_min = 3.0,                        # degree
        twist_max = 0.0,                     
        twist_interval = 1.0,
        
        airfoil_list = ['sg6043','naca4412','e216','s4022','naca2412']
        # airfoil_list = ['sg6043','s9027','hq3011','s4022']
        #airfoil_list = ['e216']
        )
    
    aerodynamicSetup = AerodynamicSetup(
        alpha_start = -10,                       # degree
        alpha_end = 15,
        alpha_step = 1,
        fuselage_cross_section_area = 19427,    # mm2
        fuselage_Cd_datapath = "data/fuselageDragCSV/fuselageDragCoefficients.csv",
        AOA_stall = 13,                         # degree
        AOA_takeoff_max = 10,
        AOA_climb_max = 8,
        AOA_turn_max = 8  
    )

    baseAircraft = Aircraft(
        m_fuselage = 2500,
        wing_area_blocked_by_fuselage = 72640,  # mm2
        wing_density = 0.00002,               # g/mm3

        mainwing_span = 1800,                   # mm
        mainwing_AR = 5.45,           
        mainwing_taper = 0.65,        
        mainwing_twist = 0.0,                   # degree
        mainwing_sweepback = 0,                 # degree
        mainwing_dihedral = 0.0,                # degree
        mainwing_incidence = 0.0,               # degree

        flap_start = [0.182, 0.402],            # spanwise ratio(0~1)
        flap_end = [0.335, 0.628],              # spanwise ratio(0~1)
        flap_angle = [20.0, 20.0],              # degree
        flap_c_ratio = [0.35, 0.35],            # chordwise ratio(0~1)

        horizontal_volume_ratio = 0.7,          
        horizontal_area_ratio = 0.25,           
        horizontal_AR = 4.0,                    
        horizontal_taper = 1,               
        horizontal_ThickChord = 8,              

        vertical_volume_ratio = 0.053,          
        vertical_taper = 0.6,        
        vertical_ThickChord = 9,  
        
        mainwing_airfoil_datapath = "data/airfoilDAT/sg6043.dat",
        horizontal_airfoil_datapath= "data/airfoilDAT/naca0008.dat",
        vertical_airfoil_datapath= "data/airfoilDAT/naca0009.dat"
        
        )

    missionParamConstraints = MissionParamConstraints (
                
                MTOW_min = 6,                     # Kg
                MTOW_max = 6,                  
                MTOW_analysis_interval = 0.2,
                
                M2_max_speed_min = 25,              # m/s
                M2_max_speed_max = 25,
                M3_max_speed_min = 24,
                M3_max_speed_max = 24,
                max_speed_analysis_interval = 2,    
                
                #Constraints for calculating mission2
                M2_climb_thrust_ratio_min = 0.7, 
                M2_climb_thrust_ratio_max = 0.7,
                M2_turn_thrust_ratio_min = 0.8,
                M2_turn_thrust_ratio_max = 0.8,
                M2_level_thrust_ratio_min = 0.75,
                M2_level_thrust_ratio_max = 0.75,
                M2_thrust_analysis_interval = 0.05,
    
                #Constraints for calculating mission3  
                M3_climb_thrust_ratio_min = 0.9,
                M3_climb_thrust_ratio_max = 0.9,
                M3_turn_thrust_ratio_min = 0.4,
                M3_turn_thrust_ratio_max = 0.4,
                M3_level_thrust_ratio_min = 0.5,
                M3_level_thrust_ratio_max = 0.5,
                M3_thrust_analysis_interval = 0.05,
                
                wing_loading_min = 0,               # kg/m2
                wing_loading_max = 15
                )
        

    return (presetValues, propulsionSpecs, aircraftParamConstraints, 
            aerodynamicSetup, baseAircraft, missionParamConstraints)