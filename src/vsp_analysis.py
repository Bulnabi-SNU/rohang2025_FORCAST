import openvsp as vsp
import csv
import numpy as np
import json
from typing import List
from dataclasses import asdict
import ast
import os 
import os.path
import math
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator
from src.internal_dataclass import PhysicalConstants, Aircraft, AircraftAnalysisResults
from src.setup_dataclass import PresetValues


class VSPAnalyzer:
    def __init__(self, presets: PresetValues, 
                 dataPath: str="data", outputPath: str="out"):
        self.constants = PhysicalConstants
        self.presets = presets
        self.dataPath = dataPath
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        self.outputPath = outputPath
        vsp.VSPCheckSetup()

    def clean(self) -> None:
        vsp.ClearVSPModel()
        vsp.VSPRenew()
          
        
    def setup_vsp_model(self, aircraft: Aircraft,vspPath:str = "Mothership.vsp3") -> None:
        """Creates or updates OpenVSP model based on aircraft parameters"""
        self.aircraft = aircraft
        self.wing_id = self.createMainWing(aircraft)
        self.flap_id = self.createFlap(aircraft)

        self.horizontal_tail_id = self.createHorizontalTailWing(aircraft)
        self.vertical_tail_R_id,self.vertical_tail_L_id = self.createVerticalTailWings(aircraft)
        
        vsp.Update()
        vsp.WriteVSPFile(os.path.join(self.outputPath,vspPath),vsp.SET_ALL)
        
    def calculateCoefficients(self, fileName:str = "Mothership.vsp3", 
                              alpha_start: float=0, alpha_end: float=1, alpha_step:float=0.5, 
                              CD_fuse: np.ndarray=np.zeros(4),
                              fuselage_cross_section_area: float=20000,
                              wing_area_blocked_by_fuselage : float=72640,
                              AOA_stall:float=13, 
                              AOA_takeoff_max:float=10,
                              AOA_climb_max:float=8,
                              AOA_turn_max:float=8,
                              Re:float=700000, Mach:float=0,
                              boom_density_2624:float = 0.121, 
                              boom_density_1008:float = 0.049,
                              boom_density_0604:float = 0.042,
                              boom_density_boom:float=0.121,
                              clearModel:bool=True):
        #print("Starting Analysis")
        # Calculate coefficients with flaps at zero
        results_no_flap = self._calculate_coeffs_helper(fileName, alpha_start, alpha_end, alpha_step,
                                                        Re, Mach, wing_area_blocked_by_fuselage,boom_density_2624, boom_density_1008,
                                                        boom_density_0604, boom_density_boom, clearModel, 0.0)
        
        # Find index closest to AOA_stall and zero
        alpha_list = results_no_flap['alpha_list']

        # Calculate coefficients with flaps at max angle
        results_flap_max = self._calculate_coeffs_helper(fileName, 0, AOA_takeoff_max, AOA_takeoff_max,
                                                        Re, Mach, wing_area_blocked_by_fuselage, boom_density_2624, boom_density_1008,
                                                        boom_density_0604, boom_density_boom, False, self.aircraft.flap_angle[0],
                                                        do_mass_analysis=False)

        # Get corresponding CL/CD values
        CL_flap_max = results_flap_max['CL'][1]
        CD_flap_max = results_flap_max['CD'][1]
        CL_flap_zero = results_flap_max['CL'][0]
        CD_flap_zero = results_flap_max['CD'][0]
    
        CD_fuse = CD_fuse * (fuselage_cross_section_area / results_no_flap['Sref']) 
        zero_index = int((0-alpha_start)/alpha_step)
        
        #print("Finished Analysis for this configuration.")
        
        return AircraftAnalysisResults(
                aircraft=self.aircraft,
                alpha_list=results_no_flap['alpha_list'],
                m_empty=results_no_flap['m_empty'],
                m_boom=results_no_flap['m_boom'],
                m_wing=results_no_flap['m_wing'],
                span=results_no_flap['span'],
                AR=results_no_flap['AR'], 
                taper=results_no_flap['taper'],
                twist=results_no_flap['twist'],
                Sref=results_no_flap['Sref'],
                Lw=results_no_flap['Lw'],
                Lh=results_no_flap['Lh'],
                CL=results_no_flap['CL'],
                CD_wing=results_no_flap['CD'],
                CD_fuse=CD_fuse,
                CD_total=results_no_flap['CD'] + CD_fuse,
                AOA_stall=AOA_stall,
                AOA_takeoff_max=AOA_takeoff_max, 
                AOA_climb_max=AOA_climb_max,
                AOA_turn_max=AOA_turn_max,
                CL_flap_max=CL_flap_max,
                CL_flap_zero=CL_flap_zero,
                CD_flap_max=CD_flap_max + CD_fuse[-1],
                CD_flap_zero=CD_flap_zero + CD_fuse[zero_index],
                max_load=self.presets.max_load
        )

    def _calculate_coeffs_helper(self, fileName, alpha_start, alpha_end, alpha_step,
                                Re, Mach, wing_area_blocked_by_fuselage, boom_density_2624, boom_density_1008,
                                boom_density_0604, boom_density_boom, clearModel, flap_angle, 
                                do_mass_analysis=True, do_geom_analysis=True):
        """Helper method to calculate coefficients for a given flap angle"""
        
        point_number = round(int((alpha_end - alpha_start) / alpha_step) + 1)
        
        if(clearModel):
            vsp.ClearVSPModel()
        vsp.VSPRenew()
        
        if(clearModel):
            if not os.path.exists(os.path.join(self.outputPath,fileName)):
                raise FileNotFoundError(f"Model file {fileName} not found.")
                
        vsp.ReadVSPFile(os.path.join(self.outputPath,fileName))
        
        # Set flap angle
        for i, flap_id in enumerate(self.flap_id):
            vsp.SetParmVal(flap_id[0], flap_angle)
            vsp.SetParmVal(flap_id[1], -flap_angle)
        # Geometric analysis
        
        if do_geom_analysis:
            #print("> Starting Geometric Analysis")
            geom_analysis = "VSPAEROComputeGeometry"
            vsp.SetAnalysisInputDefaults(geom_analysis)
            vsp.ExecAnalysis(geom_analysis)

            # Configure VSPAERO
            vsp.SetVSPAERORefWingID(self.wing_id) ##### wing_id 확인할것

            span = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalSpan","WingGeom"))
            AR = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalAR","WingGeom"))
            taper = vsp.GetParmVal(vsp.GetParm(self.wing_id,"Taper","XSec_1"))
            twist = self.aircraft.mainwing_incidence - vsp.GetParmVal(vsp.GetParm(self.wing_id,"Twist","XSec_1"))
            Sref = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalArea","WingGeom"))
            wing_c_root = vsp.GetParmVal(vsp.GetParm(self.wing_id,"Root_Chord","XSec_1"))
            tail_c_root = vsp.GetParmVal(vsp.GetParm(self.horizontal_tail_id,"Root_Chord","XSec_1"))
            #print("> Finished Geometric Analysis")
        else:
            #print("> Skipping Geometric Analysis")
            span = 0
            AR = 0
            taper = 0
            twist = 0
            Sref = 0
            wing_c_root = 0
            tail_c_root = 0
        
        if do_mass_analysis:
            # Mass Analysis
            #print("> Starting Mass Analysis")
            vsp.ComputeMassProps(0, 100, 0)
            mass_results_id = vsp.FindLatestResultsID("Mass_Properties")
            mass_data = vsp.GetDoubleResults(mass_results_id, "Total_Mass")

            span_Projected = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalProjectedSpan","WingGeom"))
            chord_Mean = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalChord","WingGeom"))
            lh = self.aircraft.horizontal_volume_ratio * chord_Mean / self.aircraft.horizontal_area_ratio
            horizontal_distance = chord_Mean/4 + lh - tail_c_root/4

            m_wing = mass_data[0] + span * (boom_density_1008 + boom_density_2624 + boom_density_0604)
            
            m_boom = horizontal_distance * boom_density_boom * 2
            
            m_device = 172 + 106 + 95 + 1700
            
            m_empty = m_wing + m_boom + self.aircraft.m_fuselage + self.presets.m_x1 + m_device
            
            mass_center_x = 120 # Calculated by CG Calculater, static margin 10%

            # Aerodynamic Center
            w_ac = 0.25 * 2/3 * wing_c_root * (1 + taper + taper ** 2) / (1 + taper)
            h_ac = w_ac + lh
            Lw = w_ac - mass_center_x
            Lh = h_ac - mass_center_x
            tail_effect = float((Lh-Lw)/Lh)

            #print("> Finished Mass Analysis")
        else:
            #print("> Skipping Mass Analysis")
            m_empty, m_boom, m_wing = 0, 0, 0
            Lw, Lh = 0, 0
            tail_effect = 1

        # Configure sweep analysis for coefficient
        #print("> Starting Sweep Analysis")
        
        sweep_analysis = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(sweep_analysis)
        vsp.SetIntAnalysisInput(sweep_analysis, "AnalysisMethod", [vsp.VORTEX_LATTICE])
        vsp.SetIntAnalysisInput(sweep_analysis, "GeomSet", [vsp.SET_ALL])
        
        # **Set the reference geometry set**
        vsp.SetDoubleAnalysisInput(sweep_analysis, "MachStart", [Mach])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "ReCref", [Re])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "AlphaStart", [alpha_start])
        vsp.SetDoubleAnalysisInput(sweep_analysis, "AlphaEnd", [alpha_end])
        vsp.SetIntAnalysisInput(sweep_analysis, "AlphaNpts", [point_number])
        
        # Number of CPUs
        vsp.SetIntAnalysisInput(sweep_analysis, "NCPU", [1])
        vsp.SetIntAnalysisInput(sweep_analysis,"FixedWakeFlag",[1])
        vsp.SetIntAnalysisInput(sweep_analysis,"NumWakeNodes",[64])
        
        
        # Redirect log to null
        vsp.SetStringAnalysisInput( "VSPAEROSweep", "RedirectFile", "" )

        # Disable CpSlice
        aero_id = vsp.FindContainer("VSPAEROSettings",0);
        vsp.SetParmVal(aero_id,"CpSliceFlag","VSPAERO",0);
        vsp.Update()
        
        # Execute sweep analysis
        sweep_results_id = vsp.ExecAnalysis(sweep_analysis)
        
        #print("> Finished Sweep Analysis")

        # Extract coefficient data
        sweepResults = vsp.GetStringResults(sweep_results_id, "ResultsVec")
        
        effective_wing_area_factor = (Sref - wing_area_blocked_by_fuselage) / Sref
        
        alpha_list = np.zeros(point_number)
        CL_list = np.zeros(point_number)
        CDwing_list =np.zeros(point_number)
                
        for i in range (point_number):
            alpha_list[i]= vsp.GetDoubleResults(sweepResults[i], "Alpha")[-1]

            CL_list[i]= vsp.GetDoubleResults(sweepResults[i], "CL")[-1]

            CDwing_list[i] = vsp.GetDoubleResults(sweepResults[i], "CDtot")[-1]

        CL_list = [cl * tail_effect * effective_wing_area_factor for cl in CL_list]
        CL_list = np.array(CL_list)
        CDwing_list = [cd * effective_wing_area_factor for cd in CDwing_list] 
        CDwing_list = np.array(CDwing_list)

        return {
            'alpha_list': alpha_list,
            'm_empty': m_empty,
            'm_boom': m_boom,
            'm_wing': m_wing,
            'span': span,
            'AR': AR,
            'taper': taper,
            'twist': twist,
            'Sref': Sref,
            'Lw': Lw,
            'Lh': Lh,
            'CL': CL_list,
            'CD': CDwing_list
        }

    def createMainWing(self, aircraft: Aircraft) -> str:

        """ Create Main Wing, Included Parameters are FIXED """
        # Main Wing ID
        wing_id = vsp.AddGeom("WING", "")
        vsp.SetGeomName(wing_id,"Main Wing")
        
        # Main Wing Settings
        vsp.SetDriverGroup(wing_id, 1, vsp.AR_WSECT_DRIVER, vsp.SPAN_WSECT_DRIVER, vsp.TAPER_WSECT_DRIVER)
        vsp.SetParmVal(wing_id, "Span", "XSec_1", aircraft.mainwing_span / 2)  # Half span of the each wing 
        vsp.SetParmVal(wing_id, "Aspect", "XSec_1", aircraft.mainwing_AR / 2)  
        vsp.SetParmVal(wing_id, "Taper", "XSec_1", aircraft.mainwing_taper) 
        vsp.SetParmVal(wing_id, "Twist", "XSec_1", aircraft.mainwing_incidence - aircraft.mainwing_twist)
        
        vsp.SetParmVal(wing_id, "Dihedral", "XSec_1", aircraft.mainwing_dihedral)
        vsp.SetParmVal(wing_id, "Twist", "XSec_0", aircraft.mainwing_incidence)
        vsp.SetParmVal(wing_id, "Sweep", "XSec_1", aircraft.mainwing_sweepback)
        vsp.SetParmVal(wing_id, "Sweep_Location", "XSec_1", 0)

        vsp.SetParmVal(wing_id, "X_Rel_Location", "XForm", 0) 
        vsp.SetParmVal(wing_id, "Y_Rel_Location", "XForm", 0)
        vsp.SetParmVal(wing_id, "Z_Rel_Location", "XForm", 0)

        vsp.SetParmVal(wing_id, "Density", "Mass_Props", aircraft.wing_density)
        vsp.Update()
        
        # Airfoil Selection
        vsp.ChangeXSecShape(vsp.GetXSecSurf(wing_id,0),0,vsp.XS_FILE_AIRFOIL)
        vsp.ChangeXSecShape(vsp.GetXSecSurf(wing_id,0),1,vsp.XS_FILE_AIRFOIL)
        xsec_0 = vsp.GetXSec(vsp.GetXSecSurf(wing_id,0),0)
        xsec_1 = vsp.GetXSec(vsp.GetXSecSurf(wing_id,0),1)
        vsp.ReadFileAirfoil(xsec_0,aircraft.mainwing_airfoil_datapath)
        vsp.ReadFileAirfoil(xsec_1,aircraft.mainwing_airfoil_datapath)
        vsp.Update()

        return wing_id

    def createFlap(self,aircraft:Aircraft) -> List[List[str]]:
        
        if(not(len(self.aircraft.flap_start) == len(self.aircraft.flap_end) == \
            len(self.aircraft.flap_angle) == len(self.aircraft.flap_c_ratio) )):
            pass
           #raise ValueError("Flap config array lengths don't match!")

        flap_id_list = []

        ## VSP uses 1-indexing
        for i in range(1,len(self.aircraft.flap_start)+1):
            flap_angle = aircraft.flap_angle[i-1]
            flap_start = aircraft.flap_start[i-1]
            flap_end = aircraft.flap_end[i-1]
            flap_c_ratio = aircraft.flap_c_ratio[i-1]

            flap_id = vsp.AddSubSurf(self.wing_id,vsp.SS_CONTROL)

            vsp.SetSubSurfName(self.wing_id, flap_id, "Flaps"+str(i))

            vsp.SetParmVal(self.wing_id,"EtaFlag","SS_Control_"+str(i), 1)
            vsp.SetParmVal(self.wing_id,"EtaStart","SS_Control_"+str(i), flap_start)
            vsp.SetParmVal(self.wing_id,"EtaEnd","SS_Control_"+str(i), flap_end)
            vsp.SetParmVal(self.wing_id,"Length_C_Start","SS_Control_"+str(i), flap_c_ratio)
            
            # Flap settings
            flap_group_l = vsp.CreateVSPAEROControlSurfaceGroup()
            flap_group_r = vsp.CreateVSPAEROControlSurfaceGroup()
    
    
            vsp.SetVSPAEROControlGroupName("Flap"+str(i)+"_l", flap_group_l)
            vsp.AddSelectedToCSGroup([1+2*(i-1)], flap_group_l)
            vsp.SetVSPAEROControlGroupName("Flap"+str(i)+"_r", flap_group_r)
            vsp.AddSelectedToCSGroup([2+2*(i-1)], flap_group_r)
    
            container_id = vsp.FindContainer("VSPAEROSettings", 0)

            # ControlSurfaceGroup 이름이 추가할때마다 1씩 증가함.
            flap_group_id_l = vsp.FindParm(container_id, "DeflectionAngle", 
                                           "ControlSurfaceGroup_"+str(0+2*(i-1)))
            flap_group_id_r = vsp.FindParm(container_id, "DeflectionAngle", 
                                           "ControlSurfaceGroup_"+str(1+2*(i-1)))
    
            vsp.SetParmVal(flap_group_id_l, flap_angle)
            vsp.SetParmVal(flap_group_id_r, -flap_angle)
            vsp.Update()

            flap_id_list.append([flap_group_id_l,flap_group_id_r])

        return flap_id_list

    def createHorizontalTailWing(self, aircraft:Aircraft,airfoilName:str="naca0008.dat") -> str:
        """ Create Horizontal Tail, Included Parameters are FIXED """

        # Horizontal Tail ID
        tailwing_id = vsp.AddGeom("WING", "")
        vsp.SetGeomName(tailwing_id,"Tail Wing")
        
        # Airfoil Selection
        vsp.ChangeXSecShape(vsp.GetXSecSurf(tailwing_id,0),0,vsp.XS_FILE_AIRFOIL)
        vsp.ChangeXSecShape(vsp.GetXSecSurf(tailwing_id,0),1,vsp.XS_FILE_AIRFOIL)
        xsec_h_0 = vsp.GetXSec(vsp.GetXSecSurf(tailwing_id,0),0)
        xsec_h_1 = vsp.GetXSec(vsp.GetXSecSurf(tailwing_id,0),1)
        vsp.ReadFileAirfoil(xsec_h_0,aircraft.horizontal_airfoil_datapath)
        vsp.ReadFileAirfoil(xsec_h_1,aircraft.horizontal_airfoil_datapath)
        vsp.Update()
        
        # Fixed Parameters
        tailwing_sweep = 0
        tailwing_yoffset = 0
        tailwing_zoffset = 0
        tailwing_option_tip = 3
        tailwing_length_tip = 5
        tailwing_offset_tip = 0
        
        # Parameters related with Main Wing
        span_Projected = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalProjectedSpan","WingGeom"))
        chord_Mean = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalChord","WingGeom"))
        area_Projected = span_Projected * chord_Mean
        horizontal_area = aircraft.horizontal_area_ratio * area_Projected
        horizontal_span = math.sqrt(horizontal_area * aircraft.horizontal_AR)
        horizontal_root = (2 * horizontal_area) / ((1 + aircraft.horizontal_taper) * horizontal_span)
        horizontal_tip = horizontal_root * aircraft.horizontal_taper
        lh = aircraft.horizontal_volume_ratio * chord_Mean / aircraft.horizontal_area_ratio
        horizontal_distance = chord_Mean/4 + lh - horizontal_root/4
        
        
        # Horizontal Tail settings
        vsp.SetParmVal(tailwing_id, "Span", "XSec_1", horizontal_span / 2)  # Span of the each wing (Half of span)
        vsp.SetParmVal(tailwing_id, "Root_Chord", "XSec_1", horizontal_root)  
        vsp.SetParmVal(tailwing_id, "Tip_Chord", "XSec_1", horizontal_tip)  
        vsp.SetParmVal(tailwing_id, "Taper", "XSec_1", aircraft.horizontal_taper) 
        vsp.SetParmVal(tailwing_id, "Sweep", "XSec_1", tailwing_sweep) #Sweep Angle
        vsp.SetParmVal(tailwing_id, "X_Rel_Location", "XForm", horizontal_distance)  # Position along X-axis
        vsp.SetParmVal(tailwing_id, "Y_Rel_Location", "XForm", tailwing_yoffset)  # Position along Y-axis
        vsp.SetParmVal(tailwing_id, "Z_Rel_Location", "XForm", tailwing_zoffset)  # Position vertically

        vsp.SetParmVal(tailwing_id, "CapUMaxOption", "EndCap" , tailwing_option_tip)
        vsp.SetParmVal(tailwing_id, "CapUMaxLength", "EndCap" , tailwing_length_tip)
        vsp.SetParmVal(tailwing_id, "CapUMaxOffset", "EndCap" , tailwing_offset_tip)

        vsp.SetParmVal(tailwing_id, "Density", "Mass_Props", aircraft.wing_density)

        vsp.Update()

        return tailwing_id

    def createVerticalTailWings(self,aircraft:Aircraft,airfoilName:str="naca0009.dat") -> List[str]:

        """ Create Vertical Wing (Right), Included Parameters are FIXED """
        # Vertical Wing (Right) ID
        verwing_right_id = vsp.AddGeom("WING", "")
        vsp.SetGeomName(verwing_right_id,"Vertical Wing Right")
        
        # Airfoil Selection
        vsp.ChangeXSecShape(vsp.GetXSecSurf(verwing_right_id,0),0,vsp.XS_FILE_AIRFOIL)
        vsp.ChangeXSecShape(vsp.GetXSecSurf(verwing_right_id,0),1,vsp.XS_FILE_AIRFOIL)
        xsec_vr_0 = vsp.GetXSec(vsp.GetXSecSurf(verwing_right_id,0),0)
        xsec_vr_1 = vsp.GetXSec(vsp.GetXSecSurf(verwing_right_id,0),1)
        vsp.ReadFileAirfoil(xsec_vr_0,aircraft.vertical_airfoil_datapath)
        vsp.ReadFileAirfoil(xsec_vr_1,aircraft.vertical_airfoil_datapath)
        vsp.Update()
        
        # Fixed Parameters
        verwing_sweep = 0
        verwing_yoffset = 290
        verwing_zoffset = 0
        verwing_xRotate = 90
        verwing_option_tip = 3
        verwing_length_tip = 5
        verwing_offset_tip = 0
        
        # Parameters related with Main Wing
        span_Projected = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalProjectedSpan","WingGeom"))
        chord_Mean = vsp.GetParmVal(vsp.GetParm(self.wing_id,"TotalChord","WingGeom"))
        area_Projected = span_Projected * chord_Mean
        horizontal_area = aircraft.horizontal_area_ratio * area_Projected
        horizontal_span = math.sqrt(horizontal_area * aircraft.horizontal_AR)
        horizontal_root = (2 * horizontal_area) / ((1 + aircraft.horizontal_taper) * horizontal_span)
        
        # Parameters related with Main, Horizontal 
        chord_Mean_horizontal = vsp.GetParmVal(vsp.GetParm(self.horizontal_tail_id,"TotalChord","WingGeom"))
        lh = aircraft.horizontal_volume_ratio * chord_Mean / aircraft.horizontal_area_ratio
        vertical_area = aircraft.vertical_volume_ratio * span_Projected * area_Projected / lh # vertical_distance = horizontal_distance
        vertical_c_root = chord_Mean_horizontal
        horizontal_distance = chord_Mean/4 + lh - horizontal_root/4
        
        # Vertical Tail settings
        vsp.SetDriverGroup(verwing_right_id, 1, vsp.AREA_WSECT_DRIVER, vsp.TAPER_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER)
        vsp.SetParmVal(verwing_right_id, "Area", "XSec_1", vertical_area / 2)  # Span of the each wing (Half of span)
        vsp.SetParmVal(verwing_right_id, "Taper", "XSec_1", aircraft.vertical_taper)  
        vsp.SetParmVal(verwing_right_id, "Root_Chord", "XSec_1", vertical_c_root) 
        vsp.SetParmVal(verwing_right_id, "Sweep", "XSec_1", verwing_sweep) #Sweep Angle
        vsp.SetParmVal(verwing_right_id, "Sweep_Location", "XSec_1", 0.99)
        vsp.SetParmVal(verwing_right_id, "X_Rel_Location", "XForm", horizontal_distance)  # Position along X-axis
        vsp.SetParmVal(verwing_right_id, "Y_Rel_Location", "XForm", verwing_yoffset)  # Position along Y-axis
        vsp.SetParmVal(verwing_right_id, "Z_Rel_Location", "XForm", verwing_zoffset)  # Position vertically
        vsp.SetParmVal(verwing_right_id, "X_Rel_Rotation", "XForm", verwing_xRotate)  # X-axis Rotation
        vsp.SetParmVal(verwing_right_id, "CapUMaxOption", "EndCap" , verwing_option_tip)
        vsp.SetParmVal(verwing_right_id, "CapUMaxLength", "EndCap" , verwing_length_tip)
        vsp.SetParmVal(verwing_right_id, "CapUMaxOffset", "EndCap" , verwing_offset_tip)
        vsp.SetParmVal(verwing_right_id, "Sym_Planar_Flag","Sym", 0)
        vsp.SetParmVal(verwing_right_id, "Density", "Mass_Props", aircraft.wing_density)
        vsp.Update()
        
        """ Create Vertical Wing (Left), Included Parameters are FIXED """
        # Vertical Wing (Left) ID
        verwing_left_id = vsp.AddGeom("WING", "")
        vsp.SetGeomName(verwing_left_id,"Vertical Wing Left")
        
        # Airfoil Selection
        vsp.ChangeXSecShape(vsp.GetXSecSurf(verwing_left_id,0),0,vsp.XS_FILE_AIRFOIL)
        vsp.ChangeXSecShape(vsp.GetXSecSurf(verwing_left_id,0),1,vsp.XS_FILE_AIRFOIL)
        xsec_vl_0 = vsp.GetXSec(vsp.GetXSecSurf(verwing_left_id,0),0)
        xsec_vl_1 = vsp.GetXSec(vsp.GetXSecSurf(verwing_left_id,0),1)
        vsp.ReadFileAirfoil(xsec_vl_0,aircraft.vertical_airfoil_datapath)
        vsp.ReadFileAirfoil(xsec_vl_1,aircraft.vertical_airfoil_datapath)
        vsp.Update()
        
        # Vertical Tail settings
        vsp.SetDriverGroup(verwing_left_id, 1, vsp.AREA_WSECT_DRIVER, vsp.TAPER_WSECT_DRIVER, vsp.ROOTC_WSECT_DRIVER)
        vsp.SetParmVal(verwing_left_id, "Area", "XSec_1", vertical_area / 2)  # Span of the each wing (Half of span)
        vsp.SetParmVal(verwing_left_id, "Taper", "XSec_1", aircraft.vertical_taper)  
        vsp.SetParmVal(verwing_left_id, "Root_Chord", "XSec_1", vertical_c_root) 
        vsp.SetParmVal(verwing_left_id, "Sweep", "XSec_1", verwing_sweep) #Sweep Angle
        vsp.SetParmVal(verwing_left_id, "Sweep_Location", "XSec_1", 0.99)
        vsp.SetParmVal(verwing_left_id, "X_Rel_Location", "XForm", horizontal_distance)  # Position along X-axis
        vsp.SetParmVal(verwing_left_id, "Y_Rel_Location", "XForm", -1 * verwing_yoffset)  # Position along Y-axis
        vsp.SetParmVal(verwing_left_id, "Z_Rel_Location", "XForm", verwing_zoffset)  # Position vertically
        vsp.SetParmVal(verwing_left_id, "X_Rel_Rotation", "XForm", verwing_xRotate)  # X-axis Rotation
        vsp.SetParmVal(verwing_left_id, "CapUMaxOption", "EndCap" , verwing_option_tip)
        vsp.SetParmVal(verwing_left_id, "CapUMaxLength", "EndCap" , verwing_length_tip)
        vsp.SetParmVal(verwing_left_id, "CapUMaxOffset", "EndCap" , verwing_offset_tip)
        vsp.SetParmVal(verwing_left_id, "Sym_Planar_Flag","Sym", 0)
        vsp.SetParmVal(verwing_left_id, "Density", "Mass_Props", aircraft.wing_density) 
        vsp.Update()  

        return [verwing_right_id,verwing_left_id]


def resetAnalysisResults(csvPath:str = "data/aircraft.csv"):
    df = pd.read_csv(csvPath, sep='|', encoding='utf-8')
    df_columns_only = pd.DataFrame(columns=df.columns)
    df_columns_only.to_csv(csvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)

def removeAnalysisResults(csvPath:str = "data/aircraft.csv"):
    if os.path.exists(csvPath):
        os.remove(csvPath)
        #print(f"{csvPath} file has been deleted.")

def writeAnalysisResults(anaResults: AircraftAnalysisResults, csvPath:str = "data/aircraft.csv"):

    if not os.path.isfile(csvPath):
        df = pd.json_normalize(asdict(anaResults))
        df['hash'] = "'" + str(hash(anaResults.aircraft)) + "'"
    else:
        new_df = pd.json_normalize(asdict(anaResults))
        new_df['hash'] = "'" + str(hash(anaResults.aircraft)) + "'"
        df = pd.read_csv(csvPath, sep='|', encoding='utf-8')
        df= pd.concat([df,new_df]).drop_duplicates(["hash"],keep='last')

    # if selected_outputs is not None:
    #     df = df.loc[:, [col for col in selected_outputs if col in df.columns]]    

    def convert_cell(x):
        if isinstance(x, np.ndarray):
            return f"{json.dumps(x.tolist())}"
        return x

    df_copy = df.copy()
    for col in df_copy.columns:
        df_copy[col] = df_copy[col].apply(convert_cell)
    
    # Save the updated DataFrame back to CSV
    df_copy.to_csv(csvPath, sep='|', encoding='utf-8', index=False, quoting=csv.QUOTE_NONE)

def loadAnalysisResults(hashValue:str, csvPath:str = "data/aircraft.csv")-> AircraftAnalysisResults:
    df = pd.read_csv(csvPath, sep='|', encoding='utf-8')
    df = df.loc[df['hash']==hashValue]
   
    for col in df.columns:
       df[col] = df[col].apply(lambda x: 
                               np.array(ast.literal_eval(x),float) if isinstance(x, str) and x.startswith('[')
                               else x)
    df.pop('hash')
    
    if df.empty:
        raise ValueError(f"No data found for hash value: {hashValue}")

    analysisResult=df.to_dict(orient='records')[0]
    return AircraftAnalysisResults.fromDict(analysisResult)

def visualize_results(results: AircraftAnalysisResults):
    """Visualize CL and CD data with flap points"""
    fig = plt.figure(figsize=(16,6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1])
    
    ax_table = fig.add_subplot(grid[0, 0])
    ax_table.axis('tight')
    ax_table.axis('off')
    table1_data=[
        ["Empty Weight (g)",f"{results.m_empty:.2f}"],
        ["m_wing (g)",f"{results.m_wing:.2f}"],
        ["m_boom (g)",f"{results.m_boom:.2f}"],
        ["m_fuse (g)",f"{results.aircraft.m_fuselage:.2f}"],
        ["S (m^2)",f"{results.Sref/1000000:.4f}"]
    ]
    table1 = ax_table.table(cellText=table1_data,cellLoc='center',loc='upper center')
    table1.auto_set_font_size(False)
    table1.set_fontsize(14)
    for (row, col), cell in table1.get_celld().items():
        if col==0:
            cell.set_width(0.6)
        elif col==1:
            cell.set_width(0.4)            
        cell.set_height(0.08)
    
    table2_data=[
        ["Span (m)",f"{results.span/1000:.2f}"],
        ["AR",f"{results.AR:.2f}"],
        ["Taper",f"{results.taper:.2f}"],
        ["Incidence (degree)",f"{results.aircraft.mainwing_incidence:.2f}"],
        ["Twist (degree)",f"{results.twist:.2f}"]
    ]
    table2 = ax_table.table(cellText=table2_data,cellLoc='center',loc='lower center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(14)
    for (row, col), cell in table2.get_celld().items():
        if col==0:
            cell.set_width(0.6)
        elif col==1:
            cell.set_width(0.4)     
        cell.set_height(0.07)
    
    
    
    ax1 = fig.add_subplot(grid[0, 1])
    ax1.plot(results.alpha_list, results.CL, 'b-', label='Flap OFF')
    ax1.scatter([0, results.AOA_takeoff_max], [results.CL_flap_zero, results.CL_flap_max], 
                c='r', marker='o', label='Flap ON')
    ax1.set_xlabel('Angle of Attack (degrees)')
    ax1.set_ylabel('Lift Coefficient (CL)')
    ax1.grid(which='major', linestyle='-', linewidth=1.0)
    ax1.grid(which='minor', linestyle='--', linewidth=0.5)  
    ax1.xaxis.set_major_locator(MultipleLocator(1.0))
    ax1.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.legend()    
    
    ax2 = fig.add_subplot(grid[0, 2])
    ax2.plot(results.alpha_list, results.CD_total, 'b-', label='CD Total')
    ax2.plot(results.alpha_list, results.CD_wing, 'r-', label='CD Wing')
    ax2.plot(results.alpha_list, results.CD_fuse, 'g-', label='CD Fuselage')
    ax2.scatter([0, results.AOA_takeoff_max], [results.CD_flap_zero, results.CD_flap_max],
                c='r', marker='o', label='Flap ON')
    ax2.set_xlabel('Angle of Attack (degrees)')
    ax2.set_ylabel('Drag Coefficient (CD)')
    ax2.grid(which='major', linestyle='-', linewidth=1.0)
    ax2.grid(which='minor', linestyle='--', linewidth=0.5)      
    ax2.xaxis.set_major_locator(MultipleLocator(1.0))
    ax2.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax2.yaxis.set_major_locator(MultipleLocator(0.01))
    ax2.yaxis.set_minor_locator(MultipleLocator(0.005))
    ax2.legend()    
 
    plt.tight_layout()
    plt.show()
