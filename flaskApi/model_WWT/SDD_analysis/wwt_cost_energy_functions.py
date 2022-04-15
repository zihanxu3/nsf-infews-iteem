# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 16:38:07 2020

@author: Shaobin
"""

import numpy as np
# import pandas as pd
# from model_WWT.SDD_analysis.influent_SDD import influent_SDD
# from model_WWT.SDD_analysis.wwt_model_SDD import WWT_SDD

def blower_energy(temp, Q_air, elec_price=0.0638):
    '''
    diffuser_submergence: m
    temp: K
    Q_air: m3/d    
    elec_price: $/kwh; 0.0638 kWh in IL 2019 industrial user
    return energy in kWh/d and cost in $/d
    '''
    P_atm = 101.325  # kPa
    P_inlet_loss = 0.02*101.325  # kPa
    diff_head_loss = 0.17*101.325  # kPa
    P_in = P_atm - P_inlet_loss    # kPa
    diffuser_submergence = 4 #meter
    P_out = P_atm + 9.81*diffuser_submergence + diff_head_loss  # kPa
    blow_eff = 0.7
    energy = 1.4161*10**-5*(temp+273.15)*Q_air*((P_out/P_in)**0.283-1)/blow_eff*24  # kWh/d
    # return energy in kWh/d    
    energy_cost = energy * elec_price
    return energy, energy_cost

# tech_as = WWT_SDD(tech='AS', multiyear=False)
# airflow = tech_as.run_model(1000)[1][:,:,-1]
# energy_blower = blower_energy(20, airflow, 0.05)[1]
# 30*energy_blower.sum(axis=0)
def pump_energy(flowrate, head, density, elec_price=0.0638):
    '''
    density: kg/m3
    hyraulic head: m
    florate: m3/d
    elec_price: $/kwh; 0.0638 kWh in IL 2019 industrial user
    return energy in kWh/d and $/d
    '''
    pump_eff = 0.7
    flowrate = flowrate/24
    energy = flowrate * density * head * 9.81 * 24 / (pump_eff*3.6*10**6)
    # return energy in kWh/d
    energy_cost = energy * elec_price
    return energy, energy_cost

# energy_pump_wasfeed = pump_energy(flowrate=800, head=3, density=1020, elec_price=0.0638)[0]
# energy_pump_waswaste = pump_energy(flowrate=4000, head=2, density=1020, elec_price=0.0638)[0]
# energy_pump_ras = pump_energy(flowrate=52761, head=10, density=1005, elec_price=0.0638)[0]

def heating_energy(sludge, gas_price = 5.25):
    '''
    sludge: kg/d
    energy_price: $/cbf; $5.25/cbf in IL 2019 industrial user
    return energy in MJ/d and $/d
    '''
    dia = 100*0.3048 # meter
    delta_T = 35 - (23.9+18.3)/2
    q_s = sludge * 4.18 * delta_T/1000    # MJ/d
    A_wall = np.pi * dia * 23.5*0.3048   # m2 high = 23.5*0.3048 m
    cone_depth = 12*0.3048 # meter
    A_cone = np.pi*dia/2*np.sqrt(cone_depth**2+dia**2/4)
    A_roof = np.pi * dia/2**2  # m2
    U_wall = 0.68  # W/(m2*C)
    U_roof = 0.91  # W/(m2*C)
    U_cone = 0.85  # W/(m2*C)
    q_l =  (U_wall*A_wall + U_cone*A_cone + U_roof*A_roof)*delta_T/(10**6)*86400  #unit, MJ/d; 1 W = 1 J/s
    energy = q_s + q_l*4
    energy_cost = energy*947.8/1050*gas_price/1000  #1 MJ = 947.8 Btu; $5.25 per thousand cbf in IL 2019, 1050 Btu/cbf
    return energy, energy_cost

# tech_as = WWT_SDD(tech='AS')
# sludge_digestor = tech_as.run_model(1000)[1][:,:,-2]
# energy_sludge_heating = heating_energy(sludge_digestor)


def mixing_energy(tech_WWT, elec_price=0.0638):
    '''
    mixing energy of aeration tank
    elec_price: $/kwh; 0.0638 kWh in IL 2019 industrial user
    '''
    P_mixing = 0.003 # kW/m3
    if tech_WWT == 'AS' or tech_WWT == 'ASCP':
        energy = P_mixing * 56816 * 24   # kWh/d
    elif tech_WWT == 'EBPR_basic' or tech_WWT == 'EBPR_acetate' or tech_WWT == 'EBPR_StR':
        energy = P_mixing * (8902+97870+48069+17498+3504) * 24  # kWh/d
    energy_cost = energy*elec_price  # $/d
    return energy, energy_cost

# energy_mixing = mixing_energy('AS', 0.0638)

def mis_energy(elec_price=0.0638):
    '''
    miscellaneous energy, including various mechanical operations (gates, arms, rakes)
    elec_price: $/kwh; 0.0638 kWh in IL 2019 industrial user
    '''
    primary_clarifier = 0.35*6*24 #kWh/d
    sec_clarifier = 1.0*8*24 #kWh/d
    nitri_clarifier = 1.0*12*24 #kWh/d
    thickner = 2.2*3*24 #kWh/d
    energy = primary_clarifier + sec_clarifier + nitri_clarifier + thickner
    energy_cost = energy*elec_price
    return energy, energy_cost

# energy_mis = mis_energy(0.0638)

def fix_energy_cost(tech_WWT, elec_price=0.0638):
    '''
    including: mixing, mis_energy, pumping energy;
    return: MJ/d; $/d
    '''
    # price = 0.0638
    energy_wasfeed, cost_wasfeed = pump_energy(flowrate=800, head=3, density=1020, elec_price=elec_price)
    energy_waswaste, cost_waswaste = pump_energy(flowrate=4000, head=2, density=1020,  elec_price=elec_price)
    energy_ras, cost_ras = pump_energy(flowrate=52761, head=10, density=1005,  elec_price=elec_price)
    energy_mis, cost_mis =  mis_energy(elec_price)
    energy_mixing, cost_mixing = mixing_energy(tech_WWT, elec_price)
    energy = (energy_wasfeed + energy_waswaste + energy_ras + energy_mis + energy_mixing)*3.6
    cost = cost_wasfeed + cost_waswaste + cost_ras + cost_mis + cost_mixing
    return energy, cost

# cost_fix_energy = fix_energy_cost('AS', 0.0638)[1]

def sludge_cost(amount):
    '''
    amount: kg/d
    return sludge cost in $/d'''
    unit_cost = 7.66                      # $/m3 
    amount_volume = amount/1050           # m3/d desntiy =  1050 kg/m3
    cost = amount_volume * unit_cost      # $/d
    return cost
    
# tech_as = WWT_SDD(tech='AS')
# sludge_hauled = tech_as.run_model(1000)[1][:,:,5]
# sludge_hauled_cost = sludge_cost(sludge_hauled)

def chemical_cost(tech_WWT, flowrate, TP, COD):
    '''
    flowrate: m3/d   # flowrate = Q_WW + Q_rain
    TP: mg/L or g/m3
    COD: mg/L
    Return cost in $/day
    '''
    disinfectant_chem = 0.0052*flowrate  # kg/d
    disinfectant_cost = disinfectant_chem * 0.30  # $/d; NaOCl=$0.30/kg
    poly_chem = 80.6 # kg/d
    poly_cost = poly_chem*2.87  # $/d
    if tech_WWT == 'ASCP':
        flowrate = np.where(flowrate>113562.0, 113562.0, flowrate)
        
        # if flowrate.all() > 113562:   # 113562 m3/d = 30 MGD
        #     flowrate = 113562
        chemical = TP*1.44*flowrate/1000*(55.8/162.2)  # kg/d, Ratio of Fe to influent TP = 1.44; (55.8/162.2): convert Fe to weight of FeCl3
        cost = chemical * 1.49 + disinfectant_cost + poly_cost  # FeCl3 = 1.49 $/kg dry weight;
    elif tech_WWT == 'EBPR_acetate':
        q_ww = np.where(flowrate>113562.0, 113562.0, flowrate)
        # if flowrate.all() > 113562:   # 113562 m3/d = 30 MGD
        # q_ww = 113562
        TP_mass = (q_ww * TP/1000) *(q_ww/flowrate)   # kg/d
        COD_mass = (q_ww * COD/1000) *(q_ww/flowrate) # kg/d
        # else:
        #     TP_mass = flowrate* TP/1000
        #     COD_mass = flowrate * COD/1000
        acetate_demand = 30 * TP_mass - COD_mass  # kg/d
        acetate_demand = np.where(acetate_demand<0, 0, acetate_demand)
        acetate_flow = acetate_demand/1049000 * 1000  # m3/d, 1049000 is the acetate density of solution
        cost = acetate_demand * 0.64/0.5 + disinfectant_cost + poly_cost # 0.64 $/kg for acetate with 50% solution
        chemical_demand = [acetate_demand, acetate_flow]    #kg/d, m3/d
    elif tech_WWT == 'EBPR_StR':
        # if flowrate.all() > 113562:   # 113562 m3/d = 30 MGD
        #     TP_conc = 113562*TP/flowrate
        # else:
        #     TP_conc = TP
        flowrate = np.where(flowrate>113562.0, 113562.0, flowrate)
        
        P_centrifuge_mass = TP*flowrate*0.6  # gram/day
        P_centrifuge_mole = P_centrifuge_mass/30.97  # moles/day
        MgCl2_mole = P_centrifuge_mole * 1.3  # moles/day
        MgCl2_mass_30prt =  MgCl2_mole*95.21/1000/0.3    # kg/d, 95.21 as molar weight; purity: 30%
        # MgCl2_flow = MgCl2_mass_30prt/1276  # m3/d; density: 1276 kg MgCl2/m3;
        # NaOH_flow = MgCl2_flow/2  # m3/d    
        NaOH_mass = MgCl2_mole*40/1000/0.5  # kg/d, NaOH molecular wieght: 40g/mol; purity: 50%
        cost= MgCl2_mass_30prt*0.153 + NaOH_mass*0.37 + disinfectant_cost + poly_cost # $0.153/kg for MgCl; $0.32/kg for NaOH
        chemical_demand = [MgCl2_mass_30prt]
    else:
        cost = disinfectant_cost + poly_cost
        chemical_demand = []
    return cost

# influent = influent_SDD_ave(1000)
# flowrate = influent[:,0]
# TP = influent[:,1]
# COD = influent[:,3]
# np.mean(cost_chemical, axi)
# cost_chemical = chemical_cost('EBPR_acetate', influent[:,0], influent[:,1], influent[:,3])*365  # $/yr

# acetate = 76.5*1049  # kg/d
# acetate_price = acetate*0.49*365



