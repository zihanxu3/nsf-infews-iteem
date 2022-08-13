# -*- coding: utf-8 -*-
"""
Project: NSF INFEWS project (Award Abstract #1739788)
PI: Ximing Cai

Author: Shaobin Li (shaobin@illinois.edu)

Purpose:
The ITEEM that includes the five component models:
1) SWAT: represented by a response matrix method
2) Wastewater treatment (WWT): represented by neural netowrks to represent different wastewater treatment technologies
3) Grain processing (GP): represented by a lookup table with different P recovery technologies
4) Economics: economics of crop yield and willingness to pay by farmer and public 
5） Dringkin water treatment (DWT): energy and chemicals needed to treat different N conc. in drinking water
"""


import numpy as np
import numpy_financial as npf
import pandas as pd
import time

# load new modules developed for ITEEM
from model_WWT.SDD_analysis.wwt_model_SDD import WWT_SDD
from model_SWAT.SWAT_functions import get_yield, loading_outlet_USRW, sediment_instream, get_P_riverine, get_P_biosolid, loading_outlet_USRW_opt_v2 
from model_SWAT.crop_yield import get_yield_crop, get_crop_cost, response_mat_crop, get_P_crop, get_P_fertilizer
from model_Grain.Grain import Grain
from model_DWT.DWT_daily import DWT
from model_Economics.Economics import Economics
from model_Economics.discount_functions import annuity_factor


class ITEEM(object):
    '''
    landuse_matrix: land use decision for BMPs (45,56)
    tech_wwt = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']
    limit_N = policy on nitrate concentration in drinking water, default: 10 mg/L
    tech_GP1: for wet milling plant 1, decision values: [1,2]
    tech_GP2: for wet milling plant 2, decision values: [1,2]
    tech_GP3: for dry grind plant, decision values: [1,2]
    '''
    def __init__(self, landuse_matrix, tech_wwt, limit_N, tech_GP1, tech_GP2, tech_GP3):
        self.landuse_matrix = landuse_matrix
        self.tech_wwt = tech_wwt
        self.limit_N = limit_N
        self.tech_GP1 = tech_GP1
        self.tech_GP2 = tech_GP2
        self.tech_GP3 = tech_GP3

    def get_load(self, name, nutrient_index=1, flow_index=1):
        '''
        used for dynamic plot. time series –> click specific watershed -> popup plots for the subwatershed [SPECIFIC]
        name = ['nitrate', 'phosphorus', 'streamflow']                                                     [33: GENERAL]
        output unit: (nitrate or phosphorus: kg/month or kg/yr) or (streamflow: m3/month, m3/yr)
        sw = subwatershed location, staring from 0; outlet: sw=33
        nutrinet_index and flow_index are used for sensitivity analysis. default value=1
        return: numpy array (16,12) = (year,month); starting year January, 2003
        '''
        load = loading_outlet_USRW(name, self.landuse_matrix, self.tech_wwt, nutrient_index, flow_index)
        # load_sw = load[:,:,sw]
        return load.tolist()
    
    def get_yield(self, name):
        '''
        used for map plot -> Gradient (dropdown)
        name = ['nitrate', 'phosphorus', 'streamflow']     
        output: nonpoint yield from SWAT, kg/hactere (kg) or m3/ha  
        return: average yield, kg/ha per yr
        '''
        yield_data = get_yield(name, self.landuse_matrix)[1]  # shape: numpy array (16,12,45) = (year, month, subwatersheds)
        yield_data = yield_data.sum(axis=1).mean(axis=0) # shape: (45, )
        return yield_data.tolist()
        
    def get_sediment_load(self):
        '''
        used for simulating dynamic sediment load, kg/month
        (subcategory of get_load)                            [33: GENERAL]
        return: numpy array (16,12,45) = (year, month, subwatersheds)
        '''
        sediment_load = np.zeros((16,12,45))
        for sw in range(45):         
            sediment_load[:,:,sw] = sediment_instream(sw, self.landuse_matrix)
        return sediment_load.tolist()
    
    def get_crop(self, crop_name):
        '''
        crop_name = ['corn', 'soybean', 'switchgrass']
        return:
            1. crop_yield, kg/ha, used for map plot (16, 45) -> Gradient (dropdown)
            2. crop_production (similar pollutant load),kg/yr, used for dynamic plot (16,45) -> time series plot [Sum Up: GENERAL]
        '''
        # crop_yield = response_mat_crop(crop_name).sum(axis=2) # (16, 45) = (year, sw)
        _, crop_production, crop_yield = get_yield_crop(crop_name, self.landuse_matrix)
        return crop_yield.tolist(), crop_production.tolist()
    
    def get_cost_energy(self, r=0.07, n_wwt=40, nutrient_index=1.0, flow_index=1.0, 
                        chem_index=1.0, utility_index=1.0, rP_index=1.0, feedstock_index=1.0, crop_index=1.0):
        '''
        return a numpy array (energy_wwt, energy_grain, energy_water): Million MJ/yr
        7% interest rate, 40 year of lifespan
        '''
        '''*** energy of drinking water in MJ***'''
        DWT_Decatur = DWT(self.limit_N, self.landuse_matrix)
        energy_dwt = DWT_Decatur.get_nitrate_energy()[2].sum()/16
        '''*** energy of GP in Million MJ ***'''
        wet_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1)
        wet_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2)
        dry_1 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3)  
        energy_grain = wet_1.get_energy_use()[-1] + wet_2.get_energy_use()[-1] + dry_1.get_energy_use()[-1]
        '''*** cost in $/yr ***'''
        cost_grain = wet_1.get_cost(feedstock_index, chem_index, utility_index)[-1] \
        + wet_2.get_cost(feedstock_index, chem_index, utility_index)[-1] \
        + dry_1.get_cost(feedstock_index, chem_index, utility_index)[-1]
        
        cost_dwt = DWT_Decatur.get_cost(r, chem_index, utility_index)
        wwt_SDD = WWT_SDD(self.tech_wwt, multiyear=True, start_yr = 2003, end_yr=2018)
        cost_energy_nutrient = wwt_SDD.get_cost_energy_nutrient(1000, self.landuse_matrix, r, n_wwt, 
                                                                nutrient_index, flow_index, 
                                                                chem_index, utility_index,
                                                                rP_index)
        cost_wwt = cost_energy_nutrient[0]
        energy_wwt = cost_energy_nutrient[4]
        rP_amount = cost_energy_nutrient[-4]
        revenue_rP = cost_energy_nutrient[-3]        
        outlet_nitrate = cost_energy_nutrient[-2]
        outlet_tp = cost_energy_nutrient[-1]

        cost_crop = Economics(self.landuse_matrix).get_crop_cost_acf(r)[-1]   # annualized cost, $/yr
        cost_total = cost_dwt + cost_grain + cost_wwt + cost_crop
        return [energy_dwt/(10**6), energy_grain/(10**6), energy_wwt/(10**6),
                cost_dwt, cost_grain, cost_wwt, cost_crop, cost_total, 
                rP_amount, revenue_rP, outlet_nitrate, outlet_tp]
    
    # def get_system_cost(self, r):
    #     '''return equivalent annualized cost from all submodels, $/yr'''
    #     DWT_Decatur = DWT(self.limit_N, self.landuse_matrix)
    #     cost_dwt = DWT_Decatur.get_cost()
    #     wet_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1)
    #     wet_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2)
    #     dry_1 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3)
    #     cost_grain = wet_1.get_cost()[-1] + wet_2.get_cost()[-1] + dry_1.get_cost()[-1]
    #     cost_wwt = WWT_SDD(self.tech_wwt,multiyear=True, start_yr = 2003, 
    #                         end_yr=2018).get_cost_energy(1000, 0.07, 40)[0]
    #     cost_crop = Economics(self.landuse_matrix).get_crop_cost_acf()[-1]
    #     total = cost_dwt + cost_grain + cost_wwt + cost_crop
    #     return cost_dwt, cost_grain, cost_wwt, cost_crop, total
    
    def get_system_revenue(self, r=0.07, grain_product_index = 1.0, rP_index=1.0, 
                           feedstock_index=1.0, chem_index=1.0, utility_index=1.0, crop_index=1.0, sg_price=0.05):
        '''return annualized benefit from all submodels'''
        wet_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1)
        wet_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2)
        dry_1 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3)
        revenue_GP = wet_1.get_revenue(grain_product_index=grain_product_index, rP_index=rP_index)[-1] \
        + wet_2.get_revenue(grain_product_index=grain_product_index, rP_index=rP_index)[-1] \
        + dry_1.get_revenue(grain_product_index=grain_product_index, rP_index=rP_index)[-1]
        
        profit_GP = wet_1.get_profit(r, grain_product_index=grain_product_index, rP_index=rP_index, 
                                     feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index)[-1] \
        + wet_2.get_profit(r, grain_product_index=grain_product_index, rP_index=rP_index, 
                                     feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index)[-1] \
        + dry_1.get_profit(r, grain_product_index=grain_product_index, rP_index=rP_index, 
                                     feedstock_index=feedstock_index, chem_index=chem_index, utility_index=utility_index)[-1]
        
        revenue_crop = Economics(self.landuse_matrix, sg_price=sg_price).get_crop_revenue_acf(r=r, crop_index=crop_index)[-1]
        revenue_total = revenue_GP + revenue_crop
        return profit_GP, revenue_GP, revenue_crop, revenue_total
    
    def get_rP(self):
        '''return rP in kg/yr'''
        rP_1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1).get_rP()[1]
        rP_2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2).get_rP()[1]
        rp_3 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3).get_rP()[1]
        rP = rP_1 + rP_2 + rp_3
        return rP
    
    def get_P_flow(self):
        '''calculate P flow between submodels, metric ton/yr'''
        '''P_riverine'''
        # P_nonpoint, P_point, P_reservoir, P_instream_store, P_total_outlet, struvite
        P_nonpoint, P_point, P_reservoir, P_instream_store, P_total_outlet, struvite = get_P_riverine(self.landuse_matrix, self.tech_wwt)
        P_SDD_influent = 676.8 # MT/yr
        # P_point_baseline = 582.4 # MT/yr
        # P_nonpoint_baseline = 292.9 # MT/yr
        in_stream_load = P_nonpoint + P_point
        
        '''P_biosolid'''
        P_in_biosolid, P_crop_biosolid, P_riverine_biosolid, P_soil_biosolid = get_P_biosolid(self.tech_wwt)
        
        '''P_crop & P_fertilizer'''
        P_fertilizer = get_P_fertilizer('corn', self.landuse_matrix) # MT/yr
        P_corn_self, P_corn_import, P_soybean, P_sg = get_P_crop(self.landuse_matrix)
        P_crop_list = [P_corn_self, P_corn_import, P_soybean, P_sg]
        # P_fertilizer_net = P_fertilizer - P_crop_biosolid
        
        '''P_corn_biorefinery'''
        P_in1, P_product1, P_other1, rP1 = Grain(plant_type=1, plant_capacity=2.1, tech_GP=self.tech_GP1).get_P_flow()
        P_in2, P_product2, P_other2, rP2 = Grain(plant_type=1, plant_capacity=5.0, tech_GP=self.tech_GP2).get_P_flow()
        P_in3, P_product3, P_other3, rP3 = Grain(plant_type=2, plant_capacity=120, tech_GP=self.tech_GP3).get_P_flow()
        P_cb_in = P_in1 + P_in2 + P_in3
        P_cb_rP = rP1 + rP2 + rP3
        P_cb_product = P_cb_in - P_cb_rP
        P_cb_list = [P_cb_in, P_cb_rP, P_cb_product]
        
        '''P_manure'''
        P_corn_silage = 24.7                 # 10487*908.6*0.26/100/1000 #10487 kg/ha, 908.6 ha, assume 0.26%
        if self.tech_GP1==1 and self.tech_GP2==1 and self.tech_GP3==1:
            P_CGF = 2726*12/1000             # 2726 ton/yr, total CGF demand for StoneDairy; 12mg/g
            P_manure = 67.8
            P_manure_runoff = 1.932
            P_manure_soil = P_manure - P_manure_runoff - P_corn_silage
        else:
            P_CGF = 2726*2.5/1000            # 2726 ton/yr, total CGF demand for StoneDairy; 12mg/g
            P_manure = 67.8 - (2726*12/1000-2726*2.5/1000)  #
            P_manure_runoff = 1.700
            P_manure_soil = P_manure - P_manure_runoff - P_corn_silage
        P_manure_list = [P_manure, P_manure_runoff, P_manure_soil, P_CGF]
        P_rP = P_cb_list[1] + struvite
        P_soil = P_soil_biosolid + P_manure_list[2] 
        P_soil_fertilizer = P_fertilizer -  P_corn_self - P_soybean - P_sg - P_soil_biosolid - P_nonpoint
        
        '''P_list'''
        P_in_list  = [P_crop_list[1], P_fertilizer, P_manure_list[0], P_SDD_influent]
        P_out_list = [P_cb_list[2], P_rP, P_crop_list[2], P_corn_silage, P_soil, P_soil_fertilizer,
                     P_total_outlet,  P_reservoir,  P_instream_store]
        
        P_intermediate_list = [in_stream_load, P_crop_list[0], P_crop_biosolid]

        '''adjustment coefficient'''
        P_in = sum(P_in_list); P_out = sum(P_out_list); coef = (P_out - P_in)/P_in
        P_out_list_adj = [(1-coef)*x for x in P_out_list]
        
        if P_soil_fertilizer > 0:
            output_list = [P_corn_import, P_nonpoint, P_corn_self, P_soybean, P_soil_fertilizer, P_sg,
                           P_manure_runoff, P_corn_silage, P_manure_soil, 
                           P_point,  P_crop_biosolid, struvite, P_soil_biosolid,
                           P_cb_product, P_cb_rP,
                           P_total_outlet, P_reservoir, P_instream_store,
                           P_corn_self+P_crop_biosolid
                           ]
            
            source = ['Imported corn', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Fertilizer',  
                      'Manure', 'Manure', 'Manure', 'Wastewater', 'Wastewater', 'Wastewater', 'Wastewater', 'Corn biorefinery', 'Corn biorefinery',
                      'In-stream load', 'In-stream load', 'In-stream load', 'Corn (local)'
                      ]
            target = ['Corn biorefineries', 'In-stream load', 'Corn (local)', 'Soybean', 'Soil',
                      'Biomass', 'In-stream load', 'Corn silage', 'Soil', 'In-stream load', 
                      'Corn (local)', 'recovered P', 'Soil', 'Products from CBs', 'recovered P', 
                      'Riverine export', 'Reservoir trapping', 'In-stream storage',
                      'Corn biorefineries'
                      ]
            

        elif P_soil_fertilizer < 0:
            output_list = [P_corn_import, P_nonpoint, P_corn_self+P_soil_fertilizer*0.65, P_soybean+P_soil_fertilizer*0.35, 
                           P_sg, P_manure_runoff, P_corn_silage, 
                           P_point,  P_crop_biosolid, struvite, P_soil_biosolid,
                           P_cb_product, P_cb_rP,
                           P_total_outlet, P_reservoir, P_instream_store,
                           P_corn_self+P_crop_biosolid,
                           P_soil_fertilizer*-0.65, P_soil_fertilizer*-0.35
                           ]
            source = ['Imported corn', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Fertilizer', 'Manure', 'Manure',
                      'Wastewater', 'Wastewater', 'Wastewater', 'Wastewater', 'Corn biorefinery', 'Corn biorefinery',
                      'In-stream load', 'In-stream load', 'In-stream load', 'Corn (local)', 'Soil', 'Soil'
                      ]
            target = ['Corn biorefineries', 'In-stream load', 'Corn (local)', 'Soybean', 'Biomass',
                      'In-stream load', 'Corn silage', 'In-stream load', 'Corn (local)', 
                      'recovered P', 'Biosolid', 'Products from CBs', 'recovered P', 
                      'Riverine export', 'Reservoir trapping', 'In-stream storage',
                      'Corn biorefineries', 'Corn (local)', 'Soybean'
                      ]

        output_list2 = []
        for i in range(len(source)):
            single_node = [source[i], target[i], output_list[i]]
            output_list2.append(single_node)
                
        return P_in_list, P_intermediate_list, P_out_list_adj, source, target, P_soil_fertilizer, output_list, output_list2
    
    
    def run_ITEEM(self, input_para=[0.152, 0.356, 40, 0.95, 0.07, 0.0638, 5.25],
                  sw=33, r=0.07, n_wwt=40, nutrient_index=1.0, flow_index=1.0, chem_index=1.0, rP_index=1.0, 
                  utility_index=1.0, grain_product_index=1.0, feedstock_index=1.0, crop_index=1.0, wtp_price=0.95):
        
        '''
        return a list containg multiple outputs of N, P, streamflow, sediment, 
        energy_dwt, energy_grain, energy_wwt,
        cost_dwt, cost_grain, rP
        '''
        
        '''
        baseline = np.array(input_para)
        indices = np.array(input_para)/baseline
        nutrient_index = indices[0]
        '''
        
        
        streamflow = self.get_streamflow_outlet(sw)
        streamflow_outlet = streamflow.sum(axis=1).mean()
        sediment_outlet = self.get_sediment_outlet(sw).sum(axis=1).mean()
        sediment_outlet_landscape = loading_outlet_USRW('sediment', self.landuse_matrix)[:,:,33].sum(axis=1).mean()
        sediment_decautr_instream = sediment_instream(32, self.landuse_matrix).sum(axis=1).mean()
        
        # cost_dwt, cost_GP, cost_wwt, cost_crop, cost_total = self.get_system_cost(r)
        cost_energy = self.get_cost_energy(r=r, n_wwt=n_wwt, nutrient_index=nutrient_index, flow_index=flow_index, 
                                           chem_index=chem_index, utility_index=utility_index, rP_index=rP_index)
        energy_dwt = cost_energy[0]
        energy_grain = cost_energy[1]
        energy_wwt = cost_energy[2]
        cost_dwt = cost_energy[3]
        cost_grain = cost_energy[4]
        revenue_rP = cost_energy[9]
        cost_wwt = cost_energy[5] - revenue_rP
        cost_crop = cost_energy[6]
        # cost_total = cost_energy[7]
        rP_amount = cost_energy[8]
        outlet_nitrate = cost_energy[-2]
        outlet_tp = cost_energy[-1]
        N_outlet = outlet_nitrate[:,:,sw].sum(axis=1).mean()
        P_outlet = outlet_tp[:,:,sw].sum(axis=1).mean()
        profit_GP, revenue_GP, revenue_crop, revenue_total = self.get_system_revenue(r=r, grain_product_index=grain_product_index,
                                                                                     rP_index=rP_index, feedstock_index=feedstock_index, 
                                                                                     chem_index=chem_index, utility_index=utility_index, 
                                                                                     crop_index=crop_index)
        
        nitrate_impro_prt = ((7240 - N_outlet/1000)/7240)/0.45  # baseline nitrate load =7240 Mg/yr
        if nitrate_impro_prt > 0 and nitrate_impro_prt <1.0:
            wtp_nitrate = nitrate_impro_prt*0.95*100*59660 # $0.95/1% nitrate improvement, 59660 household
        elif nitrate_impro_prt > 1.0:
            wtp_nitrate = wtp_price*100*59660 # 59660 households
        else:
            wtp_nitrate = 0
        
        tp_impro_prt = ((324 - P_outlet/1000)/324)/0.45  # baseline TP load = 324 Mg/yr
        if  tp_impro_prt > 0 and tp_impro_prt < 1.0:
            wtp_tp = tp_impro_prt*wtp_price*100*59660 # 59660 households
        elif tp_impro_prt > 1.0: 
            wtp_tp = wtp_price*100*59660 # 59660 households
        else:
            wtp_tp = 0
        wtp = 0.5*wtp_nitrate + 0.5*wtp_tp
        wtp_npv = npf.npv(0.07, [wtp]*16)
        wtp_acf = wtp_npv/annuity_factor(16, 0.07)
        
        sediment_credit = (27455*0.7 - sediment_decautr_instream*0.7)*21.2   # $/yr， 21.2 $/ton, 70% trapped
        sediment_credit_ac = npf.npv(0.07, [sediment_credit for i in range(16)])/annuity_factor(16, 0.07) # 16 years
        
        system_net_benefit = wtp_acf + profit_GP + revenue_crop + sediment_credit_ac \
            - cost_crop - cost_dwt - cost_wwt  # 27276 is the baseline sediment load; 21.2 $/ton if sediment is avoided
        profit_crop = revenue_crop - cost_crop

        ''' P recovery and food production '''
        rP_P_complex = self.get_rP()*0.3       # 25.5% P for wet milling and 31.5 for dry-grind, kg/yr
        
        if self.tech_wwt == 'EBPR_StR':
            rP_struvite = 1283150*0.1262   # 12.62% P in struvite, kg/yr
        else: rP_struvite = 0
        
        corn = self.get_crop('corn')[1]        # kg/yr
        soybean = self.get_soybean('soybean')[1]  # kg/yr
        biomass = self.get_biomass('switchgrass')[1]  # kg/yr
        
        environment = [N_outlet, P_outlet, sediment_outlet_landscape, sediment_outlet, streamflow_outlet]
        energy = [energy_dwt, energy_grain, energy_wwt.mean(), biomass]
        economics = [cost_dwt, cost_grain, cost_wwt, cost_crop,
                     revenue_GP, revenue_crop-cost_crop, profit_GP, wtp, system_net_benefit]
        food = [rP_P_complex, rP_amount, corn, soybean]
        
        spyder_output = [N_outlet, P_outlet, sediment_outlet, streamflow_outlet,
                         energy_dwt, energy_grain, energy_wwt.mean(), biomass,
                         cost_dwt, cost_wwt, profit_crop, profit_GP,
                         wtp, system_net_benefit,
                         corn, soybean, rP_P_complex + rP_amount, ]
        
        # spyder_output_normalized = 
        return environment, food, economics, energy, spyder_output
        

# start = time.time()
landuse_matrix_baseline = np.zeros((45,62))
landuse_matrix_baseline[:,55] = 0.25
landuse_matrix_baseline[:,37] = 0.75
baseline = ITEEM(landuse_matrix_baseline, tech_wwt='AS', limit_N=10.0, tech_GP1=1, tech_GP2=1, tech_GP3=1)
output = baseline.get_P_flow()
# end = time.time()
# print('Simulation time is: ', end - start)