# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS project - ITEEM

Purpose:
Cost and benefit analyses for ITEEM, including 
1) WTP: 
    - Social willingness to pay (WTP) by general public;
    - Willingness to pay by farmers for using recovered P products
2) Total benefit: output prices received in product markets,
    and non-market benefits associated with water quality levels.
3) Total cost: the techno-economic costs of Farm practice costs, 
    P recovery, municipal drinking and/or wastewater treatment costs
"""

# import packages
import numpy as np
import numpy_financial as npf
from model_SWAT.crop_yield import get_crop_cost, get_yield_crop
from model_Economics.discount_functions import annuity_factor
# from model_Grain.Grain import *
# from model_DWT.DWT_daily import DWT
# from model_DWT.get_distribution_para_rawnitrate import * 


class Economics(object):
    '''
    Function to calculate cost and benefit analysis
    landuse_matrix determines the BMPs cost, crop yield, raw water qualtiy going into water treatment plant
    tech_wwtp determines product price and cost of recovered P from grain processing from wwtp
    tech_grain determines product price and cost of recovered P from grain processing
    '''
    def __init__(self, landuse_matrix, corn_price=0.152, soybean_price=0.356, sg_price=0.05):
        '''
        default corn price: https://www.indexmundi.com/commodities/?commodity=corn
        default soybean price:https://www.indexmundi.com/commodities/?commodity=soybeans&months=360'''
        self.landuse_matrix = landuse_matrix
        self.corn_price = corn_price
        self.soybean_price = soybean_price
        self.sg_price = sg_price

    def get_crop_cost_acf(self, r=0.07):
        '''return $/yr'''
        corn_cost = get_crop_cost('corn', self.landuse_matrix)[1]
        soybean_cost = get_crop_cost('soybean', self.landuse_matrix)[1]
        sg_cost = get_crop_cost('switchgrass', self.landuse_matrix)[1]
        crop_cost = corn_cost + soybean_cost + sg_cost
        npv = npf.npv(r, crop_cost)
        annualized_value = npv/annuity_factor(16, r)
        return crop_cost, annualized_value
    
    def get_crop_revenue_acf(self, r=0.07, crop_index=1.0):
        '''return $/yr'''
        corn_production = get_yield_crop('corn', self.landuse_matrix)[1].sum(axis=1)  # kg/yr, (yr,)
        soybean_production = get_yield_crop('soybean', self.landuse_matrix)[1].sum(axis=1)  # kg/yr, (yr,)
        sg_production = get_yield_crop('switchgrass', self.landuse_matrix)[1].sum(axis=1)  # kg/yr, (yr,)
        # production = [corn_production, soybean_production, sg_production]
        
        corn_revenue = corn_production*self.corn_price*crop_index          # corn_price = $0.152/kg
        soybean_revenue = soybean_production*self.soybean_price*crop_index # soybean_price = $0.356/kg 
        sg_revenue = sg_production*self.sg_price*0.8*crop_index                # sg_price = $0.04/kg， 80% moisture
        
        total_revenue = corn_revenue + soybean_revenue + sg_revenue
        npv = npf.npv(r, total_revenue)
        annualized_value = npv/annuity_factor(16, r)
        return corn_revenue, soybean_revenue, sg_revenue, total_revenue, annualized_value

    def get_WTP(self, unit_pay=0.95):
        '''return $/yr'''
        # streamflow_outlet = self.get_streamflow_outlet().sum(axis=1).mean()
        energy_dwt, energy_grain, energy_wwt, cost_dwt, cost_GP, cost_wwt, cost_crop, cost_total, outlet_nitrate, outlet_tp = self.get_cost_energy()
        N_outlet = outlet_nitrate[:,:,33].sum(axis=1).mean()
        P_outlet = outlet_tp[:,:,33].sum(axis=1).mean()
        nitrate_impro_prt = ((7240 - N_outlet/1000)/7240)/0.45  # baseline nitrate load =7240 Mg/yr
        if nitrate_impro_prt > 0 and nitrate_impro_prt <1.0:
            wtp_nitrate = nitrate_impro_prt*unit_pay*100*113700 # $0.95/1% nitrate improvement, 113700 household
        elif nitrate_impro_prt > 1.0:
            wtp_nitrate = unit_pay*100*113700
        else:
            wtp_nitrate = 0
        tp_impro_prt = ((324 - P_outlet/1000)/324)/0.45  # baseline TP load = 324 Mg/yr
        if  tp_impro_prt > 0 and tp_impro_prt < 1.0:
            wtp_tp = tp_impro_prt*unit_pay*100*113700 # 113700 households
        elif tp_impro_prt > 1.0: 
            wtp_tp = unit_pay*100*113700
        else:
            wtp_tp = 0
        wtp = 0.5*wtp_nitrate + 0.5*wtp_tp
        return wtp
    
    def get_WTA(self):
        '''under development'''
        wta = 0
        return wta
    
    def get_cost_benefit(self):
        total_cost = self.get_crop_cost() + self.get_wwt_cost() + self.get_grain_profit()
        total_benefit = self.get_WTP()
        cost_benefit = total_benefit - total_cost
        return cost_benefit

    def get_cost_eff(self, name):
        '''calculate cost effectiveness for nitrate and TP removal: $/kg P saved'''
        reduction = 0   # kg of nitrate or TP reduced
        net_crop_cost = 0   # net increase of cost in order to reduce nitrate and TP.
        cost_eff = net_crop_cost/reduction
        return cost_eff
    
# landuse_matrix = np.zeros((45,62))
# landuse_matrix[:,1] = 1
# landuse_matrix[:,55] = 0.25
# test = Economics(landuse_matrix)
# crop_revenue = test.get_crop_revenue_acf(r=0.07, crop_index=1)[-1]
# crop_cost = test.get_crop_cost_acf(r=0.07)[-1]


# crop_revenue[-2]
# crop_cost = test.get_crop_cost_acf(r=0.07)

# landuse_matrix2 = np.zeros((45,62))
# landuse_matrix2[:,1] = 1
# crop_production_revenue2 = Economics(landuse_matrix2).get_crop_revenue_acf(r=0.07, crop_index=1)[-1]
