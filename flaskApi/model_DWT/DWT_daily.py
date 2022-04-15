# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
    To estimate nitrate and sediment impact on drinking water treatment
    1) estimate cost of chemicals (both sediment and nitrate) and energy (due to nitrate)
    2) estimate energy use due to nitrate removal
"""

# import packages
import pandas as pd
import numpy as np
import numpy_financial as npf
from calendar import monthrange
from model_DWT.SWAT_originalRM_daily import loading_outlet_originalRM 
from model_SWAT.SWAT_functions import sediment_instream, loading_outlet_USRW
from model_Economics.discount_functions import annuity_factor, pv, real_dr, cost_inflation

# set up global variable
df_nitrate_yield = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')

class DWT(object):
    '''
    This class evaluates the impact of nitrate and sediment on the cost and 
    energy use of drinking water treatment (DWT) plant.
    '''
    def __init__(self, limit_N, landuse_matrix):
        '''landuse_matrix: (45,56)'''
        self.limit_N = limit_N
        self.flow = 22.3*3785.4118  # unit in m3/d; 1 mgd = 3785.4118 m3/d
        self.df = df_nitrate_yield
        self.year = self.df.iloc[:,1].unique()
        self.month = self.df.iloc[:,2].unique()
        # self.ratio = 0.20  # k_w=20% of total water is diverted to NRF as default
        self.landuse_matrix = landuse_matrix[:,:56]
        self.nitrate_loading = loading_outlet_originalRM('nitrate', self.landuse_matrix)[:,:,32]  # unit: kg/day
        self.streamflow = loading_outlet_originalRM('streamflow', self.landuse_matrix)[:,:,32]    # streamflow: m3/day
        self.streamflow_month = loading_outlet_USRW('streamflow', landuse_matrix, 'AS')[:,:,32]
        self.nitrate_conc = self.nitrate_loading*(10**3)/self.streamflow # unit: g/m3 or mg/L
        # self.nitrate_unadjust = self.nitrate_conc[:,:,32] # subwatershed 33 is where the DWT plant extracts water
        self.nitrate_conc = np.where(self.nitrate_conc[:,:]>12.9, 12.9, self.nitrate_conc[:,:]) #Based on historical data, the highest monthly average is 12.9 mg/L
        self.sediment_loading = sediment_instream(32, landuse_matrix)
        self.sediment_conc = self.sediment_loading*(10**6)/self.streamflow_month # convert ton/m3 to mg/L
        self.NTU = self.sediment_conc/1.86  # TSS (mg/L) = 1.86 x NTU

    def get_nitrate_days(self):
        '''
        return 
        N_days: showing number of days, (year, day) 
        that nitrate is higher than the regulation limit
        '''
        N_days = np.where(self.nitrate_conc>0.8*self.limit_N, 1, 0) # (year, day) 
        # N_days_byyear = N_days.sum(axis=1) # (yrs)
        return N_days
    
    def get_nitrate_conc(self):
        '''return a numpy array (year, days)'''
        nitrate_conc2 = np.where(self.nitrate_conc[:,:]<0.8*self.limit_N, 0, self.nitrate_conc[:,:])  
        return nitrate_conc2 
    
    def get_nitrate_chemical(self):
        '''return a numpy array (year, day), unit: kg per day'''
        # 58.44g NaCl/14 g N, 150% regeneration ratio
        # chemical_N = self.get_nitrate_conc()*self.flow*self.ratio*58.44/14*1.5/1000  #k_r = 1.5
        # Assuming that final nitrate is 8 mg/L
        ratio = 0.20    #20% of total water is diverted to NRF as defaul
        # ratio[ratio==-inf] = 0
        chemical_N = self.get_nitrate_conc()*self.flow*ratio*58.44/14*1.5/1000
        return chemical_N, ratio

    def get_nitrate_energy(self):
        '''MJ/d, energy use of the nitrate removal facility if operating: (year,day)'''
        elec_use = 825/0.0653 * self.get_nitrate_days() # $0.0653/kWh
        gas_use = 75/5.83 * 1000 * self.get_nitrate_days() # $6.036/1000 cbf
        total_use = elec_use*3.6 + gas_use*1.06 # 1 kWh = 3.6 MJ; 1 cbf = 1.06 MJ
        return elec_use, gas_use, total_use
    
    def get_nitrate_cost(self, r=0.07, chem_index=1.0, utility_index=1.0):
        '''
        return total cost of nitrate removal facility: (year,days); 
        N_EAC = $/yr
        '''
        N_cost_capital = 633                         # depreciation cost: $633/day: equivalent annual cost per day
        overhead_cost = 72 * self.get_nitrate_days()      # $72/day in 2003
        utility_cost = 900 * self.get_nitrate_days()*utility_index      # $900/day
        labor_cost = 239 * self.get_nitrate_days()        # $239/day in 2003 (4 hrs/day for operation hour and 2 hrs/day for maintenance labor)
        N_cost_chemical = self.get_nitrate_days() * self.get_nitrate_chemical()[0] * 0.089 * 2.20462 * chem_index # 0.089 $/lb in 2003, 1 kg = 2.20462 lb
        N_cost_op = N_cost_chemical + overhead_cost + utility_cost + labor_cost  # (year, day)
        N_cost_total = N_cost_capital + N_cost_op   # return cost as a numpy array： （year, month）
        N_npv = npf.npv(r, N_cost_total.sum(axis=1))
        N_EAC = N_npv/annuity_factor(16, r)  # 16 yrs
        return [N_EAC]
    
    def get_sediment_cost(self, r=0.07, chem_index=1.0):
        '''return chemical cost for sediment treatment as a numpy array: (year,month)'''
        polymer = (0.0062*self.NTU + 0.31)/1000     # kg/m3
        alum = (0.3*self.NTU + 30.0)/1000           # kg/m3
        
        # cost_alum = cost_inflation(cost=0.0787, cost_yr=2019, start_yr=2003)  # 0.0787 $/lb
        # cost_polymer = cost_inflation(cost=1.26, cost_yr=2019, start_yr=2003) # 1.26 $/lb
        alum_cost = alum*self.flow*0.0787*2.20462*chem_index      # 0.0787 $/lb, 1 kg = 2.20462 lb
        polymer_cost = polymer*self.flow*1.26*2.20462*chem_index  # 1.26 $/lb, 1 kg = 2.20462 lb
        number_days = np.zeros(((self.year.size, self.month.size)))
        for i in range(self.year.size):
            for j in range(12):
                number_days[i,j] = monthrange(self.year[i], self.month[j])[1]
        sediment_cost = (alum_cost + polymer_cost) * number_days 

        # discount cash flow analysis
        sediment_cost_PV = npf.npv(r, sediment_cost.sum(axis=1))
        sediment_EAC = sediment_cost_PV/annuity_factor(16, r)   # 16 yrs
        return [sediment_EAC]
    
    def get_cost(self, r=0.07, chem_index=1.0, utility_index=1.0):
        '''return equivalent annual cost'''
        cost_EAC = self.get_nitrate_cost(r, chem_index, utility_index)[-1] + self.get_sediment_cost(r, chem_index)[-1]
        return cost_EAC

# landuse_matrix = np.zeros((45,62))
# landuse_matrix[:, 55] = 1

# start = time.time()   
# dwt = DWT(limit_N=10, landuse_matrix=landuse_matrix)
# days_month = dwt.get_nitrate_days().sum(axis=1).mean()
# S0_cost = dwt.get_cost()
# S0_energy = S0_dwt.get_nitrate_energy()[-1].sum(axis=1).mean()  # MJ
# S2_dwt = DWT(limit_N=10, landuse_matrix=landuse_matrix_s2)
# N_days_S2 = S2_dwt.get_nitrate_days().sum(axis=1).mean()
# DWT_nitrate_cost = S0_dwt.get_cost(r=0.07, chem_index=1.2, utility_index=1.0)
# DWT_sediment_cost = S0_dwt.get_sediment_cost()
# N_days_baseline = S0_dwt.get_nitrate_days().sum(axis=1).mean()

# N_days.sum(axis=1)
# dwt_nitrate_conc = S0_dwt.nitrate_conc
# cost_dwt = S0_dwt.get_cost()
# end = time.time()
# print('Simulation time is: ', end - start)

# days_month = S0_dwt.get_nitrate_days().sum(axis=1).mean()
# S0_energy = S0_dwt.get_nitrate_energy()[-1].sum(axis=1).mean()
# S0_cost = S0_dwt.get_cost()

# days_annual = S0.get_nitrate_days().sum(axis=1)

# chemical_N, ratio = S0.get_nitrate_chemical()
# N_days = S0.get_nitrate_days()
# days_monthly = S0_DWT.get_nitrate_days()
# S0_N_conc = S0_DWT.get_nitrate_conc()
# S0_Nitrate = S0_DWT.nitrate

# total_cost = S0.get_total_cost()
# sediment_conc = S0.sediment_conc
# ntu = S0.NTU
# op_cost = S0_DWT.get_op_cost()
# S0_conc = S0_DWT.get_nitrate_conc()
# S0_nitrate = S0_DWT.nitrate

if __name__ == "__main__":
    # set up global variable
    df_nitrate_yield = pd.read_csv('model_SWAT/response_matrix_csv/yield_nitrate.csv')