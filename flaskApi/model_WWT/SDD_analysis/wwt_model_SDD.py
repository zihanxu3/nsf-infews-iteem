# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS project - ITEEM

Purpose:
Scripts used to evaluate treatment performance of wastewater treatment plants (WWTP)
under stochastic impacts of influent characteristics
"""

# Import required packages
import pandas as pd
import numpy as np
import numpy_financial as npf

# Import machine learning packages
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Import packages for saving and loading model from keras
from model_WWT.SDD_analysis.influent_SDD import influent_SDD, influent_SDD_multiyear2, influent_SDD_ave
from model_WWT.SDD_analysis.wwt_cost_energy_functions import blower_energy, chemical_cost, heating_energy, sludge_cost, fix_energy_cost
from model_Economics.discount_functions import annuity_factor
from model_WWT.SDD_analysis.data import *


class WWT_SDD(object):
    def __init__(self, tech, multiyear, **kwargs):
        self.tech = tech
        self.tech_dict_data = {'AS': df_AS, 'ASCP':df_ASCP, 'EBPR_basic': df_EBPR_basic, 
                               'EBPR_acetate': df_EBPR_acetate,'EBPR_StR':df_EBPR_StR}
        self.tech_dict_model = {'AS': model_AS, 'ASCP':model_ASCP, 'EBPR_basic': model_EBPR_basic, 
                                'EBPR_acetate': model_EBPR_acetate, 'EBPR_StR':model_EBPR_StR}        
        self.model = self.tech_dict_model[self.tech]
        self.data = self.tech_dict_data[self.tech]
        self.multiyear = multiyear
        if self.multiyear == True:
            self.start_yr = kwargs.get('start_yr')
            self.end_yr = kwargs.get('end_yr')
        if self.tech == 'AS' or self.tech=='ASCP':
            self.data = self.data[(self.data.iloc[:,17]>self.data.iloc[:,2]*0.0052*0.99) & 
                                  (self.data.iloc[:,17]<self.data.iloc[:,2]*0.0052*1.01)]
            self.xdata = self.data.iloc[:,2:6]
            self.ydata = self.data.iloc[:,9:17]
            self.scalar_x = MinMaxScaler().fit(self.xdata)
            self.scalar_y = MinMaxScaler().fit(self.ydata)
        elif self.tech == 'EBPR_basic':
            self.data = self.data[self.data.iloc[:,21]>1]
            # self.data = self.data[self.data.iloc[:,11]<20]
            self.data = self.data[(self.data.iloc[:,17]>self.data.iloc[:,2]*0.0052*0.99) & 
                                  (self.data.iloc[:,17]<self.data.iloc[:,2]*0.0052*1.01)]
            self.xdata = self.data.iloc[:,2:6]
            self.ydata = self.data.iloc[:,9:17]
            self.scalar_x = MinMaxScaler().fit(self.xdata)
            self.scalar_y = MinMaxScaler().fit(self.ydata)
        elif self.tech=='EBPR_acetate':
            self.data = self.data[self.data.iloc[:,21]>1]
            self.data = self.data[self.data.iloc[:,11]<20]
            self.data = self.data[self.data.iloc[:,9]>0.001]
            self.data = self.data[(self.data.iloc[:,17]>self.data.iloc[:,2]*0.0052*0.99) & 
                                  (self.data.iloc[:,17]<self.data.iloc[:,2]*0.0052*1.01)]
            self.xdata = self.data.iloc[:,2:6]
            self.ydata = self.data.iloc[:,9:17]
            self.scalar_x = MinMaxScaler().fit(self.xdata)
            self.scalar_y = MinMaxScaler().fit(self.ydata)
        elif self.tech == 'EBPR_StR':
            self.data = self.data[self.data.iloc[:,21]>1]
            self.data = self.data[self.data.iloc[:,11]<15]
            self.data = self.data[self.data.iloc[:,9]>0.001]
            self.data = self.data[self.data.iloc[:,11]!=1.38062]
            self.data = self.data[self.data.iloc[:,13]!=9.69921]
            self.xdata = self.data.iloc[:,2:6]
            self.ydata = self.data.iloc[:,9:18]
            self.scalar_x = MinMaxScaler().fit(self.xdata)
            self.scalar_y = MinMaxScaler().fit(self.ydata)
        
    def run_model(self, sample_size, nutrient_index=1.0, flow_index=1.0):
        ''' return a numpy array containing the performance of WWT given selecting tech'''
        if self.multiyear == False:
            influent_tot = influent_SDD(sample_size)
            length = 12
        if self.multiyear == True:
             influent_tot = influent_SDD_multiyear2(sample_size, start_yr=self.start_yr, end_yr=self.end_yr, nutrient_index=nutrient_index, flow_index=flow_index)
             length = (self.end_yr - self.start_yr + 1)*12
        influent_tot_scaled = np.zeros((length,sample_size, len(self.xdata.iloc[0,:])))
        output_scaled = np.zeros((length,sample_size, len(self.ydata.iloc[0,:])))
        output_raw = np.zeros((length,sample_size, len(self.ydata.iloc[0,:])))
        for i in range(length):
            influent_tot_scaled[i,:,:] = self.scalar_x.transform(influent_tot[i,:,:])
            output_scaled[i,:,:] = self.model.predict(influent_tot_scaled[i,:,:])
            output_scaled[:,:,:] = np.where(output_scaled[:,:,:]<0, 0, output_scaled[:,:,:])
            output_raw[i,:,:] = self.scalar_y.inverse_transform(output_scaled[i,:,:])
        # return output by order: Nitrate, TSS, TP, COD, TN
        return output_scaled, output_raw, influent_tot
    
    def run_model_USRW(self, sample_size, landuse_matrix):
        '''used for optimzation: calculate point source and nonpoint source at the same time'''
        from model_SWAT.SWAT_functions import loading_outlet_USRW_opt
        
        influent_tot = influent_SDD_multiyear2(sample_size, start_yr=self.start_yr, end_yr=self.end_yr)
        length = (self.end_yr - self.start_yr +1)*12
        influent_tot_scaled = np.zeros((length,sample_size, len(self.xdata.iloc[0,:])))
        output_scaled = np.zeros((length,sample_size, len(self.ydata.iloc[0,:])))
        output_raw = np.zeros((length,sample_size, len(self.ydata.iloc[0,:])))
        for i in range(length):
            influent_tot_scaled[i,:,:] = self.scalar_x.transform(influent_tot[i,:,:])
            output_scaled[i,:,:] = self.model.predict(influent_tot_scaled[i,:,:])
            output_scaled[:,:,:] = np.where(output_scaled[:,:,:]<0, 0, output_scaled[:,:,:])
            output_raw[i,:,:] = self.scalar_y.inverse_transform(output_scaled[i,:,:])
        if self.tech == 'AS':
            outlet_nitrate, outlet_tp = loading_outlet_USRW_opt(landuse_matrix, 'AS')
        if self.tech !='AS':
              outlet_nitrate, outlet_tp = loading_outlet_USRW_opt(landuse_matrix, self.tech, output_raw, influent_tot)
        # return output by order: Nitrate, TSS, TP, COD, TN
        return outlet_nitrate, outlet_tp
        
    def run_model_single(self, influent):
        '''for single run'''
        influent_tot_scaled = self.scalar_x.transform(influent)
        output_scaled = self.model.predict(influent_tot_scaled)
        output_scaled = np.where(output_scaled<0, 0, output_scaled)
        output_raw = self.scalar_y.inverse_transform(output_scaled)    
        return output_scaled, output_raw        
        
    def run_model_ave(self, sample_size, seed = True):
        '''all influents vary under given distributions, including inflow'''
        influent_tot = influent_SDD_ave(sample_size, seed)
        # influent_tot_scaled = np.zeros((sample_size, len(self.xdata.iloc[0,:])))
        # output_scaled = np.zeros((sample_size, len(self.ydata.iloc[0,:])))
        # output_raw = np.zeros((sample_size, len(self.ydata.iloc[0,:])))
        influent_tot_scaled = self.scalar_x.transform(influent_tot)
        output_scaled = self.model.predict(influent_tot_scaled)
        output_scaled = np.where(output_scaled[:,:]<0, 0, output_scaled[:,:])
        output_raw= self.scalar_y.inverse_transform(output_scaled)    
        return output_scaled, output_raw, influent_tot

    def get_cost_energy_nutrient(self, sample_size, landuse_matrix, r, n_wwt, nutrient_index=1.0, flow_index=1.0, chem_index=1.0, utility_index=1.0, rP_index=1.0):
        '''
        calculating cost and energy from WWT and total nutreint loads from both point source and nonpoint source
        default: r = 0.07; n_wwt = 40 yrs
        '''
        af = r/(1-(1+r)**(-n_wwt))
        if self.tech == 'AS':
            cost_cap_fix = 135*(10**6)*af
            cost_op_labor = 4240000
            cost_main_labor = 1350000
            cost_material = 1010000
        elif self.tech == 'ASCP':
            cost_cap_fix = 137*(10**6)*af
            cost_op_labor = 4440000
            cost_main_labor = 1350000
            cost_material = 1030000
        elif self.tech == 'EBPR_basic' or self.tech == 'EBPR_acetate':
            cost_cap_fix = 154*(10**6)*af
            cost_op_labor = 4390000
            cost_main_labor = 1440000
            cost_material = 1150000
        elif self.tech == 'EBPR_StR':
            cost_cap_fix = 155*(10**6)*af
            cost_op_labor = 4430000
            cost_main_labor = 1450000
            cost_material = 1210000
        output_scaled, outputs, influent = self.run_model(sample_size, nutrient_index, flow_index)
        
        # from model_SWAT.SWAT_functions import loading_outlet_USRW_opt
        # outlet_nitrate, outlet_tp = loading_outlet_USRW_opt(landuse_matrix, self.tech, outputs, influent)
        # from model_SWAT.SWAT_functions import loading_outlet_USRW
        from model_SWAT.SWAT_functions import loading_outlet_USRW

        outlet_nitrate = loading_outlet_USRW('nitrate', landuse_matrix, self.tech, nutrient_index, flow_index)
        outlet_tp = loading_outlet_USRW('phosphorus', landuse_matrix, self.tech, nutrient_index, flow_index)
        
        airflow = outputs[:,:,-1]  #m3/d, (month, sample_size)
        sludge_digestor = outputs[:,:,-2]   #kg/d, (month, sample_size)
        sludge_hauld = outputs[:,:,5] #kg/d, (month, sample_size)
        energy_aeration, cost_aeration_day = blower_energy(temp=23, Q_air=airflow, elec_price=0.0638)  #$/d, (month, sample size)
        cost_aeration_yr = cost_aeration_day.mean(axis=1).reshape(-1,12).sum(axis=1)*30 #$/yr, (yrs, )

        cost_chemical_day = chemical_cost(self.tech, influent[:,:,0], influent[:,:,1], influent[:,:,3])*chem_index #$/d, (month, sample size)
        cost_chemical_yr = cost_chemical_day.mean(axis=1).reshape(-1,12).sum(axis=1)*30 #$/yr, (yrs, )
        
        energy_sludge_heating, cost_sludge_heating_day = heating_energy(sludge_digestor)  #$/d, (month, sample size)
        cost_sludge_heating_yr = cost_sludge_heating_day.mean(axis=1).reshape(-1,12).sum(axis=1)*30 #$/yr, (yrs, )
        
        cost_sludge_haul_day = sludge_cost(sludge_hauld)  #$/d, (month, sample size)
        cost_sludge_haul_yr = cost_sludge_haul_day.mean(axis=1).reshape(-1,12).sum(axis=1)*30 #$/yr, (yrs, )
        
        energy_fix, cost_energy_fix_day = fix_energy_cost(self.tech, elec_price=0.0638)  #$/d, singular
        cost_energy_fix_yr = cost_energy_fix_day*365  #$/yr
        
        cost_op_yrs = [cost_aeration_yr, cost_chemical_yr, cost_sludge_heating_yr,
                       cost_sludge_haul_yr, cost_energy_fix_yr] # $/yr, (yrs, )

        cost_labor_yr = cost_op_labor + cost_main_labor #$/yr, singular
        cost_material_yr = cost_material #$/yr, singular
        cost_energy_yr = (cost_aeration_yr + cost_sludge_heating_yr + cost_energy_fix_yr)*utility_index #$/yr,  (yrs, )
        
        cost_yr_ave = cost_cap_fix + cost_labor_yr + cost_material_yr + cost_energy_yr.mean() + cost_chemical_yr.mean() + cost_sludge_haul_yr.mean()
        cost_yrs = cost_cap_fix + cost_labor_yr + cost_material_yr + cost_energy_yr + cost_chemical_yr + cost_sludge_haul_yr
        cost_npv = npf.npv(r, cost_yrs)
        if self.multiyear == True:
            n_yr = self.end_yr - self.start_yr + 1
            cost_af = cost_npv/annuity_factor(n_yr, r)
        else:
            cost_af = cost_npv
        cost_list_yr = [cost_labor_yr, cost_material_yr, cost_energy_yr.mean(), 
                            cost_chemical_yr.mean(), cost_sludge_haul_yr.mean(), cost_cap_fix]
        
        energy = energy_aeration*3.6 + energy_sludge_heating + energy_fix # MJ/d, (month, sample size)
        energy_yrs = energy.mean(axis=1).reshape(-1,12).sum(axis=1)*30 # MJ/yr, (yrs, )
        energy_aeration_yrs = energy_aeration.mean(axis=1).reshape(-1,12).sum(axis=1)*30*3.6 # MJ/yr, (yrs, )
        energy_sludge_heating_yrs = energy_sludge_heating.mean(axis=1).reshape(-1,12).sum(axis=1)*30 # MJ/yr, (yrs, )
        energy_fix_yr = energy_fix*365 # MJ/yr, singular
        
        energy_list_yrs = [energy_aeration_yrs, energy_sludge_heating_yrs, energy_fix_yr]  #MJ/yr, (yrs, )
        
        nitrate_eff =  (outputs[:,:,0]*influent[:,:,0]/1000).mean()*365  # kg/yr, singular
        tp_eff = (outputs[:,:,2]*influent[:,:,0]/1000).mean()*365  # kg/yr, singular
        tp_inf = (influent[:,:,1]*influent[:,:,0]/1000).mean()*365 # kg/yr
        nutrient_red = [nitrate_eff, tp_eff, tp_inf]
        
        '''struvite revenue'''
        if self.tech == 'EBPR_StR':
            rP_amount = outputs[:,:,6].mean()*365*100/1000  # kg/yr; 100 m3/d, singular
        else:
            rP_amount = 0
        revenue_rP = rP_amount*0.5*rP_index # assuming $0.5/kg as default
        '''return the cost list'''
        return [cost_af, cost_yrs, cost_list_yr, cost_op_yrs, 
                energy_yrs, energy_list_yrs, 
                nutrient_red, rP_amount, revenue_rP, outlet_nitrate, outlet_tp]
        
    def get_cost_energy_ave(self, sample_size, r, n_wwt, seed=True):
        '''for analysis using averaged annual data'''
        af = r/(1-(1+r)**(-n_wwt))
        if self.tech == 'AS':
            cost_cap_fix = 135*(10**6)*af
            cost_op_labor = 4240000
            cost_main_labor = 1350000
            cost_material = 1010000
        elif self.tech == 'ASCP':
            cost_cap_fix = 137*(10**6)*af
            cost_op_labor = 4440000
            cost_main_labor = 1350000
            cost_material = 1030000
        elif self.tech == 'EBPR_basic' or self.tech == 'EBPR_acetate':
            cost_cap_fix = 154*(10**6)*af
            cost_op_labor = 4390000
            cost_main_labor = 1440000
            cost_material = 1150000
        elif self.tech == 'EBPR_StR':
            cost_cap_fix = 155*(10**6)*af
            cost_op_labor = 4430000
            cost_main_labor = 1450000
            cost_material = 1210000
        # sample_size = 10
        # a = WWT_SDD(tech='ASCP', multiyear=True, start_yr = 2003, end_yr=2018)
        # output_scaled, outputs, influent = a.run_model_ave(10)
        output_scaled, outputs, influent = self.run_model_ave(sample_size, seed)
        airflow = outputs[:,-1]  #m3/d, (sample_size)
        sludge_digestor = outputs[:,-2]   #kg/d, (sample_size)
        sludge_hauld = outputs[:,5] #kg/d, (sample_size)
        energy_aeration, cost_aeration_day = blower_energy(temp=23, Q_air=airflow, elec_price=0.0638)  #$/d, (sample size, )
        cost_aeration_yr = cost_aeration_day*365 #$/yr, (sample size, )

        cost_chemical_day = chemical_cost(self.tech, influent[:,0], influent[:,1], influent[:,3]) #$/d, (sample size, )
        cost_chemical_yr = cost_chemical_day*365 #$/yr, (sample size, )
        
        energy_sludge_heating, cost_sludge_heating_day = heating_energy(sludge_digestor)  #$/d, (sample size, )
        cost_sludge_heating_yr = cost_sludge_heating_day*365 #$/yr, (sample size, )
        
        cost_sludge_haul_day = sludge_cost(sludge_hauld)  #$/d, (sample size)
        cost_sludge_haul_yr = cost_sludge_haul_day*365 #$/yr, (sample size, )
        
        energy_fix, cost_energy_fix_day = fix_energy_cost(self.tech, elec_price=0.0638)  #$/d, singular
        cost_energy_fix_yr = cost_energy_fix_day*365  #$/yr, singular
        
        cost_labor_yr = cost_op_labor + cost_main_labor #$/yr, singular
        cost_material_yr = cost_material #$/yr, singular
        cost_energy_yr = cost_aeration_yr + cost_sludge_heating_yr + cost_energy_fix_yr #$/yr,  (sample size, )

        cost_op_samples = [cost_energy_yr, cost_chemical_yr, cost_sludge_heating_yr, cost_aeration_yr, 
                            cost_sludge_haul_yr, cost_energy_fix_yr] # $/yr, (sample size, ) and singular

        total_cost_yr_ave = cost_cap_fix + cost_labor_yr + cost_material_yr + cost_energy_yr.mean() + cost_chemical_yr.mean() + cost_sludge_haul_yr.mean()
        cost_yr_samples = cost_cap_fix + cost_labor_yr + cost_material_yr + cost_energy_yr + cost_chemical_yr + cost_sludge_haul_yr #$/yr,  (sample size, )
        cost_list_yr_ave = [cost_labor_yr, cost_material_yr, cost_energy_yr.mean(),
                            cost_chemical_yr.mean(), cost_sludge_haul_yr.mean(), cost_cap_fix]
        
        energy = energy_aeration*3.6 + energy_sludge_heating + energy_fix # MJ/d, (sample size,)
        energy_yrs = energy*365 # MJ/yr, (sample size,)
        energy_aeration_yrs = energy_aeration*3.6*365 # MJ/yr, (sample size,)
        energy_sludge_heating_yrs = energy_sludge_heating*365 # MJ/yr, (sample size,)
        energy_fix_yr = energy_fix*365 # MJ/yr, singular
        energy_list_yrs = [energy_aeration_yrs, energy_sludge_heating_yrs, energy_fix_yr]  #MJ/yr, (sample size,) and singular
        
        nitrate_eff =  (outputs[:,0]*influent[:,0]/1000)*365  # kg/yr, (sample_size,)
        tp_eff = (outputs[:,2]*influent[:,0]/1000)*365        # kg/yr, (sample_size,)
        tp_inf = (influent[:,1]*influent[:,0]/1000)*365       # kg/yr, (sample_size,)
        nutrient = [nitrate_eff, tp_eff, tp_inf]
        
        #struvite revenue
        if self.tech == 'EBPR_StR':
            rP = outputs[:,6]*365*100/1000  # kg/yr; 100 m3/d, (sample size,)
        else:
            rP = 0
        revenue_rP = rP*0.5 # assuming $0.5/kg as default
        rP_struvite = [rP, revenue_rP]  #  both (sample size,)
        '''return the cost list'''
        return [total_cost_yr_ave, cost_yr_samples, cost_list_yr_ave, cost_op_samples, 
                energy_yrs, energy_list_yrs, nutrient, rP_struvite]       

    def get_rP_struvite(self, sample_size, rP_price=0.5):
        '''kg/yr; $/yr'''
        if self.tech == 'EBPR_StR' and self.multiyear:
            rP = self.run_model(sample_size)[1][:,:,6].mean(axis=1)*365*100/1000  # kg/yr; 100 m3/d
        elif self.tech == 'EBPR_StR' and not self.multiyear:
            rP = self.run_model_ave(sample_size)[1][:,6].mean()*365*100/1000
        else:
            rP = 0    
        revenue_rP = rP*rP_price # assuming $0.5/kg as default
        return rP, revenue_rP

if __name__ == "__main__":
    landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1]=1
    tech_ebpr = WWT_SDD(tech='ASCP', multiyear=True, start_yr=2003, end_yr=2018)
    tech_ebpr_sim = tech_ebpr.get_cost_energy_nutrient(1000, landuse_matrix, r=0.07, n_wwt=40,
                                                    nutrient_index=1.0, flow_index=1.0, chem_index=1.0, 
                                                    utility_index=1.0, rP_index=1.0)