# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS project - ITEEM

@author: Shaobin

Purpose: get parameters for NH3, TP, Inflow from the following three distributions:
    1) lognormal
    2) normal
    3) triangular
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, lognorm, triang, gamma, beta
from fitter import Fitter

def get_monthly_ave(name):
    '''return a np(year,month) for effluent and prec of SDD'''
    if name == 'effluent':
        df = pd.read_excel('./model_WWT/SSD_effluent.xlsx', parse_dates=['Date'], index_col='Date')
        df = df.dropna()
        df = df[(df.iloc[:,0]<115)]
        mask = (df != 0).any(axis=1)
        df = df.loc[mask]
        df_min, df_max = min(df.iloc[:,0]), max(df.iloc[:,0])  
        df['month'] = df.index.month
        df['year'] = df.index.year
    elif name =='precipitation':
         df = pd.read_excel('./model_WWT/SSD_effluent.xlsx', parse_dates=['Date'], index_col='Date', sheet_name=1)
         df = df.dropna()
         df_min, df_max = min(df.iloc[:,0]), max(df.iloc[:,0])  
         df['month'] = df.index.month
         df['year'] = df.index.year
    output_np = np.zeros((2020-1990,12))
    for i in range (2020-1990):
        for j in range(12):
            output_np[i,j] = df[(df.year==1990+i) & (df.month==1+j)].iloc[:,0].mean()
    output_np = output_np[13:29,:]
    return output_np

# output_eff = get_monthly_ave('effluent').flatten()
# output_prec = get_monthly_ave('precipitation').flatten()

def get_lognormal_effluent():
    df = pd.read_excel('./model_WWT/SSD_effluent.xlsx', parse_dates=['Date'], index_col='Date')
    df = df.dropna()
    df = df[(df.iloc[:,0]<115)]
    mask = (df != 0).any(axis=1)
    df = df.loc[mask]
    df_min, df_max = min(df.iloc[:,0]), max(df.iloc[:,0])  
    df['month'] = df.index.month
    # df['year'] = df.index.year
    # eff_np = np.zeros((12,2020-1990))
    # for i in range (2020-1990):
    #     for j in range(12):
    #         eff_np[j,i] = df[(df.year==1990+i) & (df.month==1+j)].iloc[:,0].mean()
    
    sigma = []
    mu = []
    for i in range(12):
        data = df[(df.month==i+1)].iloc[:,0]
        parm = lognorm.fit(data,floc=0)
        sigma.append(parm[0])
        mu.append(np.log(parm[2]))
#        
#        mean = np.exp(mu + 1/2*(sigma**2))
#        mean_data = data.mean()
#        median = np.exp(mu)
#        cv = np.sqrt(np.exp(sigma**2) - 1)
#        sd = mean*np.sqrt(np.exp(sigma**2) - 1)
    data = df.iloc[:,0]
    parm = lognorm.fit(data,floc=0)
    sigma.append(parm[0])
    mu.append(np.log(parm[2]))
    
    return mu, sigma

#mu, sigma = get_lognormal_effluent()

def get_lognormal_para(name):
    name = 'Inflow'
    df = pd.read_excel('./model_WWT/SDD_N_P_2012-2019.xlsx', 
                       parse_dates=['Date'], index_col='Date', sheet_name=1)
    start_date = '2012-08-01'
    end_date = '2019-8-15'
    mask = (df.index > start_date) & (df.index <= end_date)
    df = df.loc[mask]
    
    if name=='NH3':
        data = df.iloc[:,0]
        data = data.replace(0, np.nan)
        data = data.dropna()
    elif name=='TP':
        data = df.iloc[:,3]
        data = data.replace(0, np.nan)
        data = data.dropna()
    elif name=='Inflow':
        data = df.iloc[:,2]
        data = data.replace(0, np.nan)
        data = data.dropna()
        data = pd.DataFrame(data)
        data.iloc[:,0].value_counts()
        data = data[data.iloc[:,0] != 30]  # inflow data used is only sewage flow, 30 MGD needs to be removed.
        data = np.array(data)
    else:
        print ('wrong inputs')
    
    parm = lognorm.fit(data,floc=0)  #parm[0] = sigma; parm[1]=location, 0; parm[2]=median, m
    sigma = parm[0]
    mu = np.log(parm[2])  # mu is not equal to arithmetic mean
    mean = np.exp(mu + 1/2*(sigma**2))
    mean_data = data.mean()
    median = np.exp(mu)
    cv = np.sqrt(np.exp(sigma**2) - 1)
    sd = mean*np.sqrt(np.exp(sigma**2) - 1)
    
        
    return {'mu': mu, 'sigma': sigma, 'cv':cv, 'median (scale, m)': parm[2], 
            'mean (E[X])': mean, 'mean_realdata': mean_data,  'SD[X]': sd,
            'location': parm[1]}
    
#parameter_NH3_lognorm = get_lognormal_para('NH3')
#parameter_TP_lognorm = get_lognormal_para('TP')
#parameter_inflow_lognorm = get_lognormal_para('Inflow')

def get_normal_para(name):
    df = pd.read_excel('./model_WWT/SDD_N_P_2012-2019.xlsx', 
                       parse_dates=['Date'], index_col='Date',sheet_name=1)
    start_date = '2012-08-01'
    end_date = '2019-8-15'
    mask = (df.index > start_date) & (df.index <= end_date)
    df = df.loc[mask]
    
    if name=='NH3':
        data = df.iloc[:,0]
        data = data.replace(0, np.nan)
        data = data.dropna()
    elif name=='TP':
        data = df.iloc[:,3]
        data = data.replace(0, np.nan)
        data = data.dropna()
    elif name=='Inflow':
        data = df.iloc[:,2]
        data = data.replace(0, np.nan)
        data = data.dropna()
        # data = df.iloc[:,2]
        # data = data.replace(0, np.nan)
        # data = data.dropna()
        data = pd.DataFrame(data)
        data.iloc[:,0].value_counts()
        data = data[data.iloc[:,0] != 30]
        data = np.array(data)
    else:
        print ('wrong inputs')
    
    m, s = norm.fit(data)
    cv = s/m
    
    return {'mu': m, 'sigma': s, 'cv': cv}
    
    
#parameter_NH3_norm = get_normal_para('NH3')
#parameter_TP_norm = get_normal_para('TP')
#parameter_inflow_norm = get_normal_para('Inflow')

def get_triang_para(name):
    df = pd.read_excel('./model_WWT/SDD_N_P_2012-2019.xlsx',
                       parse_dates=['Date'], index_col='Date',sheet_name= 1)
    start_date = '2012-08-01'
    end_date = '2019-8-15'
    mask = (df.index > start_date) & (df.index <= end_date)
    df = df.loc[mask]
    
    if name=='NH3':
        data = df.iloc[:,0]
        data = data.replace(0, np.nan)
        data = data.dropna()
    elif name=='TP':
        data = df.iloc[:,3]
        data = data.replace(0, np.nan)
        data = data.dropna()
    elif name=='Inflow':
        data = df.iloc[:,2]
        data = data.replace(0, np.nan)
        data = data.dropna()
        data = pd.DataFrame(data)
        data.iloc[:,0].value_counts()
        data = data[data.iloc[:,0] != 30]
        data = np.array(data)
    else:
        print ('wrong inputs')
    
    c, loc, scale = triang.fit(data)
    left = loc
    mode = loc + c*scale
    right = loc + scale    
    return {'left': left, 'mode': mode, 'right': right, 'c': c, 'loc': loc, 'scale': scale}

#parameter_inflow_triang = get_triang_para('Inflow')
#parameter_inflow_triang['mode']
