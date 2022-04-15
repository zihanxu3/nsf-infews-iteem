# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:56:58 2020

@author: Shaobin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_SWAT.crop_yield import basic_landuse2
from model_SWAT.SWAT_functions import basic_landuse

# set up global variable
df_nitrate = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')
df_TP = pd.read_csv('./model_SWAT/response_matrix_csv/yield_phosphorus.csv')
df_sediment = pd.read_csv('./model_SWAT/response_matrix_csv/yield_sediment.csv')
df_streamflow = pd.read_csv('./model_SWAT/response_matrix_csv/yield_streamflow.csv')
df_corn = pd.read_csv('./model_SWAT/response_matrix_csv/yield_corn.csv')
df_soybean = pd.read_csv('./model_SWAT/response_matrix_csv/yield_soybean.csv')

def response_mat_crop(crop_name):
    '''
    return a numpy (year, subwatershed, bmp)
    unit: kg/ha for soy, corn, corn silage
    '''
    if crop_name == 'soybean':
        df = df_soybean
    elif crop_name == 'corn':
        df = df_corn
    df[np.isnan(df)] = 0
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    df = df.drop(df.columns[[0,1]], axis=1)
    bmp_size = df.shape[1]
    df_to_np = np.zeros((year.size, subwatershed.size, bmp_size))
    for i in range(year.size):
            df_to_np[i,:,:] = df.iloc[subwatershed.size*(i):subwatershed.size*(i+1),:]
    return df_to_np
# response_mat_crop('corn')

def get_yield_crop(crop_name):
    '''
    calculate crop yield for each subwatershed
    return a tuple: 
    crop_production_after_forgone (kg/ha): size = (year, subwatershed, bmp) 
    total_area (ha):                       size = (year, subwatershed, bmp)
    crop_production (kg): total            size = (year, subwatershed, bmp)
    '''
    # crop_name='corn'
    df_corn = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_corn')
    # df_soybean = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_soybean')
    alpha = df_corn.iloc[:,7]
    crop = response_mat_crop(crop_name)                    #size = (year, subwatershed, bmp)
    # landuse_matrix = landuse_mat(scenario_name)          #(45,56)
    landuse_matrix = np.ones((45,62))
    cs_area = basic_landuse2()[0]                          #(45,1)
    cc_area = basic_landuse2()[1]
    sc_area = basic_landuse2()[2]
    fraction_after_forgone = np.mat(1-alpha)               #(56,1)
    ag_area = np.zeros((crop.shape[1],crop.shape[0]))      #(45,16)
    
    if crop_name =='corn':
        for i in range(0,16,2):
            ag_area[:,i] = (sc_area + cc_area).T
        for i in range(1,17,2):
            ag_area[:,i] = (cs_area + cc_area).T
    if crop_name =='soybean':
        for i in range(0,16,2):
            ag_area[:,i] = (cs_area).T
        for i in range(1,17,2):
            ag_area[:,i] = (sc_area).T
    
    area = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2]))        #(16,45,56)
    total_area = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2]))  #(16,45,56)
    for i in range(area.shape[0]):
            area[i,:,:] = np.multiply(landuse_matrix, np.mat(ag_area[:,i]).T)
            total_area[i,:,:] = np.multiply(area[i,:,:], fraction_after_forgone)
    # area = np.multiply(landuse_matrix, ag_area)            #(45,56)
    # total_area = np.multiply(area, fraction_after_forgone) #(45,56)
    crop_yield = np.zeros((crop.shape[0], crop.shape[1]))    #(16,45)
    crop_production = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2]))  #(16,45,56)
    for i in range(crop.shape[0]):
        for j in range(crop.shape[1]):
            bmp_yield = crop[i,:,:]*landuse_matrix    # bmp_yield = (45,56)
            crop_yield[i,j] = bmp_yield[i,j]          # crop_yield = (16,45) 
        crop_production[i,:,:] = np.multiply(bmp_yield, total_area[i,:,:])     #(16,45,56)
    crop_production_after_forgone = crop_production/area    # (16,45)
    # crop_production3 = crop_production2.sum(axis=1)
    return crop_production_after_forgone, total_area, crop_production

# revenue_corn = get_yield_crop('corn')[0]*0.152   # $0.152/kg of corn, $/ha

def response_mat(name):
    '''
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=0)
        df = df_nitrate
    elif name == 'phosphorus':
        df = df_TP
    elif name == 'sediment':
        df = df_sediment
    elif name == 'streamflow':
        df = df_streamflow
    else:
        raise ValueError('please enter the correct names, e.g., nitrate, phosphorus, sediment')
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    month = df.iloc[:,2].unique()
    area_sw = df.iloc[:,3].unique()
#    response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
#            df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw

# yield_tp_yr = response_mat('phosphorus')[0].sum(axis=1).mean(axis=0) # kg/(TP*yr)

def get_crop_cost(crop_name):
    '''
    return:
    cost_BMP_total: (year, subwatershed, BMP); $/yr for all hactares
    cost_BMP_ha: (year, subwatershed, BMP); $/(yr*ha)
    '''
    # scenario_name = 'BMP00'
    df_corn = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_corn')
    df_soybean = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_soybean')
    cost_corn = np.mat(df_corn.iloc[:,6]).T                  # $/ha, (56,1)
    cost_soybean = np.mat(df_soybean.iloc[:,6]).T            # $/ha, (56,1)
    crop = response_mat_crop(crop_name)                      #size = (year, subwatershed, BMP)
    landuse_matrix = np.ones((45,62))                        #(subwatershed, BMP) = (45,62)
    cs_area = basic_landuse2()[0]                            #(45,1)
    cs_area2 = np.repeat(cs_area, df_corn.shape[0], axis=1)  #(45,56)
    cc_area = basic_landuse2()[1]
    cc_area2 = np.repeat(cc_area, df_corn.shape[0], axis=1)  #(45,56)
    sc_area = basic_landuse2()[2]
    sc_area2 = np.repeat(sc_area, df_corn.shape[0], axis=1)  #(45,56)
    ag_area = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2])) #(16,45,56)
    
    # crop_name = 'soybean'
    if crop_name =='corn':
        for i in range(0,16,2):
            ag_area[i,:,:] = sc_area2 + cc_area2 # important: sc_area2 start corn in from 2001, 2003, 2005... 
        for i in range(1,17,2):
            ag_area[i,:,:] = cs_area2 + cc_area2 # important: cs_area2 start corn in 2000, 2002, 2004...
        cost = np.repeat(cost_corn, crop.shape[1], axis=1).T # (45,56)
    if crop_name =='soybean':
        for i in range(0,16,2):
            ag_area[i,:,:] = cs_area2
        for i in range(1,17,2):
            ag_area[i,:,:] = sc_area2
        cost = np.repeat(cost_soybean, crop.shape[1], axis=1).T      # (45,56)
    # ag_area = basic_landuse()[1]                                   # (45,1)
    # ag_area2 = np.repeat(ag_area, df_corn.shape[0], axis=1)        # (45,56)
    # landuse_matrix = landuse_mat(scenario_name)                    # (45,56)
    # area = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2])) # (16,45,56)
    cost_BMP_total = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2])) #(16,45,56)
    for i in range(ag_area.shape[0]):
        # area[i,:,:] = np.multiply(landuse_matrix, ag_area[i,:,:])  # (16,45,56)
        cost_BMP_total[i,:,:] = np.multiply(ag_area[i,:,:], cost)    # $/yr; (16,45,56)
    # cost_annual = cost_BMP.sum(axis=1).sum(axis=1)
    # ag_land = basic_landuse()[1]
    # ag_land[7] = 0  # 0 agricultural area for subwatershed 8   
    np.seterr(all='ignore')
    cost_BMP_ha = cost_BMP_total/ag_area
    return cost_BMP_ha, cost_BMP_total

# crop_BMP_ha, cost_BMP_total = get_crop_cost('soybean')

def bmp_compare_data(name, sw, percent=False):
    '''
    name: nutrient name
    sw: subwatershed index, starting from 0
    yield_data: (45,56); kg/ha
    cost_data: (45,56); $/ha
    return: x = nutrient or sediment reduction (kg/ha)
            y = crop net revenue loss ($/ha) 
    '''
    # name = 'phosphorus'
    ag_land = basic_landuse()[1]
    ag_land[7] = 0  # 0 agricultural area for subwatershed 8  
    yield_data = response_mat(name)[0].sum(axis=1).mean(axis=0)[:,1:56]  # kg nutrient/yr, (45,54)
    cost_data = (get_crop_cost('corn')[1] + get_crop_cost('soybean')[1]).mean(axis=0)/ag_land   # total cost, (16, 45, 56)
    cost_data = cost_data[:,1:56]
    cost_data[np.isnan(cost_data)] = 0
    cost_data[np.isinf(cost_data)] = 0

    revenue_corn = get_yield_crop('corn')[-1].mean(axis=0)*0.152  # $0.152/kg of corn, $, (45,56)
    revenue_corn = revenue_corn[:,1:56]                          # remove baseline and perenial grass
    revenue_soy = get_yield_crop('soybean')[-1].mean(axis=0)*0.356# $0.356/kg of soybean, $/
    revenue_soy = revenue_soy[:,1:56]
    
    revenue_total = (revenue_corn + revenue_soy)/ag_land
    benefit = revenue_total - cost_data
    
    cost_eff = (benefit - benefit[:,0])/yield_data
    yield_baseline = yield_data[sw,0]   # second column is 0 because it represents BMP1 (baseline)
    benefit_baseline = benefit[sw,0]

    x = yield_baseline - yield_data[sw,:]
    if percent == True:
        x = (yield_baseline - yield_data[sw,:])/yield_baseline 
    
    y = np.array(benefit_baseline - benefit[sw,:]).T
    return x, y, yield_baseline

# x,y, _ = bmp_compare_data('sediment', 32, percent=True)


def bmp_compare_plot(name,sw):
    '''
    name: nutrient name
    sw: subwatershed index, starting from 0
    yield_data: (45,56); kg/ha
    cost_data: (45,56); $/ha
    '''
    # name = 'phosphorus'
    # sw = 32
    ag_land = basic_landuse()[1]
    ag_land[7] = 0  # 0 agricultural area for subwatershed 8  
    yield_data = response_mat(name)[0].sum(axis=1).mean(axis=0)[:,1:56]  # kg nutrient/yr, (45,54)
    cost_data = (get_crop_cost('corn')[1] + get_crop_cost('soybean')[1]).mean(axis=0)/ag_land   # total cost, (16, 45, 56)
    cost_data = cost_data[:,1:56]
    cost_data[np.isnan(cost_data)] = 0
    cost_data[np.isinf(cost_data)] = 0

    revenue_corn = get_yield_crop('corn')[-1].mean(axis=0)*0.152  # $0.152/kg of corn, $, (45,56)
    revenue_corn = revenue_corn[:,1:56]                           # remove baseline and perenial grass
    revenue_soy = get_yield_crop('soybean')[-1].mean(axis=0)*0.356# $0.356/kg of soybean, $/
    revenue_soy = revenue_soy[:,1:56]
    
    revenue_total = (revenue_corn + revenue_soy)/ag_land
    benefit = revenue_total - cost_data
    
    # cost_eff = (benefit - benefit[:,0])/yield_data
    yield_baseline = yield_data[sw,0]   # singular, kg/yr on average
    benefit_baseline = benefit[sw,0]    # singular, kg/yr on average

    x = np.array(yield_baseline - yield_data[sw,:]) #
    y = np.array(benefit_baseline - benefit[sw,:])
    z = y[0]/x
    # z[24] = 0
    # y[np.isnan(y)] = 0
    BMP_list = ['BMP' + str(i) for i in range(1,55)]
    
    fig, ax = plt.subplots(figsize=(12,6))  
    ax.scatter(x[:54],y[0,:54])
    for i, txt in enumerate(BMP_list):
        ax.annotate(txt, (x[i], y[0,i]))
    
    ax.set_xlabel(name + ' reduction (kg/ha)', fontsize= 12)
    ax.set_ylabel('Crop net revenue loss ($/ha)', fontsize=12)    
    plt.show()
    return x, y

# cost_eff = bmp_compare_plot('phosphorus', 32)
# a = cost_eff[0][:54]
# b = cost_eff[1][0,:54]
# cost_eff = bmp_compare_plot('nitrate', 32)
# cost_eff = bmp_compare_plot('sediment', 32)
# cost_eff = bmp_compare_plot('streamflow', 32)