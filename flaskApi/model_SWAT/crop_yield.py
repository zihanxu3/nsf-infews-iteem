# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 22:53:52 2020

@author: Shaobin

Purpose: functions for calculating crop yield and cost
"""

# import packages
import pandas as pd
import numpy as np
from model_SWAT.SWAT_functions import basic_landuse, landuse_mat

df_corn = pd.read_csv('./model_SWAT/response_matrix_csv/yield_corn.csv')
df_soybean = pd.read_csv('./model_SWAT/response_matrix_csv/yield_soybean.csv')
df_corn_silage = pd.read_csv('./model_SWAT/response_matrix_csv/yield_corn_silage.csv')
df_switchgrass = pd.read_csv('./model_SWAT/response_matrix_csv/yield_switchgrass.csv')
# sensitivity analysis
# parset = 'Set11'
# file_name = './model_SWAT/sensitivity_analysis/MonthlyYields_Feb6_Parameter'+ parset +'.xlsx'
# xls = pd.ExcelFile(file_name)
# df_corn = pd.read_excel(xls, 'Corn')
# df_soybean = pd.read_excel(xls, 'Soy')
# df_corn_silage = pd.read_excel(xls, 'Corn Silage')


def response_mat_crop(crop_name):
    '''
    return a numpy (year, subwatershed, bmp)
    unit: kg/ha for soy, corn, corn silage
    '''
    if crop_name == 'soybean':
        df = df_soybean
    elif crop_name == 'corn':
        df = df_corn
    elif crop_name == 'corn sillage':
        df = df_corn_silage
    elif crop_name == 'switchgrass': 
        df = df_switchgrass
    df[np.isnan(df)] = 0
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    df = df.drop(df.columns[[0,1]], axis=1)
    bmp_size = df.shape[1]
    df_to_np = np.zeros((year.size, subwatershed.size, bmp_size))
    for i in range(year.size):
            df_to_np[i,:,:] = df.iloc[subwatershed.size*(i):subwatershed.size*(i+1),:]
    return df_to_np

# switchgrass = response_mat_crop('switchgrass')
# corn = response_mat_crop('corn')

#-----Functions for land use fraction of each BMP at each subwatershed-----
def basic_landuse2():
    '''basic case of land use: (45,1)'''
    landuse = pd.read_excel('./model_SWAT/landuse.xlsx', sheet_name='landuse2').fillna(0)
    cs = np.mat(landuse.iloc[:,1]).T
    cc = np.mat(landuse.iloc[:,2]).T
    sc = np.mat(landuse.iloc[:,3]).T
    ##return as pandas dataframe##
    return cs, cc, sc

# land_cs, land_cc, land_sc = basic_landuse2()


#-----Function for calculating crop yield for each subwatershed-----
def get_yield_crop(crop_name, landuse_matrix):
    '''
    calculate crop yield for each subwatershed
    return a tuple: 
    crop_production (kg): size= (year, subwatershed, BMP)
    crop_production2 (kg): size = (year, subwatershed)
    total_area (ha): size = (year, subwatershed, BMP)
    '''
    df_corn = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_corn')
    # df_soybean = pd.read_excel(r'C:\ITEEM\Submodel_Economics\Economics.xlsx', sheet_name='Crop_soybean')
    alpha = df_corn.iloc[:,7]
    # crop_name = 'soybean'
    crop = response_mat_crop(crop_name)                    #size = (year, subwatershed, bmp)    
    # landuse_matrix = landuse_mat(scenario_name)          #(45,56)
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
    # crop_yield = np.zeros((crop.shape[0], crop.shape[1]))  #(16,45)
    crop_production = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2]))  #(16,45,56)
    for i in range(crop.shape[0]):
        # for j in range(crop.shape[1]):
            # bmp_yield = crop[i,:,:]*landuse_matrix    # bmp_yield = (45,56)
            # crop_yield[i,j] = np.sum(bmp_yield[j,:])  # crop_yield = (16,45) 
        # crop_production[i,:,:] = np.multiply(bmp_yield, total_area[i,:,:])   #(16,45,56)
        crop_production[i,:,:] = np.multiply(crop[i,:,:], total_area[i,:,:])
    if crop_name == 'switchgrass':
        # landuse_matrix[:,55]
        ag_area = basic_landuse()[1]
        sg_area = np.multiply(landuse_matrix[:,55], ag_area.T)
        crop_production[:,:,55] = np.multiply(crop[:,:,55], sg_area)
        # total_area = sg_area.sum()  # ha, (1,45)
    crop_production2 = crop_production.sum(axis=2)
    # crop_production3 = crop_production2.sum(axis=1).mean()
    return crop_production, crop_production2, total_area

# landuse_matrix = np.zeros((45,62))
# landuse_matrix[:,37] = 0.75
# landuse_matrix[:,55] = 0.25
# soy_yield = get_yield_crop('switchgrass', landuse_matrix)
# sg_yield = get_yield_crop('switchgrass', landuse_matrix)[1].sum()
# corn_yield = get_yield_crop('corn', landuse_matrix)[1].sum(axis=1)
# corn_production = get_yield_crop('soybean', 'BMP00')[1]
# corn_production = get_yield_crop('corn', landuse_matrix)[1].sum(axis=1).mean()
# soybean_yield = get_yield_crop('soybean','BMP00')[0]
# soybean_production = get_yield_crop('soybean','BMP00')[1]

#-----Function for calculating crop production cost -----
def get_crop_cost(crop_name, landuse_matrix):
    '''
    return total production cost of crop in $/yr: 
    cost_BMP: (year, subwatershed, BMP) for all hactares
    cost_annual: (year,1) for all subwatersheds, all BMPs, all hactares
    '''
    df_corn = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_corn')
    df_soybean = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='Crop_soybean')
    df_switchgrass = pd.read_excel('./model_Economics/Economics.xlsx', sheet_name='switchgrass')
    cost_corn = np.mat(df_corn.iloc[:,6]).T                  # $/ha, (56,1)
    cost_soybean = np.mat(df_soybean.iloc[:,6]).T            # $/ha, (56,1)

    crop = response_mat_crop(crop_name)                      # size = (year, subwatershed, BMP)
    # landuse_matrix = landuse_mat(scenario_name)            # (subwatershed, BMP) = (45,56)
    cs_area = basic_landuse2()[0]                            # (45,1)
    cs_area2 = np.repeat(cs_area, df_corn.shape[0], axis=1)  # (45,56)
    cc_area = basic_landuse2()[1]
    cc_area2 = np.repeat(cc_area, df_corn.shape[0], axis=1)  # (45,56)
    sc_area = basic_landuse2()[2]
    sc_area2 = np.repeat(sc_area, df_corn.shape[0], axis=1)  # (45,56)
    ag_area = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2]))        # (16,45,56)
    area = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2]))     # (16,45,56)
    cost_BMP = np.zeros((crop.shape[0], crop.shape[1], crop.shape[2])) # (16,45,56)
    if crop_name =='corn':
        for i in range(0,16,2):
            ag_area[i,:,:] = sc_area2 + cc_area2             # important: sc_area2 start corn in from 2001, 2003, 2005... 
        for i in range(1,17,2):
            ag_area[i,:,:] = cs_area2 + cc_area2             # important: cs_area2 start corn in 2000, 2002, 2004...
        cost = np.repeat(cost_corn, crop.shape[1], axis=1).T # (45,56)
        cost[:,55] = 0
        for i in range(ag_area.shape[0]):
            area[i,:,:] = np.multiply(landuse_matrix, ag_area[i,:,:])
            cost_BMP[i,:,:] = np.multiply(area[i,:,:], cost)
            
    elif crop_name =='soybean':
        for i in range(0,16,2):
            ag_area[i,:,:] = cs_area2
        for i in range(1,17,2):
            ag_area[i,:,:] = sc_area2
        cost = np.repeat(cost_soybean, crop.shape[1], axis=1).T        # (45,56)
        cost[:,55] = 0
        # ag_area = basic_landuse()[1]                                 # (45,1)
        # ag_area2 = np.repeat(ag_area, df_corn.shape[0], axis=1)      # (45,56)
        for i in range(ag_area.shape[0]):
            area[i,:,:] = np.multiply(landuse_matrix, ag_area[i,:,:])
            cost_BMP[i,:,:] = np.multiply(area[i,:,:], cost)

    elif crop_name == 'switchgrass':
        landuse_matrix[:,55]
        cost_switchgrass = df_switchgrass.iloc[:,3]              # (16,1)  opportunity cost included: df_switchgrass.iloc[:,2]
        ag_area = basic_landuse()[1]                             # (45,1)
        sg_area = np.multiply(landuse_matrix[:,55], ag_area.T)   # (1,45)
        for i in range(crop.shape[0]):
            cost_BMP[i,:,55] = np.multiply(sg_area, cost_switchgrass[i])
    
    cost_annual = cost_BMP.sum(axis=1).sum(axis=1)
    return cost_BMP, cost_annual

# landuse_matrix = np.zeros((45,62))
# landuse_matrix[:,1] = 0.5
# landuse_matrix[:,55] = 1
# cost_switchgrass = get_crop_cost('switchgrass', landuse_matrix)[1]
# corn = response_mat_crop('corn')
# corn = get_crop_cost('corn',landuse_matrix)
# soybean = get_crop_cost('soybean',landuse_matrix)
# crop_cost2 = get_crop_cost('soybean','BMP01')[1]
# crop_corn = get_crop_cost('corn','BMP01')[1]
# crop_soybean = get_crop_cost('soybean','BMP01')[0].sum(axis=2)

'''P content'''    
def get_P_crop(landuse_matrix):
    '''return P content of corn in Metric ton/yr'''
    corn = get_yield_crop('corn', landuse_matrix)[1].sum(axis=1).mean()/1000 # MT/yr
    P_corn_self = corn*0.85*0.26/100  # MT/yr, 0.26% P in corn, dry basis
    P_corn_import = 17966 - P_corn_self  # MT/yr, 17966 MT is the fixed total P in corn
    
    soybean = get_yield_crop('soybean', landuse_matrix)[1].sum(axis=1).mean()/1000    
    P_soybean = soybean*0.85*0.41/100  #0.41% P in soybean, dry basis
    
    sg = get_yield_crop('switchgrass', landuse_matrix)[1].sum(axis=1).mean()/1000
    P_sg = sg*0.85*0.1/100  #0.1% P in switchgrass, assumed    
    return P_corn_self, P_corn_import, P_soybean, P_sg


def get_P_fertilizer(crop_name, landuse_matrix):
    '''return P content of fertilizer in Metric ton/yr'''
    total_area = get_yield_crop('corn', landuse_matrix)[2].mean(axis=0).sum(axis=0)  # (16,45,62)
    bmp_P_baseline = [1 for i in range(19)]
    bmp_P_15 = [0.85 for i in range(18)]
    bmp_P_30 = [0.7 for i in range(18)]
    bmp_P_0 = [55/207]
    bmp_P_biosolid = [0]*6
    
    bmp_P = []
    bmp_P = bmp_P_baseline + bmp_P_15 + bmp_P_30 + bmp_P_0 + bmp_P_biosolid
    P_fertilizer = total_area*bmp_P*207*0.2/1000   # 207 kg/ha DAP, 20% P in DAP 
    return P_fertilizer.sum()

# landuse_matrix = np.zeros((45,62))
# landuse_matrix[:,1] = 0.5
# landuse_matrix[:,55] = 0.5
# P_fertilizer = get_P_fertilizer('corn', landuse_matrix)
    