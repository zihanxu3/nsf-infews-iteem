# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
Prepare three key components of response matrix method:
    1) connectivity matrix
    2) response matrix
    3) landuse matrix
"""

# import basic packages
import pandas as pd
import numpy as np
import time
from calendar import monthrange
from model_SWAT.data import *
from model_DWT.data import df_nitrate_daily, df_streamflow_daily

#-----Function for connectivity matrix-----
def watershed_linkage(**kwargs):
    linkage = df_linkage  # df_linkage stored in model_SWAT.data
    nodes = linkage.shape[0]
    linkage_W = np.zeros((nodes,nodes))
    
    for j in range(1,5):
        for i in range (0,nodes):
            if linkage.iloc[i,j] != 0:
                col = int(linkage.iloc[i,j]) - 1
                linkage_W[i,col] = 1
    np.fill_diagonal(linkage_W,-1)
    linkage_W_inv = np.linalg.inv(linkage_W)
    if kwargs:
        print('Outlet is at subbasin', *kwargs.values())
    return linkage_W, linkage_W_inv      
            
# linkage_W = watershed_linkage(outlet=34)
# linkage_W = watershed_linkage()[0]
# linkage_W_inv = watershed_linkage()[1]


#-----Function for response matrix-----
def response_mat(name):
    '''
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        df = df_nitrate_daily
    elif name == 'streamflow':
        df = df_streamflow_daily
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    # month = df.iloc[:,2].unique()
    # month=35
    day = 365
    area_sw = df.iloc[:,3].unique()
#    response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, 365, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(365):
            df2 = df.iloc[365*subwatershed.size*(i):365*subwatershed.size*(i+1),:]
#            df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, day, df.shape[1], area_sw

# response_mat_all = response_mat('streamflow')
#response_nitrate = response_mat_all[0]
#reseponse_nitrate_yr1_month1 = response_nitrate[0,0,:,:]

#-----Functions for land use fraction of each BMP at each subwatershed-----
def basic_landuse():
    '''basic case of land use'''
    landuse = df_landuse
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
    ##return as pandas dataframe##
    return landuse, land_agri

# landuse, land_agri = basic_landuse()

def landuse_mat(scenario_name):
    '''
    Return a decison matrix (# of subwatershed, # of BMPs) to decide land use fractions
    of each BMP application in each subwatershed
    '''
    linkage = df_linkage
    df = df_nitrate
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    n = int(scenario_name[-2:])
    landuse_matrix[:,n] = 1.0
#     '''Creating a matrix of arbitrary size where rows sum to 1'''
#     if args:
#         if random == 0:
#             landuse_matrix = np.random.rand(row_sw,col_BMP)
#             landuse_matrix = landuse_matrix/landuse_matrix.sum(axis=1)[:,None]
# #            np.sum(landuse_matrix, axis=1)
    return landuse_matrix

# landuse_matrix = landuse_mat('BMP55')
# landuse_matrix[:,0:5] = 0.2

#-----Function for calculating yield of N, P, sediment, streamflow for each subwatershed-----
def get_yield(name, landuse_matrix):
    '''
    return a tuple containing two numpy array: 
        1) yield_per_BMP: (year, month, subwatershed, BMP)
        2) yield_sum: (year, month, subwatershed)
    unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm for water yield
    '''    
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    day = 365
    BMP_num = response[4]
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    # landuse_matrix = landuse_mat(scenario_name)
    
    yield_per_BMP = np.zeros((year.size, 365, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(365):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)

    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# yield_sw = get_yield('streamflow', 'BMP50')[1]
# yield_sw_flat = yield_sw.flatten()
# yield_sw_yr1 = yield_sw[0,:,:,:][0]
# yield_sw_yr2 = yield_sw[1,:,:,:][0]

#-----Function for calculating crop yield for each subwatershed-----
def get_yield_crop(name, landuse_matrix):
    '''
    return a tuple: (crop yield per unit (kg/ha) [subwatershed, year], 
    total crop yield per subwatershed (kg) [subwatershed, year] ) 
    calculate crop yield for each subwatershed
    '''
    crop = loading_per_sw(name, landuse_matrix)
    crop[np.isnan(crop)] = 0
    crop_total = np.zeros((crop.shape[2], crop.shape[0]))
    for i in range(crop.shape[2]):
        for j in range(crop.shape[0]):
            crop_total[i,j] = np.sum(crop[j,:,i,:])
    crop_unit = crop_total/basic_landuse()[1]
    crop_unit[np.isnan(crop_unit)] = 0
    return crop_total, crop_unit
    
#crop_corn = get_yield_crop('corn')

#-----Function for calculating loadings of N, P, sediment, streamflow for each subwatershed-----
def loading_per_sw(name, landuse_matrix):
    '''
    return a numpy array (year, month, subwatershed)
    calculate the background loading from the yield at each subwatershe
    unit: kg for nitrate, phosphorus; ton for sediment; mm for water 
    '''
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    # landuse_matrix = landuse_mat(scenario_name)
    
    '''landuse for agri, expressed in ha'''
    land_agri = np.mat(basic_landuse()[1])
    landuse  = basic_landuse()[0]
    total_land = np.mat(landuse.iloc[:,-1]).T
    '''total landuse for agri, expressed in ha'''
    total_land_agri = np.multiply(landuse_matrix, land_agri)
    loading = np.zeros((year.size, 365, subwatershed.size))
    '''get yield data'''
    yield_data = get_yield(name, landuse_matrix)[1]
    # yield_data = get_yield('nitrate', 'Sheet1')
    # test = np.multiply(np_yield_s1[0,0,:],total_land.T)
    '''get loading'''
    for i in range(year.size):
        for j in range(365):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    # '''add nutrient contribution from urban'''
    # loading[:,:,30] = response_matrix[:,:,30,0]*total_land[30,0]
    return loading

# loading_per_sw_test = loading_per_sw('nitrate','BMP00')
# loading_per_sw_yr1_month1 = loading_per_sw_test[0,0,:,:]
# loading_per_sw_yr1_month2 = loading_per_sw_test[0,1,:,:]
# loading_per_sw_yr2 = loading_per_sw_test[1,:,:,:]

#-----Function for calculating outlet loading of N, P, sediment, streamflow for each subwatershed-----
def loading_outlet_originalRM(name, landuse_matrix):
    '''
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
    # name = 'nitrate'
    # scenario_name = 'BMP00'
    # df = pd.read_excel('./model_SWAT\results_validation\NitrateAndStreamflowAtSub32.xlsx', sheet_name=2)
    # df[np.isnan(df)] = 0
        
    linkage_W_inv = watershed_linkage()[1]
    loading_BMP_sum = loading_per_sw(name, landuse_matrix)
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        loading_BMP_sum_minus = np.mat(loading_BMP_sum[i,:,:] * -1).T
        outlet[i,:,:]= np.dot(linkage_W_inv, loading_BMP_sum_minus)
    
    outlet = np.swapaxes(outlet,1,2)
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
        
    return outlet

# nitrate_load_sw33 = loading_outlet_originalRM('nitrate', 'BMP00')[:,:,32]  # kg/d
# streamflow_m3_sw33 = loading_outlet_originalRM('streamflow', 'BMP00')[:,:,32]  # m3/d
# nitrate_conc_sw33 = nitrate_load_sw33/streamflow_m3_sw33*1000  # mg/L
# nitrate_conc_sw33_list = nitrate_conc_sw33.flatten()

# count = np.where(nitrate_conc_sw33>8.0,1,0)
# count_byyear = count.sum(axis=1)