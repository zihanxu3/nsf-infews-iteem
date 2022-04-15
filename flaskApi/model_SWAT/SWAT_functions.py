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

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from calendar import monthrange
import scipy.io
from model_WWT.SDD_analysis.wwt_model_SDD import WWT_SDD
from model_SWAT.data import *

#-----Function for connectivity matrix-----
def watershed_linkage(**kwargs):
    linkage = df_linkage  # df_linkage stored in data.py file in the same folder
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

#-----Function for response matrix-----
def response_mat(name):
    '''
    sa = sensitivity analysis
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
    # response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
            # df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
    return df_to_np, subwatershed, year, month, df.shape[1], area_sw

# response_mat_all = response_mat('phosphorus')[0][:,:,7,0]
#response_nitrate = response_mat_all[0]

#-----Functions for land use fraction of each BMP at each subwatershed-----
def basic_landuse():
    '''basic case of land use'''
    landuse = df_landuse
    land_agri = landuse.iloc[:,1] + landuse.iloc[:,2]
    land_agri = np.mat(land_agri).T
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
    return landuse_matrix

# landuse_matrix = landuse_mat('BMP55')
# landuse_matrix[:,0:5] = 0.2
# landuse_matrix = np.zeros((45,56))
# landuse_matrix[:, 5] = 1.0

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
    BMP_num = response[4]
    # landuse_matrix = landuse_matrix_combinedS1
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    # landuse_matrix = landuse_mat(scenario_name)
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)

    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1] = 1
# yield_sw = get_yield('streamflow', landuse_matrix)[1]
# yield_sum = yield_sw.sum(axis=1)
# yield_ave = yield_sum.mean(axis=0)
# yield_sw_flat = yield_sw.flatten()
# yield_sw_yr1 = yield_sw[0,:,:,:][0]
# yield_sw_yr2 = yield_sw[1,:,:,:][0]

#-----Function for calculating loadings of N, P, sediment, streamflow for each subwatershed-----
def loading_landscape(name, landuse_matrix):
    '''
    return
    loading: calculate the sum of landscape loss from at each subwatershed: (year, month, subwatershed)
    outlet: outlet at each subwatershed: (year, month, subwatershed)
    unit of loading and outlet: kg/month for nitrate, phosphorus; ton/month for sediment; m3/month for streamflow
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
    loading = np.zeros((year.size, month.size, subwatershed.size))
    '''get yield data'''
    yield_data = get_yield(name, landuse_matrix)[1]
    # yield_data = get_yield('nitrate', 'Sheet1')
    # test = np.multiply(np_yield_s1[0,0,:],total_land.T)
    '''get background loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    # '''add nutrient contribution from urban'''
    # loading[:,:,30] = response_matrix[:,:,30,0]*total_land[30,0]
    '''get landscape outlet''' 
    linkage_W_inv = watershed_linkage()[1]
    loading_BMP_sum = loading
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[2], loading_BMP_sum.shape[1]))
    for i in range(loading_BMP_sum.shape[0]):
        loading_BMP_sum_minus = np.mat(loading_BMP_sum[i,:,:] * -1).T
        outlet[i,:,:]= np.dot(linkage_W_inv, loading_BMP_sum_minus)
    
    outlet = np.swapaxes(outlet,1,2)
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
    return loading, outlet

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1] = 1
# loading, outlet = loading_landscape('sediment', landuse_matrix)
# loading_sw_yr2 = loading_sw_test[1,:,:,:]

#-----Function for calculating outlet loading of N, P, sediment, streamflow for each subwatershed-----
def loading_outlet_USRW(name, landuse_matrix, tech_wwt='AS', nutrient_index=1.0, flow_index=1.0):
    '''
    return a numpy array: (year, month,subwatershed)
    reservoir watershed: 33; downstream of res: 32
    outlet: 34
    '''
    df = df_linkage2
    df[np.isnan(df)] = 0
    # name = 'nitrate'
    # scenario_name = 'BMP00'
    loading_BMP_sum = loading_landscape(name, landuse_matrix)[0]
    outlet = np.zeros((loading_BMP_sum.shape[0], loading_BMP_sum.shape[1], loading_BMP_sum.shape[2]))
    for i in range(33):
        a = df.loc[i].unique().astype('int')
        a = a[a!=0]
        for j in a:
            # print (j)
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]     
    # Total loading in sw32 = res_out + background loading
    '''******************Start of reservior trapping effect*******************'''
    res_in = outlet[:,:,32]
    if name == 'nitrate':
        res_out = res_in * 0.8694 - 46680.0 # equationd derived from data
    elif name =='phosphorus':
        res_out = res_in * 0.8811 - 2128.1  # equationd derived from data
    elif name =='sediment':
        res_out = 14.133*res_in**0.6105     # equationd derived from data
        # res_out = res_in
    elif name =='streamflow':
        res_out = res_in * 1.0075 - 1.9574  # equationd derived from data
    res_out = np.where(res_out<0, 0, res_out)
        
    # sw32 is the downstream of reservoir
    outlet[:,:,31] = loading_BMP_sum[:,:,31] + res_out
    '''******************End of reservior trapping effect*******************'''
    
    # update loading in SDD (sw31)
    outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31]
    
    '''***********************Start of point source*************************'''
    # name = 'nitrate'
    if tech_wwt == 'AS':
        if name == 'nitrate' or name == 'phosphorus':
            df_point = df_point_SDD
            if name == 'nitrate':
                df_point = pd.DataFrame(df_point.iloc[:,0])
            elif name == 'phosphorus':
                df_point = pd.DataFrame(df_point.iloc[:,1])  
            df_point['month'] = df_point.index.month
            df_point['year'] = df_point.index.year
            df2_point = np.zeros((16,12))
            for i in range(16):
                for j in range(12):
                    df2_point[i,j] = df_point.loc[(df_point.year==2003+i) & (df_point.month==1+j)].iloc[:,0].astype('float').sum()
            # Calculate loading in sw31 with point source
            # loading_BMP_sum[i,j,30] = ANN...
            if name =='nitrate':
                # point_Nitrate = 1315.43*30 # kg/month, average
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
            elif name == 'phosphorus':
                # point_TP = 1923.33*30# kg/month, average
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
                
    # Calculate loading in sw31 with point source
    elif tech_wwt != 'AS':
        if name == 'nitrate' or name == 'phosphorus':
            instance = WWT_SDD(tech=tech_wwt, multiyear=True, start_yr=2003, end_yr=2018)
            output_scaled, output_raw, influent_tot = instance.run_model(sample_size=1000, nutrient_index=nutrient_index, flow_index=flow_index)
    
            if name == 'nitrate':
                nitrate_load = output_raw[:,:,0]*influent_tot[:,:,0]
                loading_day = nitrate_load.mean(axis=1)/1000  # loading: kg/d
                loading_day = loading_day.reshape(16,12)
                
            elif name == 'phosphorus':
                tp_load = output_raw[:,:,2]*influent_tot[:,:,0]
                loading_day = tp_load.mean(axis=1)/1000  # loading: kg/d
                loading_day = loading_day.reshape(16,12)

            loading_month = np.zeros((16,12))    #16 yr, 12 month
            for i in range(16):
                for j in range(12):
                        loading_month[i,j] = loading_day[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month
            if name =='nitrate':
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + loading_month
            elif name == 'phosphorus':
                outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + loading_month
    '''***********************End of point source***************************'''
    
    # b contains all upstream subwatersheds for sw31
    b = df.loc[30].unique().astype('int')
    b = b[b!=0]
    # get unique subwatersheds that do not contribute to reservoir
    for i in range(33,45):
        c = df.loc[i].unique().astype('int')
        c = c[c!=0]
        d = list(set(c) - set(b))
        # Key step: the following equation takes the trapping efficiency into account. 
        # All upstream contributions of sw32 is reconsidered with trapping efficiency 
        if 31 in list(c):
            # print ('true, i=',i)
            outlet[:,:,i] = outlet[:,:,30]
        for j in d:
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]
    # update the loadings for upperstream that has higher values
    e = b[b>33] 
    for i in e:
        f = df.loc[i-1].unique().astype('int')
        f = f[f!=0]
        for j in f:
            outlet[:,:,i-1] += loading_BMP_sum[:,:,j-1]
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
    # add adjustment coefficient
    if name =='phosphorus':
        outlet = outlet/1.07  # 7% overestimates across all BMPs
    return outlet

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,1]=1
# tp = loading_outlet_USRW('phosphorus', landuse_matrix, 'EBPR_acetate')
# nitrate = loading_outlet_USRW('sediment', landuse_matrix, 'ASCP')[:,:,33].sum(axis=1).mean()
# landuse_matrix = np.zeros((45,56))
# landuse_matrix[:,48]=1
# sediment = loading_outlet_USRW('sediment', landuse_matrix, 'AS')[:,:,33].sum(axis=1).mean()

def loading_outlet_USRW_opt(landuse_matrix, tech_wwt, output_raw, influent_tot):
    '''
    return two numpy arrays: (year, month,subwatershed) for nitrate and TP at the same time
    reservoir watershed: 33; downstream of res: 32
    outlet: 34
    '''
    df = df_linkage2
    df[np.isnan(df)] = 0
    loading_BMP_sum_nitrate = loading_landscape('nitrate', landuse_matrix)[0]
    loading_BMP_sum_tp = loading_landscape('phosphorus', landuse_matrix)[0]
    
    outlet_nitrate = np.zeros((loading_BMP_sum_nitrate.shape[0], loading_BMP_sum_nitrate.shape[1], loading_BMP_sum_nitrate.shape[2]))
    outlet_tp = np.zeros((loading_BMP_sum_tp.shape[0], loading_BMP_sum_tp.shape[1], loading_BMP_sum_tp.shape[2]))
    
    for i in range(33):
        a = df.loc[i].unique().astype('int')
        a = a[a!=0]
        for j in a:
            # print (j)
            outlet_nitrate[:,:,i] += loading_BMP_sum_nitrate[:,:,j-1]    
            outlet_tp[:,:,i] += loading_BMP_sum_tp[:,:,j-1]  
    # Total loading in sw32 = res_out + background loading
    '''******************Start of reservior trapping effect*******************'''
    res_in_nitrate = outlet_nitrate[:,:,32]
    res_out_nitrate = res_in_nitrate * 0.8694 - 46680.0 # equationd derived from data
    res_in_tp = outlet_tp[:,:,32]
    res_out_tp = res_in_tp * 0.8811 - 2128.1  # equationd derived from data
        
    res_out_nitrate = np.where(res_out_nitrate<0, 0, res_out_nitrate)
    res_out_tp = np.where(res_out_tp<0, 0, res_out_tp)      
    # sw32 is the downstream of reservoir
    outlet_nitrate[:,:,31] = loading_BMP_sum_nitrate[:,:,31] + res_out_nitrate
    outlet_tp[:,:,31] = loading_BMP_sum_tp[:,:,31] + res_out_tp
    '''******************End of reservior trapping effect*******************'''

    # update loading in SDD (sw31)
    outlet_nitrate[:,:,30] = loading_BMP_sum_nitrate[:,:,30] + outlet_nitrate[:,:,31]
    outlet_tp[:,:,30] = loading_BMP_sum_tp[:,:,30] + outlet_tp[:,:,31]
    
    '''***********************Start of point source*************************'''
    # name = 'nitrate'
    if tech_wwt == 'AS':
        df_point = df_point_SDD
        df_point_nitrate = pd.DataFrame(df_point.iloc[:,0])
        df_point_tp = pd.DataFrame(df_point.iloc[:,1])  
        df_point_nitrate['month'] = df_point_nitrate.index.month
        df_point_nitrate['year'] = df_point_nitrate.index.year
        df_point_tp['month'] = df_point_tp.index.month
        df_point_tp['year'] = df_point_tp.index.year
        
        df2_point_nitrate = np.zeros((16,12))
        df2_point_tp = np.zeros((16,12))
        for i in range(16):
            for j in range(12):
                df2_point_nitrate[i,j] = df_point_nitrate.loc[(df_point_nitrate.year==2003+i) & (df_point_nitrate.month==1+j)].iloc[:,0].astype('float').sum()
                df2_point_tp[i,j] = df_point_tp.loc[(df_point_tp.year==2003+i) & (df_point_tp.month==1+j)].iloc[:,0].astype('float').sum()
        outlet_nitrate[:,:,30] = loading_BMP_sum_nitrate[:,:,30] + outlet_nitrate[:,:,31] + df2_point_nitrate
        outlet_tp[:,:,30] = loading_BMP_sum_tp[:,:,30] + outlet_tp[:,:,31] + df2_point_tp
        
    elif tech_wwt != 'AS':
        nitrate_load = output_raw[:,:,0]*influent_tot[:,:,0]
        loading_day_nitrate = nitrate_load.mean(axis=1)/1000  # loading: kg/d
        loading_day_nitrate = loading_day_nitrate.reshape(16,12)

        tp_load = output_raw[:,:,2]*influent_tot[:,:,0]
        loading_day_tp = tp_load.mean(axis=1)/1000  # loading: kg/d
        loading_day_tp = loading_day_tp.reshape(16,12)

        loading_month_nitrate = np.zeros((16,12))    #16 yr, 12 month
        loading_month_tp = np.zeros((16,12))    #16 yr, 12 month        

        for i in range(16):
            for j in range(12):
                    loading_month_nitrate[i,j] = loading_day_nitrate[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month
                    loading_month_tp[i,j] = loading_day_tp[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month        

        outlet_nitrate[:,:,30] = loading_BMP_sum_nitrate[:,:,30] + outlet_nitrate[:,:,31] + loading_month_nitrate
        outlet_tp[:,:,30] = loading_BMP_sum_tp[:,:,30] + outlet_tp[:,:,31] + loading_month_tp
    '''***********************End of point source***************************'''
    # b contains all upstream subwatersheds for sw31
    b = df.loc[30].unique().astype('int')
    b = b[b!=0]
    # get unique subwatersheds that do not contribute to reservoir
    for i in range(33,45):
        c = df.loc[i].unique().astype('int')
        c = c[c!=0]
        d = list(set(c) - set(b))
        # Key step: the following equation takes the trapping efficiency into account. 
        # All upstream contributions of sw32 is reconsidered with trapping efficiency 
        if 31 in list(c):
            # print ('true, i=',i)
            outlet_nitrate[:,:,i] = outlet_nitrate[:,:,30]
            outlet_tp[:,:,i] = outlet_tp[:,:,30]
        for j in d:
            outlet_nitrate[:,:,i] += loading_BMP_sum_nitrate[:,:,j-1]
            outlet_tp[:,:,i] += loading_BMP_sum_tp[:,:,j-1]
    # update the loadings for upperstream that has higher values
    e = b[b>33] 
    for i in e:
        f = df.loc[i-1].unique().astype('int')
        f = f[f!=0]
        for j in f:
            outlet_nitrate[:,:,i-1] += loading_BMP_sum_nitrate[:,:,j-1]
            outlet_tp[:,:,i-1] += loading_BMP_sum_tp[:,:,j-1]
    outlet_tp = outlet_tp/1.07 # 11% overestimates across all BMPs
    return outlet_nitrate, outlet_tp

# landuse_matrix = np.zeros((45,61))
# landuse_matrix[:,0] = 1
# outlet_nitrate, outlet_tp = loading_outlet_USRW_opt(landuse_matrix,'AS')
# outlet_nitrate[:,:,33].sum(axis=1).mean()
# outlet_tp[:,:,33].sum(axis=1).mean()

def loading_outlet_USRW_opt_v2(landuse_matrix, tech_wwt):
    '''
    simplified version of loading_outlet_USRW_opt; precalculate point source results of WWT and store in mat
    '''
    df = df_linkage2
    df[np.isnan(df)] = 0
    loading_BMP_sum_nitrate = loading_landscape('nitrate', landuse_matrix)[0]
    loading_BMP_sum_tp = loading_landscape('phosphorus', landuse_matrix)[0]
    
    outlet_nitrate = np.zeros((loading_BMP_sum_nitrate.shape[0], loading_BMP_sum_nitrate.shape[1], loading_BMP_sum_nitrate.shape[2]))
    outlet_tp = np.zeros((loading_BMP_sum_tp.shape[0], loading_BMP_sum_tp.shape[1], loading_BMP_sum_tp.shape[2]))
    
    for i in range(33):
        a = df.loc[i].unique().astype('int')
        a = a[a!=0]
        for j in a:
            # print (j)
            outlet_nitrate[:,:,i] += loading_BMP_sum_nitrate[:,:,j-1]    
            outlet_tp[:,:,i] += loading_BMP_sum_tp[:,:,j-1]
            
    # Total loading in sw32 = res_out + background loading
    '''******************Start of reservior trapping effect*******************'''
    res_in_nitrate = outlet_nitrate[:,:,32]
    res_out_nitrate = res_in_nitrate * 0.8694 - 46680.0 # equationd derived from data
    res_in_tp = outlet_tp[:,:,32]
    res_out_tp = res_in_tp * 0.8811 - 2128.1  # equationd derived from data
        
    res_out_nitrate = np.where(res_out_nitrate<0, 0, res_out_nitrate)
    res_out_tp = np.where(res_out_tp<0, 0, res_out_tp)      
    # sw32 is the downstream of reservoir
    outlet_nitrate[:,:,31] = loading_BMP_sum_nitrate[:,:,31] + res_out_nitrate
    outlet_tp[:,:,31] = loading_BMP_sum_tp[:,:,31] + res_out_tp
    '''******************End of reservior trapping effect*******************'''
    
    # update loading in SDD (sw31)
    outlet_nitrate[:,:,30] = loading_BMP_sum_nitrate[:,:,30] + outlet_nitrate[:,:,31]
    outlet_tp[:,:,30] = loading_BMP_sum_tp[:,:,30] + outlet_tp[:,:,31]
    
    '''***********************Start of point source*************************'''
    # name = 'nitrate'
    if tech_wwt == 'AS':
        df_point = df_point_SDD
        df_point_nitrate = pd.DataFrame(df_point.iloc[:,0])
        df_point_tp = pd.DataFrame(df_point.iloc[:,1])  
        df_point_nitrate['month'] = df_point_nitrate.index.month
        df_point_nitrate['year'] = df_point_nitrate.index.year
        df_point_tp['month'] = df_point_tp.index.month
        df_point_tp['year'] = df_point_tp.index.year
        
        df2_point_nitrate = np.zeros((16,12))
        df2_point_tp = np.zeros((16,12))
        for i in range(16):
            for j in range(12):
                df2_point_nitrate[i,j] = df_point_nitrate.loc[(df_point_nitrate.year==2003+i) & (df_point_nitrate.month==1+j)].iloc[:,0].astype('float').sum()
                df2_point_tp[i,j] = df_point_tp.loc[(df_point_tp.year==2003+i) & (df_point_tp.month==1+j)].iloc[:,0].astype('float').sum()     
                
        outlet_nitrate[:,:,30] = loading_BMP_sum_nitrate[:,:,30] + outlet_nitrate[:,:,31] + df2_point_nitrate
        outlet_tp[:,:,30] = loading_BMP_sum_tp[:,:,30] + outlet_tp[:,:,31] + df2_point_tp

    if tech_wwt != 'AS': 
        if tech_wwt == 'ASCP':
            loading_day_nitrate = scipy.io.loadmat('./model_WWT/SDD_analysis/ASCP_nitrate_matrix.mat')['out']
            loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/ASCP_tp_matrix.mat')['out']
    
        elif tech_wwt == 'EBPR_basic':
            loading_day_nitrate = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_basic_nitrate_matrix.mat')['out']
            loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_basic_tp_matrix.mat')['out']        
            
        elif tech_wwt == 'EBPR_acetate':
            loading_day_nitrate = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_acetate_nitrate_matrix.mat')['out']
            loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_acetate_tp_matrix.mat')['out']
    
        elif tech_wwt == 'EBPR_StR':
            loading_day_nitrate = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_StR_nitrate_matrix.mat')['out']
            loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_StR_tp_matrix.mat')['out']

        loading_month_nitrate = np.zeros((16,12))    #16 yr, 12 month
        loading_month_tp = np.zeros((16,12))       
        
        for i in range(16):
            for j in range(12):
                    loading_month_nitrate[i,j] = loading_day_nitrate[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month
                    loading_month_tp[i,j] = loading_day_tp[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month        
        outlet_nitrate[:,:,30] = loading_BMP_sum_nitrate[:,:,30] + outlet_nitrate[:,:,31] + loading_month_nitrate
        outlet_tp[:,:,30] = loading_BMP_sum_tp[:,:,30] + outlet_tp[:,:,31] + loading_month_tp
    '''***********************End of point source***************************'''
    # b contains all upstream subwatersheds for sw31
    b = df.loc[30].unique().astype('int')
    b = b[b!=0]
    # get unique subwatersheds that do not contribute to reservoir
    for i in range(33,45):
        c = df.loc[i].unique().astype('int')
        c = c[c!=0]
        d = list(set(c) - set(b))
        # Key step: the following equation takes the trapping efficiency into account. 
        # All upstream contributions of sw32 is reconsidered with trapping efficiency 
        if 31 in list(c):
            # print ('true, i=',i)
            outlet_nitrate[:,:,i] = outlet_nitrate[:,:,30]
            outlet_tp[:,:,i] = outlet_tp[:,:,30]
        for j in d:
            outlet_nitrate[:,:,i] += loading_BMP_sum_nitrate[:,:,j-1]
            outlet_tp[:,:,i] += loading_BMP_sum_tp[:,:,j-1]
    # update the loadings for upperstream that has higher values
    e = b[b>33] 
    for i in e:
        f = df.loc[i-1].unique().astype('int')
        f = f[f!=0]
        for j in f:
            outlet_nitrate[:,:,i-1] += loading_BMP_sum_nitrate[:,:,j-1]
            outlet_tp[:,:,i-1] += loading_BMP_sum_tp[:,:,j-1]
    outlet_tp = outlet_tp/1.07  # 11% overestimates across all BMPs
    return outlet_nitrate, outlet_tp

# landuse_matrix = np.zeros((45,62)); landuse_matrix[:,0]=1
# outlet_nitrate, outlet_tp = loading_outlet_USRW_opt_v2(landuse_matrix, 'AS')
# outlet_nitrate[:,:,33].sum(axis=1).mean()
# outlet_tp[:,:,33].sum(axis=1).mean()

def sediment_instream(sw, landuse_matrix):
    streamflow = loading_outlet_USRW('streamflow', landuse_matrix, 'AS')
    streamflow = streamflow[:,:,sw]
    pd_coef_poly = df_pd_coef_poly
    sediment = pd_coef_poly.iloc[sw,0]*streamflow**2 + pd_coef_poly.iloc[sw,1]*streamflow + pd_coef_poly.iloc[sw,2]
    sediment = np.where(sediment<0, 0, sediment)
    return sediment

# start = time.time()
# test_sed = sediment_instream(32, landuse_matrix).sum(axis=1).mean()
# BMP0_sed_outlet = sediment_instream(33, landuse_matrix).sum(axis=1).mean()
# BMP0_sed_above = sediment_instream(26, landuse_matrix).sum(axis=1).mean()
# BMP0_sed_lake = sediment_instream(32, 'BMP00').sum(axis=1).mean()
# end = time.time()
# print('simulation time is {:.1f} seconds'.format(end-start))

def get_P_riverine(scenario_name, tech_wwt):
    '''return annual P_nonpoint, P_point, P_reservoir, P_instream_loss, P_total_outlet, kg/yr'''
    P_point = 582.4 # MT/yr
    struvite = 0
    if tech_wwt == 'ASCP':
        loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/ASCP_tp_matrix.mat')['out']
    elif tech_wwt == 'EBPR_basic':
        loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_basic_tp_matrix.mat')['out']        
    elif tech_wwt == 'EBPR_acetate':
        loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_acetate_tp_matrix.mat')['out']
    elif tech_wwt == 'EBPR_StR':
        loading_day_tp = scipy.io.loadmat('./model_WWT/SDD_analysis/EBPR_StR_tp_matrix.mat')['out']
        struvite = 450 # kg/year, TP in pellect collection (mixtured of struvite and others), calculated from Aryan.
    
    if tech_wwt != 'AS':
        P_point = np.zeros((16,12))       
        for i in range(16):
            for j in range(12):
                P_point[i,j] = loading_day_tp[i,j]*monthrange(2003+i,j+1)[1] # loading: kg/month 
        P_point = P_point.sum(axis=1).mean()/1000   # MT/yr
    
    P_nonpoint = loading_landscape('phosphorus', scenario_name)[1][:,:,33].sum(axis=1).mean()/1000   # MT/yr
    P_reservoir = loading_landscape('phosphorus', scenario_name)[1][:,:,31].sum(axis=1).mean()*(1-0.8812)/1000   # MT/yr # trapping coefficient estimated from SWAT data
    P_total_outlet = loading_outlet_USRW('phosphorus', scenario_name, tech_wwt)[:,:,33].sum(axis=1).mean()/1000  # MT/yr
    P_instream_store = P_nonpoint+P_point - P_total_outlet - P_reservoir   # MT/yr, 7% P are deposited 
    
    return P_nonpoint, P_point, P_reservoir, P_instream_store, P_total_outlet, struvite

def get_P_biosolid(tech_wwt):
    if tech_wwt =='AS':
        P_in = 94.4; P_crop = 52.9; P_riverine = 1.1; P_soil = 40.4
    elif tech_wwt =='ASCP':
        P_in = 661.5; P_crop = 53.3; P_riverine = 3.3; P_soil = 604.9
    elif tech_wwt =='EBPR_basic':
        P_in = 592.0; P_crop = 54.0; P_riverine = 3.0; P_soil = 535.0
    elif tech_wwt =='EBPR_acetate':
        P_in = 626.4; P_crop = 54.3; P_riverine = 3.2; P_soil = 568.9        
    elif tech_wwt =='EBPR_StR':
        P_in = 209.1; P_crop = 53.4; P_riverine = 1.9; P_soil = 153.8
    return P_in, P_crop, P_riverine, P_soil



'''***************************** Performance metrics **************************************'''
def pbias(obs, sim):
    '''
    obs and sim should be array
    The optimal value of PBIAS is 0.0, with low-magnitude values indicating accurate model simulation. 
    Positive values indicate overestimation bias, whereas negative values indicate model underestimation bias
    '''
    # obs = df_swat
    # sim = df_iteem_sw
    obs_flat = obs.flatten()
    sim_flat = sim.flatten()
    bias = 100*sum(sim_flat-obs_flat)/sum(obs_flat)
    return bias
    
def nse(obs, sim):
    '''
    obs and sim should be array
    An efficiency of 1 (NSE = 1) corresponds to a perfect match of modeled discharge to the observed data.
    An efficiency of 0 (NSE = 0) indicates that the model predictions are as accurate as the mean of the observed data, 
    whereas an efficiency less than zero (NSE < 0) occurs when the observed mean is a better predictor than the model
    '''
    # obs = df_swat
    # sim = df_iteem_sw
    obs_flat = obs.flatten()
    obs_ave = obs.mean()
    sim_flat = sim.flatten()
    nse0 = 1 - sum((obs_flat - sim_flat)**2)/sum((obs_flat-obs_ave)**2) 
    return nse0


'''***************************** Plot **************************************'''
def dynamic_plot(name, time_period, sw):
    # name = 'streamflow'
    # time_period = 'cumulative'
    test = loading_outlet_USRW(name, 'BMP00')
    test_1D = test[:,:,sw]
    test_1D_cum = np.cumsum(test_1D)
    test_annual = test_1D.sum(axis=1)
    
    test1 = loading_outlet_USRW(name, 'BMP01', 'ASCP')
    test1_1D = test1[:,:,sw]
    test1_1D_cum = np.cumsum(test1_1D)
    test1_annual = test1_1D.sum(axis=1)
    
    test2 = loading_outlet_USRW(name, 'BMP23', 'ASCP')
    test2_1D = test2[:,:,sw]
    test2_1D_cum = np.cumsum(test2_1D)
    test2_annual = test2_1D.sum(axis=1)
    
    test3 = loading_outlet_USRW(name, 'BMP50', 'EBPR')
    test3_1D = test3[:,:,sw]
    test3_1D_cum = np.cumsum(test3_1D)
    test3_annual = test3_1D.sum(axis=1)

    fig = plt.figure(figsize=(6.5,5))
    if time_period == 'monthly':
        plt.plot(test_1D.flatten(), color='blue', label='Baseline', linewidth=3)
        plt.plot(test1_1D.flatten(), color='purple', label='S1(BMP1)', linewidth=2.5)
        plt.plot(test2_1D.flatten(), color='green', label='S2(BMP23)', linewidth=2)
        plt.plot(test3_1D.flatten(), color='red', label='S3(BMP50)', linewidth=1.5)
        
    elif time_period == 'cumulative':
        plt.plot(np.cumsum(test_1D), color='blue', label='Baseline', linewidth=3)
        plt.plot(np.cumsum(test1_1D), color='purple', label='S1(BMP1)', linewidth=2.5)
        plt.plot(np.cumsum(test2_1D), color='green', label='S2(BMP23)', linewidth=2)
        plt.plot(np.cumsum(test3_1D), color='red', label='S3(BMP50)', linewidth=1.5)
        
    elif time_period == 'annual' and name == 'nitrate':
        plt.plot(test_annual, color='blue', marker='o', label='Baseline', linewidth=3)
        # 7240 Mg/yr for Nitrate-N as 1980-1996 baseline
        plt.plot(7240*1000*0.85, color='blue', marker='o', linestyle='dashdot', label='15% Reductional Goal by 2025', alpha=.5, linewidth=2)
        plt.plot(7240*1000*0.55, color='blue', marker='o', linestyle=':', label='45% Reductional Goal by 2045', alpha=.5, linewidth=2)
        plt.plot(test1_annual, color='purple', marker='o', label='S1(BMP1)', linewidth=2.5)
        plt.plot(test2_annual, color='green', marker='o', label='S2(BMP23)', linewidth=2)
        plt.plot(test3_annual, color='red', marker='o', label='S3(BMP50)', linewidth=1.5)
    
    elif time_period == 'annual' and name == 'phosphorus':
        plt.plot(test_annual, color='blue', marker='o', label='Baseline', linewidth=3)
        # 324 Mg/yr for TP as 1980-1996 baseline
        plt.plot(324*1000*0.75, color='blue', marker='o', linestyle='dashdot', label='25% Reductional Goal by 2025', alpha=.5, linewidth=2)
        plt.plot(324*1000*0.55, color='blue', marker='o', linestyle=':', label='45% Reductional Goal by 2045', alpha=.5, linewidth=2)
        plt.plot(test1_annual, color='purple', marker='o', label='S1(BMP1)', linewidth=2.5)
        plt.plot(test2_annual, color='green', marker='o', label='S2(BMP23)', linewidth=2)
        plt.plot(test3_annual, color='red', marker='o', label='S3(BMP50)', linewidth=1.5)
    
    if name == 'streamflow':
        plt.ylabel(name +' at outlet (m3)', fontsize=14)
    elif name =='sediment':
        plt.ylabel(name +' loading at outlet (ton)', fontsize=14)
    else:
        plt.ylabel(name +' loading at outlet (kg)', fontsize=14)
            
    plt.xlabel('Time (2003-2018)', fontsize=14)
    if time_period == 'monthly' or time_period == 'cumulative':
        labels = [2003] + [str(i)[-2:] for i in range(2004,2020)]
        plt.xticks(np.arange(0, 192+1, 12), labels)
    else:
        labels = [2003] + [str(i)[-2:] for i in range(2004,2019)]
        plt.xticks(np.arange(0,16+1), labels)
    # ax.set_xticklabels([i for i in range(2003,2019)])
    # plt.yticks(np.arange(0, 25, 2))
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    # plt.legend(loc='upper left', fontsize=12 )
    plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(0.03, 1.2), ncol=2)
    plt.tight_layout()
    # plt.savefig('./ITEEM_figures/July//'+name+'_'+ time_period + str(sw) +'_loading.tif', dpi=150)
    plt.show()
    return

# start = time.time()
# dynamic_plot('nitrate', 'monthly'))
# dynamic_plot('sediment', 'annual')