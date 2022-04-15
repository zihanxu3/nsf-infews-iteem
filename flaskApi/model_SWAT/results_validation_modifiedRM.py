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
import matplotlib.pyplot as plt
import time
from calendar import monthrange


#-----Function for response matrix-----
def response_mat(name):
    '''
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')
    elif name == 'phosphorus':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=1)
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_phosphorus.csv')
    elif name == 'sediment':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=2)
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_sediment.csv')
    elif name == 'streamflow':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=3)
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_streamflow.csv')
    elif name == 'soybean':
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_soybean.csv')
    elif name == 'corn':
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_corn.csv')
    elif name == 'corn sillage':
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_corn_silage.csv')
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


# response_mat_all = response_mat('streamflow')
#response_nitrate = response_mat_all[0]
#reseponse_nitrate_yr1_month1 = response_nitrate[0,0,:,:]


#-----Functions for land use fraction of each BMP at each subwatershed-----
def basic_landuse():
    '''basic case of land use'''
    landuse = pd.read_excel('./model_SWAT/landuse.xlsx').fillna(0)
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
    linkage = pd.read_excel('./model_SWAT/Watershed_linkage.xlsx').fillna(0)
    df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    # scenario_name = 'BMP01' 
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
def get_yield(name, scenario_name):
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
    
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    landuse_matrix = landuse_mat(scenario_name)
    
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)

    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# yield_sw = get_yield('streamflow', 'BMP50')[1]
# yield_sw_flat = yield_sw.flatten()
# yield_sw_yr1 = yield_sw[0,:,:,:][0]
# yield_sw_yr2 = yield_sw[1,:,:,:][0]

#-----Function for calculating crop yield for each subwatershed-----
def get_yield_crop(name, scenario_name):
    '''
    return a tuple: (crop yield per unit (kg/ha) [subwatershed, year], 
    total crop yield per subwatershed (kg) [subwatershed, year] ) 
    calculate crop yield for each subwatershed
    '''
    crop = loading_per_sw(name, scenario_name)
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
def loading_per_sw(name, scenario_name):
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
    landuse_matrix = landuse_mat(scenario_name)
    
    '''landuse for agri, expressed in ha'''
    land_agri = np.mat(basic_landuse()[1])
    landuse  = basic_landuse()[0]
    total_land = np.mat(landuse.iloc[:,-1]).T
    '''total landuse for agri, expressed in ha'''
    total_land_agri = np.multiply(landuse_matrix, land_agri)
    loading = np.zeros((year.size, month.size, subwatershed.size))
    '''get yield data'''
    yield_data = get_yield(name, scenario_name)[1]
    # yield_data = get_yield('nitrate', 'Sheet1')
    # test = np.multiply(np_yield_s1[0,0,:],total_land.T)
    '''get loading'''
    for i in range(year.size):
        for j in range(month.size):
            loading[i,j,:] = np.multiply(yield_data[i,j,:], total_land.T)
    # '''add nutrient contribution from urban'''
    # loading[:,:,30] = response_matrix[:,:,30,0]*total_land[30,0]
    return loading

# loading_per_sw_test = loading_per_sw('streamflow')
# loading_per_sw_yr1_month1 = loading_per_sw_test[0,0,:,:]
# loading_per_sw_yr1_month2 = loading_per_sw_test[0,1,:,:]
# loading_per_sw_yr2 = loading_per_sw_test[1,:,:,:]

#-----Function for calculating outlet loading of N, P, sediment, streamflow for each subwatershed-----
def loading_outlet_modifiedRM(name, scenario_name):
    '''
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
    # name = 'streamflow'
    # scenario_name = 'BMP00'
    df = pd.read_excel('./model_SWAT/Watershed_linkage_v2.xlsx')
    df[np.isnan(df)] = 0
    loading_BMP_sum = loading_per_sw(name, scenario_name)
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
    res_in_cum = np.zeros((16,12))
    for i in range(16):
        res_in_cum[i,:] = np.cumsum(res_in)[i*12: (i+1)*12]
    if name == 'nitrate':
        res_out_cum = res_in_cum * 0.725  # assuming 27.5% trapping efficiency for nitrate
    elif name =='phosphorus':
        res_out_cum = res_in_cum * 0.50   # assuming 50-51% trapping efficiency for phosphorus
    elif name =='sediment':
        res_out_cum = res_in_cum * 0.129  # assuming 87.1% trapping efficiency for sediment
    elif name =='streamflow':
        res_out_cum = res_in_cum * 0.926  # assuming 7.4% trapping efficiency for streamflow
        
    res_out_cum_flatten = res_out_cum.flatten()
    res_out = np.zeros((192,1))
    res_out[0] = res_out_cum_flatten[0]
    res_out[1:192,0] = np.diff(res_out_cum_flatten)
    res_out2 = np.zeros((16,12))
    for i in range(16):
        res_out2[i,:] = res_out[i*12: (i+1)*12].T
    # sw32 is the downstream of reservoir
    outlet[:,:,31] = loading_BMP_sum[:,:,31] + res_out2
    # outlet[:,:,30] = loading_BMP_sum[:,:,30]
    
    '''******************End of reservior trapping effect*******************'''
    # update loading in SDD (sw31)
    outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31]
    
    '''***********************Start of point source*************************'''
    if name == 'nitrate' or name == 'phosphorus':
        df_point = pd.read_csv('./model_SWAT/results_validation/SDD_interpolated_2000_2018_Inputs.csv', 
                          parse_dates=['Date'],index_col='Date')
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
            
    '''***********************End of point source***************************'''
    # b contains all upstream subwatersheds for sw31
    b = df.loc[30].unique().astype('int')
    b = b[b!=0]
    # calculate loadings of subwatersheds that do not contribute to reservoir
    for i in range(33,45):
        c = df.loc[i].unique().astype('int')
        c = c[c!=0]
        # d represents other upperstreams of sw(i+1) in addition to sw31 and upstreams of sw31
        d = list(set(c) - set(b))
        # g = set(c) | set(b)
        # h = list(g - set(b))
        # Key step: the following equation takes the trapping efficiency into account. 
        # All upstream contributions of sw32 is reconsidered with trapping efficiency 
        if 31 in list(c):
            # print ('true, i=',i)
            # all upstream of SDD (sw31) are now using updated loading of SDD (sw31) 
            outlet[:,:,i] = outlet[:,:,30]
        for j in d:
            outlet[:,:,i] += loading_BMP_sum[:,:,j-1]
            
    # update the loadings for sw31's upperstreams that has higher values than 33
    e = b[b>33]
    for i in e:
        f = df.loc[i-1].unique().astype('int')
        f = f[f!=0]
        for j in f:
            outlet[:,:,i-1] += loading_BMP_sum[:,:,j-1]         
    if name == 'streamflow':
        outlet = outlet*10   # convert mm*ha to m3 by 10
        
    return outlet

def loading_Decatur(name):
    '''return a numpy array (192,56)'''
    BMP_list = ['BMP0'+str(i) for i in range(10)]
    for i in range(10,56):
        BMP_list.append('BMP'+str(i))
    allArrays = np.zeros((192,0))
    for i in range(56):
        a = loading_outlet_modifiedRM(name, BMP_list[i])[:,:,32].flatten()
        allArrays = np.column_stack((allArrays, a))
    return allArrays
# loading_Decatur_streamflow = loading_Decatur('streamflow')
# loading_Decatur_sediment = loading_Decatur('sediment')

def loading_outlet(name):
    '''return a numpy array (192,56)'''
    BMP_list = ['BMP0'+str(i) for i in range(10)]
    for i in range(10,56):
        BMP_list.append('BMP'+str(i))
    allArrays = np.zeros((192,0))
    for i in range(56):
        a = loading_outlet_modifiedRM(name, BMP_list[i])[:,:,33].flatten()
        allArrays = np.column_stack((allArrays, a))
    return allArrays

# loading_outlet_sediment = loading_outlet('sediment')
# loading_outlet_sediment.sum(axis=0)

'''***************************** Plot **************************************'''
def outlet_scatter_plot(name):
    # name = 'sediment'
    # time_period = 'cumulative'
    BMP_list = ['BMP0'+str(i) for i in range(10)]
    for i in range(10,56):
        BMP_list.append('BMP'+str(i))
        
    df = []
    for i in range(56):
        a = np.sum(loading_outlet_modifiedRM(name, BMP_list[i])[:,:,33])
        df.append(a)

    fig = plt.figure(figsize=(10,6.5))
    plt.scatter(x=BMP_list, y = df)
    plt.xticks(rotation=90)
    if name == 'streamflow':
        plt.ylabel(name +' at outlet (m3)', fontsize=14)
    elif name =='sediment':
        plt.ylabel(name +' loading at outlet (ton)', fontsize=14)
    else:
        plt.ylabel(name +' loading at outlet (kg)', fontsize=14)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    plt.tight_layout()
    plt.savefig('./model_SWAT/figures_validation/'+name+'_'+ 'modified.tif', dpi=80)
    plt.show()
    return df

# df_nitrate_modifiedRM = outlet_scatter_plot('nitrate')
# df_phosphorus_modifiedRM = outlet_scatter_plot('phosphorus')
# df_sediment_modifiedRM = outlet_scatter_plot('sediment')
# df_streamflow_modifiedRM = outlet_scatter_plot('streamflow')
