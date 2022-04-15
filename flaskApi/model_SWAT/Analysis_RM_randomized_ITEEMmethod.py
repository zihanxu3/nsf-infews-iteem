# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose: validation test of response matrix method for SWAT
"""

# Import required packages for data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from SWAT_functions import basic_landuse, pbias, nse
from results_validation_originalRM import loading_outlet_originalRM

# df = pd.read_excel('./model_SWAT/results_validation/100Randomizations/100RandomizeHRUstorePercentage_5BMPsDec82020.xlsx')
xls = pd.ExcelFile(r'./model_SWAT/results_validation/100Randomizations/100RandomizeHRUstorePercentage_5BMPsDec82020.xlsx')
# df1 = pd.read_excel(xls, 'Sheet01')

def response_mat(name):
    '''
    return as a tuple
    unit: kg/ha for nitrate, phosphorus, soy, corn, corn silage; ton/ha for sediment; mm for water yield
    '''
    if name == 'nitrate':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')
    elif name == 'phosphorus':
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_phosphorus.csv')
    elif name == 'sediment':
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_sediment.csv')
    elif name == 'streamflow':
        df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_streamflow.csv')
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


def landuse_mat(*args):
    '''
    Return a decison matrix (# of subwatershed, # of BMPs) to decide land use fractions
    of each BMP application in each subwatershed
    '''
    linkage = pd.read_excel('./model_SWAT/Watershed_linkage.xlsx').fillna(0)
    # df = pd.read_csv('./model_SWAT/results_validation/response_matrix_csv/yield_nitrate.csv')
    df = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')
    row_sw = linkage.shape[0]
    '''minus 4 to subtract first two columns of subwatershed and area'''
    col_BMP = df.shape[1] - 4
    landuse_matrix = np.zeros((row_sw,col_BMP))
    
    return landuse_matrix  # return as an (45,56), all zeros


def get_area_prcnt(sheet_name):
    '''
    return area percentage of agricultural land for each BMP
    '''
    # df = pd.read_excel('./model_SWAT/results_validation/StorePercentageHRU_First5BMPs.xlsx', 
    #                     sheet_name=sheet_name)
    # df = pd.read_excel('./model_SWAT/results_validation/100Randomizations/100RandomizeHRUstorePercentage_5BMPsDec82020.xlsx', 
    #                     sheet_name)
    df = pd.read_excel(xls, sheet_name)
    df2 = df.iloc[:,6:10].reindex()
    BMPs = df2.iloc[0:45,2].unique()
    BMPs = np.sort(BMPs)
    BMP_list = [int(i) for i in list(BMPs)]
    df_BMP = pd.DataFrame()
    df_temp = pd.DataFrame()
    for i in range(45):
        df_temp = pd.DataFrame()
        for j in range(len(BMP_list)):
            df3 = df2[(df2.SUBBASIN==i+1) & (df2.BMPsAdopted==BMP_list[j])]
            df4 = pd.DataFrame([df3.iloc[:,-1].sum()])
            df_temp = df_temp.append(df4, ignore_index=True) 
            df_temp_T = df_temp.T
        df_BMP = df_BMP.append(df_temp_T, ignore_index=True)
    
    landuse, land_agri = basic_landuse()
    # total_land = np.mat(landuse.iloc[:,-1]).T
    
    df_BMP_Prcnt = df_BMP/land_agri
    df_BMP_Prcnt.columns = BMP_list
    # np_BMP_Prcnt = np.array(df_BMP_Prcnt)
    return df_BMP_Prcnt, BMP_list


# scenario_01, BMP_list = get_area_prcnt('Sheet01')
# scenario_02, BMP_list = get_area_prcnt('Sheet02')
# scenario_03, BMP_list = get_area_prcnt('Sheet03')

# scenario.sum(axis=1)
# scenario2 = get_area_prcnt('Sheet02')[0]

def get_yield(name, scenario_name):
    '''
    return a tuple containing two numpy array: 
        1) yield_per_BMP: (year, month, subwatershed, BMP)
        2) yield_sum: (year, month, subwatershed)
    unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm/ha for water yield
    '''    
    # name = 'nitrate'
    # scenario_name = 'Sheet01'
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    landuse_matrix = landuse_mat()  # (45,56)
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
        
    yield_per_BMP = np.zeros((year.size, month.size, subwatershed.size, BMP_num))
    for i in range(year.size):
        for j in range(month.size):
            yield_per_BMP[i,j,:,:] = np.multiply(response_matrix[i,j,:,:], landuse_matrix)
            
    yield_sum = np.sum(yield_per_BMP, axis=3)
    yield_sum[:,:,30] = response_matrix[:,:,30,0]
    return yield_per_BMP, yield_sum

# yield_sum_sheet01 = get_yield('nitrate','Sheet01')[1]
# sw8_sheet01 = yield_sum_sheet01[:,:,8].flatten()


def get_yield_1D(name, sheet_name):  
    ''' for yield data validations
    name represents pollutant category
    sheet_name represents scenarios 
    '''
    scenario = get_area_prcnt(sheet_name)[0]
    scenario = np.array(scenario)
    yield_s1 = get_yield(name, sheet_name)[1]
    # df_1D_s1 = pd.DataFrame(yield_s1.flatten())
    return yield_s1.flatten()

# yield_data_s1_tot_1D = get_yield_1D('nitrate','Sheet1')
# yield_data_1D_streamflow = get_yield_1D('streamflow','Sheet1')

def loading_per_sw(name, scenario_name):
    '''
    return a numpy array (year, month, subwatershed)
    calculate the landscape loading from the yield at each subwatershe
    unit: kg for nitrate, phosphorus; ton for sediment; mm for water 
    '''
    response = response_mat(name)
    response_matrix = response[0]
    subwatershed = response[1]
    year = response[2]
    month = response[3]
    BMP_num = response[4]
    landuse_matrix = landuse_mat()
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    scenario, BMP_list= get_area_prcnt(scenario_name)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i] 
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

# landscape_loading_nitrate = loading_per_sw('nitrate', 'Sheet01')

def loading_outlet_USRW(name, scenario_name):
    '''
    return a numpy (year, month, watershed)
    reservoir watershed: 33; downstream of res: 32; outlet: 34
    '''
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
    if name == 'nitrate':
        res_out = res_in * 0.8694 - 46680.0 # equationd derived from data
    elif name =='phosphorus':
        res_out = res_in * 0.8811 - 2128.1  # equationd derived from data
    elif name =='sediment':
        res_out = 14.133*res_in**0.6105     # equationd derived from data
    elif name =='streamflow':
        res_out = res_in * 1.0075 - 1.9574  # equationd derived from data
    res_out = np.where(res_out<0, 0, res_out)
        
    # sw32 is the downstream of reservoir
    outlet[:,:,31] = loading_BMP_sum[:,:,31] + res_out
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
        # df2_point_1D = df2_point.flatten()
        # Calculate loading in sw31 with point source
        # loading_BMP_sum[i,j,30] = ANN...
        if name =='nitrate':
            # point_Nitrate = 1315.43*30 # kg/month, average
            outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
        elif name == 'phosphorus':
            # point_TP = 1923.33*30# kg/month, average
            outlet[:,:,30] = loading_BMP_sum[:,:,30] + outlet[:,:,31] + df2_point
    '''***********************End of point source*************************'''

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
        outlet = outlet/1.11  # 11% overestimates across all BMPs
    return outlet

# test_N = loading_outlet_USRW('nitrate', 'Sheet01')
# test_N_1D = test_N.flatten()
# test_TP = loading_outlet_USRW('phosphorus', 'Sheet1')
# test_TP_1D = test_TP.flatten()

def sediment_instream(sw, scenario_name):
    streamflow = loading_outlet_USRW('streamflow', scenario_name)
    streamflow = streamflow[:,:,sw]
    # if sw ==33:  # y=2E-14x^2 + 3E-05x - 378.52
    #     sediment = 2*10**-14*streamflow**2 + 3*10**-5*streamflow - 378.52
    # if sw ==32:  #  y = 2E-13x2 + 0.0002x - 1775.3
    #     sediment = 2*10**-13*streamflow**2 + 2*10**-4*streamflow - 1775.3
    pd_coef_poly = pd.read_excel('./model_SWAT/results_validation/sediment_streamflow_regression_coefs.xlsx', sheet_name='poly', usecols='B:D')
    sediment = pd_coef_poly.iloc[sw,0]*streamflow**2 + pd_coef_poly.iloc[sw,1]*streamflow + pd_coef_poly.iloc[sw,2]
    sediment = np.where(sediment<0, 0, sediment)
    return sediment

# sediment_instream(32, 'Sheet01')

def swat_vs_iteem(name, sw, ag_scenario, plot=True):
    '''note: only works for one specificed sw: traditional RM and modified RM'''
    # ag_scenario = 'BMP00'
    # name = 'nitrate'
    # sw = 32
    # Step 1: get loading from original SWAT results...
    if name == 'nitrate':
        # df = pd.read_excel('./model_SWAT/Response_matrix_BMPs.xlsx',sheet_name=0)
        df = pd.read_csv('./model_SWAT/results_validation/100Randomizations/loading_nitrate.csv')
    elif name == 'phosphorus':
        df = pd.read_csv('./model_SWAT/results_validation/100Randomizations/loading_phosphorus.csv')
    elif name == 'sediment':
        df = pd.read_csv('./model_SWAT/results_validation/100Randomizations/loading_sediment.csv')
    elif name == 'streamflow':
        df = pd.read_csv('./model_SWAT/results_validation/100Randomizations/loading_streamflow.csv')
        df = df*30*60*60*24
    subwatershed = df.iloc[:,0].unique()
    year = df.iloc[:,1].unique()
    month = df.iloc[:,2].unique()
    # area_sw = df.iloc[:,3].unique()
    # response_matrix = df.set_index(['Year','Month'])
    df = df.drop(df.columns[[0,1,2,3]], axis=1)
    df_to_np = np.zeros((year.size, month.size, subwatershed.size, df.shape[1]))
    for i in range(year.size):
        for j in range(month.size):
            df2 = df.iloc[month.size*subwatershed.size*(i):month.size*subwatershed.size*(i+1),:]
#            df = df.reset_index(inplace=False, drop= True)
            df_to_np[i,j,:,:] = df2.iloc[45*(j):45*(j+1),:]
            
    n = int(ag_scenario[-2:])-1
    df_swat = df_to_np[:,:,sw,n]
    df_iteem = loading_outlet_USRW(name, ag_scenario)
    df_iteem_sw = df_iteem[:,:,sw]
    
    if name =='sediment':
        df_iteem_sw = sediment_instream(sw, ag_scenario)
    pbias0 = pbias(obs=df_swat, sim=df_iteem_sw).round(2)
    nse0 = nse(obs=df_swat, sim=df_iteem_sw).round(3)
    
    '''original RM results'''
    landuse_matrix = landuse_mat()  # (45,56)
    '''landuse_matrix is expressed as %, changed as land decision changes'''
    scenario, BMP_list= get_area_prcnt(ag_scenario)
    for i in BMP_list:
        landuse_matrix[:,i] = scenario.loc[:,i]
        df_iteem_originalRM = loading_outlet_originalRM(name, landuse_matrix) # use original RM to predict 
    df_iteem_sw_originalRM = df_iteem_originalRM[:,:,sw]
    
    pbias_originalRM = pbias(obs=df_swat, sim=df_iteem_sw_originalRM).round(2)
    nse_originalRM = nse(obs=df_swat, sim=df_iteem_sw_originalRM).round(3)
    '''End: original RM results'''
    
    if plot == True:
        fig, ax = plt.subplots()
        # plt.text(x=0.03, y=0.79, s= 'P-bias: ' + str(pbias0) + '%', transform=ax.transAxes, fontsize=10)
        # plt.text(x=0.03, y=0.73, s= 'NSE: ' + str(nse0), transform=ax.transAxes, fontsize=10)
        
        plt.text(x=0.8, y=0.9, s= 'P-bias: ' + str(pbias0) + '%', transform=ax.transAxes, fontsize=10)
        plt.text(x=0.8, y=0.84, s= 'NSE: ' + str(nse0), transform=ax.transAxes, fontsize=10)
        # plt.text(x=0.1, y=1.02, s= 'P-bias of traditional RM: ' + str(pbias_originalRM) + '%',transform=ax.transAxes)
        # plt.text(x=0.55, y=1.02, s= 'NSE of traditional RM: ' + str(nse_originalRM),transform=ax.transAxes)
        
        plt.plot(df_swat.flatten(), color='red', label='SWAT', linewidth=1.5)
        plt.plot(df_iteem_sw.flatten(), color='blue', linestyle='dashed', label='Response matrix', linewidth=2.5)
        # plt.plot(df_iteem_sw_originalRM.flatten(), color='blue', linestyle='dashed', label='Traditional RM', linewidth=1.5)
        
        if name == 'streamflow':
            plt.ylabel(name.capitalize() +' (m3/month)', fontsize=10)
        elif name =='sediment':
            plt.ylabel(name.capitalize() +' loads (ton/month)', fontsize=10)
        else:
            plt.ylabel(name.capitalize() +' loads (kg/month)', fontsize=10)
        
        plt.xlabel('Time (2003-2018)', fontsize=10)
        labels = [2003] + [str(i)[-2:] for i in range(2004,2020)]
        plt.xticks(np.arange(0, 192+1, 12), labels)
        
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # plt.text(0.3, 1.07, 'Subwatershed ' + str(sw+1), transform=ax.transAxes, fontsize=10)
        # plt.text(0.3, 1.02, 'Scenario ' + ag_scenario, transform=ax.transAxes, fontsize=10)        
        # plt.legend(loc='upper left', fontsize=12 )
        plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(0.01, 0.88),frameon=False)
        plt.tight_layout()
        plt.savefig('./ITEEM_figures/RM_Jan_2021/randomized/ITEEM_method'+name+'_' + ag_scenario +'_sw'+str(sw+1) +'_coefficientonly.tif', dpi=300)
        plt.show()
    
    return pbias0, nse0, pbias_originalRM, nse_originalRM

# swat_vs_iteem('phosphorus', sw=33, ag_scenario='Sheet01')
# swat_vs_iteem('sediment', sw=33, ag_scenario='Sheet01')
# swat_vs_iteem('nitrate', sw=33, ag_scenario='Sheet01')
# swat_vs_iteem('streamflow', sw=33, ag_scenario='Sheet01')
