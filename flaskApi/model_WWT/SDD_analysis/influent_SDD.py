# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
Scripts used to MC sampling of influents
"""

# Import required packages for data processing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# set global variable
df_inflow_SDD_1yr = pd.read_excel('./model_WWT/SDD_N_P_2012-2019.xlsx', sheet_name=3,
                                  parse_dates=['Date'], index_col='Date')
df_outflow_SDD_yrs = pd.read_excel('./model_WWT/SDD_effluent.xlsx', sheet_name=0, 
                                  parse_dates=['Date'], index_col='Date')
df_influent_SDD_yrs = pd.read_excel('./model_WWT/SDD_N_P_1989_2020.xlsx', parse_dates=['Date'],
                        index_col='Date')

def influent_SDD(sample_size):
    '''    
    return influent characteristics in 1yr, inflow changes by month; others remain same across months
    sampel size = n
    as a numpy array: (month, MC samplings, influent_var)
    '''
    df = df_inflow_SDD_1yr
    df2 = df.iloc[:,0]
    df2 = pd.DataFrame(df2)
    df2['month'] = df.index.month
    df2['week'] = df.index.isocalendar().week
    df2['day'] = df.index.day
    # weekly_mean = df2.iloc[:,0].resample('W').mean()
    # monthly_mean = df2.iloc[:,0].resample('M').mean()
    df_month = df2.groupby([df.index.month]).agg('mean')
    df_week  = df2.groupby([df.index.isocalendar().week]).agg('mean')
    influent_var = 4
    month = 12
    week = 52
    period='month'
    if period == 'week':
        influent_sewage = np.zeros((week, sample_size, influent_var))
        np_influent = np.repeat(np.matrix(df_week.iloc[:,0]), sample_size, axis=0).T
        length = 52
    elif period == 'month':
        influent_sewage = np.zeros((month, sample_size, influent_var))
        np_influent = np.repeat(np.matrix(df_month.iloc[:,0]), sample_size, axis=0).T
        length = 12
    # sample_size = 1000
    # random.seed(30)
    '''influent sewage'''
    for i in range(12):
            influent_sewage[i,:,0] = np_influent[i,0]
            influent_sewage[i,:,0] = np.random.triangular(left=np_influent[i,0]*0.85, 
                                                          mode=np_influent[i,0], 
                                                          right=np_influent[i,0]*1.15, 
                                                          size=(sample_size))
            # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] > 75, 75, influent_sewage[:,:,k])
            # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] < 16, 16, influent_sewage[:,:,k])
    influent_sewage[:,:,0] = influent_sewage[:,:,0]*3785.4118  # 1 mgd = 3785.4118 m3/d  
    '''type1: normal distribution'''
    # parameter_TP_norm = get_normal_para('TP')
    # mu = parameter_TP_norm['mu']
    # sd = parameter_TP_norm['sigma']
    # influent_sewage[:,:,k] = np.random.normal(mu, sd, size=(length,sample_size))
    # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] > 38.9, 38.9, influent_sewage[:,:,k])
    # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] < 11.7, 11.7, influent_sewage[:,:,k])
    '''type2: triangle distribution'''
    left = 11.7 # mg/L from SDD data
    mode = 17   # mg/L from SDD data
    right = 38.9  # mg/L from SDD data
    influent_sewage[:,:,1] = np.random.triangular(left=left, mode=mode, 
                                                  right=right, size=(length,sample_size))
    '''type1: lognormal distribution'''
    # parameter_NH3_lognorm = get_lognormal_para('NH3')
    # mu = parameter_NH3_lognorm['mu']
    # sd = parameter_NH3_lognorm['sigma']
    # influent_sewage[:,:,k] = np.random.lognormal(mean=mu, sigma=sd, size=(length,sample_size))
    # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] > 49.4, 49.4,influent_sewage[:,:,k])
    # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] < 14.6, 14.6,influent_sewage[:,:,k])
    '''type2: triangle distribution'''
    left = 14.6 # mg/L from SDD data
    mode = 32   # mg/L from SDD data
    right = 49.4  # mg/L from SDD data
    influent_sewage[:,:,2] = np.random.triangular(left=left, mode=mode, 
                                                  right=right, size=(length,sample_size))
    left = 447.4 # mg/L from SDD data
    mode = 600  # mg/L from SDD data
    right = 752  # mg/L from SDD data
    influent_sewage[:,:,3] = np.random.triangular(left=left, mode=mode, 
                                                  right=right, size=(length,sample_size))            
    return influent_sewage

# a = influent_SDD(sample_size=1000)
#start = time.time()    
# influent_month = influent_SDD(1000, period='month')
#end = time.time()
#print('MC sampling time: ', end-start)
#influent_trial_eff = influent_trial[:,:,0]
# TP_ave, TP_min, TP_max = np.mean(influent_month[0,:,2]), min(influent_month[0,:,2]), max(influent_month[0,:,2])

def influent_SDD_ave(sample_size, seed=False):
    
    if seed == True:
        np.random.seed(0)
        
    df = df_inflow_SDD_1yr
    df2 = df.iloc[:,0]
    df2 = pd.DataFrame(df2)
    df2['month'] = df.index.month
    df2['week'] = df.index.isocalendar().week
    df2['day'] = df.index.day
    # weekly_mean = df2.iloc[:,0].resample('W').mean()
    # monthly_mean = df2.iloc[:,0].resample('M').mean()
    # df_month = df2.groupby([df.index.month], squeeze=True).agg('mean')
    # df_week  = df2.groupby([df.index.isocalendar().week], squeeze=True).agg('mean')
    # sample_size= 1000
    influent_var = 4
    # month = 12
    influent_sewage = np.zeros((sample_size, influent_var))
    # np_influent = np.repeat(np.matrix(df_month.iloc[:,0]), sample_size, axis=0).T
    
    left_inflow = 25 # MGD, minimum annual flow from SDD
    mode_inflow = 33   # MGD from SDD data
    right_inflow = 40.7  # MGD minimum annual flow from SDD
    influent_sewage[:,0] = np.random.triangular(left=left_inflow, mode=mode_inflow, 
                                                  right=right_inflow, size=sample_size)*3785.4118

    left_tp = 11.7 # mg/L from SDD data
    mode_tp = 17   # mg/L from SDD data
    right_tp = 38.9  # mg/L from SDD data
    influent_sewage[:,1] = np.random.triangular(left=left_tp, mode=mode_tp, 
                                                  right=right_tp, size=sample_size)

    left_tkn = 14.6 # mg/L from SDD data
    mode_tkn = 32   # mg/L from SDD data
    right_tkn = 49.4  # mg/L from SDD data
    influent_sewage[:,2] = np.random.triangular(left=left_tkn, mode=mode_tkn, 
                                                  right=right_tkn, size=sample_size)

    left_cod = 447.4 # mg/L from SDD data
    mode_cod = 600  # mg/L from SDD data
    right_cod = 752  # mg/L from SDD data
    influent_sewage[:,3] = np.random.triangular(left=left_cod, mode=mode_cod, 
                                                  right=right_cod, size=sample_size)
    return influent_sewage
    
# a = influent_SDD_ave(1000)
# a.max()

def influent_SDD_multiyear(sample_size, start_yr, end_yr):
    '''
    return influent characteristics (i.e., total flowrate, TP, TKN, COD)
    the distribution of nutrient influent characteristics is same across months
    as a numpy array: (month_series, MC samplings, influent_var)'''
    influent_var = 4
    # start_yr = 2000
    # end_yr = 2006
    # sample_size = 10
    yr_duration = end_yr - start_yr + 1
    df = df_outflow_SDD_yrs
    df2 = df.iloc[:,0]
    df2 = pd.DataFrame(df2)
    df2['month'] = df.index.month
    df3 = df.resample('M').mean()
    df3['year'] = df3.index.year
    # df.resample('M').min()
    df4 = df3.loc[(df3.iloc[:,1]>=start_yr) & (df3.iloc[:,1]<=end_yr)]
    
    # fill missing data with SDD monthly average:
    df = df_inflow_SDD_1yr
    df2 = df.iloc[:,0]
    df2 = pd.DataFrame(df2)
    df2['month'] = df.index.month
    df2['week'] = df.index.isocalendar().week
    df2['day'] = df.index.day
    # weekly_mean = df2.iloc[:,0].resample('W').mean()
    # monthly_mean = df2.iloc[:,0].resample('M').mean()
    df_month = df2.groupby([df.index.month]).agg('mean')
        
    if start_yr<=2005:
        a = 2005 - start_yr
        df4.iloc[a*12:a*12+12,0] = np.array(df_month.iloc[:,0])
    #start = time.time()    
    # weekly_mean = df2.iloc[:,0].resample('W').mean()
    # monthly_mean = df2.iloc[:,0].resample('M').mean()
    # df_month = df2.groupby([df.index.month], squeeze=True).agg('mean')
    # df_week  = df2.groupby([df.index.isocalendar().week], squeeze=True).agg('mean')
    length = yr_duration*12
    influent_sewage = np.zeros((length, sample_size, influent_var))
    np_influent = np.repeat(np.matrix(df4.iloc[:,0]), sample_size, axis=0).T

    for i in range(length):
        # influent_sewage[i,:,0] = np_influent[i,0]
        influent_sewage[i,:,0] = np.random.triangular(left=np_influent[i,0]*0.85, 
                                                      mode=np_influent[i,0], 
                                                      right=np_influent[i,0]*1.15, 
                                                      size=(sample_size))
        # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] > 75, 75, influent_sewage[:,:,k])
        # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] < 16, 16, influent_sewage[:,:,k])
    influent_sewage[:,:,0] = influent_sewage[:,:,0]*3785.4118  # 1 mgd = 3785.4118 m3/d  

    '''type2: triangle distribution'''
    left = 11.7 # mg/L from SDD data
    mode = 17   # mg/L from SDD data
    right = 38.9  # mg/L from SDD data
    influent_sewage[:,:,1] = np.random.triangular(left=left, mode=mode, 
                                                  right=right, size=(length,sample_size))
    # elif k == 2:  #TKN, using distribution of NH3 as proxy distribution, lognormal
    '''type1: lognormal distribution'''
    # parameter_NH3_lognorm = get_lognormal_para('NH3')
    # mu = parameter_NH3_lognorm['mu']
    # sd = parameter_NH3_lognorm['sigma']
    # influent_sewage[:,:,k] = np.random.lognormal(mean=mu, sigma=sd, size=(length,sample_size))
    # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] > 49.4, 49.4,influent_sewage[:,:,k])
    # influent_sewage[:,:,k] = np.where(influent_sewage[:,:,k] < 14.6, 14.6,influent_sewage[:,:,k])
    '''type2: triangle distribution'''
    left = 14.6 # mg/L from SDD data
    mode = 32   # mg/L from SDD data
    right = 49.4  # mg/L from SDD data
    influent_sewage[:,:,2] = np.random.triangular(left=left, mode=mode, 
                                                  right=right, size=(length,sample_size))
    # elif k == 3:  #COD, assume triangular distribution 
    left = 447.4 # mg/L from SDD data
    mode = 600  # mg/L from SDD data
    right = 752  # mg/L from SDD data
    influent_sewage[:,:,3] = np.random.triangular(left=left, mode=mode, 
                                                  right=right, size=(length,sample_size))          
    return influent_sewage

# b =  influent_SDD_multiyear(10, 2003, 2015)
# start_yr = 2006
# end_yr = 20

def influent_SDD_multiyear2(sample_size, start_yr, end_yr, nutrient_index=1.0, flow_index=1.0):
    '''return influent characteristcs (e.g., COD, TKN, TP, inflow) based min, max, ave on each month'''
    SDD_multiyear_data = SDD_multiyear(start_yr, end_yr, 'loading')
    yr_duration = end_yr - start_yr + 1
    length = yr_duration*12
    influent_var = 4
    influent_sewage = np.zeros((length, sample_size, influent_var))
    
    np_influent = SDD_multiyear_data[0]
    np_tp = SDD_multiyear_data[1]
    np_tkn = SDD_multiyear_data[2]
    
    r = pd.date_range(start=str(start_yr) + '-01-01', end=str(end_yr) + '-12-31', freq='M')
    df_tp = pd.DataFrame(np_tp).T
    df_tp.columns = ['min', 'mean', 'max']
    df_tp = df_tp.reindex(r)
    df_tp = df_tp[['min', 'mean', 'max']].fillna(df_tp[['min', 'mean', 'max']].mean())
    
    df_tkn = pd.DataFrame(np_tkn).T
    df_tkn.columns = ['min', 'mean', 'max']
    df_tkn = df_tkn.reindex(r)
    df_tkn = df_tkn[['min', 'mean', 'max']].fillna(df_tkn[['min', 'mean', 'max']].mean())
    
    for i in range(length):  
        # length = # of months
        '''inflow: triangle distribution'''
        if np_influent[0][i]==np_influent[2][i]:
            np_influent[0][i] = np_influent[1][i]*0.9
            np_influent[2][i] = np_influent[1][i]*1.1
        # influent_sewage[i,:,0] = np.random.triangular(left=np_influent[0][i], 
        #                                               mode=np_influent[1][i], 
        #                                               right=np_influent[2][i], size=(sample_size))
        # influent_sewage[i,:,0] = np.where(influent_sewage[i,:,0] > 55*3785, 55*3785, influent_sewage[i,:,0])*flow_index
    
        influent_sewage[i,:,0] = np.random.triangular(left=np_influent[0][i]*flow_index, 
                                                      mode=np_influent[1][i]*flow_index, 
                                                      right=np_influent[2][i]*flow_index, size=(sample_size))
        influent_sewage[i,:,0] = np.where(influent_sewage[i,:,0] > 55*3785, 55*3785, influent_sewage[i,:,0])
        
        '''TP: triangle distribution'''
        if df_tp.iloc[i][0]==df_tp.iloc[i][0]:
            df_tp.iloc[i][0] = df_tp.iloc[i][1]*0.9
            df_tp.iloc[i][2] = df_tp.iloc[i][1]*1.1
        influent_sewage[i,:,1] = np.random.triangular(left=df_tp.iloc[i][0], mode=df_tp.iloc[i][1], 
                                                      right=df_tp.iloc[i][2], size=(sample_size))*nutrient_index
        '''TKN: triangle distribution'''
        if df_tkn.iloc[i][0]==df_tkn.iloc[i][2]:
            df_tkn.iloc[i][0] = df_tkn.iloc[i][1]*0.9
            df_tkn.iloc[i][2] = df_tkn.iloc[i][1]*1.1
        # influent_sewage[i,:,2] = np.random.triangular(left=df_tkn.iloc[i][0], mode=df_tkn.iloc[i][1], 
        #                                               right=df_tkn.iloc[i][2], size=(sample_size))*nutrient_index
        
        influent_sewage[i,:,2] = np.random.triangular(left=df_tkn.iloc[i][0]*nutrient_index,
                                                      mode=df_tkn.iloc[i][1]*nutrient_index, 
                                                      right=df_tkn.iloc[i][2]*nutrient_index,
                                                      size=(sample_size))
    # '''TP: triangle distribution'''
    # left = 11.7 # mg/L from SDD data
    # mode = 17   # mg/L from SDD data
    # right = 38.9  # mg/L from SDD data
    # influent_sewage[:,:,1] = np.random.triangular(left=left, mode=mode, 
    #                                               right=right, size=(length,sample_size))
    # '''TKN: triangle distribution'''
    # left = 14.6 # mg/L from SDD data
    # mode = 32   # mg/L from SDD data
    # right = 49.4  # mg/L from SDD data
    # influent_sewage[:,:,2] = np.random.triangular(left=left, mode=mode, 
    #                                               right=right, size=(length,sample_size))
    '''COD: triangle distribution '''
    left = 447.4 # mg/L from SDD data
    mode = 600  # mg/L from SDD data
    right = 752  # mg/L from SDD data
    # influent_sewage[:,:,3] = np.random.triangular(left=left, mode=mode, 
    #                                               right=right, size=(length, sample_size))*nutrient_index
    
    influent_sewage[:,:,3] = np.random.triangular(left=left*nutrient_index, mode=mode*nutrient_index, 
                                              right=right*nutrient_index, size=(length, sample_size))
    
    return influent_sewage

# SDD_multiyear_data2 = influent_SDD_multiyear2(100, 2003, 2018)

def SDD_multiyear(start_yr, end_yr, unit):
    '''
    return monthly loading of influents and effluents: TN, Nitrate, TP, etc.
    '''    
    # df = pd.read_excel('./model_WWT/SDD_N_P_1989_2020.xlsx', parse_dates=['Date'],
    #                     index_col='Date')
    df = df_influent_SDD_yrs
    # start_yr = 2003
    # end_yr = 2005
    start_date = str(start_yr) + '-01-01'  # '2012-01-01' 
    end_date = str(end_yr) + '-12-31'    # '2019-12-31' 
    df[df.columns[5]] = df[df.columns[5]].replace('<0.12', '0.12')
    df[df.columns[6]] = df[df.columns[6]].replace('<', '0.12')
    # setting errors=’coerce’,transform the non-numeric values into NaN.
    df[df.columns[5]] = pd.to_numeric(df[df.columns[5]], errors='coerce') 
    mask = (df.index > start_date) & (df.index <= end_date)
    df = df.loc[mask]
    
    df_influent_inflow = df.iloc[:,10]  # m3/d
    df_influent_inflow_mean = df_influent_inflow.groupby(pd.Grouper(freq='M')).mean()
    df_influent_inflow_min = df_influent_inflow.groupby(pd.Grouper(freq='M')).min()
    df_influent_inflow_max = df_influent_inflow.groupby(pd.Grouper(freq='M')).max()

    df_influent_tp = df.dropna(subset=[df.columns[3]]).iloc[:,3]
    df_influent_tp_mean = df_influent_tp.groupby(pd.Grouper(freq='M')).mean()
    df_influent_tp_min = df_influent_tp.groupby(pd.Grouper(freq='M')).min()
    df_influent_tp_max = df_influent_tp.groupby(pd.Grouper(freq='M')).max()
    
    df_influent_N = df.dropna(subset=[df.columns[0], df.columns[2]])
    df_influent_TKN = df_influent_N.iloc[:,0] +df_influent_N.iloc[:,2]
    df_influent_TKN_mean = df_influent_TKN.groupby(pd.Grouper(freq='M')).mean()
    df_influent_TKN_min = df_influent_TKN.groupby(pd.Grouper(freq='M')).min()
    df_influent_TKN_max = df_influent_TKN.groupby(pd.Grouper(freq='M')).max()
    
    df_effluent_nitrate = df.dropna(subset=[df.columns[6]]).iloc[:,6]
    df_effluent_N = df.dropna(subset=[df.columns[5], df.columns[6], df.columns[7]])
    df_effluent_tn = df_effluent_N.iloc[:,5] + df_effluent_N.iloc[:,6] + df_effluent_N.iloc[:,7]
    df_effluent_tp = df.dropna(subset=[df.columns[8]]).iloc[:,8]
    
    if unit == 'concentration':
        df_effluent_nitrate = df_effluent_nitrate.astype(float).groupby(pd.Grouper(freq='M')).mean()
        df_effluent_tn = df_effluent_tn.astype(float).groupby(pd.Grouper(freq='M')).mean()
        df_effluent_tp = df_effluent_tp.astype(float).groupby(pd.Grouper(freq='M')).mean()
    
    if unit == 'loading':
        df_effluent_nitrate = df_effluent_nitrate*df.dropna(subset=[df.columns[6]]).iloc[:,10]/1000 # kg/d
        df_effluent_nitrate = df_effluent_nitrate.astype(float).groupby(pd.Grouper(freq='M')).mean()*30
        df_effluent_tn = df_effluent_tn*df.dropna(subset=[df.columns[5], df.columns[6], df.columns[7]]).iloc[:,10]/1000 # kg/d
        df_effluent_tn = df_effluent_tn.astype(float).groupby(pd.Grouper(freq='M')).mean()*30
        df_effluent_tp = df_effluent_tp *df.dropna(subset=[df.columns[8]]).iloc[:,10]/1000 # kg/d
        df_effluent_tp = df_effluent_tp.astype(float).groupby(pd.Grouper(freq='M')).mean()*30
    
    df_influent_inflow = [df_influent_inflow_min, df_influent_inflow_mean, df_influent_inflow_max]
    df_influent_tp = [df_influent_tp_min, df_influent_tp_mean, df_influent_tp_max]
    df_influent_tkn = [df_influent_TKN_min, df_influent_TKN_mean, df_influent_TKN_max]
    
    return df_influent_inflow, df_influent_tp, df_influent_tkn, df_effluent_nitrate, df_effluent_tn, df_effluent_tp

# SDD_multiyear_data = SDD_multiyear(2006, 2015, 'loading')

# np_cod = np.ones(120)*600
# np_influent = np.column_stack((SDD_multiyear_data[0][1], 
#                                SDD_multiyear_data[1][1], 
#                                SDD_multiyear_data[2][1], 
#                                np_cod
#                                ))

def plot_sdd_multiyear_influent(output_name):
    # SDD_multiyear_data = SDD_multiyear(2006, 2015, 'loading')
    SDD_multiyear_data2 = influent_SDD_multiyear2(1000, 2006, 2015)
    # uncertainty range
    t = np.arange(120)
    if output_name == 'TP':
        i = 1
    elif output_name == 'TKN':
        i = 2
    elif output_name == 'COD':
        i = 3
    elif output_name == 'Inflow':
        i = 0  
    data = SDD_multiyear_data2[:,:,i]
    color = ['cornflowerblue', 'darkblue', 'lightcoral', 'red', 'maroon']
    
    fig, ax = plt.subplots(figsize=(5,3.5))
    upper = np.percentile(data,95, axis=1)
    lower = np.percentile(data,5, axis=1)
    
    plt.xlabel('Time (2006-2015)', fontsize=14)
    labels = [str(i) for i in range(2006,2017)]
    plt.xticks(np.arange(0, 120+1, 12), labels)
    
    ax.plot(data.mean(axis=1), color=color[i], linewidth=1)
    ax.fill_between(t, upper, lower, facecolor=color[i], alpha=0.5)

    if output_name == 'TP':
        i = 1
        plt.ylabel('Influent TP (mg/L)', fontsize=14)
    elif output_name == 'TKN':
        i = 2
        plt.ylabel('Influent TKN (mg/L)', fontsize=14)
    elif output_name == 'COD':
        i = 3
        plt.ylabel('Influent COD (mg/L)', fontsize=14)
    elif output_name == 'Inflow':
        i = 0      
        plt.ylabel('Inflow (m3/d)', fontsize=14)
         
    plt.xlabel('Time (2006-2015)', fontsize=14)
    fig.tight_layout()
    plt.savefig('./model_WWT/SDD_analysis/figures/SDD_Jan2021/influent'+ output_name+'.tif', dpi=80)
    plt.show()
    return

# plot_sdd_multiyear_influent('TKN')
# plot_sdd_multiyear_influent('COD')
# plot_sdd_multiyear_influent('Inflow')
# plot_sdd_multiyear_influent('TP')    
    