# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
Plot results for WWT_ANN 
"""

# Import required packages for data processing
import seaborn as sns
import joypy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
# import time
# Import machine learning packages
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


# Import packages from ITEEM
from get_distribution_para import *
from SDD_analysis.influent_SDD import *
from SDD_analysis.wwt_model_SDD import WWT_SDD
from SDD_N_P_effluent_boxplot import *
from SWAT_functions import nse

df_inflow_SDD_1yr = pd.read_excel('./model_WWT/SDD_N_P_2012-2019.xlsx', sheet_name=3,
                                  parse_dates=['Date'], index_col='Date')
df_outflow_SDD_yrs = pd.read_excel('./model_WWT/SDD_effluent.xlsx', sheet_name=0, 
                                  parse_dates=['Date'], index_col='Date')

''''Box plot'''
def box_data(sample_size, period):
    '''
    return a pandas dataframe: rows = sample_size*# of tech; cols = 5 pollutant + period
    period = ['month', 'week']
    '''
    output_list = ['Nitrate', 'TSS', 'TP', 'COD', 'TN', 'Hauled sludge']
    tech_list = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    week_list = ['Week_' + str(i) for i in range(1,53)]
    # year_list = [start_yr + i for i in range(end_yr-start_yr+1)]
    df1 = pd.DataFrame()
    df = df_inflow_SDD_1yr
    df2 = df.iloc[:,0]
    df2 = pd.DataFrame(df2)
    df2['month'] = df.index.month
    df2['week'] = df.index.isocalendar().week 
    df2['year'] = df.index.year
    df2['tech'] = 'SDD'
    # df2 = df2.loc[(df2.iloc[:,3]>=start_yr) & (df2.iloc[:,3]<=end_yr)]
    # df2.iloc[start_yr:end_yr,1]
    df_month = df2.groupby([df.index.month]).agg('mean')
    df_week  = df2.groupby([df.index.isocalendar().week]).agg('mean')
    
    for k in range(len(tech_list)):
        # k=0
        instance = WWT_SDD(tech=tech_list[k], multiyear=False)
        df = pd.DataFrame()        
        if period == 'month' or period =='week':
            np_instance_scaled, np_instance, influent_tot = instance.run_model(sample_size)  # (month/week, sample_size, output_var)
        elif period == 'year': 
            np_instance_scaled, np_instance, influent_tot = instance.run_model_ave(sample_size) # (sample_size, output_var)
        
        if period =='week':
            for i in range(5):
                df_temp = pd.DataFrame()
                for j in range(52):
                    df2 = pd.DataFrame(np_instance[j,:,i], columns=[output_list[i]])               
                    df2['tech'] = tech_list[k]    
                    df2['week'] = week_list[j]
                    df2[output_list[i] + '_loading'] = df2.iloc[:,0]*influent_tot[j,:,0]/1000 # kg/d
                    df_temp = df_temp.append(df2, ignore_index=True) 
                df = pd.concat([df, df_temp], axis=1)
            df1 = df1.append(df, ignore_index=True)
        elif period =='month':
            for i in range(5):
                df_temp = pd.DataFrame()
                for j in range(12):
                    df2 = pd.DataFrame(np_instance[j,:,i], columns=[output_list[i]])               
                    df2['tech'] = tech_list[k]    
                    df2['month'] = month_list[j]
                    df2[output_list[i] + '_loading'] = df2.iloc[:,0]*influent_tot[j,:,0]/1000  # kg/d
                    df_temp = df_temp.append(df2, ignore_index=True) 
                df = pd.concat([df, df_temp], axis=1)
            df1 = df1.append(df, ignore_index=True)
        elif period == 'year':
            for i in range(5):
                df_temp = pd.DataFrame()
                # for j in range(12):
                df2 = pd.DataFrame(np_instance[:,i], columns=[output_list[i]])               
                df2['tech'] = tech_list[k]    
                # df2['month'] = month_list[j]
                df2[output_list[i] + '_loading'] = df2.iloc[:,0]*influent_tot[:,0]/1000  # kg/d
                df_temp = df_temp.append(df2, ignore_index=True) 
                df = pd.concat([df, df_temp], axis=1)
            df1 = df1.append(df, ignore_index=True)
    df1 = df1.loc[:,~df1.columns.duplicated()]
    df1.loc[(df1.tech == 'EBPR_acetate') & (df1.Nitrate<0.1),'Nitrate'] = 0.1
    df1.loc[(df1.tech == 'EBPR_acetate') & (df1.Nitrate_loading<0.1*30*3875/1000),'Nitrate_loading'] = 0.1*30*3875/1000 #kg/d
    df1.loc[(df1.tech == 'EBPR_StR') & (df1.Nitrate<0.1),'Nitrate'] = 0.1
    df1.loc[(df1.tech == 'EBPR_StR') & (df1.Nitrate_loading<0.1*30*3875/1000),'Nitrate_loading'] = 0.1*30*3875/1000 #kg/d
    return df1

# box_year = box_data(1000, 'year')
# box_month = box_data(1000, 'month')
# box_month[box_month.iloc[:,1]=='EBPR_StR'].min()
# box_month[box_month.iloc[:,1]=='EBPR_acetate']
# box_month['month'].append(data_sdd['month'])
# type(box_month['month'])

def box_plot(output_name, sample_size, period, unit):
    data_df = box_data(sample_size, period)
    # period = 'year'
    # output_name = 'TP_loading'
    data_sdd = box_data_SDD()
    if unit == 'loading':
        output_name = output_name + '_loading'
    '''This part only works for nitrate, TN, TP'''
    if period == 'year' and unit=='loading':
        
        y = data_sdd[output_name].append(data_df[output_name], ignore_index=True)*30  # kg/month
        by_category = data_sdd['tech'].append(data_df['tech'], ignore_index=True)
        plt.figure(figsize=(3.5,3.25))
    elif period =='month' or period == 'week':    
        '''---------------------------------------------------------------------'''
        x = data_sdd[period].append(data_df[period], ignore_index=True)
        y = data_sdd[output_name].append(data_df[output_name], ignore_index=True)
        by_category = data_sdd['tech'].append(data_df['tech'], ignore_index=True)
        '''---------------------------------------------------------------------'''
        plt.figure(figsize=(6.5,3.25))
    
    # tab20 = sns.color_palette("tab20")
    # tab20c = sns.color_palette("tab20c")
    # set1 = sns.color_palette("Set1")
    # color = [set1[4], tab20[0], tab20[1], tab20c[11], tab20c[9], 'green']
    color = ['grey', 'cornflowerblue', 'darkblue', 'lightcoral', 'red', 'maroon']
    
    if period == 'year':
        g = sns.boxplot(x=by_category, y=y, palette=color, 
                showfliers = True, fliersize=0.5, linewidth=0.5)
        # labels = [item.get_text() for item in g.get_xticklabels()]
        # labels = ['SDD','AS', 'ASCP', 'EBPR_\nbasic', 'EBPR_\nacetate', 'EBPR_\nStR']
        # g.set_xticklabels(labels)
        plt.xticks(fontsize=10, rotation=60)
    elif period =='month' or period == 'week':
        g = sns.boxplot(x=x, y=y, hue=by_category, palette=color, 
                        # showmeans=True, meanline=True,
                        showfliers = True, fliersize=0.5,linewidth=0.5)
        plt.xticks(fontsize=10, rotation=60)
    g.set_yscale('log')
    if unit == 'loading' and period =='year':
        plt.ylabel(output_name[:-8] + ' effluent\nloading (kg/month)', fontdict={'family':'Arial', 'size':10})
        
    elif unit == 'loading' and period =='month':
        plt.ylabel(output_name[:-8] + ' effluent\nloading (kg/d)', fontdict={'family':'Arial', 'size':10})
        
    else:
        plt.ylabel(output_name + ' effluent\nconcentration(mg/L)', fontdict={'family':'Arial', 'size':10})
    
    # plt.xticks(fontsize=10, rotation=60)
    plt.xlabel('')
    labels = ['Historical','AS','ASCP', 'EBPR', 'EBPR-A', 'EBPR-S']
    g.set_xticklabels(labels)
    plt.yticks(fontsize=10)
    plt.grid(False)
    if period == 'week' or period == 'month':
        plt.legend(loc='lower left', mode='expand', ncol=3, 
                         bbox_to_anchor=(0, 1.02, 1, 0.2), fontdict={'family':'Arial', 'size':10})
    plt.tight_layout()
    plt.savefig('./model_WWT/SDD_analysis/figures/SDD_Jan2021/Boxplot'+output_name+'_'+unit+'_'+period +'_exlowN_May2021.tif', dpi=300, bbox_inches = 'tight')
    plt.show()

box_plot('Nitrate', 1000, 'year', 'loading')
box_plot('TP', 1000, 'year', 'loading')
# box_plot('TP', 1000, 'month', 'concentration')
# box_plot('TN', 1000, 'month', 'concentration')
# box_plot('TN', 1000, 'month', 'loading')
# box_plot('Nitrate', 1000, 'month', 'loading')


def boxplot_costeff():
    '''for manuscript Fig. 7'''
    tech_as = WWT_SDD(tech='AS', multiyear=True, start_yr = 2006, end_yr=2015)
    tech_ascp = WWT_SDD(tech='ASCP', multiyear=True, start_yr = 2006, end_yr=2015)    
    tech_ebpr_basic = WWT_SDD(tech='EBPR_basic', multiyear=True, start_yr = 2006, end_yr=2015)
    tech_ebpr_acetate = WWT_SDD(tech='EBPR_acetate', multiyear=True, start_yr = 2006, end_yr=2015)
    tech_ebpr_str = WWT_SDD(tech='EBPR_StR', multiyear=True, start_yr = 2006, end_yr=2015)
    
    cost_energy_as = tech_as.get_cost_energy_ave(1000, 0.07, 40)
    cost_energy_ascp = tech_ascp.get_cost_energy_ave(1000, 0.07, 40)
    cost_energy_ebpr_basic = tech_ebpr_basic.get_cost_energy_ave(1000, 0.07, 40)
    cost_energy_ebpr_acetate = tech_ebpr_acetate.get_cost_energy_ave(1000, 0.07, 40)
    cost_energy_ebpr_str = tech_ebpr_str.get_cost_energy_ave(1000, 0.07, 40)

    cost_as = cost_energy_as[1]; nutrient_as = cost_energy_as[-2]    # $/yr, (sample_size,)
    nitrate_eff_as = nutrient_as[0]; tp_in_as = nutrient_as[-1]; tp_eff_as = nutrient_as[-2]          # kg/yr, (sample_size,)
    cost_ascp = cost_energy_ascp[1]; nutrient_ascp = cost_energy_ascp[-2]
    nitrate_eff_ascp = nutrient_ascp[0]; tp_in_ascp = nutrient_ascp[-1]; tp_eff_ascp = nutrient_ascp[-2]   
    cost_ebpr_basic = cost_energy_ebpr_basic[1]; nutrient_ebpr_basic = cost_energy_ebpr_basic[-2]
    nitrate_eff_ebpr_basic = nutrient_ebpr_basic[0]; tp_in_ebpr_basic = nutrient_ebpr_basic[-1]; tp_eff_ebpr_basic = nutrient_ebpr_basic[-2]   
    cost_ebpr_acetate = cost_energy_ebpr_acetate[1]; nutrient_ebpr_acetate = cost_energy_ebpr_acetate[-2]
    nitrate_eff_ebpr_acetate = nutrient_ebpr_acetate[0]; tp_in_ebpr_acetate = nutrient_ebpr_acetate[-1]; tp_eff_ebpr_acetate = nutrient_ebpr_acetate[-2]      
    cost_ebpr_str = cost_energy_ebpr_str[1] - cost_energy_ebpr_str[-1][1].mean(); nutrient_ebpr_str = cost_energy_ebpr_str[-2]
    nitrate_eff_ebpr_str = nutrient_ebpr_str[0]; tp_in_ebpr_str = nutrient_ebpr_str[-1]; tp_eff_ebpr_str = nutrient_ebpr_str[-2]      
    
    df1 = pd.DataFrame(); df2 = pd.DataFrame(); df3 = pd.DataFrame(); df4 = pd.DataFrame(); df5 = pd.DataFrame()
    '''TP'''
    # df1['cost'] = cost_as/(tp_in_as-tp_eff_as); df1['tech'] = 'AS'     #$/kg 
    # df2['cost'] = cost_ascp/(tp_in_ascp-tp_eff_ascp); df2['tech'] = 'ASCP'     #$/kg 
    # df3['cost'] = cost_ebpr_basic/(tp_in_ebpr_basic-tp_eff_ebpr_basic); df3['tech'] = 'EBPR_basic'     #$/kg 
    # df4['cost'] = cost_ebpr_acetate/(tp_in_ebpr_acetate-tp_eff_ebpr_acetate); df4['tech'] = 'EBPR_acetate'     #$/kg 
    # df5['cost'] = cost_ebpr_str/(tp_in_ebpr_str-tp_eff_ebpr_str); df5['tech'] = 'EBPR_StR'     #$/kg 
    # df = pd.concat([df1, df2, df3, df4, df5])
    '''additonal TP moreval'''
    df2['cost'] = (cost_as - cost_ascp)/((tp_in_as - tp_eff_as)-(tp_in_ascp-tp_eff_ascp)) ; df2['tech'] = 'ASCP'     #$/kg 
    df3['cost'] = (cost_as - cost_ebpr_basic)/((tp_in_as - tp_eff_as) - (tp_in_ebpr_basic-tp_eff_ebpr_basic)); df3['tech'] = 'EBPR_basic'     #$/kg 
    df4['cost'] = (cost_as - cost_ebpr_acetate)/((tp_in_as - tp_eff_as) - (tp_in_ebpr_acetate-tp_eff_ebpr_acetate)); df4['tech'] = 'EBPR_acetate'     #$/kg 
    df5['cost'] = (cost_as - cost_ebpr_str)/((tp_in_as - tp_eff_as) - (tp_in_ebpr_str-tp_eff_ebpr_str)); df5['tech'] = 'EBPR_StR'     #$/kg 
    df = pd.concat([df2, df3, df4, df5])

    '''additional nitrate removal'''
    # df2['cost'] = (cost_as - cost_ascp)/(nitrate_eff_as.mean() - nitrate_eff_ascp.mean()) ; df2['tech'] = 'ASCP'     #$/kg 
    # df3['cost'] = -(cost_as - cost_ebpr_basic)/(nitrate_eff_as - nitrate_eff_ebpr_basic); df3['tech'] = 'EBPR_basic'     #$/kg 
    # df4['cost'] = -(cost_as - cost_ebpr_acetate)/(nitrate_eff_as - nitrate_eff_ebpr_acetate); df4['tech'] = 'EBPR_acetate'     #$/kg 
    # df5['cost'] = -(cost_as - cost_ebpr_str)/(nitrate_eff_as - nitrate_eff_ebpr_str); df5['tech'] = 'EBPR_StR'     #$/kg 
    # df4 = df4.loc[(df4['cost'] < df4.quantile(0.99)[0])] 
    # df5 = df5.loc[(df5['cost'] < df5.quantile(0.99)[0])] 
    # df = pd.concat([df2, df3, df4, df5])
    # df = pd.concat([df3, df4, df5])
    
    
    '''plot'''
    plt.figure(figsize=(3.5,3.25))    
    # color = [ 'cornflowerblue', 'darkblue', 'lightcoral', 'red', 'maroon']
    color = [ 'darkblue', 'lightcoral', 'red', 'maroon']
    # color = ['lightcoral', 'red', 'maroon']
    g = sns.boxplot(x=df['tech'], y=df['cost'], palette=color, 
            showfliers = True, fliersize=0.5, linewidth=0.5)
    # g.set_yscale('log')
    # labels = [item.get_text() for item in g.get_xticklabels()]
    # labels = ['AS', 'ASCP', 'EBPR_\nbasic', 'EBPR_\nacetate', 'EBPR_\nStR']
    # labels = ['EBPR', 'EBPR-A', 'EBPR-S']
    
    labels = ['ASCP','EBPR', 'EBPR-A', 'EBPR-S']
    g.set_xticklabels(labels, fontdict={'family':'Arial', 'size':10})
    plt.xticks(fontsize=10, rotation=60) 
    plt.xlabel('')
    # plt.ylabel('Cost effectiveness for removing extral\nNitrate compared to AS ($/Δkg N)', fontsize=10)
    plt.ylabel('Normalized cost of enhanced\nP removal ($/Δkg P)', fontdict={'family':'Arial', 'size':11})
    plt.yticks(fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./model_WWT/SDD_analysis/figures/SDD_Jan2021/Boxplot_Pcosteff_May_2021v2.tif', 
                dpi=300, bbox_inches = 'tight')
    plt.show()

# boxplot_costeff()
# SDD_multiyear_data = SDD_multiyear(2006, 2015, 'loading')
# np_cod = np.ones(120)*600
# np_influent = np.column_stack((SDD_multiyear_data[0][1], 
#                                SDD_multiyear_data[1][1], 
#                                SDD_multiyear_data[2][1], 
#                                np_cod
#                                ))

'''Dynaymic plot'''
def dynamic_nutrient_plot(output_name, unit):
    if output_name == 'TP':
        i = 2
    elif output_name == 'TN':
        i = 4
    elif output_name == 'Nitrate':
        i = 0
    # output_name = 'nitrate'
    # unit = 'loading'
    tech_as = WWT_SDD(tech='AS', multiyear=True, start_yr = 2006, end_yr=2015)
    eff_as_scale, eff_as_raw, influent_tot = tech_as.run_model(1000)
    eff_as = eff_as_raw[:,:,i]
    
    # inflow = influent_tot[:,:,0]
    if unit == 'loading':
        eff_as = eff_as * influent_tot[:,:,0]/1000*30
    eff_as_mean = eff_as.mean(axis=1) # np.median(eff_as, axis=1) # eff_as_mean =
    
    # '''start of selected runs using monthly average'''
    # output_scaled, output_raw = tech_as.run_model_single(np_influent)
    # if unit == 'loading':
    #     eff_as_mean = output_raw[:,i] * np_influent[:,0]/1000*30
    # '''end of selected runs using monthly average'''
    
    tech_ascp = WWT_SDD(tech='ASCP', multiyear=True, start_yr = 2006, end_yr=2015)
    eff_ascp_scale, eff_ascp_raw, influent_tot = tech_ascp.run_model(1000)
    eff_ascp = eff_ascp_raw[:,:,i]
    if unit == 'loading':
        eff_ascp = eff_ascp * influent_tot[:,:,0]/1000*30
    eff_ascp_mean = eff_ascp.mean(axis=1) #np.median(eff_ascp, axis=1) #
    
    # '''start of selected runs using monthly average'''
    # output_scaled, output_raw = tech_ascp.run_model_single(np_influent)
    # if unit == 'loading':
    #     eff_ascp_mean = output_raw[:,i] * np_influent[:,0]/1000*30
    # '''end of selected runs using monthly average'''
    
    tech_ebpr_basic = WWT_SDD(tech='EBPR_basic', multiyear=True, start_yr = 2006, end_yr=2015)
    eff_ebpr_basic_scale, eff_ebpr_basic_raw, influent_tot = tech_ebpr_basic.run_model(1000)
    eff_ebpr_basic = eff_ebpr_basic_raw[:,:,i]
    if unit == 'loading':
        eff_ebpr_basic = eff_ebpr_basic * influent_tot[:,:,0]/1000*30
    eff_ebpr_basic_mean = eff_ebpr_basic.mean(axis=1) #np.median(eff_ebpr_basic, axis=1) #
    
    # '''start of selected runs using monthly average'''
    # output_scaled, output_raw = tech_ebpr_basic.run_model_single(np_influent)
    # if unit == 'loading':
    #     eff_ebpr_basic_mean = output_raw[:,i] * np_influent[:,0]/1000*30
    # '''end of selected runs using monthly average'''

    tech_ebpr = WWT_SDD(tech='EBPR_acetate', multiyear=True, start_yr = 2006, end_yr=2015)
    eff_ebpr_scale, eff_ebpr_raw, influent_tot = tech_ebpr.run_model(1000)
    eff_ebpr = eff_ebpr_raw[:,:,i]
    if unit == 'loading':
        eff_ebpr = eff_ebpr * influent_tot[:,:,0]/1000*30
    eff_ebpr_mean = eff_ebpr.mean(axis=1) #np.median(eff_ebpr, axis=1) #
    
    # '''start of selected runs using monthly average'''
    # output_scaled, output_raw = tech_ebpr.run_model_single(np_influent)
    # if unit == 'loading':
    #     eff_ebpr_mean = output_raw[:,i] * np_influent[:,0]/1000*30
    # '''end of selected runs using monthly average'''

    tech_ebpr_str = WWT_SDD(tech='EBPR_StR', multiyear=True, start_yr = 2006, end_yr=2015)
    eff_ebpr_str_scale, eff_ebpr_str_raw, influent_tot = tech_ebpr_str.run_model(1000)
    eff_ebpr_str = eff_ebpr_str_raw[:,:,i]
    if unit == 'loading':
        eff_ebpr_str = eff_ebpr_str * influent_tot[:,:,0]/1000*30
    eff_ebpr_str_mean = eff_ebpr_str.mean(axis=1) #np.median(eff_ebpr_str, axis=1)#
    
    # '''start of selected runs using monthly average'''
    # output_scaled, output_raw = tech_ebpr_str.run_model_single(np_influent)
    # if unit == 'loading':
    #     eff_ebpr_str_mean = output_raw[:,i] * np_influent[:,0]/1000*30
    # '''end of selected runs using monthly average'''
    
    fig, ax1 = plt.subplots(figsize=(6,3.5))
    ''' add SDD nitrate, TN and TP'''
    df_point = pd.read_csv('./model_SWAT/results_validation/SDD_interpolated_2000_2018_Inputs.csv', 
                      parse_dates=['Date'],index_col='Date')
    SDD_multiyear_data = SDD_multiyear(2006, 2015, unit)
    
    if output_name == 'Nitrate':
        df_point = pd.DataFrame(df_point.iloc[:,0])
        sdd_loading = SDD_multiyear_data[-3].reset_index(drop=True)
        # sdd_conc = SDD_multiyear_data[2][1].reset_index(drop=True)

    elif output_name == 'TP':
        df_point = pd.DataFrame(df_point.iloc[:,1])
        sdd_loading = SDD_multiyear_data[-1].reset_index(drop=True)
        # sdd_conc = SDD_multiyear_data[1][1].reset_index(drop=True)
        
    elif output_name == 'TN':
        sdd_loading = SDD_multiyear_data[-2].reset_index(drop=True)
    # df_point['month'] = df_point.index.month
    # df_point['year'] = df_point.index.year
    # df_month = df_point.groupby(pd.Grouper(freq='M')).sum().iloc[36:36+10*12,:]
    # df_month2 = df_month.reset_index()
    # df_month2 = df_month2.iloc[:,1]

    ax1.plot(sdd_loading, color='grey', label='Historical data', linewidth=1)
    # if unit == 'concentration' and output_name=='TP':
        # ax1.plot(sdd_conc, color='grey', linestyle='dashed', label='SDD_TP_influent', linewidth=1)
    # elif (unit == 'concentration' and output_name=='Nitrate') or (unit == 'concentration' and output_name=='TN'):
        # ax1.plot(sdd_conc, color='grey', linestyle='dashed', label='SDD_TKN_influent', linewidth=1)    
    
    # color = sns.color_palette("Set1")[0:4]
    color = ['cornflowerblue', 'darkblue', 'lightcoral', 'red', 'maroon']
    ax1.plot(eff_as_mean, color=color[0], label='AS', linewidth=1)
    ax1.plot(eff_ascp_mean, color=color[1], label='ASCP', linewidth=1)
    ax1.plot(eff_ebpr_basic_mean, color=color[2], label='EBPR', linewidth=1)
    ax1.plot(eff_ebpr_mean, color=color[3], label='EBPR-A', linewidth=1)
    ax1.plot(eff_ebpr_str_mean, color=color[4], label='EBPR-S', linewidth=1)
    
    # uncertainty range
    # t = np.arange(120)
    # for j in range(5):
    #     data = [eff_as, eff_ascp, eff_ebpr_basic, eff_ebpr, eff_ebpr_str]
    #     upper = np.percentile(data[j],95, axis=1)
    #     lower = np.percentile(data[j],5, axis=1)
    #     ax1.fill_between(t, upper, lower, facecolor=color[j], alpha=0.5)
        
    if unit == 'loading':
        plt.ylabel(output_name +' loading (kg/month)', fontdict={'family':'Arial', 'size':11})
    elif unit == 'concentration':
        plt.ylabel(output_name +' concentration (mg/L)', fontdict={'family':'Arial', 'size':11})
    plt.yscale('log')
    plt.xlabel('Time (2006-2015)', fontdict={'family':'Arial', 'size':11})
    labels = [str(i) for i in range(2006,2017)]
    plt.xticks(np.arange(0, 120+1, 12), labels)
    # ax1.set_xticklabels(np.arange(0, 120+1, 12),fontdict={'family':'Arial', 'size':10})
    plt.legend(loc='lower left', mode='expand', ncol=3, 
                     bbox_to_anchor=(0, 1.02, 1, 0.2), prop={'family':'Arial', 'size':10})
    
    '''add SDD inflow''' 
    # df = pd.DataFrame(influent_SDD_multiyear2(1000, 2006, 2015)[:,:,0].mean(axis=1))
    # df.columns = ['flow']
    # ax2 = ax1.twinx()
    # ax2.plot(df, color='orange', label='inflow (m3/d)', linewidth=1)
    # ax2.set_ylabel('Inflow (m3/d)', color='orange')
    # ax2.tick_params(axis='y', labelcolor='orange')
    plt.savefig('./model_WWT/SDD_analysis/figures/SDD_Jan2021/Dynamic' + output_name+'_'+unit + '_exlowN_SDD_monthlyaverage_noFlow_May2021.tif', 
                dpi=300, bbox_inches = 'tight')
    fig.tight_layout()
    plt.show()

    # nse_metric = nse(np.array(sdd_loading), np.array(eff_as_mean))
    p_corr = [pearsonr(eff_as_mean, eff_ascp_mean),
              pearsonr(eff_ebpr_basic_mean, eff_ebpr_mean),
              pearsonr(eff_ebpr_str_mean, eff_ebpr_mean),
              pearsonr(eff_as_mean,sdd_loading)]
    
    r = r2_score(eff_as_mean, sdd_loading)
    # pearsonr return 1) r; 2) p-value
    return p_corr, r, eff_as_mean, sdd_loading

# plt.hist(eff_ebpr_basic[80,:])
p_corr = dynamic_nutrient_plot('TP', 'loading')
# p_corr, r, eff_as_mean_nitrate, sdd_loading_nitrate = dynamic_nutrient_plot('Nitrate', 'loading')
# dynamic_nutrient_plot('Nitrate', 'concentration')
dynamic_nutrient_plot('Nitrate', 'loading')
# dynamic_nutrient_plot('TN', 'concentration')
# p_corr, r, eff_as_mean_tp, sdd_loading_tp = dynamic_nutrient_plot('TP', 'loading')
# dynamic_nutrient_plot('TP', 'concentration')


def dynamic_inflow_plot():
    import matplotlib.dates as mdates
    df = df_inflow_SDD_1yr
    df2 = df.iloc[:,0]
    df2 = pd.DataFrame(df2)
    df2['month'] = df.index.month
    df2['week'] = df.index.isocalendar().week
    df2['day'] = df.index.day
    weekly_mean = df2.iloc[:,0].resample('W').mean()
    monthly_mean = df2.iloc[:,0].resample('M').mean()
    df_month = df2.groupby([df.index.month]).agg('mean')
    df_week  = df2.groupby([df.index.isocalendar().week]).agg('mean')
    # # Months as axis ticks; Set the locator
    # locator = mdates.MonthLocator()  # every month
    # Specify the format - %b gives us Jan, Feb...
    fmt = mdates.DateFormatter('%b')
 
    plt.figure(figsize=(6.5,5))
    plt.plot(df2['Average'], marker='o', markersize=5, linestyle='-', label='Daily influent', color='grey')
    plt.plot(weekly_mean, marker='^', markersize=5, linestyle='-', label='Weekly average', color='red')
    plt.plot(monthly_mean, marker='D', markersize=5, linestyle='-', label='Monthly average', color ='goldenrod')
    # plt.xticks(labels=['Jan','Feb','Mar','Apr'])
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    # Specify formatter
    X.set_major_formatter(fmt)
    plt.xlabel('Time (2012-2019)')
    plt.ylabel('Influent (MGD)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./model_WWT/SDD_analysis/figures/Dec2020/influent.tif', dpi=80)
    plt.show()



''''Parallel coordinate'''
# process data into requried format for parallel coordinate plot
def parallel_data(sample_size):
    '''
    return a pandas dataframe: rows = sample_size*# of tech; cols = 5 pollutant + month
    '''
    output_list = ['Nitrate', 'TSS', 'TP', 'COD', 'TN', 'Hauled\nsludge', 'Annual\ncost']
    tech_list = ['AS', 'ASCP', 'EBPR_basic', 'EBPR_acetate', 'EBPR_StR']
    df1 = pd.DataFrame()
    # sample_size = 10
    for k in range(len(tech_list)):
        instance = WWT_SDD(tech=tech_list[k], multiyear=False)
        np_instance = instance.run_model_ave(sample_size)[1][:,0:6]
        instance_cost = instance.get_cost_energy(sample_size, 0.07, 40)[0]
        instance_cost = np.tile(instance_cost,(12,1))
        # np_instance = np.dstack((np_instance, instance_cost))
        df = pd.DataFrame()
        for i in range(len(output_list)-1):
            df_temp = pd.DataFrame()
            # for j in range(12):
            # only use month 1 as example.
            df2 = pd.DataFrame(np_instance[:,i], columns=[output_list[i]])
            df2['tech'] = tech_list[k]       
            df_temp = df_temp.append(df2, ignore_index=True) 
            df = pd.concat([df, df_temp], axis=1)
        df1 = df1.append(df, ignore_index=True)
    df1 = df1.loc[:,~df1.columns.duplicated()]
    return df1

# parallel_data_month = parallel_data(1000)
# parallel_data_month.columns

def parallel_plot(sample_size):
    '''
    return a parallel coordinate plot
    '''
    data_df = parallel_data(sample_size)
    data_notech = data_df.drop(columns=['tech'])
    # data_notech = data_notech[['Nitrate','TN','TP','COD','TSS']]
    scalar = MinMaxScaler()
    data_scaled_np = scalar.fit_transform(data_notech)
    data_scaled_df = pd.DataFrame(data_scaled_np, columns=data_notech.columns)
    data_scaled_df['tech'] = data_df['tech']
    data_scaled_df2 = data_scaled_df.groupby('tech').mean()
    data_scaled_df2.reset_index(inplace=True)
    data_scaled_df2 = data_scaled_df2.reindex([0,1,4,3,2])
    # data_scaled_df2.reset_index()
    # data_scaled_df[:,'tech'] = 
    
    # Make the plot
    fig, ax = plt.subplots(figsize=(6.5,5))
    color = ['cornflowerblue', 'darkblue', 'lightcoral', 'red', 'maroon']  
    parallel_coordinates(data_scaled_df, 'tech', color=color, linewidth=0.2)
    parallel_coordinates(data_scaled_df2, 'tech', color=color,linewidth=2.0)
    # legend_without_duplicate_labels(ax)
    # ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    # ax.legend(*zip(*unique), loc='center left', bbox_to_anchor=(1, 0.5),fontsize=11)
    plt.ylabel('Normalized value [0,1]',fontsize=11)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0,1)
    plt.grid(False)
    leg = ax.legend(*zip(*unique),loc='center left', bbox_to_anchor=(1, 0.5),fontsize=11)
    for line, text in zip(leg.get_lines(), leg.get_texts()):
        text.set_color(line.get_color())
    for legobj in leg.legendHandles:
        legobj.set_linewidth(2.0)
    plt.tight_layout()
    plt.savefig('./model_WWT/SDD_analysis/figures/Dec2020/Parallel_coordinate.tif', 
                dpi=300, bbox_inches = 'tight')
    plt.show()

# parallel_plot(sample_size=1000)

'''Joyplot'''
def joy_plot(name, sample_size):
    # Get data
    name = 'nitrate'
    sample_size = 100
    month_list = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December']
    # week_list = [i+1 for i in range(52)]
    tech_list = ['AS', 'ASCP']
    df_AS = pd.DataFrame()
    df_ASCP = pd.DataFrame()
    df_EBPR = pd.DataFrame()
    df_EBPR_FBR = pd.DataFrame()
    instance_AS =  WWT_SDD(tech=tech_list[0])
    np_instance_AS = instance_AS.run_model(sample_size, period='month')[1]
    instance_ASCP =  WWT_SDD(tech=tech_list[1])
    np_instance_ASCP = instance_ASCP.run_model(sample_size, period='month')[1]
    # process data into required format
    # Nitrate, TSS, TP, COD, TN
    if name == 'nitrate':
        p = 0
    elif name == 'TSS':
        p = 1
    elif name == 'TP':
        p = 2
    elif name == 'COD':
        p = 3
    else:
        p = 4
    for i in range(12):
        df2 = pd.DataFrame(np_instance_AS[i,:,p])
        df2.columns = [name +'_' + tech_list[0]]
        df2['Month'] = i+1
        df_AS = df_AS.append(df2, ignore_index=True)
    for i in range(12):
        df2 = pd.DataFrame(np_instance_ASCP[i,:,p])
        df2.columns = [name +'_' + tech_list[1]]
        df2['Month'] = i+1
        df_ASCP = df_ASCP.append(df2, ignore_index=True)
    df_combined = pd.concat([df_AS,df_ASCP], axis=1)
    df_combined = df_combined.loc[:,~df_combined.columns.duplicated()]
    # df_combined = df_combined[[df_AS.columns[0], df_ASCP.columns[0], 'Month']]
    # df_combined_log = np.log10(df_combined.iloc[:,0:2])
    # df_combined_log['Month'] = df_combined['Month']
    plt.figure(figsize=(6.5,11))
    fig, axes = joypy.joyplot(df_combined, column=[df_AS.columns[0], df_ASCP.columns[0]], by="Month", 
                              ylim='own', figsize=(6.5,11), legend=True, alpha=0.8,
                              labels=month_list, ylabelsize=14,color=['blue','green'])
    plt.title(name + ' effluent by month', fontsize=14)
    plt.xlabel(name + ' effluent (mg/L)',  fontsize=14)
    # axes[:-1][0].xaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    # ax = axes[:-1][-1]
    # axes[:-1][0].xaxis.set_ticks([np.log10(x) for p in range(-6,1) 
    #                     for x in np.linspace(10**p, 10**(p+1), 10)], minor=True)
    # plt.xlim(0,20)
    # for a in axes[:-1]:
    #     a.set_xscale('log')
    plt.tight_layout()
    # plt.savefig('./model_WWT/SDD_analysis'+ '/Joyplot_'+name+'.tif',dpi=300)
    plt.show()
    
# joy_plot(name='nitrate', sample_size=1000)

'''violin plot'''
def violin_plot(output_name, sample_size, period):
    data_df = box_data(sample_size, period)
    output_list = ['Nitrate', 'TSS', 'TP', 'COD', 'TN', 'Hauled sludge production']
    # Make the plot
    plt.figure(figsize=(10,6.5))
    g = sns.violinplot(x=data_df[period], y=data_df[output_name], hue=data_df['tech'],palette="Set1",showfliers = False)
    if output_name == 'TP':
        g.set_yscale('log')
    plt.ylabel(output_name + ' effluent (mg/L)',fontsize=14)
    plt.xticks(fontsize=11, rotation=90)
    plt.xlabel('Time interval' +' (' + period + ')', fontsize=14)
    plt.yticks(fontsize=12)
    leg = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
    plt.tight_layout()
    plt.show()
    
# violin_plot('TP', 1000, 'month')    