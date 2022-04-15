# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:30:05 2020

@author: Shaobin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_SWAT.BMP_cost_eff_analysis.BMPs_comparison import bmp_compare_data
import seaborn as sns
from pylab import *
from matplotlib import colors
import scipy.io
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import ConvexHull

def scores(name, sw, percent=False):
    x, y, yield_baseline = bmp_compare_data(name, sw, percent) # '_' stores  yield_baseline data, not important for this case
    if percent== False:
        x = x.reshape(-1,1)
    elif percent==True:
        x = x.reshape(-1,1)*100
    y = y*-1
    scores = np.array([x,y])[:,:,0].T
    # x = scores[:, 0]
    # y = scores[:, 1]
    return scores, yield_baseline
# scores = scores('phosphorus', 32, percent=True)

def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def pareto_front_plot(name, percent=False, *args):
    # scores1 = scores('phosphorus', 33)
    '''
    return: 
    xx = kg P/ha removed from baseline, if percent=False; xx= % reduction/ha from baseline, 
    if percent=True: yy = $/ha; z: $/kg P removed
    score = all raw data before averaging
    df_score2: data used for plot in pandas dataframe
    yield_baseline:  kg/ha for one subwatershed or all subwatersheds
    '''
    if not args:
        score = np.zeros((45,54,2))
        yield_baseline = np.zeros((45))
        for i in range(45):
            score[i,:,:], yield_baseline[i] = scores(name, i, percent)
        score2 = np.delete(score, [7,30], axis=0).mean(axis=0)
        df_score2 = pd.DataFrame(score2)
        df_score2['z'] = df_score2[1]*-1/df_score2[0]
        df_score2['BMP'] = ['BMP' + str(i) for i in range(1,55)]
        df_score2 = df_score2.fillna(0)
        xx = score2[:,0]
        yy = score2[:,1]*-1
        z = yy/(xx)
        z[np.isnan(z)] = 0        
        if percent==True:
            label = '_ave' + '_pct'
        else: label = '_ave'
        data_list = [xx, yy, z, score, df_score2, yield_baseline]
        
    else:
        score2, yield_baseline = scores(name, args[0], percent)
        df_score2 = pd.DataFrame(score2)
        df_score2['z'] = df_score2[1]*-1/df_score2[0]
        df_score2['BMP'] = ['BMP' + str(i) for i in range(1,55)]
        df_score2 = df_score2.fillna(0)
        xx = score2[:,0]
        yy = score2[:,1]*-1
        z = yy/xx
        z[np.isnan(z)] = 0
        if percent==True:
            label = '_single_sw_' + str(args[0]+1) + '_pct'
        else: label = '_single_sw_' + str(args[0]+1)                
        data_list = [xx, yy, z, score2, df_score2, yield_baseline]
    if name== 'phosphorus':
        z[6] = 0
        
    pareto = identify_pareto(score2)
    pareto = np.insert(pareto,[0], 0)
    pareto_front = score2[pareto]
    pareto_front[:,1] = pareto_front[:,1]*-1
    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values
    BMP_list = ['BMP' + str(i) for i in range(1,55)]
    BMP_list2 = ['' for i in range(1,55)]
    for i in range(len(pareto)):
        index = pareto[i]
        BMP_list2[index] = BMP_list[index]
    
    f, ax = plt.subplots(figsize=(10,5))
    vmin = z.min(); vmax = z.max()
    norm = colors.DivergingNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap = cm.get_cmap('seismic')
    points = ax.scatter(xx, yy, c=z, s=50, edgecolor='k', cmap=cmap, norm=norm)
    ax.plot(pareto_front[:,0], pareto_front[:,1], color='r')
    for i, txt in enumerate(BMP_list2):
        ax.annotate(txt, (xx[i], yy[i]))
    if percent == False:
        ax.set_xlabel(name + ' reduction (kg/ha)', fontsize= 12)
        f.colorbar(points, label = 'Cost ($/kg removal)', orientation="horizontal", aspect=40)
    elif percent ==True:
        ax.set_xlabel(name + ' reduction (%)', fontsize= 12)
        f.colorbar(points, label = 'Cost ($/% removal)', orientation="horizontal", aspect=40)
    ax.set_ylabel('Crop net revenue loss ($/ha)', fontsize=12)   
    # f.colorbar(points, label = 'Cost ($/kg removal)', orientation="horizontal", aspect=40)
    plt.tight_layout()
    plt.savefig('./model_SWAT/BMP_cost_eff_analysis/cost_eff_analysis_figures/plot_' + name + label+ '.tif', dpi=150)
    plt.show()
    return data_list

# data_list_phosphorus_pct = pareto_front_plot('phosphorus', True)
# data_list_nitrate_pct = pareto_front_plot('nitrate', True)
# data_list_streamflow_pct = pareto_front_plot('streamflow', True)
# data_list_sediment_pct = pareto_front_plot('sediment', False)
# data_list_phosphorus = pareto_front_plot('phosphorus')
# data_list_nitrate = pareto_front_plot('nitrate')
# data_list_phosphorus_pct_singlesw = pareto_front_plot('nitrate', True, 32)

# score_nitrate = data_list_nitrate[3]
# score_phosphorus = data_list_phosphorus[3]
# score_phosphorus_pct = data_list_phosphorus_pct[3]
# score_nitrate_pct = data_list_nitrate_pct[3]

# scipy.io.savemat('score_nitrate.mat', mdict={'out': score_nitrate}, oned_as='row')
# load data again
# matdata = scipy.io.loadmat('score_nitrate.mat')
# score_nitrate_load = matdata['out']

def bmp_for_allsw(bmp, percent=False):
    '''
    purpose: find clusters that can aggregate subwatersheds
    return: x: kg/ha nitrate removal or %/ nitrate removal
    '''
    if percent == False:
        score_phosphorus = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_phosphorus.mat')['out']
        score_nitrate = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_nitrate.mat')['out']
    elif percent == True:
        score_phosphorus = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_phosphorus_pct.mat')['out']
        score_nitrate = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_nitrate_pct.mat')['out']
    '''bmp numbering start from 1, not 0'''
    f, ax = plt.subplots(figsize=(10,5))
    x = score_nitrate[:,bmp-1,0]
    x = np.delete(x, [7,30], axis=0) 
    y = score_phosphorus[:,bmp-1,0]
    y = np.delete(y, [7,30], axis=0)
    plt.scatter(x=x, y=y)
    sw_list1 = ['sw' + str(i+1) for i in range(7)]
    sw_list2 = ['sw' + str(i+1) for i in range(8,30)]
    sw_list3 = ['sw' + str(i+1) for i in range(31,45)]
    sw_list = sw_list1 + sw_list2 + sw_list3
    for i, txt in enumerate(sw_list):
        ax.annotate(txt, (x[i], y[i]))
    
    if percent==False:
        ax.set_xlabel('Nitrate reduction (kg/ha)', fontsize=12, family = 'Arial')
        ax.set_ylabel('Phosphorus reduction (kg/ha)', fontsize=12, family = 'Arial')   
    if percent==True:
        ax.set_xlabel('Nitrate reduction from baseline (%)', fontsize=12, family = 'Arial')
        ax.set_ylabel('Phosphorus reduction from baseline (%)', fontsize=12, family = 'Arial')   
    plt.show()
    return x, y
    
# bmp_for_allsw(bmp=46, percent=True)

def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)
    

def cluster_sw(bmp_set, n_clusters, cluster_method, percent=False):
    '''average all bmps first; then cluster 45 subwatersheds into desired clusters
    bmp_set: a tuple containing bmp_set fron pareto front
    cluster_method = 'Kmeans' or 'AgglomerativeClustering'
    bmp_set = (36, 38, 45, 46, 47) # slice BMP 37, 39, 46, 47, 48; note that 36 = BMP37
    '''
    # bmp_set = (46)
    # bmp_set = (36, 38, 45, 46, 47)
    # cluster_method = 'AgglomerativeClustering'
    # n_clusters = 10
    # percent=False
    if percent == False:
        score_phosphorus = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_phosphorus.mat')['out']
        score_nitrate = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_nitrate.mat')['out']
    elif percent == True:
        score_phosphorus = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_phosphorus_pct.mat')['out']
        score_nitrate = scipy.io.loadmat('./model_SWAT/BMP_cost_eff_analysis/score_nitrate_pct.mat')['out']
    
    x = score_nitrate[:,bmp_set,:]  # slice BMP 37, 39, 46, 47, 48; note that 36 = BMP37
    x = np.delete(x, [7,30], axis=0)[:,:,0].mean(axis=1)
    y = score_phosphorus[:,bmp_set,:]  # slice BMP 37, 39, 46, 47, 48; note that 36 = BMP37
    y = np.delete(y, [7,30], axis=0)[:,:,0].mean(axis=1)

    if percent==False:
        x=x/20
        df = pd.DataFrame((x,y)).T
    else: df = pd.DataFrame((x,y)).T
    df.columns = ['Nitrate', 'Phosphorus']

    # colormap = sns.color_palette("hls", n_clusters)
    # tab10 = sns.color_palette("tab10")
    if cluster_method == 'Kmeans':
        cluster = KMeans(n_clusters=n_clusters).fit(df)
        centroids = cluster.cluster_centers_
        f, ax = plt.subplots(figsize=(6,4))
        # label = ['Cluster' + str(i+1) for i in cluster.labels_]
        scatter = ax.scatter(df['Nitrate'], df['Phosphorus'],c=cluster.labels_.astype(float)+1, cmap='tab10', s=25)
        ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=10, edgecolor='k', label = 'centroid')
        ax.legend(*scatter.legend_elements(), loc="lower left",
                        title=" Clusters\n(K-means)", bbox_to_anchor=(1, 0))
        
    elif cluster_method == 'AgglomerativeClustering':
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')  
        cluster.fit_predict(df)
        f, ax = plt.subplots(figsize=(6,4))
        # label = ['Cluster' + str(i+1) for i in cluster.labels_]
        scatter = ax.scatter(df.iloc[:,0], df.iloc[:,1], c=cluster.labels_.astype(float)+1, cmap='tab10', s=25)

        ax.legend(*scatter.legend_elements(), loc="lower left",
                        title="     Clusters\n(Agglomerative\n    Clustering)", bbox_to_anchor=(1, 0))
        
    # handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
    sw_list1 = ['sw' + str(i+1) for i in range(7)]
    sw_list2 = ['sw' + str(i+1) for i in range(8,30)]
    sw_list3 = ['sw' + str(i+1) for i in range(31,45)]
    sw_list = sw_list1 + sw_list2 + sw_list3
    for i, txt in enumerate(sw_list):
        ax.annotate(txt, (x[i], y[i]))
        
    if percent==False:
        ax.set_xlabel('Nitrate reduction (20*kg/ha)', fontsize=12, family = 'Arial')
        ax.set_ylabel('Phosphorus reduction (kg/ha)', fontsize=12, family = 'Arial')
        label = ''
    if percent==True:
        ax.set_xlabel('Nitrate reduction from baseline (%)', fontsize=12, family = 'Arial')
        ax.set_ylabel('Phosphorus reduction from baseline (%)', fontsize=12, family = 'Arial')
        label = 'pct'
        
    # raw polygon surrounding vertices 
    tab10 = sns.color_palette("tab10")
    for i in range(n_clusters):
        if df.iloc[cluster.labels_ == i, 0].size > 2:
            encircle(df.iloc[cluster.labels_ == i, 0], df.iloc[cluster.labels_ == i, 1], 
                      ec="k", fc=tab10[i], alpha=0.2, linewidth=0)
    # cluster_list = ['Cluster' + str(i+1) for i in range(n_clusters) ]
    plt.tight_layout()
    plt.savefig('./model_SWAT/BMP_cost_eff_analysis/cost_eff_analysis_figures/cluster_bmp' + 
                str(bmp_set) + label + cluster_method + '.tif', dpi=150)
    plt.show()

# bmp_set = (36, 38, 45, 46, 47)
# cluster_sw(bmp_set, n_clusters = 10, cluster_method='Kmeans', percent=False)
# cluster_sw(bmp_set, n_clusters = 10, cluster_method='AgglomerativeClustering', percent=False)
