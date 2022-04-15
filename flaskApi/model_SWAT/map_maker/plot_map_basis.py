# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
To visualize the map of Upper Sangamon River Basin
"""

import numpy as np
import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import geometry, ops
from mpl_toolkits.axes_grid1 import make_axes_locatable

# load new modules developed for ITEEM
#from model_SWAT.data_prep import N_outlet, N_yield

#watershed_USRB = gpd.read_file('Shapefiles/Watershed.shp')
#watershed_USRB.head()
#watershed_USRB.columns

'''Using Merge on a column and Index in Pandas'''
#watershed_with_N = watershed_USRB.merge(N_loading, left_index=True, right_index=True)

#fig, ax = plt.subplots(1,1)
#watershed_USRB.plot(legend=True, ax=ax, 
#                    color="white", edgecolor="grey", linewidth=0.5)
#reach_USRB = gpd.read_file(r'Shapefiles/reach.shp')
#watershed_USRB.plot(alpha=1)

def plot_map_basis():
    '''month = {1,2,3..., 12}; 1 repersent January and so on.'''
    watershed_USRB = gpd.read_file(r'./Shapefiles2/Watershed.shp')
    reach_USRB = gpd.read_file(r'./Shapefiles2/reach.shp')
#    dairy_sw = gpd.read_file(r'./Shapefiles2/StoneRidgeShapeFile.shp')
#    Decatur_sw = gpd.read_file(r'./Shapefiles2/ReservoirSubbasin.shp')
    Decatur_lake = gpd.read_file(r'./Shapefiles2/LakeDecatur/LakeDecaturPolygon.shp')
    # extract latitude and longitude of point-source plants
    df = pd.read_excel('./model_SWAT/map_maker/plant_location.xlsx')
    gdf = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, 
                           geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    gdf = gdf.to_crs({'init': 'epsg:26916'})
    

    ##plot figure
    fig, ax = plt.subplots(figsize=(5,4))
    fig.tight_layout(pad=1)
    ax.set_axis_off()
    ax.set_aspect('equal')
    divider = make_axes_locatable(ax)
    
    watershed_USRB.plot(legend=True, ax=ax, color='white', edgecolor="grey", linewidth=0.5)
#    dairy_sw.plot(legend=True, ax=ax, color='c', edgecolor="grey", linewidth=0.5)
#    Decatur_sw.plot(legend=True, ax=ax, color='c', edgecolor="grey", linewidth=0.5)
    Decatur_lake.plot(legend=True, ax=ax, color='lightskyblue', edgecolor="grey", linewidth=0.5)
    reach_USRB.plot(ax=ax, color='blue', linewidth=0.8)
 
    #three markers for plants
    markers = ['*','D', '+', '+', '+', 'v']
    colors = ['green','red','m','green','red','brown']
    for x, y, label, i,j in zip(gdf.geometry.x, gdf.geometry.y, gdf.Name, markers,colors):
        ax.scatter(x, y, color=j, marker=i, s=60, label=label)
    
#    # add an North arrow 
#    x, y, arrow_length = 0.1, 0.75, 0.2
#    ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
#                arrowprops=dict(facecolor='black', width=5, headwidth=15),
#                ha='center', va='center', fontsize=20,
#                xycoords=ax.transAxes)
    fig.suptitle('Upper Sangamon River Watershed', y = 1.0, fontname='Arial', fontsize=14)
    leg = ax.legend(loc='upper left', prop={'family': 'Arial', 'size':10},
              bbox_to_anchor=(0.0, 0.4, 0.6, 0.5), frameon=False)
    leg.set_title('Plant location', prop={'size':14})
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.0, wspace=0.0)
    # plt.savefig('USRW_withriver.tif',dpi=300)
    plt.show()
    
plot_map_basis()
