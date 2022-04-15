# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
To visualize the map of Upper Sangamon River Basin
"""

import numpy as np
import pandas as pd
import time
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely import geometry, ops
from mpl_toolkits.axes_grid1 import make_axes_locatable

# load new modules developed for ITEEM
from model_SWAT.SWAT_functions import yield_unit, crop_yield
 
#watershed_USRB = gpd.read_file('Shapefiles/Watershed.shp')
#watershed_USRB.head()
#watershed_USRB.columns

def plot_map(name):
    '''month = {1,2,3..., 12}; 1 repersent January and so on.'''
    watershed_USRB = gpd.read_file('./Shapefiles2/Watershed.shp')
    reach_USRB = gpd.read_file('./Shapefiles2/reach.shp')

    ## extract latitude and longitude of point-source plants
    df = pd.read_excel('./model_SWAT/map_maker/plant_location.xlsx')
    gdf = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, 
                           geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
    gdf = gdf.to_crs({'init': 'epsg:26916'})

    ## select what type of outputs for plotting
    if name == 'nitrate' or name == 'phosphorus':
        yield_per_sw = yield_unit(name)[1]
        unit = 'kg/ha'
    elif name == 'sediment':
        yield_per_sw = yield_unit(name)[1]
        unit = 'ton/ha'
    elif name == 'streamflow':
        yield_per_sw = yield_unit(name)[1]
        unit = 'mm'
    elif name == 'soybean' or name == 'corn' or name == 'corn sillage':
        yield_per_sw = crop_yield(name)[1]
        unit = 'kg/ha'
#    min_value = yield_per_sw.min()
    max_value = yield_per_sw.max()
#    max_value = math.trunc(yield_per_sw.max()) + 1

    for i in range(yield_per_sw.shape[0]):
        yield_per_sw_df = pd.DataFrame(yield_per_sw[i,:,:]).T
        for j in range(yield_per_sw.shape[1]):
            ## plot figure
            fig, ax = plt.subplots(figsize=(6.5,6))
            fig.tight_layout(pad=1)
            ax.set_axis_off()
            ax.set_aspect('equal')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            
            yield_per_sw_df_month = yield_per_sw_df.iloc[:,j]
            
            ## Using Merge on a column and Index in Pandas 
            watershed = watershed_USRB.merge(yield_per_sw_df_month, left_index=True, right_index=True)
            watershed.plot(column=j, legend=True, ax=ax, cmap= mpl.cm.get_cmap('viridis_r'),
                                edgecolor="grey", linewidth=0.5, cax=cax, vmin=0, vmax=max_value,
                                legend_kwds={'orientation': "horizontal"})
            reach_USRB.plot(ax=ax, color='blue', linewidth=0.8)
            
            #three markers for plants
            markers = ['*','D', '+', '+', '+', 'v']
            colors = ['green','red','m','m','m','brown']
            for x, y, label, m, c in zip(gdf.geometry.x, gdf.geometry.y, gdf.Name, markers,colors):
                ax.scatter(x, y, color=c, marker=m, s=60, label=label)
                
            fig.suptitle('Upper Sangamon River Watershed', y = 0.9, fontname='Arial', fontsize=14)
            
            # add text for year and month
            plt.text(x=0.25, y=0.95, fontsize=14, s=str(name).capitalize() + 
                     '_Year' +str(1+i) + '_month' + str(1+j),transform=ax.transAxes)     
            # add text for legend
            plt.text(x=0.5, y=0.02, fontsize=14, s=str(name).capitalize() + "_yield" + 
                     ' (' + unit + ')',transform=ax.transAxes)

            # add total loadings as text
#            plt.text(3.1, 307, fontsize=14, s='Month '+ str(month)+' (Total loadings: ' + 
#                     str(round(outlet_sw.iloc[34,month-1]/1000,1)) + ' ton)')
            leg = ax.legend(loc='upper left', prop={'family': 'Arial', 'size':12},
                            bbox_to_anchor=(0.00, 0.32, 0.5, 0.5),frameon=False)
            leg.set_title('Plant location', prop={'size':14})
        
            #fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.0, wspace=0.0)
            # plt.savefig('./model_SWAT/figures//' + str(name).capitalize() + '_year' +str(i+1) + '_month' + str(j+1) + '.tif', dpi=60)
            plt.show()


#yield_per_sw = yield_unit('nitrate')
            
            
start = time.time()
#plot_map(name='nitrate')
plot_map(name='sediment')
end = time.time()
print('Running time is {:.1f} seconds'.format(end-start))
