# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM

Purpose:
Program used to extract latitude and longitude data of plants
@author: Shaobin
"""


import pandas as pd
import numpy as np
## geocode locations of different plants in the USRB
import geopandas as gpd
from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="specify_your_app_name_here")

# geocode location of Decatur wastewater treatment plant
location_wwtp = geolocator.geocode("501 S Dipper Ln, Decatur, IL 62522")
print((location_wwtp.latitude, location_wwtp.longitude))

# geocode location of drinking water plant
location_N_removal = geolocator.geocode("1155 S Martin Luther King Jr Dr, Decatur, IL 62521")
print((location_N_removal.latitude, location_N_removal.longitude))

## geocode location of dairy 
#location_dairy = geolocator.geocode("5 N 4000 East Road, Mansfield, IL 61854")
#print((location_dairy.latitude, location_dairy.longitude))

df = pd.read_excel('plant_location.xlsx')
gdf = gpd.GeoDataFrame(df, crs={'init': 'epsg:4326'}, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
gdf = gdf.to_crs({'init': 'epsg:26916'})
gdf.crs
gdf.plot()
gdf['geometry'].x
