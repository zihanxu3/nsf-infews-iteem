# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 02:58:05 2021

@author: Shaobin
"""

import pandas as pd
from pathlib import Path

# set up global variables, relative path start from ITEEM/...

# yield_streamflow_daily.csv file larger than 100 MB, cannot upload to github; need to find another way to upload
df_nitrate_daily_path = Path('./model_SWAT/response_matrix_csv/yield_nitrate_daily.csv')  
df_streamflow_daily_path = Path('./model_SWAT/response_matrix_csv/yield_streamflow_daily.csv')

df_nitrate_daily = pd.read_csv(df_nitrate_daily_path)
df_streamflow_daily = pd.read_csv(df_streamflow_daily_path)

# url = pd.read_csv('https://uofi.box.com/s/rvhc89buo0t1tcnqlpxftp7ecuuqqwm2.csv')
