# -*- coding: utf-8 -*-

import pandas as pd

df_nitrate_yield = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate.csv')
df_nitrate_daily = pd.read_csv('./model_SWAT/response_matrix_csv/yield_nitrate_daily.csv')
df_streamflow_daily = pd.read_csv('./model_SWAT/response_matrix_csv/yield_streamflow_daily.csv')
    