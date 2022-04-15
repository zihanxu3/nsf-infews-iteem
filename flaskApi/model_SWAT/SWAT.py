# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: INFEWS - ITEEM (NSF award number: 1739788)

Purpose:
To calculate  SWAT simulation rsults using response matrix method for environmental indicators, 
including 1) N/P loadings and 2) stream flow, 3) sediment and crop yield under different BMPs
"""

# import packages
from model_SWAT.SWAT_functions import get_yield, loading_landscape, loading_outlet_USRW
from model_SWAT.crop_yield import get_crop_yield

class SWAT(object):
    def __init__(self, scenario_name, tech_WWT, *args):
        self.scenario_name = scenario_name
        self.tech_WWT = tech_WWT
        # if args:
        #     self.tech = args[0]
# -------------------- Pollutant yield and outlet per subwatershed---------------     
    def get_yield_data(self, name):
        '''
        return a numpy array: (year, month, subwatershed)
        unit: kg/ha for nitrate, phosphorus; ton/ha for sediment; mm for water yield
        '''
        yield_data = get_yield(name, self.scenario_name)
        return yield_data[1]
    
    def get_loading_landscape(self, name):
        '''
        return a numpy array (year, month, subwatershed)
        calculate the landscape (background) loading from the yield at each subwatershe
        unit: kg for nitrate, phosphorus; ton for sediment; mm for water 
        '''
        loading_landscape_data = loading_landscape(name, self.scenario_name)
        return loading_landscape_data

    def get_loading_outlet(self, name, *args): 
        '''
        return a numpy array: (year, month,subwatershed)
        reservoir watershed: 33; downstream of res: 32; outlet: 34
        '''
        # args should be *args
        loading_outlet = loading_outlet_USRW(name, self.scenario_name, *args)
        return loading_outlet

# -------------------- Crop yield per subwatershed-----------------------------
    def get_crop_yield(self, name):
        '''
        calculate crop yield for each subwatershed
        return a tuple: 
        crop_yield (kg/ha): size = (year, subwatershed) 
        crop_production (kg): size = (year, subwatershed) 
        '''
        crop = get_crop_yield(name, self.scenario_name)
        return crop

# A = SWAT('BMP00')
# N_yield = A.get_yield_data('nitrate')
# A.get_N_yield(month=1, year=1)

# B = SWAT('BMP23','ASCP')
# nitrate_outlet_loading = B.get_loading_outlet('nitrate')
# nitrate_landscape_loading = B.get_loading_landscape('nitrate')

# C = SWAT('BMP23')
# C_nitrate = C.get_loading_outlet('nitrate')
# C_nitrate_ASv2 = C.get_loading_outlet('nitrate','AS')
# C_nitrate_ASCP = C.get_loading_outlet('nitrate','ASCP')
# C_nitrate_EBPR = C.get_loading_outlet('nitrate','EBPR_basic')
