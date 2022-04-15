# -*- coding: utf-8 -*-
"""
Author: Shaobin Li (shaobin@illinois.edu)
Project: NSF INFEWS project - ITEEM

Purpose: 
    Calculate 1) real discount rate; 2) NPV, 3) EAC;
"""

import pandas as pd
import numpy as np


def real_dr(discount_f, inflation_f=0.021):
    '''default inflation factor is 2.1%'''
    rate = (discount_f - inflation_f)/(1 + inflation_f)
    return rate

# real_dr(0.07)

def annuity_factor(n, rate):
    '''
    convert P (present) to A (annual)
    used for EAC calculation
    '''
    AF = (1 - 1/(1+rate)**n)/rate
    return AF

# af = annuity_factor(n=40, rate=real_dr(0.03))
# annuity_factor(n=20, rate = 0.048)
# annuity_factor(n=16, rate = real_dr(0.03,0.021))

def cost_inflation(cost, cost_yr, start_yr, inflation_f=0.021):    
    '''used for converting cost value in any given year to the start year at t=0'''
    cost_adjusted = cost/((1+inflation_f)**(cost_yr-start_yr+1))
    return cost_adjusted

# cost_inflation(wet_1.get_cap_cost()[-1], 2017, 2003, inflation_f=0.021)

# cost_inflation(2189136161, 2017, 2003)
# cost_inflation(0.136, 2017, 2003)
# S0 = DWT(limit_N=10, ag_scenario='BMP00')
# cost_list_N = nitrate_cost.sum(axis=1)
# total_nitrate = nitrate_cost.sum()
# nitrate_cost = S0.get_nitrate_cost()[2]


def pv(cost, discount_f, n): 
    '''
    return present value of asset or cost
    if n=0, cost needs to be a numpy array;
    otherwise, cost is a single value.
    '''
    pv = 0
    r = real_dr(discount_f)
    if n==0:     
        for i in range(len(cost)):
            pv += cost[i]/(1+r)**i 
    if n!=0:
        for i in range(n):
            pv += cost/(1+r)**i
    return pv

# pv(cost_list_N, 0.03, n=0)
# eac = pv(cost=100, discount_f=0.10, n=10)/annuity_factor(n=10, rate=real_dr(0.10))

# npv_dwt_cap = 7595955  # asset price in 2002
# EAC_dwt_cap = npv_dwt_cap/annuity_factor(n=40, rate=real_dr(0.03))/365
