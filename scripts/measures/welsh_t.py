# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:41:47 2021

@author: KeliDu
"""
import pandas as pd
from scipy import stats
import numpy as np

def Welshs_t_test (absolute1, absolute2, p_value = False):
    """
    This function implements Welch's t-test (https://en.wikipedia.org/wiki/Welch%27s_t-test)
    The input "absolute1" and "absoulte2" should be 2 dataframes. Columns represent documents and rows represents features.
    """
    welsh_t_results = stats.ttest_ind(absolute1.T, absolute2.T, equal_var = False)
    welsh_t_df = pd.DataFrame(welsh_t_results)
    welsh_t_df = welsh_t_df.T
    welsh_t_df.index = absolute1.index
    welsh_t_df.columns = ['t_value', 'p_value']
    welsh_t_value = welsh_t_df['t_value']
    #change inf to a much larger value
    welsh_t_value.replace(np.inf, max(welsh_t_value.replace(np.inf, np.nan))*10, inplace=True)
    #change -inf to a much smaller value 
    welsh_t_value.replace(-np.inf, min(welsh_t_value.replace(np.inf, np.nan))/10, inplace=True)
    if p_value == False:
        return welsh_t_value
    if p_value == True:
        return welsh_t_df

def main (absolute1, absolute2, p_value = False):
    welsh_t = Welshs_t_test(absolute1, absolute2, p_value)
    return welsh_t