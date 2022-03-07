# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:08:02 2021

@author: KeliDu
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy

def KLD (absolute1, absolute2, log_base, logaddition):
    '''
    This function implements Kullback–Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) using scipy.stats.entropy(https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html).
    The input "absolute1" and "absolute2" should be 2 dataframes. Columns represent documents and rows represents features.
    '''
    
    KLD_results = []
    KLD_count = 0
    while KLD_count < len(absolute1):
        KLD_row_result = entropy(absolute1.iloc[KLD_count], qk=absolute2.iloc[KLD_count], base = log_base)
        KLD_results.append(KLD_row_result[0])
        KLD_count +=1
    KLD_full = pd.DataFrame(KLD_results, columns = ['KL_Divergence'], index = absolute1.index)    
    KLD_value = KLD_full['KL_Divergence']
    return KLD_value

def main (absolute1, absolute2, log_base, logaddition):
    KL_Divergence = KLD(absolute1, absolute2, log_base, logaddition)
    return KL_Divergence




