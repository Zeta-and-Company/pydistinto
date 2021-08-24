# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 17:08:02 2021

@author: KeliDu
"""

import pandas as pd
import numpy as np

def KLD (relfreqs1, relfreqs2, log_base, logaddition):
    '''
    This function implements Kullback–Leibler divergence (https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
    The input "docprops1" and "docprops2" should be two lists of values like pd.Series.
    '''
    divaddition = 0.00000000001
    
    KLD_results = []
    KLD_count = 0
    while KLD_count < len(relfreqs1):
        KLD_row_result = relfreqs1.iloc[KLD_count] * (np.log((relfreqs1.iloc[KLD_count] / (relfreqs2.iloc[KLD_count] + divaddition)) + logaddition) / np.log(log_base))
        KLD_results.append(KLD_row_result)
        KLD_count +=1
    KLD_full = pd.DataFrame(KLD_results, columns = ['KL_Divergence'], index = relfreqs1.index)    
    KLD_value = KLD_full['KL_Divergence']
    return KLD_value

def main (relfreqs1, relfreqs2, log_base, logaddition):
    KL_Divergence = KLD(relfreqs1, relfreqs2, log_base, logaddition)
    return KL_Divergence




