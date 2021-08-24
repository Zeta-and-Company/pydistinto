# -*- coding: utf-8 -*-
"""
Created on Sun Apr 18 23:28:46 2021

@author: KeliDu
"""

import numpy as np
import pandas as pd
import welsh_t
import wilcoxon_ranksum
import KL_Divergence
import LLR

def measure_variants (value1, value2, variants = 'subtraction', logaddition=0.1, log_base = None):

    logaddition = logaddition+1
    divaddition = 0.00000000001
    
    if variants not in ['subtraction', 'division', 'welsh', 'wilcoxon', 'KL_Divergence', 'LLR']:
        raise TypeError("Error!!! First variants parameter must be one of the following options: 'subtraction', 'division', 'welsh', 'wilcoxon', 'KL_Divergence', 'LLR'")
    
    
    if variants == 'subtraction':
        '''
        value1 and value2 have to be two lists of values like pd.Series
        '''
        if log_base == None:
            print("--- calculating: subtraction, untransformed. If the input values are docprops, 'original Zeta' has been calculated. ---")
            s_0 = value1 - value2
            print("--- calculation successfully done! ---")
            return s_0
        if log_base != None:
            print("--- calculating: subtraction, log" + str(log_base) + "-transformed  ---")
            s_log = (np.log(value1 + logaddition + 1) / np.log(log_base)) - (np.log(value2 + logaddition + 1) / np.log(log_base))
            print("--- calculation successfully done! ---")
            return s_log
            
    if variants == 'division':
        '''
        value1 and value2 have to be two lists of values like pd.Series
        '''
        if log_base == None:
            print("--- calculating: division, untransformed  ---")
            d_0 = (value1 + divaddition) / (value2 + divaddition)
            print("--- calculation successfully done! ---")
            return d_0
        if log_base != None:
            print("--- calculating: division, log" + str(log_base) + "-transformed  ---")
            d_log = (np.log(value1 + logaddition + 1) / np.log(log_base)) / (np.log(value2 + logaddition + 1) / np.log(log_base))
            print("--- calculation successfully done! ---")
            return d_log
        
    if variants == 'welsh':
        '''
        value1 and value2 have to be two tables like pd.DataFrame
        '''
        print("--- calculating: Welch's t-test  ---")
        welsh_t_results = welsh_t.main(value1, value2)
        print("--- calculation successfully done! ---")
        return welsh_t_results
    
    if variants == 'wilcoxon':
        '''
        value1 and value2 have to be two tables like pd.DataFrame
        '''
        print("--- calculating: Wilcoxon rank sum test  ---")
        wilcoxon_ranksum_results = wilcoxon_ranksum.main(value1, value2)
        print("--- calculation successfully done! ---")
        return wilcoxon_ranksum_results

    if variants == 'KL_Divergence':
        '''
        value1 and value2 have to be two lists of values like pd.Series
        '''
        print("--- calculating: Kullback-Leibler Divergence  ---")
        if log_base != None:
            KLD_results = KL_Divergence.main(value1, value2, log_base, logaddition)
            print("--- calculation successfully done! ---")
            return KLD_results
        if log_base == None:
            raise ValueError("Error!!! For KL_Divergence, a log_base must be set.")
        else:
            raise ValueError("Error!!! invalid value encountered in log, please check the setting of log_base.")

    if variants == 'LLR':
        '''
        value1 and value2 have to be two tables like pd.DataFrame
        '''
        print("--- calculating: Log-Likelihood-Ratio test  ---")
        LLR_results = LLR.main(value1, value2)
        print("--- calculation successfully done! ---")
    return LLR_results







sd0 = measure_variants(docprops1, docprops2, variants = 'subtraction', log_base=10)

KL_Divergence = measure_variants(docprops1, docprops2, variants = 'KL_Divergence', log_base=10)


















