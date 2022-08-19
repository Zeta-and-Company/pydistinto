# -*- coding: utf-8 -*-
"""
Created on Tue May 25 16:08:49 2021

@author: KeliDu
"""

import pandas as pd
from scipy import stats

def LLR_test (absolute1, absolute2, p_value = False):
    """
    This function implements Log-likelihood-Ratio test (https://en.wikipedia.org/wiki/G-test)
    The input "absolute1" and "absoulte2" should be 2 dataframes. Columns represent documents and rows represents features.
    """
    LLR_results = []
    LLR_count = 0
    corpus1 = sum(absolute1.sum())
    corpus2 = sum(absolute2.sum())
    absolute1_sum = absolute1.sum(axis = 1)
    absolute2_sum = absolute2.sum(axis = 1)
    while LLR_count < len(absolute1):
        obs1 = absolute1_sum[LLR_count]
        obs2 = absolute2_sum[LLR_count]
        exp1 = (corpus1 * (obs1 + obs2) ) / (corpus1 + corpus2)
        exp2 = (corpus2 * (obs1 + obs2) ) / (corpus1 + corpus2)
        LLR_row_result = stats.power_divergence([obs1, obs2], f_exp= [exp1, exp2], lambda_='log-likelihood')
        LLR_results.append(LLR_row_result)
        LLR_count+=1
    LLR_full = pd.DataFrame(LLR_results, columns = ['LLR_value', 'p_value'], index=absolute1.index)
    LLR_value = LLR_full['LLR_value']
    if p_value == False:
        return LLR_value
    if p_value == True:
        return LLR_full

def main (absolute1, absolute2, p_value = False):
    LLR = LLR_test(absolute1, absolute2, p_value)
    return LLR