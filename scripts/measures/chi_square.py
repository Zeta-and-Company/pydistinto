# -*- coding: utf-8 -*-
"""
Created on Mon May 24 22:44:09 2021

@author: KeliDu
"""
import pandas as pd
from scipy import stats

def chisquare_test (absolute1, absolute2, p_value = False):
    """
    This function implements Chi-squared test (https://en.wikipedia.org/wiki/Chi-squared_test)
    The input "absolute1" and "absoulte2" should be 2 dataframes. Columns represent documents and rows represents features.
    """
    chi_square_results = []
    chi_square_count = 0
    corpus1 = sum(absolute1.sum())
    corpus2 = sum(absolute2.sum())
    absolute1_sum = absolute1.sum(axis = 1)
    absolute2_sum = absolute2.sum(axis = 1)
    while chi_square_count < len(absolute1):
        obs1 = absolute1_sum[chi_square_count]
        obs2 = absolute2_sum[chi_square_count]
        exp1 = (corpus1 * (obs1 + obs2) ) / (corpus1 + corpus2)
        exp2 = (corpus2 * (obs1 + obs2) ) / (corpus1 + corpus2)
        chi_square_row_result = stats.chisquare([obs1, obs2], f_exp= [exp1, exp2])
        chi_square_results.append(chi_square_row_result)
        chi_square_count+=1
    chi_square_full = pd.DataFrame(chi_square_results, columns = ['chisquare_value', 'p_value'], index=absolute1.index)
    chi_square_value = chi_square_full['chisquare_value']
    if p_value == False:
        return chi_square_value
    if p_value == True:
        return chi_square_full

def main (absolute1, absolute2, p_value = False):
    chi_square = chisquare_test(absolute1, absolute2, p_value)
    return chi_square