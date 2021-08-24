# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 18:29:43 2021

@author: KeliDu
"""

import pandas as pd
from scipy import stats

def Wilcoxon_ranksum_test (absolute1, absolute2, p_value = False):
    """
    This function implements Wilcoxon rank sum test (https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
    """
    
    ranksum_t_results = []
    ranksum_count = 0
    while ranksum_count < len(absolute1):
        ranksum_row_result = stats.ranksums(absolute1.iloc[ranksum_count], absolute2.iloc[ranksum_count])
        ranksum_t_results.append(ranksum_row_result)
        ranksum_count+=1
    ranksum_full = pd.DataFrame(ranksum_t_results, columns = ['ranksumtest_value', 'p_value'], index=absolute1.index)
    ranksumtest_value = ranksum_full['ranksumtest_value']
    if p_value == False:
        return ranksumtest_value
    if p_value == True:
        return ranksum_full

def main (absolute1, absolute2, p_value = False):
    wilcoxon_ranksum = Wilcoxon_ranksum_test(absolute1, absolute2, p_value)
    return wilcoxon_ranksum