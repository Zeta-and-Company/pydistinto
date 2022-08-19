# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 15:11:25 2022

@author: KeliDu
"""
import numpy as np

def Deviation_of_proportions (absolute, segmentlength):
    """
    This function implements Gries "deviation of proportions" (Gries, 2008. DOI: https://doi.org/10.1075/ijcl.13.4.02gri)
    """
    segnum = len(absolute.columns.values)
    if segmentlength == 'text':
        seglens = list(absolute.sum())
    else:
        seglens = [segmentlength] * segnum
    crpsize = sum(seglens)
    totalfreqs = np.sum(absolute, axis=1)
    expprops = np.array(seglens) / crpsize
    obsprops = absolute.div(totalfreqs, axis=0)
    obsprops = obsprops.fillna(0)#(expprops1[0]) # was: expprops1[0]
    devprops = (np.sum(abs(expprops - obsprops), axis=1) /2 ) #/ (1 - min(expprops1))
    return devprops

def main (absolute, segmentlength):
    DP = Deviation_of_proportions(absolute, segmentlength)
    return DP