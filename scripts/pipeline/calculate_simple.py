# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:38:38 2021

@author: KeliDu
"""
# =================================
# Import statements
# =================================

import os
import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
import random
from measures import welsh_t
from measures import wilcoxon_ranksum
#from measures import KL_Divergence
from measures import chi_square
from measures import LLR
from measures import DP_Gries
# =================================
# Functions: calculate
# =================================

def make_idlists(metadatafile, separator, contrast):
    """
    This function creates lists of document identifiers based on the metadata.
    Depending on the contrast defined, the two lists contain various identifiers.
    """
    with open(metadatafile, "r", encoding='utf-8') as infile:
        metadata = pd.read_csv(infile, sep=separator)
        print("\nmetadata\n", metadata.head())
        #metadata = metadata.drop("Unnamed: 0", axis=1)
        metadata.set_index("idno", inplace=True)
        print("\nmetadata\n", metadata.head())
        if contrast[0] != "random":
            list1 = list(metadata[metadata[contrast[0]].isin([contrast[1]])].index)
            list2 = list(metadata[metadata[contrast[0]].isin([contrast[2]])].index)
            #print("list1", list1)
            #print("list2", list2)
        elif contrast[0] == "random":
            allidnos = list(metadata.loc[:, "idno"])
            allidnos = random.sample(allidnos, len(allidnos))
            list1 = allidnos[:int(len(allidnos)/2)]
            list2 = allidnos[int(len(allidnos)/2):]
            #print(list1[0:5])
            #print(list2[0:5])
        idlists = [list1, list2]
        #print(idlists)
        return idlists


def filter_dtm(dtmfolder, parameterstring, idlists, absolutefreqs, relativefreqs, binaryfreqs, tf_frame):
    """
    This function splits the DTM in two parts.
    Each part consists of the segments corresponding to one partition.
    Each segment is chosen based on the file id it corresponds to.
    """
    ids1 = "|".join([str(idno)+".*" for idno in idlists[0]])
    print(ids1)
    ids2 = "|".join([str(id)+".*" for id in idlists[1]])
    binary = binaryfreqs
    relative = relativefreqs
    absolute = absolutefreqs
    binary1 = binary.T.filter(regex=ids1, axis=1)
    binary2 = binary.T.filter(regex=ids2, axis=1)
    relative1 = relative.T.filter(regex=ids1, axis=1)
    relative2 = relative.T.filter(regex=ids2, axis=1)
    absolute1 = absolute.T.filter(regex=ids1, axis=1)
    absolute2 = absolute.T.filter(regex=ids2, axis=1)
    tf_frame1 = tf_frame.T.filter(regex=ids1, axis=1)
    tf_frame2 = tf_frame.T.filter(regex=ids2, axis=1)
    print("\nbinary1\n", binary1.head())
    return binary1, binary2, relative1, relative2, absolute1, absolute2, tf_frame1, tf_frame2
    
    
def get_indicators(binary1, binary2, relative1, relative2, tf_frame1, tf_frame2):
    """
    Indicators are the mean relative frequency or the document proportions,
    depending on the method chosen.
    """
    docprops1 = np.mean(binary1, axis=1)
    docprops1 = pd.Series(docprops1, name="docprops1")
    docprops2 = np.mean(binary2, axis=1)
    docprops2 = pd.Series(docprops2, name="docprops2")
    relfreqs1 = np.mean(relative1, axis=1)*1000
    relfreqs1 = pd.Series(relfreqs1, name="relfreqs1")
    relfreqs2 = np.mean(relative2, axis=1)*1000
    relfreqs2 = pd.Series(relfreqs2, name="relfreqs2")
    tf_framefreqs1 = np.mean(tf_frame1, axis=1)
    tf_framefreqs2 = np.mean(tf_frame2, axis=1)
    print("\ndocprops1\n", docprops1.head(20))
    print("\ndocprops2\n", docprops2.head(20))
    print("\nrelfreqs1\n", relfreqs1.head())
    print("\nrelfreqs2\n", relfreqs2.head())
    print("\tf_framefreqs2\n", tf_framefreqs2.head())
    print("\tf_framefreqs1\n", tf_framefreqs1.head())
    return docprops1, docprops2, relfreqs1, relfreqs2, tf_framefreqs1, tf_framefreqs2

def scaling_results (scaler, Series):
    Series_index = Series.index
    Series = scaler.fit_transform(Series.values.reshape(-1, 1))
    Series = [value[0] for value in Series]
    scaled_Series = pd.Series(data=Series, index=Series_index)
    return scaled_Series
    
def calculate_scores(docprops1, docprops2, absolute1, absolute2, relfreqs1, relfreqs2, logaddition, segmentlength, idlists, tf_framefreqs1, tf_framefreqs2, scaling):
    """
    This function implements several distinctive measures.
    """
    # Define logaddition and division-by-zero avoidance addition
    logaddition = logaddition+1
    divaddition = 0.00000000001
    print("---calculating scores: 1/9, 'original Zeta'---")
    try:
        # sd0 - Subtraction, docprops, untransformed a.k.a. "original Zeta"
        zeta_sd0 = docprops1 - docprops2
        zeta_sd0 = pd.Series(zeta_sd0, name="zeta_sd0")
        # Prepare scaler to rescale variants to range of sd0 (original Zeta)
        scaler = prp.MinMaxScaler(feature_range=(min(zeta_sd0),max(zeta_sd0)))
    except:
        print("Something went wrong while calculating 'original Zeta'")
        zeta_sd0 = pd.Series()
        scaler = prp.MinMaxScaler(feature_range=(-1,1))
    
    print("---calculating scores: 2/9, 'Zeta_log2-transformed'---")
    try:
        # sd2 - Subtraction, docprops, log2-transformed
        zeta_sd2 = np.log2(docprops1 + logaddition) - np.log2(docprops2 + logaddition)
        zeta_sd2 = pd.Series(zeta_sd2, name="zeta_sd2")
    except:
        print("Something went wrong while calculating 'Zeta_log2-transformed'")
        zeta_sd2 = pd.Series()
        
    print("---calculating scores: 3/9, 'ratio of relative frequencies'---")
    try:
        rrf_dr0 = (relfreqs1 + divaddition) / (relfreqs2 + divaddition)
        rrf_dr0 = pd.Series(rrf_dr0, name="rrf_dr0")
    except:
        print("Something went wrong while calculating 'ratio of relative frequencies'")
        rrf_dr0 = pd.Series()
        
    print("---calculating scores: 4/9, 'Eta, deviation of proportions based distinctiveness'---")
    # == Calculate subtraction variants ==
    # sg0 - Subtraction, devprops, untransformed a.k.a. "dpd", ("g" for Gries)
    try:
        devprops1 = DP_Gries.main(absolute1, segmentlength)
        devprops2 = DP_Gries.main(absolute2, segmentlength)
        eta_sg0 = (devprops1 - devprops2) * -1
        eta_sg0 = pd.Series(eta_sg0, name="eta_sg0")
    except:
        print("Something went wrong while calculating 'Eta, deviation of proportions based distinctiveness'")
        eta_sg0 = pd.Series()
        
    #Calculate Welshs-t-test
    print("---calculating scores: 5/9, 'Welshs-t-test'---")
    try:
        welsh_t_value = welsh_t.main(absolute1, absolute2)
    except:
        print("Something went wrong while calculating 'Welshs-t-test'")
        welsh_t_value = pd.Series()    
    
    #Calculate Wilcoxon rank-sum test
    print("---calculating scores: 6/9, 'Wilcoxon rank-sum test'---")
    try:
        ranksumtest_value = wilcoxon_ranksum.main(absolute1, absolute2)
    except:
        print("Something went wrong while calculating 'Wilcoxon rank-sum test'")
        ranksumtest_value = pd.Series()
    '''    
    #Calculate Kullback-Leibler Divergence
    print("---calculating scores: 7/10, 'Kullback-Leibler Divergence'---")
    try:
        KLD_value = KL_Divergence.main(absolute1, absolute2, 2, logaddition)
    except:
        print("Something went wrong while calculating 'Kullback-Leibler Divergence'")
        KLD_value = pd.Series()
    '''    
    #Calculate Chi-squared test
    print("---calculating scores: 7/9, 'Chi-squared test'---")
    try:
        chi_square_value = chi_square.main(absolute1, absolute2)
    except:
        print("Something went wrong while calculating 'Chi-squared test'")
        chi_square_value = pd.Series()
        
    #Calculate Log-likelihood-Ratio test
    print("---calculating scores: 8/9, 'Log-likelihood-Ratio test'---")
    try:
        LLR_value = LLR.main(absolute1, absolute2)
    except:
        print("Something went wrong while calculating 'Log-likelihood-Ratio test'")
        LLR_value = pd.Series()
        
    print("---calculating scores; 9/9, 'Tf-idf weighted absolutefreqs absbased distinctiveness'---")
    try:
        tf_idf = tf_framefreqs1 - tf_framefreqs2
    except:
        print("Something went wrong while calculating 'Tf-idf weighted absolutefreqs absbased distinctiveness'")
        tf_idf = pd.Series()
    
    if scaling == True:
        zeta_sd2 = scaling_results (scaler, zeta_sd2)
        rrf_dr0 = scaling_results (scaler, rrf_dr0)
        eta_sg0 = scaling_results (scaler, eta_sg0)
        welsh_t_value = scaling_results (scaler, welsh_t_value)
        ranksumtest_value = scaling_results (scaler, ranksumtest_value)
        chi_square_value = scaling_results (scaler, chi_square_value)
        LLR_value = scaling_results (scaler, LLR_value)
        tf_idf = scaling_results (scaler, tf_idf)
        return zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh_t_value, ranksumtest_value, chi_square_value, LLR_value, tf_idf #KLD_value
    
    if scaling == False:
        return zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh_t_value, ranksumtest_value, chi_square_value, LLR_value, tf_idf #KLD_value


def get_meanrelfreqs(dtmfolder, parameterstring, relativefreqs):
    meanrelfreqs = relativefreqs.T
    print("\nrelfreqs_df\n", meanrelfreqs.head())
    meanrelfreqs_index = meanrelfreqs.index
    meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
    meanrelfreqs = pd.Series(data=meanrelfreqs, index=meanrelfreqs_index)
    print("\nmeanrelfreqs_series\n", meanrelfreqs.head(10))
    return meanrelfreqs

def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, meanrelfreqs, zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh, ranksum, chi_square, LLR, tf_idf):
    results = pd.DataFrame({
    "docprops1" : docprops1,
    "docprops2" : docprops2,
    "relfreqs1" : relfreqs1,
    "relfreqs2" : relfreqs2,
    "meanrelfreqs" :meanrelfreqs,
    "zeta_sd0" : zeta_sd0,
    "zeta_sd2" : zeta_sd2,
    "rrf_dr0" : rrf_dr0,
    "eta_sg0" : eta_sg0,
    "welsh" : welsh,
    "ranksum" : ranksum,
    #"KL_Divergence": KL_Divergence,
    "chi_square": chi_square,
    "LLR": LLR,
    "tf_idf": tf_idf})
    results = results[[
    "docprops1",
    "docprops2",
    "relfreqs1",
    "relfreqs2",
    "meanrelfreqs",
    "zeta_sd0",
    "zeta_sd2",
    "rrf_dr0",
    "eta_sg0",
    "welsh",
    "ranksum",
    #"KL_Divergence",
    "chi_square",
    "LLR",
    "tf_idf"]]
    results.sort_values(by="zeta_sd0", ascending=False, inplace=True)
    print("\nresults-head\n", results.head(10), "\nresults-tail\n", results.tail(10))
    return results


def save_results(results, resultsfile):
    with open(resultsfile, "w", encoding='utf-8') as outfile:
        results.to_csv(outfile, sep="\t")


# =================================
# Function: main
# =================================


def main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype, absolutefreqs, relativefreqs, binaryfreqs, tf_frame, scaling=True):
    print("--calculate")
    if not os.path.exists(resultsfolder):
        os.makedirs(resultsfolder)
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    #print(parameterstring)
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    #print(contraststring)
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    idlists = make_idlists(metadatafile, separator, contrast)
    #print(idlists)
    binary1, binary2, relative1, relative2, absolute1, absolute2, tf_frame1, tf_frame2 = filter_dtm(dtmfolder, parameterstring, idlists, absolutefreqs, relativefreqs, binaryfreqs, tf_frame)
    #print(binary1)
    docprops1, docprops2, relfreqs1, relfreqs2, tf_framefreqs1, tf_framefreqs2 = get_indicators(binary1, binary2, relative1, relative2, tf_frame1, tf_frame2)
    zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh, ranksum, chi_square, LLR, tf_idf = calculate_scores(docprops1, docprops2, absolute1, absolute2, relfreqs1, relfreqs2, logaddition, segmentlength, idlists, tf_framefreqs1, tf_framefreqs2, scaling)
    meanrelfreqs = get_meanrelfreqs(dtmfolder, parameterstring, relativefreqs)
    results = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, meanrelfreqs, zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh, ranksum, chi_square, LLR, tf_idf)
    save_results(results, resultsfile)