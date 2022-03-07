# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 17:38:38 2021

@author: KeliDu
"""
# =================================
# Import statements
# =================================

import os
import re
import csv
import glob
import pandas as pd
import numpy as np
from sklearn import preprocessing as prp
import random
from measures import welsh_t
from measures import wilcoxon_ranksum
#from measures import KL_Divergence
from measures import chi_square
from measures import LLR

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
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_binaryfreqs.csv"
    #print(dtmfile)
    #print(idlists)
    ids1 = "|".join([str(idno)+".*" for idno in idlists[0]])
    print(ids1)
    ids2 = "|".join([str(id)+".*" for id in idlists[1]])
    #print(ids2)
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
    """
    with open(dtmfile, "r", encoding='utf-8') as infile:
        binary = pd.read_hdf(infile, sep="\t", index_col="idno")
        print("\nbinary\n", binary.head())
        binary1 = binary.T.filter(regex=ids1, axis=1)
        print("\nbinary1\n", binary1.head())
        binary2 = binary.T.filter(regex=ids2, axis=1)
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r", encoding='utf-8') as infile:
        relative = pd.read_hdf(infile, sep="\t", index_col="idno")
        #print("\nrelative\n", relative.head())
        relative1 = relative.T.filter(regex=ids1, axis=1)
        relative2 = relative.T.filter(regex=ids2, axis=1)
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_absolutefreqs.csv"
    with open(dtmfile, "r", encoding='utf-8') as infile:
        absolute = pd.read_hdf(infile, sep="\t", index_col="idno")
        #print("\nabsolute\n", absolute.head())
        absolute1 = absolute.T.filter(regex=ids1, axis=1)
        absolute2 = absolute.T.filter(regex=ids2, axis=1)
    """
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


def calculate_scores(docprops1, docprops2, absolute1, absolute2, relfreqs1, relfreqs2, logaddition, segmentlength, idlists, tf_framefreqs1, tf_framefreqs2):
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
        #print("\nsd0\n", zeta_sd0.head(10))
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
        zeta_sd2_index = zeta_sd2.index
        zeta_sd2 = scaler.fit_transform(zeta_sd2.values.reshape(-1, 1))
        zeta_sd2 = [value[0] for value in zeta_sd2]
        zeta_sd2 = pd.Series(data=zeta_sd2, index=zeta_sd2_index)
    except:
        print("Something went wrong while calculating 'Zeta_log2-transformed'")
        zeta_sd2 = pd.Series()
        
    print("---calculating scores: 3/9, 'ratio of relative frequencies'---")
    try:
        rrf_dr0 = (relfreqs1 + divaddition) / (relfreqs2 + divaddition)
        rrf_dr0 = pd.Series(rrf_dr0, name="rrf_dr0")
        rrf_dr0_index = rrf_dr0.index
        rrf_dr0 = scaler.fit_transform(rrf_dr0.values.reshape(-1, 1))
        rrf_dr0 = [value[0] for value in rrf_dr0]
        rrf_dr0 = pd.Series(data=rrf_dr0, index=rrf_dr0_index)
    except:
        print("Something went wrong while calculating 'ratio of relative frequencies'")
        rrf_dr0 = pd.Series()
        
    print("---calculating scores: 4/9, 'Eta, deviation of proportions based distinctiveness'---")
    # == Calculate subtraction variants ==
    # sg0 - Subtraction, devprops, untransformed a.k.a. "dpd", ("g" for Gries)
    try:
        devprops1 = Deviation_of_proportions(absolute1, segmentlength)
        devprops2 = Deviation_of_proportions(absolute2, segmentlength)
        eta_sg0 = (devprops1 - devprops2) * -1
        eta_sg0 = pd.Series(eta_sg0, name="eta_sg0")
        eta_sg0_index = eta_sg0.index
        eta_sg0 = scaler.fit_transform(eta_sg0.values.reshape(-1, 1))
        eta_sg0 = [value[0] for value in eta_sg0]
        eta_sg0 = pd.Series(data=eta_sg0, index=eta_sg0_index)
    except:
        print("Something went wrong while calculating 'Eta, deviation of proportions based distinctiveness'")
        eta_sg0 = pd.Series()
        
    #Calculate Welshs-t-test
    print("---calculating scores: 5/9, 'Welshs-t-test'---")
    try:
        welsh_t_value = welsh_t.main(absolute1, absolute2)
        welsh_t_index = welsh_t_value.index
        welsh_t_value = scaler.fit_transform(welsh_t_value.values.reshape(-1, 1))
        welsh_t_value = [value[0] for value in welsh_t_value]
        welsh_t_value = pd.Series(data=welsh_t_value, index=welsh_t_index)
    except:
        print("Something went wrong while calculating 'Welshs-t-test'")
        welsh_t_value = pd.Series()    
    
    #Calculate Wilcoxon rank-sum test
    print("---calculating scores: 6/9, 'Wilcoxon rank-sum test'---")
    try:
        ranksumtest_value = wilcoxon_ranksum.main(absolute1, absolute2)
        ranksumtest_index = ranksumtest_value.index
        ranksumtest_value = scaler.fit_transform(ranksumtest_value.values.reshape(-1, 1))
        ranksumtest_value = [value[0] for value in ranksumtest_value]
        ranksumtest_value = pd.Series(data=ranksumtest_value, index=ranksumtest_index)
    except:
        print("Something went wrong while calculating 'Wilcoxon rank-sum test'")
        ranksumtest_value = pd.Series()
    '''    
    #Calculate Kullback-Leibler Divergence
    print("---calculating scores: 7/10, 'Kullback-Leibler Divergence'---")
    try:
        KLD_value = KL_Divergence.main(absolute1, absolute2, 2, logaddition)
        KLD_index = KLD_value.index
        KLD_value = scaler.fit_transform(KLD_value.values.reshape(-1, 1))
        KLD_value = [value[0] for value in KLD_value]
        KLD_value = pd.Series(data=KLD_value, index=KLD_index)
    except:
        print("Something went wrong while calculating 'Kullback-Leibler Divergence'")
        KLD_value = pd.Series()
    '''    
    #Calculate Chi-squared test
    print("---calculating scores: 7/9, 'Chi-squared test'---")
    try:
        chi_square_value = chi_square.main(absolute1, absolute2)
        chi_square_index = chi_square_value.index
        chi_square_value = scaler.fit_transform(chi_square_value.values.reshape(-1, 1))
        chi_square_value = [value[0] for value in chi_square_value]
        chi_square_value = pd.Series(data=chi_square_value, index=chi_square_index)
    except:
        print("Something went wrong while calculating 'Chi-squared test'")
        chi_square_value = pd.Series()
        
    #Calculate Log-likelihood-Ratio test
    print("---calculating scores: 8/9, 'Log-likelihood-Ratio test'---")
    try:
        LLR_value = LLR.main(absolute1, absolute2)
        LLR_index = LLR_value.index
        LLR_value = scaler.fit_transform(LLR_value.values.reshape(-1, 1))
        LLR_value = [value[0] for value in LLR_value]
        LLR_value = pd.Series(data=LLR_value, index=LLR_index)
    except:
        print("Something went wrong while calculating 'Log-likelihood-Ratio test'")
        LLR_value = pd.Series()
        
    print("---calculating scores; 9/9, 'Tf-idf weighted absolutefreqs absbased distinctiveness'---")
    try:
        tf_idf = tf_framefreqs1 - tf_framefreqs2
        tf_idf_index = tf_idf.index
        tf_idf = scaler.fit_transform(tf_idf.values.reshape(-1, 1))
        tf_idf = [value[0] for value in tf_idf]
        tf_idf = pd.Series(tf_idf, index=tf_idf_index)
    except:
        print("Something went wrong while calculating 'Tf-idf weighted absolutefreqs absbased distinctiveness'")
        tf_idf = pd.Series()
        
    return zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh_t_value, ranksumtest_value, chi_square_value, LLR_value, tf_idf #KLD_value, chi_square_value, LLR_value, tf_idf


def get_meanrelfreqs(dtmfolder, parameterstring, relativefreqs):
    meanrelfreqs = relativefreqs.T
    print("\nrelfreqs_df\n", meanrelfreqs.head())
    meanrelfreqs_index = meanrelfreqs.index
    meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
    meanrelfreqs = pd.Series(data=meanrelfreqs, index=meanrelfreqs_index)
    print("\nmeanrelfreqs_series\n", meanrelfreqs.head(10))
    return meanrelfreqs
    """
    dtmfile = dtmfolder + "dtm_"+parameterstring+"_relativefreqs.csv"
    with open(dtmfile, "r", encoding='utf-8') as infile:
        meanrelfreqs = pd.read_csv(infile, sep="\t", index_col="idno").T
        print("\nrelfreqs_df\n", meanrelfreqs.head())
        meanrelfreqs_index = meanrelfreqs.index
        meanrelfreqs = np.mean(meanrelfreqs, axis=1)*1000
        meanrelfreqs = pd.Series(data=meanrelfreqs, index=meanrelfreqs_index)
        print("\nmeanrelfreqs_series\n", meanrelfreqs.head(10))
        return meanrelfreqs
    """

def combine_results(docprops1, docprops2, relfreqs1, relfreqs2, meanrelfreqs, zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh, ranksum, chi_square, LLR, tf_idf):
    #print(len(docprops1), len(docprops2), len(relfreqs1), len(relfreqs2), len(devprops1), len(devprops2), len(meanrelfreqs), len(sd0), len(sd2), len(sr0), len(sr2), len(sg0), len(sg2), len(dd0), len(dd2), len(dr0), len(dr2), len(dg0), len(dg2))
    #print(type(docprops1), type(docprops2), type(relfreqs1), type(relfreqs2), type(devprops1), type(devprops2), type(meanrelfreqs), type(sd0), type(sd2), type(sr0), type(sr2), type(sg0), type(sg2), type(dd0), type(dd2), type(dr0), type(dr2), type(dg0), type(dg2))
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
    #print(results.head())
    #print(results.columns.tolist())
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


def main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype, absolutefreqs, relativefreqs, binaryfreqs, tf_frame):
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
    zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh, ranksum, chi_square, LLR, tf_idf = calculate_scores(docprops1, docprops2, absolute1, absolute2, relfreqs1, relfreqs2, logaddition, segmentlength, idlists, tf_framefreqs1, tf_framefreqs2)
    meanrelfreqs = get_meanrelfreqs(dtmfolder, parameterstring, relativefreqs)
    results = combine_results(docprops1, docprops2, relfreqs1, relfreqs2, meanrelfreqs, zeta_sd0, zeta_sd2, rrf_dr0, eta_sg0, welsh, ranksum, chi_square, LLR, tf_idf)
    save_results(results, resultsfile)