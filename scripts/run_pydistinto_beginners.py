# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 23:34:41 2021

@author: KeliDu
"""
print("pydistinto is running, this may take a few minutes...")

"""
The pyzeta set of script is a Python implementation of Craig's Zeta and related measures.
Zeta is a measure of keyness or distinctiveness for contrastive analysis of two groups of texts.
This set of scripts does preprocessing, data preparation, score calculation, and visualization.
See the readme.md and howto.md files for help on how to run the script.
"""
import warnings
warnings.filterwarnings('ignore')
# =================================
# Import statements
# =================================
import os
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
from pipeline import prepare
from pipeline import calculate_simple
from pipeline import visualize
from os.path import abspath
from os.path import join

# =================================
# read parameters from parameters.txt
# =================================
print('reading parameters...')
parameters_lines = open(r'parameters.txt', 'r', encoding='utf-8').read().split('\n')
parameters = {}
for line in parameters_lines:
    line_split = line.split('=')
    if len(line_split) == 2:
        parameters[line_split[0]] = line_split[1]

# =================================
# Parameters: files and folders
# =================================

# You need to adapt these
corpus = parameters['corpus']
workdir = parameters['workdir']
dtmfolder = join(corpus, "dtms", "")

# It is recommended to name your files and folders accordingly
datadir = abspath(os.path.join(corpus, os.pardir))
plaintextfolder = join(datadir, "corpus", "")
metadatafile = join(datadir, "metadata.csv")
stoplistfile = join(datadir, "stoplist.txt")

# It is recommended not to change these
outputdir = join(workdir, "output_" + os.path.basename(datadir))
taggedfolder = join(outputdir, "tagged", "")
segmentfolder = join(outputdir, "segments1000", "")
datafolder = join(outputdir, "results", "")
resultsfolder = join(outputdir, "results", "")
plotfolder = join(outputdir, "plots", "")


# =================================
# Prepare
# =================================

"""
This module performs several steps in preparing the data for analysis.
First, it splits each text into segments of a given length.
Second, it either takes all segments or samples an equal number of segments per text.
Third, it selects the desired features from each segment (form and pos)
Fourth, it creates document-term matrixes with absolute, relative and binary feature counts.
This function needs to be run again when a parameter is changed.
"""

try:
  segmentlength = int(parameters['segmentlength'])
except ValueError:
  segmentlength = parameters['segmentlength']
max_num_segments = -1
featuretype = [parameters['forms'], parameters['pos']] # forms, lemmatapos, pos: ["all", "ADJ", "NOM", 'VER']
absolutefreqs, relativefreqs, binaryfreqs, absolutefreqs_sum, tf_frame = prepare.main(taggedfolder, segmentfolder, datafolder, dtmfolder, segmentlength, max_num_segments, stoplistfile, featuretype)


# =================================
# Calculate
# =================================

"""
This module performs the actual distinctiveness measure for each feature.
The calculation can be based on relative or binary features.
The calculation can work in several ways: by division, subtraction as well as with or without applying some log transformation.
The contrast parameter takes ["category", "group1", "group2"] as in the metadata table.
"""
separator = "\t"
contrast = [parameters['contrast'], parameters['target_corpus'], parameters['comparison_corpus']] # example for roman20 [blanche, policier, scifi, sentimental]
#contrast = ["random", "two", "one"] # for splitting groups randomly
logaddition= 0.1 # has effect on log calculation.
calculate_simple.main(datafolder, dtmfolder, metadatafile, separator, contrast, logaddition, resultsfolder, segmentlength, featuretype, absolutefreqs, relativefreqs, binaryfreqs, tf_frame)



# =================================
# Visualize
# =================================

"""
This module provides several plotting functionalities.
"zetabarchart" shows the n words with the most extreme, negative and postive, scores.
"typescatterplot" provides a scatterplot in which each dot is one feature.
"""
# This is for a horizontal barchart for plotting Zeta and similar scores per feature.
numfeatures = int(parameters['no_of_features'])
measures = parameters['measures'].split(',')
#measures = ["zeta_sd0", "rrf_dr0", "dpd_sg0", "welsh", "ranksum", "KL_Divergence", "chi_square", "LLR"]
#droplist = ["anything", "everything", "anyone", "nothing"]
droplist = []
visualize.zetabarchart(segmentlength, featuretype, contrast, measures, numfeatures, droplist, resultsfolder, plotfolder)


print('Job done. Now you can find the figures in: ' + plotfolder)