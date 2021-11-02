# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 00:55:31 2021

@author: KeliDu
"""

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
from pipeline import preprocess_spacy
from os.path import join
from os.path import abspath


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


corpus = parameters['corpus']
workdir = parameters['workdir']
dtmfolder = join(workdir, "output", corpus, "dtms", "")

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
# Preprocess
# =================================

"""
This module performs part-of-speech tagging on each text.
This module usually only needs to be called once when preparing a collection of texts.
Currently, this module uses Spacy (https://spacy.io/)
"""
language = parameters['language']
sanitycheck = "no" # yes|no
preprocess_spacy.main(plaintextfolder, taggedfolder, language, sanitycheck)

print('Preprocessing done! Now you can run pydistinto!')