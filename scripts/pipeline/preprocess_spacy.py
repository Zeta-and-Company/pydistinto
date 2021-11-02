#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: preprocess_2.py
# author: #jd
# version: 0.1.0


"""
The "preprocess" module is the first step in the pyzeta pipeline.
This module deals with linguistic annotation of the texts.
Subsequent modules are: prepare, calculate and visualize.
"""

# =================================
# Import statements
# =================================

import spacy
# download spacy via command line python -m spacy download fr_core_news_sm
import os
import re
import csv
import glob

def read_plaintext(file):
    """
    reads plaintext files
    """
    with open(file, "r", encoding="utf-8") as infile:
        text = infile.read()
        text = re.sub("’", "'", text)
        return text



def save_tagged(taggedfolder, filename, tagged):
    """
    Takes the spacy output and writes it to a CSV file.
    """
    taggedfilename = taggedfolder + "/" + filename + ".csv"
    with open(taggedfilename, "w", encoding="utf-8") as outfile:
        writer = csv.writer(outfile, delimiter='\t')
        for token in tagged:
            if token.pos_ != 'SPACE':
                #print(token_row)
                writer.writerow([token.text, token.pos_, token.lemma_])

def sanity_check(text, tagged): 
    """
    Performs a simple sanity check on the data. 
    Checks number of words in inpu text. 
    Checks number of lines in tagged output. 
    If these numbers are similar, it looks good. 
    """
    text = re.sub("([,.:;!?])", " \1", text)
    text = re.split("\s+", text)
    print("number of words", len(text)) 
    print(text[0:10])
    print("number of lines", len(tagged))
    print(tagged[0:10])
    if len(tagged) == 0: 
        print("Sanity check: Tagging error: nothing tagged.")
    elif len(tagged) / len(text) < 0.8  or len(tagged) / len(text) > 1.2: 
        print("Sanity check: Tagging error: strong length difference.")
    else: 
        print("Sanity check: Tagging seems to have worked.")


def main(plaintextfolder, taggedfolder, language, sanitycheck='no'):
    """
    coordinationsfuction
    :param plaintextfolder:
    :param taggedfolder:
    :param language
    """
    if language == "English":
        nlp = spacy.load("en_core_web_sm")
        nlp.max_length = 10000000
    # french models
    elif language == "French":
        nlp = spacy.load("fr_core_news_sm")
        nlp.max_length = 10000000
    # Catalan models
    elif language == "Catalan":
        nlp = spacy.load("ca_core_news_sm")
        nlp.max_length = 10000000
    # Chinese models
    elif language == "Chinese":
        nlp = spacy.load("zh_core_web_sm")
        nlp.max_length = 10000000
    # Danish models
    elif language == "Danish":
        nlp = spacy.load("da_core_news_sm")
        nlp.max_length = 10000000
    # Dutch models
    elif language == "Dutch":
        nlp = spacy.load("nl_core_news_sm")
        nlp.max_length = 10000000
    # German models
    elif language == "German":
        nlp = spacy.load("de_core_news_sm")
        nlp.max_length = 10000000
    # Greek models
    elif language == "Greek":
        nlp = spacy.load("el_core_news_sm")
        nlp.max_length = 10000000
    # Italian models
    elif language == "Italian":
        nlp = spacy.load("it_core_news_sm")
        nlp.max_length = 10000000
    # Japanese models
    elif language == "Japanese":
        nlp = spacy.load("ja_core_news_sm")
        nlp.max_length = 10000000
    # Lithuanian models
    elif language == "Lithuanian":
        nlp = spacy.load("lt_core_news_sm")
        nlp.max_length = 10000000
    # Macedonian models
    elif language == "Macedonian":
        nlp = spacy.load("mk_core_news_sm")
        nlp.max_length = 10000000
    # Norwegian Bokmål models
    elif language == "Norwegian Bokmål":
        nlp = spacy.load("nb_core_news_sm")
        nlp.max_length = 10000000
    # Polish models
    elif language == "Polish":
        nlp = spacy.load("pl_core_news_sm")
        nlp.max_length = 10000000
    # Portuguese models
    elif language == "Portuguese":
        nlp = spacy.load("pt_core_news_sm")
        nlp.max_length = 10000000
    # Romanian models
    elif language == "Romanian":
        nlp = spacy.load("ro_core_news_sm")
        nlp.max_length = 10000000
    # Russian models
    elif language == "Russian":
        nlp = spacy.load("ru_core_news_sm")
        nlp.max_length = 10000000
    # Spanish models
    elif language == "Spanish":
        nlp = spacy.load("es_core_news_sm")
        nlp.max_length = 10000000
    else:
        raise ValueError("Unable to load spacy model. Please check your language settings.")
    print("\n--preprocess. \n\nDepending on the size of your corpus, this may take an hour or two. Let's take a coffee break, you deserve it!")
    if not os.path.exists(taggedfolder):
        os.makedirs(taggedfolder)
    if not os.path.exists(plaintextfolder):
        raise ValueError("Please make sure the the name of the folder with your plain text data is 'corpus'!")
    else:
        counter = 0
        for file in glob.glob(plaintextfolder + "*.txt"):
            filename, ext = os.path.basename(file).split(".")
            counter += 1
            print("next: file", counter, ":", filename)
            text = read_plaintext(file)
            tagged = nlp(text)
            save_tagged(taggedfolder, filename, tagged)
            if sanitycheck == "yes": 
                sanity_check(text, tagged)
