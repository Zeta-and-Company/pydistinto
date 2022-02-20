# pydistinto: how-to

[![DOI](https://zenodo.org/badge/384188711.svg)](https://zenodo.org/badge/latestdoi/384188711)


## Purpose of this document

This document contains some notes intended to help people use pydistinto.


## What are the requirements?

Requirements:

- Python 3
- Packages pandas, sklearn, numpy, spacy, pygal and seaborn


## How to install pydistinto?

- Simply download or clone the pydistinto repository


## How to run pydistinto?

- Adapt the parameters in `scripts\parameters.txt` to you needs
- First, run `preprocessing_before_running_pydistinto.py` from the Terminal or from an IDE
- After that, run `run_pydistinto_beginners.py` from the Terminal or from an IDE


## What is necessary to run the analyses?

The script expects the following as input. See the `data` folder for an example. 

- A folder with plain text files. They need to be in UTF-8 encoding. The files should all be in one folder (here, the `corpus`folder). 
- A metadata file called "metadata.csv" with category information about each file, identified through the column header called "idno" and which contains the filenames (without the extension). The metadata file should be a CSV file, with the "\t" (tab character) used as the separator character. This metadata file should be at the same level as the `corpus` folder (here, it is in the `data` folder)
- A file with stopwords, called `stoplist.txt`, with one stopword per line. (This can be empty but should be there.)


## What kind of output does pydistinto produce?

The folder `working_dir\output` contains some examples of what pydistinto produces:

- A folder (`data`) containing the text segments with selected features, as used in the calculation (useful for checking)
- In the folder `results`, a matrix containing the features used with their proportions in each partition and their resulting zeta score
- In the folder `plots`, a plot showing the most distinctive words as a horizontal bar chart and a plot showing the feature distribution as a scatterplot.


## What processes and options are supported?

Currently, the following standard processes are supported:

- Prepare a text collection by tagging it using Spacy (run once per collection)
- There are options to choose word forms or lemmata or POS as features. There is the possibility to filter features based on their POS.
- Visualize the most distinctive words as a horizontal bar chart.


## What parameters are there to control pydistinto behavior?

You can set the following parameters in `scripts\parameters.txt`:

- `corpus`: directory of your plain text data
- `workdir`: directory for saving results
- `language`: Catalan, Chinese, Danish, Dutch, English, French, German, Greek, Italian, Japanese, Lithuanian, Macedonian, Norwegian Bokmål, Polish, Portuguese, Romanian, Russian, Spanish (see [Spacy](https://spacy.io/usage) and install the trained pipelines in order to run POS-Tagging. “Multi-language” is not supported)
- `segmentlength`: a number, e. g. 5000; or “text” which means no segmentation
- `forms`: lemmata
- `pos`: all
- `contrast`: detective
- `target_corpus`: yes
- `comparison_corpus`: no
- `no_of_features`: a number, e. g. 20
- `measures`: following measures are implemented:
	- zeta_sd0: Zeta
	- zeta_sd2: Zeta_log2-transformed
	- rrf_dr0: ratio of relative frequencies
	- eta_sg0: Gris’ DP based measure
	- welsh: Welch's t-test
	- ranksum: Wilcoxon rank-sum test
	- chi_square: Chi-Squared Test
	- LLR: Log-Likelihood-Ratio test
	- tf-idf: tf-idf weighted absolute frequencies based measure


## When using pydistinto for research, how can it be referenced?

Software: Du, Keli; Dudar, Julia; Schöch, Christof (2021). pydistinto - a Python implementation of different measures of distinctiveness for contrastive text analysis (Version 0.1.1) [Computer software]. https://doi.org/10.5281/zenodo.5245096

Reference publication: Schöch, Christof (2018): ‘Zeta für die kontrastive Analyse literarischer Texte. Theorie, Implementierung, Fallstudie’, in Quantitative Ansätze in den Literatur- und Geisteswissenschaften. Systematische und historische Perspektiven, ed. by Toni Bernhart, Sandra Richter, Marcus Lepper, Marcus Willand, and Andrea Albrecht (Berlin: de Gruyter), pp. 77–94 <https://www.degruyter.com/view/books/9783110523300/9783110523300-004/9783110523300-004.xml>. 
