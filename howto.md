# pydistinto: how-to

[![DOI](https://zenodo.org/badge/384188711.svg)](https://zenodo.org/badge/latestdoi/384188711)


## Purpose of this document

This document contains some notes intended to help people use pydistinto.


## What are the requirements?

Requirements:

- Python 3
- Packages pandas, numpy, spacy and pygal


## How to install pydistinto?

- Simply download or clone the pydistinto repository


## How to run pydistinto?

- Adapt the parameters in `scripts\parameters.txt` to you needs
- First, run `preprocessing_before_running_pydistinto.py` from the Terminal
- After that, run `run_pydistinto_beginners.py` from the Terminal


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
- `measures`: 
	- `zeta_sd0`: Zeta
	- `rrf_dr0`: ratio of relative frequencies
	- `eta_sg0`: Gris’ DP based measure
	- `welsh`: Welch's t-test
	- `ranksum`: Wilcoxon rank-sum test
	- `KL_Divergence`: Kullback-Leibler divergence
	- `chi_square`: Chi-Squared Test
	- `LLR`: Log-Likelihood-Ratio test


## When using pydistinto for research, how can it be references?

You can either cite the software itself