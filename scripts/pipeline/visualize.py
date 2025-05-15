#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# file: prepare.py
# author: #cf
# version: 0.3.0


# =================================
# Import statements
# =================================

import os
import pandas as pd
import pygal
from pygal import style
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import seaborn as sns
from datetime import datetime

# =================================
# Pygal style
# =================================

zeta_style = pygal.style.Style(
    background='white',
    plot_background='white',
    font_family="FreeSans",
    title_font_size = 18,
    legend_font_size = 14,
    label_font_size = 12,
    major_label_font_size = 12,
    value_font_size = 12,
    major_value_font_size = 12,
    tooltip_font_size = 12,
    opacity_hover=0.2)


# =================================
# Functions: plot barchart
# =================================


def get_zetadata(resultsfile, measure, numfeatures, droplist):
    with open(resultsfile, "r", encoding="utf8") as infile:
        alldata = pd.read_csv(infile, sep="\t")
        if measure == "eta_sg0":
            alldata = alldata[(alldata['relfreqs1'] != 0) & (alldata['relfreqs2'] != 0)]
        ## TODO: This should not be "unnamed: 0": avoid this.
        alldata = alldata.set_index("Unnamed: 0")
        print("\nalldata\n", alldata.head())
        zetadata = alldata.loc[:, [measure, "docprops1"]]
        zetadata.sort_values(measure, ascending=False, inplace=True)
        zetadata.drop("docprops1", axis=1, inplace=True)
        for item in droplist:
            zetadata.drop(item, axis=0, inplace=True)
        zetadata = zetadata.dropna()

        zetadata = pd.concat(zetadata.head(numfeatures),zetadata.tail(numfeatures))
        zetadata = zetadata.reset_index(drop=False)
        print("\nzetadata\n", zetadata.head())
        return zetadata


def make_barchart(zetadata, zetaplotfile, parameterstring, contraststring, measure, numfeatures):
    range_min = min(zetadata[measure]) * 1.1
    range_max = max(zetadata[measure]) * 1.1
    plot = pygal.HorizontalBar(style = zeta_style,
                               print_values = False,
                               print_labels = True,
                               show_legend = False,
                               range = (range_min, range_max),
                               title = ("Contrastive Analysis with " + str(measure) + "\n("+contraststring+")"),
                               y_title = str(numfeatures) + " distinctive features",
                               x_title = "Parameters: "+ measure +"-"+ parameterstring)
    for i in range(len(zetadata)):
        if i < numfeatures:
            color = "#29a329"
        else:
            color = "#60799f"
        '''
        if zetadata.iloc[i, 1] > 0.8:
            color = "#00cc00"
        if zetadata.iloc[i, 1] > 0.7:
            color = "#14b814"
        if zetadata.iloc[i, 1] > 0.6:
            color = "#29a329"
        elif zetadata.iloc[i, 1] > 0.1:
            color = "#3d8f3d"
        elif zetadata.iloc[i, 1] > 0.05:
            color = "#4d804d"
        elif zetadata.iloc[i, 1] < -0.8:
            color = "#0066ff"
        elif zetadata.iloc[i, 1] < -0.7:
            color = "#196be6"
        elif zetadata.iloc[i, 1] < -0.6:
            color = "#3370cc"
        elif zetadata.iloc[i, 1] < -0.1:
            color = "#4d75b3"
        elif zetadata.iloc[i, 1] < -0.05:
            color = "#60799f"
        else:
            color = "#585858"
        '''
        plot.add(zetadata.iloc[i, 0], [{"value": float(zetadata.iloc[i, 1]), "label": zetadata.iloc[i, 0], "color": color}])
    plot.render_to_file(zetaplotfile)


def zetabarchart(segmentlength, featuretype, contrast, measures, numfeatures, droplist, resultsfolder, plotfolder):
    print("--barchart (zetascores)")
    plotfolder = plotfolder + datetime.now().isoformat(sep="_", timespec="seconds").replace(':', '').replace('-', '_')
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    os.chdir(plotfolder)
    html_file = open("merged_results_" + str(segmentlength) + ".html",'w',encoding='utf-8')
    html_file.write("<html><head>merged distinctive analysis results</head><body>"+"\n")

    for measure in measures:
        # Define some strings and filenames
        zetaplotfile = "zetabarchart_" + parameterstring +"_"+ contraststring +"_" + str(numfeatures) +"-"+str(measure) + ".svg"
        # Get the data and plot it
        zetadata = get_zetadata(resultsfile, measure, numfeatures, droplist)
        try:
            make_barchart(zetadata, zetaplotfile, parameterstring, contraststring, measure, numfeatures)
        except:
            print("Something went wrong while vasualizing " + measure)
            with open(zetaplotfile, 'w', encoding='utf-8') as fout:
                fout.write("Something went wrong while vasualizing " + measure)
                fout.close()
        html_file.write('      <object type="image/svg+xml" data="' + zetaplotfile + '"></object>' + '\n' )
    
    html_file.write("</body></html>")
    html_file.close()
'''

def zetabarchart(segmentlength, featuretype, contrast, measures, numfeatures, droplist, resultsfolder, plotfolder):
    print("--barchart (zetascores)")
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    zetaplotfile = plotfolder + "output.jpg"
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    
    if len(measures) == 1:
        f, ax = plt.subplots(1, 1, figsize=(15, int(numfeatures*0.5)), sharex=False)
        zetadata = get_zetadata(resultsfile, measures[0], numfeatures, droplist)
        zetadata.columns = ['words', measures[0]]
        sns.set(font_scale=1.5)
        sns.barplot(y='words', x=measures[0], data = zetadata, ax=ax, palette="RdBu")
        title = ("Contrastive Analysis with " + str(measures[0]) + "\n("+contraststring+")")
        ax.title.set_text(title)
        plt.savefig(zetaplotfile)
    else:
        measures_count = 0
        axs = []
        while measures_count < len(measures):
            ax_m = 'ax' + str(measures_count)
            axs.append(ax_m)
            measures_count +=1
        
        f, axs = plt.subplots(1, len(measures), figsize=(15 * len(measures), int(numfeatures*0.5)), sharex=False)
        
        i = 0
        while i < len(measures):
            # Get the data and plot it
            zetadata = get_zetadata(resultsfile, measures[i], numfeatures, droplist)
            zetadata.columns = ['words', measures[i]]
            sns.set(font_scale=1.5)
            sns.barplot(y='words', x=measures[i], data = zetadata, ax=axs.item(i), palette="RdBu")
            title = ("Contrastive Analysis with " + str(measures[i]) + "\n("+contraststring+")")
            axs.item(i).title.set_text(title)
            i+=1
        
        sns.despine(bottom=True)
        plt.tight_layout(h_pad=2)
        plt.savefig(zetaplotfile)
'''

# ==============================================
# Scatterplot of types
# ==============================================


def get_scores(resultsfile, numfeatures, measure):
    with open(resultsfile, "r", encoding="utf8") as infile:
        zetascores = pd.read_csv(infile, sep="\t")
        zetascores.sort_values(by=measure, ascending=False, inplace=True)
        positivescores = zetascores.head(numfeatures)
        negativescores = zetascores.tail(numfeatures)
        scores = pd.concat([positivescores, negativescores])
        #print(scores.head())
        return scores


def make_data(scores, measure):
    thetypes = list(scores.index)
    propsone = list(scores.loc[:, "docprops1"])
    propstwo = list(scores.loc[:, "docprops2"])
    zetas = list(scores.loc[:, measure])
    return thetypes, propsone, propstwo, zetas


def make_typesplot(types, propsone, propstwo, zetas, numfeatures, cutoff, contrast, measure, typescatterfile):
    plot = pygal.XY(style=zeta_style,
                    show_legend=False,
                    range=(0, 1),
                    show_y_guides=True,
                    show_x_guides=True,
                    title="Document proportions and " + str(measure),
                    x_title="document proportions in " + str(contrast[1]),
                    y_title="document proportions in " + str(contrast[2]))
    for i in range(0, numfeatures * 2):
        if zetas[i] > cutoff:
            color = "green"
            size = 4
        elif zetas[i] < -cutoff:
            color = "blue"
            size = 4
        else:
            color = "grey"
            size = 3
        plot.add(str(types[i]), [
            {"value": (propsone[i], propstwo[i]), "label": "zeta " + str(zetas[i]), "color": color,
             "node": {"r": size}}])
        plot.add("orientation", [(0, 0.3), (0.7, 1)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0, 0.6), (0.4, 1)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0.3, 0), (1, 0.7)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0.6, 0), (1, 0.4)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
        plot.add("orientation", [(0, 0), (1, 1)], stroke=True, show_dots=False,
                 stroke_style={'width': 0.3, 'dasharray': '2, 6'})
    plot.render_to_file(typescatterfile)


def typescatterplot(numfeatures, cutoff, contrast, segmentlength, featuretype, measure, resultsfolder, plotfolder):
    """
    Function to make a scatterplot with the type proprtion data.
    """
    print("--typescatterplot (types)")
    parameterstring = str(segmentlength) +"-"+ str(featuretype[0]) +"-"+ str(featuretype[1])
    contraststring = str(contrast[0]) +"_"+ str(contrast[2]) +"-"+ str(contrast[1])
    resultsfile = resultsfolder + "results_" + parameterstring +"_"+ contraststring +".csv"
    typescatterfile = plotfolder + "typescatterplot_" + parameterstring +"_"+ contraststring +"_" +str(numfeatures) +"-" + str(cutoff) +"-"+str(measure)+".svg"
    if not os.path.exists(plotfolder):
        os.makedirs(plotfolder)
    scores = get_scores(resultsfile, numfeatures, measure)
    thetypes, propsone, propstwo, zetas = make_data(scores, measure)
    make_typesplot(thetypes, propsone, propstwo, zetas, numfeatures, cutoff, contrast, measure, typescatterfile)

