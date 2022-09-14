"""
@author JuliaDudar
"""

import os
import csv
import glob
import random

"""
This script takes as input tagged texts in csv-format and gives a randomized version of it as output in order to protect copyrighted texts and enable its analysis at the same time. Randomizing is based on a segment length, this means that the text is randomized within a segment of certain length. More about randomized texts: https://zfdg.de/2020_006
"""


def read_csvfile(file):
    with open(file, "r", newline="\n", encoding="utf-8") as csvfile:
        filename, ext = os.path.basename(file).split(".")
        content = csv.reader(csvfile, delimiter='\t')
        alllines = [line for line in content if len(line) == 3]
        return filename, alllines


def segment_files(filename, alllines, random_segment_size):
    numsegments = int(len(alllines) / random_segment_size)
    segments = []
    segmentids = []
    for i in range(0, numsegments):
        segmentid = filename + "-" + "{:04d}".format(i)
        segmentids.append(segmentid)
        segment = alllines[i * random_segment_size:(i + 1) * random_segment_size]
        random.seed(0)
        random.shuffle(segment)
        segments.append(segment)
    return segmentids, segments


def make_segments(file, randomized_folder, random_segment_size):
    if not os.path.exists(randomized_folder):
        os.makedirs(randomized_folder)
    filename, alllines = read_csvfile(file)
    segmentids, segments = segment_files(filename, alllines, random_segment_size)
    return segmentids, segments



def save_segment(filename, segment, randomized_folder):
    resultfile = randomized_folder + filename + ".csv"
    with open(resultfile, "a", encoding="utf-8", newline='') as outfile:
        writer = csv.writer(outfile,  delimiter='\t')
        for item in segment:
            writer.writerow(item)



def main(taggedfolder, randomized_folder, random_segment_size):
    for file in glob.glob(taggedfolder + "*.csv"):
        filename, ext = os.path.basename(file).split(".")
        segmentids, segments = make_segments(file, randomized_folder, random_segment_size)
        for i in range(len(segmentids)):
            segnum = segmentids[i][:-5]
            print(segnum)
            if segnum == filename:
                segment = segments[i]
                save_segment(filename, segment, randomized_folder)
                continue

