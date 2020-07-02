"""
Author:         Jey Han Lau
Date:           Jun 19
"""

import argparse
import sys
import pickle
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import os
import numpy as np
import operator
import csv
import codecs
from scipy.stats.mstats import pearsonr as corr
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
from numpy.polynomial.polynomial import polyfit
from matplotlib import rc, rcParams


#parameters
debug = True

###########
#functions#
###########
def get_sentence_data(input_file):
    sentencexdata = defaultdict(list)
    with codecs.open(input_file, "r", "utf-8") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar="\"")
        column_names = []
        for row_id, row in enumerate(reader):
            if row_id == 0:
                column_names = row
            else:
                data ={}
                for ri, r in enumerate(row):
                    data[column_names[ri]] = r
                sentencexdata[row[0]] = data

    return sentencexdata

######
#main#
######

def main(args):

    if args.input_csv_p1 and args.input_csv_p2:
        p1_sentences = get_sentence_data(args.input_csv_p1)
        p2_sentences = get_sentence_data(args.input_csv_p2)

    p1 = pickle.load(open(args.no_context_pickle, "rb"))
    p2 = pickle.load(open(args.with_context_pickle, "rb"))

    overlap = list(set(p1.keys()) & set(p2.keys()))
    print("No. overlapping sentences =", len(overlap))

    x, y = [], []
    xy_diff = {}
    for k in overlap:
        x.append(np.mean(p1[k]))
        y.append(np.mean(p2[k]))
        xy_diff[k] = x[-1] - y[-1]

    #correlation
    c = corr(x, y)[0]
    xy = np.array([x,y])
    xy = xy.transpose()
    ds = DescrStatsW(xy)
    print("Correlation =", c, ds.corrcoef[0][1])

    #fit x y
    b, m = polyfit(x, y, 1)

    #plot
    rcParams["font.size"] = 16
    rcParams["font.family"] = "Times New Roman"
    #rc('text', usetex=True)
    #rc('text.latex', preamble=r'\usepackage{amssymb} \usepackage[utf8]{inputenc}')
    #rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': '16'})
    plt.scatter(x, y, c="g", marker=".")
    plt.plot(np.linspace(1,4), np.linspace(1,4), "k-")
    plt.plot(([1.0] + x + [4.0]), b + m * np.array([1.0] + x + [4.0]), 'r-')
    plt.xlabel("Context = Real")
    plt.ylabel("Context = Random")
    plt.xlim(1.0, 4.0)
    plt.ylim(1.0, 4.0)
    plt.savefig(os.path.join(args.output_dir, "plot.pdf"))
    #plt.savefig("/Users/jeyhan/Desktop/plot.pdf")

    #display top-N and bottom-N diffs
    if args.input_csv_p1 and args.input_csv_p2:
        diff1 = codecs.open(os.path.join(args.output_dir, "diff1.tsv"), "w", "utf-8")
        diff2 = codecs.open(os.path.join(args.output_dir, "diff2.tsv"), "w", "utf-8")
        xy_diff_sorted = sorted(xy_diff.items(), key=operator.itemgetter(1))

        def write_diff(xy_diff_list, output):
            for k, _ in xy_diff_list:
                output.write(k + "\t")
                output.write(p1_sentences[k]["SENTENCE"] + "\t")
                output.write(p1_sentences[k]["LANGUAGE"] + "\t")
                output.write(p1_sentences[k]["CONTEXT"].replace("\t", " ") + "\t")
                output.write(p2_sentences[k]["CONTEXT"].replace("\t", " ") + "\t")
                output.write(str(np.mean(p1[k])) + "\t")
                output.write(str(np.mean(p2[k])) + "\n")
            output.flush()
            output.close()
    
        write_diff(xy_diff_sorted[:10], diff1)
        write_diff(xy_diff_sorted[-10:], diff2)

if __name__ == "__main__":

        #parser arguments
        desc = "Calculate correlation given 2 pickle files containing sentence ratings"
        parser = argparse.ArgumentParser(description=desc)

        #arguments
        parser.add_argument("-p1", "--no-context-pickle", required=True, help="pickle file containing no-context ratings")
        parser.add_argument("-p2", "--with-context-pickle", required=True, help="pickle file containing with-context ratings")
        parser.add_argument("-o", "--output-dir", required=True, help="output directory for plot")
        parser.add_argument("-i1", "--input-csv-p1", help="input csv file for first pickle")
        parser.add_argument("-i2", "--input-csv-p2", help="input csv file for second pickle")
        args = parser.parse_args()

        main(args)

