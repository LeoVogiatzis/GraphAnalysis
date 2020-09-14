import community 
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import glob
import ujson
from collections import defaultdict 
import itertools
import sys
import numpy as np
import os
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering, spectral_clustering
from datetime import datetime
from py2neo import Graph

def cosine_similarity(a, b):
    res = 0
    for index, value in enumerate(a):
        res += len(set(value) & set(b[index])) / float(len(set(value) | set(b[index])))
    return res/len(a)

def read_results():
    for _file in os.listdir(os.getcwd()):
        if 'sp' in _file:
            if 'top' in _file:
                if 'undir' in _file:
                    nx_top_undir = pd.read_csv(_file, header=0)
                    nx_top_undir.columns = ['last_index', 'source', 'target', 'score']
                    nx_top_undir.drop('last_index', axis=1, inplace=True)
                    nx_top_undir_l = nx_top_undir.drop('score', axis=1).values.tolist()
                    print('1.....' + _file)
                else:
                    nx_top_dir = pd.read_csv(_file, header=0)
                    nx_top_dir.columns = ['last_index', 'source', 'target', 'score']
                    nx_top_dir.drop('last_index', axis=1, inplace=True)
                    nx_top_dir_l = nx_top_dir.drop('score', axis=1).values.tolist()
                    print('2.....' + _file)
            elif 'min' in _file:
                if 'undir' in _file:
                    nx_min_undir = pd.read_csv(_file, header=0)
                    nx_min_undir.columns = ['last_index', 'source', 'target', 'score']
                    nx_min_undir.drop('last_index', axis=1, inplace=True)
                    nx_min_undir_l = nx_min_undir.drop('score', axis=1).values.tolist()
                    print('3.....' + _file)
                else:
                    nx_min_dir = pd.read_csv(_file, header=0)
                    nx_min_dir.columns = ['last_index', 'source', 'target', 'score']
                    nx_min_dir.drop('last_index', axis=1, inplace=True)
                    nx_min_dir_l = nx_min_dir.drop('score', axis=1).values.tolist()
                    print('4.....' + _file)
        elif '200best' in _file:
            if 'min' in _file:
                neo_min = pd.read_csv(_file, header=0)
                neo_min.columns = ['last_index', 'source', 'target', 'score']
                neo_min.drop('last_index', axis=1, inplace=True)
                neo_min_l = neo_min.drop('score', axis=1).values.tolist()
                print('5.....' + _file)
            else:
                neo_top = pd.read_csv(_file, header=0)
                neo_top.columns = ['last_index', 'source', 'target', 'score']
                neo_top.drop('last_index', axis=1, inplace=True)
                neo_top_l = neo_top.drop('score', axis=1).values.tolist()
                print('6.....' + _file)

    return nx_top_dir_l, nx_top_undir_l, nx_min_dir_l, nx_min_undir_l, neo_min_l, neo_top_l            

def main():
    nx_top_dir_l, nx_top_undir_l, nx_min_dir_l, nx_min_undir_l, neo_min_l, neo_top_l = read_results()
    
    cos_nxTD_nxTU = cosine_similarity(nx_top_dir_l, nx_top_undir_l)
    cos_nxTD_nxMD = cosine_similarity(nx_top_dir_l, nx_min_dir_l)
    cos_nxTD_nxMU = cosine_similarity(nx_top_dir_l, nx_min_undir_l)
    cos_nxTD_neoM = cosine_similarity(nx_top_dir_l, neo_min_l)
    cos_nxTD_neoT = cosine_similarity(nx_top_dir_l, neo_top_l)
    cos_nxTU_nxMD = cosine_similarity(nx_top_undir_l, nx_min_dir_l)
    cos_nxTU_nxMU = cosine_similarity(nx_top_undir_l, nx_min_undir_l)
    cos_nxTU_neoM = cosine_similarity(nx_top_undir_l, neo_min_l)
    cos_nxTU_neoT = cosine_similarity(nx_top_undir_l, neo_top_l)
    cos_nxMD_nxMU = cosine_similarity(nx_min_dir_l, nx_min_undir_l)
    cos_nxMD_neoM = cosine_similarity(nx_min_dir_l, neo_min_l)
    cos_nxMD_neoT = cosine_similarity(nx_min_dir_l, neo_top_l)
    cos_nxMU_neoM = cosine_similarity(nx_min_undir_l, neo_min_l)
    cos_nxMU_neoT = cosine_similarity(nx_min_undir_l, neo_top_l)
    cos_neoM_neoT = cosine_similarity(neo_min_l, neo_top_l)

    with open('sp_compare_test2.txt', 'a') as f:
        f.write('Shortest path results.....\n')
        f.write('200 from neo, whole graph from nx\n')
        f.write('500top kept from both and 500min\n')
        f.write('Results........\n')
        f.write('Cosine nxTopDirected-nxTopUndirected: ' + str(cos_nxTD_nxTU) + '\n')
        f.write('Cosine nxTopDirected-nxMinDirected: ' + str(cos_nxTD_nxMD) + '\n')
        f.write('Cosine nxTopDirected-nxMinUndirected: ' + str(cos_nxTD_nxMU) + '\n')
        f.write('Cosine nxTopDirected-neoMin: ' + str(cos_nxTD_neoM) + '\n')
        f.write('Cosine nxTopDirected-neoTop: ' + str(cos_nxTD_neoT) + '\n')
        f.write('Cosine nxTopUndirected-nxMinDirected: ' + str(cos_nxTU_nxMD) + '\n')
        f.write('Cosine nxTopUndirected-nxMinUndirected: ' + str(cos_nxTU_nxMU) + '\n')
        f.write('Cosine nxTopUndirected-neoMin: ' + str(cos_nxTU_neoM) + '\n')
        f.write('Cosine nxTopUndirected-neoTop: ' + str(cos_nxTU_neoT) + '\n')
        f.write('Cosine nxMinDirected-nxMinUndirected: ' + str(cos_nxMD_nxMU) + '\n')
        f.write('Cosine nxMinDirected-neoMin: ' + str(cos_nxMD_neoM) + '\n')
        f.write('Cosine nxMinDirected-neoTop: ' + str(cos_nxMD_neoT) + '\n')
        f.write('Cosine nxMinUndirected-neoMin: ' + str(cos_nxMU_neoM) + '\n')
        f.write('Cosine nxMinUndirected-neoTop: ' + str(cos_nxMU_neoT) + '\n')
        f.write('Cosine neoMin-neoTop: ' + str(cos_neoM_neoT) + '\n')

if __name__ == '__main__':
    main()