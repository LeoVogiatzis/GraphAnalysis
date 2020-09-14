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

def createGraphs():
    files = glob.glob("*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for filename in files:
        print('Currently using file - ', filename)

        df = pd.read_csv(filename, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']

        if 'attack' in filename:
            rel_type = 'attacks'
        elif 'trade' in filename:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['label'] = rel_type

        all_dfs = pd.concat([all_dfs, df])
        
    g_directed = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
        create_using=nx.MultiDiGraph(name='Travian_Graph'))

    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
        create_using=nx.MultiGraph(name='Travian_Graph'))

    dir_labels = {e: g_directed.edges[e]['label'] for e in g_directed.edges}
    
    undir_labels = {e: g_undirected.edges[e]['label'] for e in g_undirected.edges}
    
    return g_directed, g_undirected, all_dfs, dir_labels, undir_labels

def compute_shortest_path(_type, graph):
    print('Computing shortest paths for ' + _type + ' of graph.....')
    t = datetime.now()
    sp = list(nx.algorithms.shortest_paths.generic.shortest_path_length(graph))
    time_taken = datetime.now() - t
    print(time_taken)

    print('Converting results.....')
    results = pd.DataFrame([[item[0],k,v] for item in sp for k,v in item[1].items()])\
                .rename(columns={0:'source',1:'target',2:'score'}).reset_index(drop=True)
    results = results[results.score > 0]
    
    print('Find top100 and min100.....')    
    top500 = results.nlargest(500, 'score')
    min500 = results.nsmallest(500, 'score')

    print('Export to csv.....')
    top500.to_csv('sp_top500_' + _type + '.csv')
    min500.to_csv('sp_min500_' + _type + '.csv')
    
    print('Shortest path for ' + _type + ' of graph completed.....')
    return time_taken

def main():
    print('Compute Shortest Paths with NetworkX.....')
    g_d, g_und, all_dfs, d_l, und_l = createGraphs()
    del all_dfs, d_l, und_l
    print('Graphs created.....')

    undirected_time = compute_shortest_path('undirected', g_und)
    directed_time = compute_shortest_path('directed', g_d)

    with open('sp_nx_times.txt', 'a') as f:
        f.write('Time to compute shortest paths with nx:\n')
        f.write('For undirected graph: ' + str(undirected_time) + '\n')
        f.write('For directed graph: ' + str(directed_time) + '\n')
    
    print('Process finished.....')

if __name__ == '__main__':
    main()