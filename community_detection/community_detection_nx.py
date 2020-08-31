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

def louvain(unGraph):
    print('Louvain')
    print('Computing parition.....')
    t = datetime.now()
    partition = community.best_partition(unGraph)
    time_taken = datetime.now() - t
    
    print('Converting results.......')
    louvain_results = defaultdict(list)
    for node, comm in sorted(partition.items()):
        louvain_results[comm].append(node)
    
    print('Computing modularity.....')
    modularity = nx.algorithms.community.modularity(unGraph, louvain_results.values())

    extract_results('louvain', unGraph, louvain_results, time_taken, modularity)

    print('Visualization.....')
    values = [partition.get(node) for node in unGraph.nodes()]
    
    nx.draw_spring(unGraph, cmap = plt.get_cmap('jet'), node_color = values,
        node_size=20, with_labels=False)
    plt.show()    
    nx.draw_circular(unGraph, cmap = plt.get_cmap('jet'), node_color = values, 
    node_size=20, with_labels=False)
    plt.show()
    
    nodes = list(unGraph.nodes())
    nodes.sort()
    plt.scatter(nodes, values, c=values, s=50, cmap='viridis')
    plt.show()

    print('Louvain Finished.....')

def label_propagation(unGraph):
    print('Label Propagation')
    print('Computing partition.....')
    t = datetime.now()
    lbcomms = nx.algorithms.community.label_propagation.label_propagation_communities(unGraph)
    time_taken = datetime.now() - t

    print('Converting results.....')
    ress = {}
    res = list(lbcomms)
    count = 0
    for item in res:
        ress[str(count)] = list(item)
        count += 1
    
    print('Computing modularity.....')
    modularity = nx.algorithms.community.modularity(unGraph, ress.values())

    extract_results('label_propagation', unGraph, ress, time_taken, modularity)

    print('Visualization.....')    
    values = calc_color_values(unGraph, ress)  

    nx.draw_spring(unGraph, cmap = plt.get_cmap('jet'), node_color = values, 
    node_size=20, with_labels=False)
    plt.show()  
    nx.draw_circular(unGraph, cmap = plt.get_cmap('jet'), node_color = values, 
    node_size=20, with_labels=False)
    plt.show()

    nodes = list(unGraph.nodes())
    nodes.sort()
    plt.scatter(nodes, values, c=values, s=50, cmap='viridis')
    plt.show()
    
    print('Label Propagation Finished.....')

def k_clique(unGraph, k):
    print('K-Clique')
    
    print('Computing parition for ' + str(k) + '-sized cliques.....')
    t = datetime.now()
    comms = list(nx.algorithms.community.k_clique_communities(unGraph, k))
    time_taken = datetime.now() - t
    
    print('Converting results.....')
    count = 0
    ress = dict()
    for item in comms:
        ress[str(count)] = list(item)
        count += 1

    print('Find and add nodes without a community.....')
    no_community_nodes = list()
    for node in unGraph.nodes():
        for comm in comms:
            if node in list(comm):
                continue
            no_community_nodes.append(node)

    ress[str(count)] = list(no_community_nodes)

    modularity = None
    print('Computing modularity.....')
    try:
        modularity = nx.algorithms.community.modularity(unGraph, ress.values())
    except Exception as e:
        print(repr(e))
    
    extract_results('kClique_' + str(k), unGraph, ress, time_taken, modularity)
    
    print('Visualization.....')
    values = calc_color_values(unGraph, ress)    
    
    proc = False
    if len(values)!=len(unGraph.nodes()):
        print('len values' + str(len(values)))#4000
        print('len nodes' + str(len(unGraph.nodes())))#4611
        diff = len(unGraph.nodes())-len(values)#611
        color = len(values)+1
        non_exist = [color for i in range(diff)]
        new_vals = values + non_exist
        info = 'Nodes without a community: ' + str(diff) + '.\n with label: ' \
            + str(color)
        proc = True

    if proc: 
        print('Colors added')
        values = new_vals

    print(len(values))
    
    nx.draw_spring(unGraph, cmap = plt.get_cmap('jet'), node_color = values,
        node_size=20, with_labels=False)
    plt.show()
    nx.draw_circular(unGraph, cmap = plt.get_cmap('jet'), node_color = values, 
    node_size=20, with_labels=False)
    plt.show()

    nodes = list(unGraph.nodes())
    nodes.sort()
    
    plt.scatter(nodes, values, c=values, s=50, cmap='viridis')
    plt.show()

def clauset_newman_moore(unGraph):
    print('Clauset-Newman-Moore')
    print('COmputing partition.....')
    t = datetime.now()
    c = list(nx.algorithms.community.greedy_modularity_communities(unGraph))
    time_taken = datetime.now() - t
    
    print('Converting results.....')
    res = []
    ress = dict()
    count=0
    for item in (c):
        res.append(list(item))
        ress[str(count)] = list(item)
        count += 1

    print('Computing modularity.....')
    modularity = nx.algorithms.community.modularity(unGraph, ress.values())
        
    
    extract_results('CNM', unGraph, ress, time_taken, modularity)

    print('Visualization.....')
    values = calc_color_values(unGraph, ress)   
    
    nx.draw_spring(unGraph, cmap = plt.get_cmap('jet'), node_color = values,
        node_size=20, with_labels=False)
    plt.show()
    nx.draw_circular(unGraph, cmap = plt.get_cmap('jet'), node_color = values, 
    node_size=20, with_labels=False)
    plt.show()

    nodes = list(unGraph.nodes())
    nodes.sort()
    plt.scatter(nodes, values, c=values, s=50, cmap='viridis')
    plt.show()

    print('Clauset-Newman-Moore finished.....')

def sc(unGraph):
    np.set_printoptions(threshold=sys.maxsize)
    A = nx.convert_matrix.to_numpy_matrix(unGraph)

    nodes = list(unGraph.nodes())
    nodes.sort()
    
    clusters = SpectralClustering(affinity = 'precomputed', 
        assign_labels="kmeans",random_state=0,n_clusters=200).fit_predict(A)
    
    plt.scatter(nodes, clusters, c=clusters, s=50, cmap='viridis')
    plt.show()

    results = defaultdict(list)
    for node, cluster in zip(nodes, clusters):
        results[str(cluster)].append(node) 
    print(results)
    x=1
    with open('sc_res_test200.txt','a') as f:
        f.write(ujson.dumps(results))

    nodes.sort(reverse=True)
    
def calc_color_values(graph, data):
    values = []
    for i in graph.nodes():
        for k,v in data.items():
            for e in v:
                if i == e:
                    values.append(int(k)) 
    return values

def extract_results(alg_name, graph, data, time, modularity, **kwargs):
    _max_l = list() 
    _min_l = list()
    error = False
    if modularity is not None:
        error = True
    extra_text = kwargs.get('extra_text')

    print('Writing results.....')
    with open(alg_name + '_results.txt', 'a') as f:
        f.write('Communities Computed: \n')
        f.write(ujson.dumps(data))
        f.write('================================================================\n')
        f.write('Information about computed partition: \n')
        try:
            if error:
                f.write('modularity= ' + str(modularity) + '\n')
            else:
                f.write('modularity= ' + str('{:f}'.format(modularity)) + ' or ' + str(modularity) + '\n')
        except Exception as e3:
            print(repr(e3))
        f.write('time taken to compute: ' + str(time) + '\n')
        cnt = 0
        d = 0
        for item in data:
            cnt+=1
            f.write('For community ' + str(item) + ':\n')
            subG = graph.subgraph(data[item])
            f.write('NumOfEdges '+str(len(subG.edges())) + '\n')
            f.write('NumOfNodes '+str(len(subG.nodes())) + '\n')
            density = nx.classes.function.density(subG)
            d += density
            f.write('Density of community (/possible) = '+str(density) + '\n')
            _max = max(dict(subG.degree()).items(), key = lambda x : x[1])
            _min = min(dict(subG.degree()).items(), key = lambda x : x[1])
            _max_l.append(_max[1])
            _min_l.append(_min[1])
            f.write('Max (node, degree) -> ' + str(_max) + '\n')
            f.write('Min (node, degree) -> ' + str(_min) + '\n')
            if len(data.keys())!=cnt:
                f.write('Next community info===============================================\n')
            
        f.write('Information after computing.....\n')
        f.write('Max degree = ' + str(max(_max_l)) + '\n')
        f.write('Min degree = ' + str(min(_min_l)) + '\n')
        f.write('Average Density = ' + str(d/len(data)) + '\n')
        if extra_text is not None:
            print(extra_text)
            f.write('Extra Info: \n')
            f.write(extra_text)

def main():
    g_directed, g_undirected, all_dfs, dir_labels, undir_labels = createGraphs()
    del g_directed, all_dfs, dir_labels, undir_labels
    print('Graph creation finished.....')
    
    # louvain(g_undirected)
    # working

    #working 
    # label_propagation(g_undirected)

    # working
    # clauset_newman_moore(g_undirected)

    k_clique(g_undirected, 8)
    k_clique(g_undirected, 7)
    k_clique(g_undirected, 6)

    # sc(g_undirected)

if __name__ == '__main__':
    main()