import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

float_formatter = lambda x: "%.3f" % x
np.set_printoptions(formatter={'float_kind': float_formatter})
from sklearn.datasets.samples_generator import make_circles
from sklearn.cluster import SpectralClustering, KMeans

import seaborn as sns

sns.set()


def createGraphs():
    file_names = glob.glob("../data_users_moves/*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for file in file_names:
        print('Currently using file - ', file)

        df = pd.read_csv(file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df['date'] = [d.date() for d in df['Timestamp']]
        df['time'] = [d.time() for d in df['Timestamp']]
        # time_stamp = int(['Timestamp'])
        # formatted_time_stamp = datetime.utcfromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
        # splitted_time_stamp = formatted_time_stamp.split()
        # df['date']= splitted_time_stamp[0]
        # df['time'] = splitted_time_stamp[1]

        x = 1
        if 'attack' in file:
            rel_type = 'attacks'

        elif 'trade' in file:
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


def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)


def spectral_clustering(g_directed):
    # X=np.array.g_directed.edges()
    # W = pairwise_distances(X, metric="euclidean")
    # vectorizer = np.vectorize(lambda x: 1 if x < 5 else 0)
    # W = np.vectorize(vectorizer)(W)
    # print(W)
    W = nx.adjacency_matrix(g_directed)
    print(W.todense())
    D = np.diag(np.sum(np.array(W.todense()), axis=1))
    print('degree matrix:')
    print(D)
    L = D - W
    print('laplacian matrix:')
    print(L)
    e, v = np.linalg.eig(L)
    # eigenvalues
    print('eigenvalues:')
    print(e)
    # eigenvectors
    print('eigenvectors:')
    print(v)
    i = np.where(e < 0.5)[0]
    x = 1
    U = np.array(v[:, i[1]])

    km = KMeans(init='k-means++', n_clusters=3)
    km.fit(U)
    km.labels_
    X, clusters = make_circles(n_samples=1000, noise=.05, factor=.5, random_state=0)
    plt.scatter(X[:, 0], X[:, 1])

    # Using K-means
    km = KMeans(init='k-means++', n_clusters=2)
    km_clustering = km.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=km_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
    # Using Spectral Clustering  scitkit-learn’s implementation
    sc = SpectralClustering(n_clusters=2, affinity='nearest_neighbors', random_state=0)
    sc_clustering = sc.fit(X)
    plt.scatter(X[:, 0], X[:, 1], c=sc_clustering.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')


def main():
    g_directed, g_undirected, all_dfs, dir_labels, undir_labels = createGraphs()
    print('Graph created...')

    spectral_clustering(g_undirected)


if __name__ == '__main__':
    main()
