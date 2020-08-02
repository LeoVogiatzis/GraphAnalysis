import glob

import networkx as nx
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler


def read_csv_files():
    """
    Read data and create MultiDiGraph.Each Node has an id and All edges have 2 attributes.
    The first is Timestamp and the second is the type of edge (Attacks, Trades, Messages)
    :return: G, all_dfs, labels
    """
    file_names = glob.glob("../data_users_moves/*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for file in file_names:
        print(str(file))
        df = pd.read_csv(file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        # df['date'] = [d.date() for d in df['Timestamp']]
        # df['time'] = [d.time() for d in df['Timestamp']]
        if 'attack' in file:
            rel_type = 'attacks'
        elif 'trade' in file:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['type'] = rel_type
        df['weight'] = 1
        df['label'] = 1
        all_dfs = pd.concat([all_dfs, df])

    graph = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                    create_using=nx.MultiDiGraph(name='Travian_Graph'))
    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                           create_using=nx.Graph(name='Travian_Graph'))

    labels = {e: graph.edges[e]['type'] for e in graph.edges}
    return graph, all_dfs, labels, g_undirected


def degree_distribution(graph):
    degrees = graph.degree()
    degree_values = sorted(set(degrees.values()))
    histogram = [list(degrees.values()).count(i) / float(nx.number_of_nodes(G)) for i in degree_values]
    return histogram


def graph_identification(graph):
    """
    Preferential Attachment ('PA')
    Small World with low probability of rewiring ('SW_L')
    Small World with high probability of rewiring ('SW_H')

    display which type of algorithm the graph stems from
    :param graph
    :return: methods
    """
    methods = []
    clustering = nx.average_clustering(graph)
    shortest_path = nx.average_shortest_path_length(graph)
    degree_hist = degree_distribution(graph)
    if len(degree_hist) > 10:
        methods.append('PA')
    elif clustering < 0.1:
        methods.append('SW_H')
    else:
        methods.append('SW_L')

    print(nx.average_clustering(graph), nx.average_shortest_path_length(graph), len(degree_distribution(graph)))
    return methods


def graph_features(graph):
    df = pd.DataFrame(index=graph.nodes())
    df['clustering'] = pd.Series(nx.clustering(graph))
    df['degree'] = pd.Series(graph.degree())
    df['degree_centrality'] = pd.Series(nx.degree_centrality(graph))
    df['closeness'] = pd.Series(nx.closeness_centrality(graph, normalized=True))
    df['betweeness'] = pd.Series(nx.betweenness_centrality(graph, normalized=True))
    df['pr'] = pd.Series(nx.pagerank(graph))
    df['Common Neighbors'] = df.index.map(lambda city: len(list(nx.common_neighbors(g_undirected, city[0], city[1]))))
    df_train = df[~pd.isnull(df['label'])]
    df_test = df[pd.isnull(df['label'])]

    features = ['clustering', 'degree', 'degree_centrality', 'closeness', 'betweeness', 'pr', 'Common Neighbors']
    return features, df_train, df_test


def ml_classification_nodes(features, df_train, df_test):
    X_train = df_train[features]
    Y_train = df_train['is_management']
    X_test = df_test[features]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=[10, 5], alpha=5,
                        random_state=0, solver='lbfgs', verbose=0)
    clf.fit(X_train_scaled, Y_train)
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    return pd.Series(test_proba, X_test.index)


def main():
    graph, all_dfs, labels, g_undirected = read_csv_files()
    # histogram = degree_distribution(graph)
    # methods = graph_identification()
    print(nx.info(graph))
    graph.nodes(data=True)
    df = pd.DataFrame(index=graph.nodes())
    print(df.head())
    #  features, df_train, df_test = graph_features()
    df['Common Neighbors'] = df.index.map(lambda city: len(list(nx.common_neighbors(g_undirected, city[0], city[1]))))
    x = 1
    df.to_pickle("./node_based_features")


if __name__ == '__main__':
    main()
