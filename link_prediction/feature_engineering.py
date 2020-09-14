import glob
import random
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import json


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
    # Create negative samples ---!
    source = all_dfs['id1'].tolist()
    destination = all_dfs['id2'].tolist()
    # combine all nodes in a list
    node_list = source + destination
    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))
    adj_G = nx.to_numpy_matrix(graph, nodelist=node_list)
    # get unconnected node-pairs
    all_unconnected_pairs = []
    #   print(nx.non_edges(G))
    ommisible_links_data = pd.DataFrame(nx.non_edges(graph)).sample(frac=1).reset_index(drop=True)
    dates = pd.date_range('2009-12-01 00:00:00', '2009-12-31 23:59:59', periods=200000)
    gen_df = ommisible_links_data.iloc[:200000, :]
    gen_df.columns = ['id1', 'id2']
    gen_df[['id1', 'id2']] = gen_df[['id1', 'id2']].applymap(np.int64)
    gen_df['Timestamp'] = dates
    gen_df['label'] = 0
    gen_df['weight'] = 1
    gen_df['type'] = random.choices(['attacks', 'messages', 'trades'], weights=(50, 25, 25), k=200000)
    gen_df['Preferential_Attachment'] = 0
    gen_df['Resource_allocation'] = 0

    # Merge dataset with links that doesnt exist

    # print(gen_df)

    labels = {e: graph.edges[e]['type'] for e in graph.edges}
    return graph, all_dfs, labels, g_undirected, gen_df


def centrality_measures(g_undirected, graph, all_dfs):
    bet_cen = nx.betweenness_centrality(g_undirected)
    close_cen = nx.closeness_centrality(graph)

    all_dfs['betweeness_centrality'] = all_dfs['id1'].map(bet_cen)
    all_dfs['betweeness_centrality fot id2'] = all_dfs['id2'].map(bet_cen)
    all_dfs['closeness_centrality'] = all_dfs['id1'].map(close_cen)
    all_dfs['closeness_centrality for id2'] = all_dfs['id2'].map(close_cen)
    return all_dfs


def map_predictions_to_df(predictions, row):
    if (row['id1'], row['id2']) in predictions:
        return predictions[(row['id1'], row['id2'])]
    else:
        return predictions[(row['id2'], row['id1'])]


def link_scores(graph, all_dfs, labels, g_undirected):
    lst = []
    lst2 = []
    predictions1 = nx.preferential_attachment(g_undirected, g_undirected.edges())

    [lst.append((u, v, p)) for u, v, p in predictions1]
    predictions1 = {(k, v): n for k, v, n in lst}

    all_dfs['Preferential_Attachment'] = all_dfs.apply(lambda x: map_predictions_to_df(predictions1, x), axis=1)

    predictions3 = nx.resource_allocation_index(g_undirected, g_undirected.edges())

    try:
        [lst2.append((u, v, p)) for u, v, p in predictions3]
        predictions3 = {(k, v): n for k, v, n in lst2}

        all_dfs['Resource_allocation'] = all_dfs.apply(lambda x: map_predictions_to_df(predictions3, x), axis=1)

    except ZeroDivisionError:
        print("ZeroDivisionError: float division by zero")

    return all_dfs
