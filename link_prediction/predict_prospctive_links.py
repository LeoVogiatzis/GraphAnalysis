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


def aggregated_dataset(all_dfs, g_undirected):
    aggregated_df = all_dfs.sort_values(by='Timestamp', ascending=True)
    aggregated_df = all_dfs.groupby(['id1', 'id2', 'type'], as_index=False)['weight'].sum()
    aggregated_df = aggregated_df.set_index(['id1', 'id2'])

    aggregated_df['preferential attachment'] = [i[2] for i in
                                                nx.preferential_attachment(g_undirected, aggregated_df.index)]
    aggregated_df['Common Neighbors'] = aggregated_df.index.map(
        lambda id: len(list(nx.common_neighbors(g_undirected, id[0], id[1]))))
    aggregated_df['label'] = 1
    aggregated_df.to_pickle("./dummy.pkl")
    return aggregated_df


def calc_weeights_without_aggregate(all_dfs):
    pairs = {}
    for index, row in all_dfs.iterrows():
        if (row['id1'], row['id2']) not in pairs:
            pairs[(row['id1'], row['id2'])] = 0
        pairs[(row['id1'], row['id2'])] += 1  # it could be row['weight']

    for index, row in all_dfs.iterrows():
        if (row['id1'], row['id2']) in pairs:
            all_dfs.at[index, 'weight'] = pairs[(row['id1'], row['id2'])]
    all_dfs.to_pickle("./aggregated_df.pkl")
    # df = pd.read_pickle("./aggregated_df.pkl")
    return all_dfs


def generate_timestamps():
    df = pd.read_pickle("./aggregated_df.pkl")


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


def new_connections_predictions(dataset):
    df_train = dataset[~pd.isnull(dataset['label'])]
    df_test = dataset[pd.isnull(dataset['label'])]
    features = ['type', 'weight', 'Preferential_Attachment', 'Resource_allocation', 'betweeness_centrality',
                'betweeness_centrality fot id2', 'closeness_centrality', 'closeness_centrality for id2']
    X_train = df_train[features]
    Y_train = df_train['label']
    X_test = df_test[features]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=[10, 5], alpha=5,
                        random_state=0, solver='lbfgs', verbose=0)
    clf.fit(X_train_scaled, Y_train)
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    predictions = pd.Series(test_proba, X_test.index)
    target = dataset['label']
    target['prob'] = [predictions[x] for x in target.index]
    return target['prob']


def main():
    # graph, all_dfs, labels, g_undirected, gen_df = read_csv_files()
    # all_dfs = link_scores(graph, all_dfs, labels, g_undirected)
    # dataset = pd.concat([all_dfs, gen_df])
    #centrality_measures(g_undirected, graph, dataset)
    #dataset = dataset.sort_values(by='Timestamp', ascennding= True)
    dataset = pd.read_pickle('./dataset_model2.pkl')
    dataset = dataset.set_index(['id1', 'id2'])
    target = new_connections_predictions(dataset)
    lst = [ ]
    [lst.append((u, v, p)) for u, v, p in dataset]
    predictions = {(k, v): n for k, v, n in lst}
    # pred = json.loads(predictions)

    with open('result.json', 'w') as fp:
        json.dump(predictions, fp)

    x = 1


if __name__ == '__main__':
    main()

    # all_dfs['Common Neighbors'] = all_dfs['id1'].map(lambda city: (list(nx.common_neighbors(graph, city[0], city[1]))))

    # predictions1 = nx.common_neighbors(g_undirected, g_undirected.edges())
    # df = pd.read_pickle("./dataset_model.pkl")
