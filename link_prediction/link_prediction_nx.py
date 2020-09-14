import glob

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def read_csv_files():
    """
    Read data and create MultiDiGraph.Each Node has an id and All edges have 2 attributes.
    The first is Timestamp and the second is the type of edge (Attacks, Trades, Messages)
    :return: G, all_dfs, labels
    """
    file_names = glob.glob("../data_users_moves/*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for file in file_names:
        print('Currently using file - ', file)

        df = pd.read_csv(file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df['date'] = [d.date() for d in df['Timestamp']]
        df['time'] = [d.time() for d in df['Timestamp']]

        if 'attack' in file:
            rel_type = 'attacks'

        elif 'trade' in file:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['label'] = rel_type

        all_dfs = pd.concat([all_dfs, df])
    G = nx.from_pandas_edgelist(all_dfs, 'id1', 'id2', edge_attr=True,
                                create_using=nx.MultiDiGraph(name='Travian_Graph'))
    source = all_dfs['id1'].tolist()
    destination = all_dfs['id2'].tolist()
    # combine all nodes in a list
    node_list = source + destination
    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))
    adj_G = nx.to_numpy_matrix(G, nodelist=node_list)
    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if (i in source) and (j in destination) and ((i, j) in G.edges):
                if i != j:
                    if nx.shortest_path_length(G, int(i), int(j)) <= 2:
                        if adj_G[i, j] == 0:
                            all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    print(len(all_unconnected_pairs))
    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0
    initial_node_count = len(G.nodes)

    all_dfs_temp = all_dfs.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(all_dfs.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(all_dfs_temp.drop(index=i), "id1", "id2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            all_dfs_temp = all_dfs_temp.drop(index=i)

    # create dataframe of removable edges
    travian_df_ghost = all_dfs.loc[omissible_links_index]

    # add the target variable 'link'
    travian_df_ghost['link'] = 1

    data = data.append(travian_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    print(data['link'].value_counts())

    # drop removable edges
    travian_df_partial = all_dfs.drop(index=travian_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(travian_df_partial, "node_1", "node_2", create_using=nx.Graph())

    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                    test_size=0.3,
                                                    random_state=35)
    lr = LogisticRegression(class_weight="balanced")

    lr.fit(xtrain, ytrain)

    predictions = lr.predict_proba(xtest)

    roc_auc_score(ytest, predictions[:, 1])

    print(roc_auc_score)
    #
    # plt.figure(figsize=(10, 10))
    #
    # pos = nx.random_layout(G, seed=23)
    # nx.draw(G, with_labels=False, pos=pos, node_size=40, alpha=0.6, width=0.7)
    #
    # plt.show()
    #
    # pos = nx.spring_layout(G, k=10)
    # nx.draw(G, pos, with_labels=True)
    labels = {e: G.edges[e]['label'] for e in G.edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # G = nx.from_pandas_edgelist(edges, edge_attr=True)

    # plt.show()

    # build adjacency matrix
    adj_G = nx.to_numpy_matrix(G, nodelist=node_list)
    return G, all_dfs, labels


def read_csv_files():
    """
    Read data and create MultiDiGraph.Each Node has an id and All edges have 2 attributes.
    The first is Timestamp and the second is the type of edge (Attacks, Trades, Messages)
    :return: G, all_dfs, labels
    """
    file_names = glob.glob("../data_users_moves/*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for file in file_names:
        print('Currently using file - ', file)

        df = pd.read_csv(file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        df['date'] = [d.date() for d in df['Timestamp']]
        df['time'] = [d.time() for d in df['Timestamp']]

        if 'attack' in file:
            rel_type = 'attacks'

        elif 'trade' in file:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['label'] = rel_type

        all_dfs = pd.concat([all_dfs, df])
    G = nx.from_pandas_edgelist(all_dfs, 'id1', 'id2', edge_attr=True,
                                create_using=nx.MultiDiGraph(name='Travian_Graph'))
    source = all_dfs['id1'].tolist()
    destination = all_dfs['id2'].tolist()
    # combine all nodes in a list
    node_list = source + destination
    # remove duplicate items from the list
    node_list = list(dict.fromkeys(node_list))
    adj_G = nx.to_numpy_matrix(G, nodelist=node_list)
    # get unconnected node-pairs
    all_unconnected_pairs = []

    # traverse adjacency matrix
    offset = 0
    for i in tqdm(range(adj_G.shape[0])):
        for j in range(offset, adj_G.shape[1]):
            if (i in source) and (j in destination) and ((i, j) in G.edges):
                if i != j:
                    if nx.shortest_path_length(G, int(i), int(j)) <= 2:
                        if adj_G[i, j] == 0:
                            all_unconnected_pairs.append([node_list[i], node_list[j]])

        offset = offset + 1

    print(len(all_unconnected_pairs))
    node_1_unlinked = [i[0] for i in all_unconnected_pairs]
    node_2_unlinked = [i[1] for i in all_unconnected_pairs]

    data = pd.DataFrame({'node_1': node_1_unlinked,
                         'node_2': node_2_unlinked})

    # add target variable 'link'
    data['link'] = 0
    initial_node_count = len(G.nodes)

    all_dfs_temp = all_dfs.copy()

    # empty list to store removable links
    omissible_links_index = []

    for i in tqdm(all_dfs.index.values):

        # remove a node pair and build a new graph
        G_temp = nx.from_pandas_edgelist(all_dfs_temp.drop(index=i), "id1", "id2", create_using=nx.Graph())

        # check there is no spliting of graph and number of nodes is same
        if (nx.number_connected_components(G_temp) == 1) and (len(G_temp.nodes) == initial_node_count):
            omissible_links_index.append(i)
            all_dfs_temp = all_dfs_temp.drop(index=i)

    # create dataframe of removable edges
    travian_df_ghost = all_dfs.loc[omissible_links_index]

    # add the target variable 'link'
    travian_df_ghost['link'] = 1

    data = data.append(travian_df_ghost[['node_1', 'node_2', 'link']], ignore_index=True)

    print(data['link'].value_counts())

    # drop removable edges
    travian_df_partial = all_dfs.drop(index=travian_df_ghost.index.values)

    # build graph
    G_data = nx.from_pandas_edgelist(travian_df_partial, "node_1", "node_2", create_using=nx.Graph())



    xtrain, xtest, ytrain, ytest = train_test_split(np.array(x), data['link'],
                                                    test_size=0.3,
                                                    random_state=35)
    lr = LogisticRegression(class_weight="balanced")

    lr.fit(xtrain, ytrain)

    predictions = lr.predict_proba(xtest)

    roc_auc_score(ytest, predictions[:, 1])

    print(roc_auc_score)


def main():
    graph, all_dfs, labels = read_csv_files()
    dataset = pd.read_pickle('./dataset_model2.pkl')


if __name__ == '__main__':
    main()
