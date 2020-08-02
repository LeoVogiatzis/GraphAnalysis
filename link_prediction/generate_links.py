import glob
import random

import networkx as nx
import numpy as np
import pandas as pd


def read_csv_files():
    """
    Read data and create MultiDiGraph.Each Node has an id and All edges have 2 attributes.
    The first is Timestamp and the second is the type of edge (Attacks, Trades, Messages)
    :return: G, all_dfs, labels
    """
    file_names = glob.glob("../data_users_moves/*.csv")

    all_dfs = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])
    # ommisible_links_data = pd.DataFrame(columns=['Timestamp', 'id1', 'id2', 'label'])

    for file in file_names:
        print('Currently using file - ', file)
        df = pd.read_csv(file, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='s')
        if 'attack' in file:
            rel_type = 'attacks'

        elif 'trade' in file:
            rel_type = 'trades'
        else:
            rel_type = 'messages'

        df['Label'] = 1
        df['Type'] = rel_type
        df['weight'] = 1
        all_dfs = pd.concat([all_dfs, df])
        break
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
    #   print(nx.non_edges(G))
    ommisible_links_data = pd.DataFrame(nx.non_edges(G)).sample(frac=1).reset_index(drop=True)
    dates = pd.date_range('2019-01-01 00:00:00', '2019-01-30 23:59:59', periods=200000)
    gen_df = ommisible_links_data.iloc[:200000, :]
    gen_df.columns = ['id1', 'id2']
    gen_df[['id1', 'id2']] = gen_df[['id1', 'id2']].applymap(np.int64)
    gen_df['Timestamp'] = dates
    gen_df['Label'] = 0
    gen_df['weight'] = 1
    gen_df['Type'] = random.choices(['attacks', 'messages', 'trades'], weights=(50, 25, 25), k=200000)
    x = pd.concat([all_dfs, gen_df])
    print(gen_df)
    # while len(gen_df) <= 200000:
    #     x=1

    #     if(row_count%2 == 0 and row_count%4 == 0):
    #         pass
    #     else:
    #         pass

    # for index, value in dates.iterrows():
    #     pass

    # for index, row in all_dfs.iterrows():
    return G, all_dfs


def main():
    graph, all_dfs = read_csv_files()


if __name__ == '__main__':
    main()
