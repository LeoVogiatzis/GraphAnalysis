import glob

import networkx as nx
import pandas as pd


def __read_csv_files():
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

    # pos = nx.spring_layout(G, k=10)
    # nx.draw(G, pos, with_labels=True)
    labels = {e: g_directed.edges[e]['label'] for e in g_directed.edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # G = nx.from_pandas_edgelist(edges, edge_attr=True)
    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                           create_using=nx.Graph(name='Travian_Graph'))
    # plt.show()
    return g_directed, g_undirected, all_dfs, labels
