import glob

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
    G = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                create_using=nx.DiGraph(name='Travian_Graph'))

    # pos = nx.spring_layout(G, k=10)
    # nx.draw(G, pos, with_labels=True)
    labels = {e: G.edges[e]['label'] for e in G.edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # G = nx.from_pandas_edgelist(edges, edge_attr=True)

    # plt.show()
    return G, all_dfs, labels


def link_prediction(G):
    # predictions = []
    predictions1 = nx.resource_allocation_index(G, G.edges())
    predictions2 = nx.jaccard_coefficient(G, G.edges())
    predictions3 = nx.adamic_adar_index(G, G.edges())
    predictions4 = nx.preferential_attachment(G, G.edges())
    # predictions.extend([predictions1, predictions2, predictions3, predictions4])
    lst = []
    try:
        for u, v, p in predictions1:
            lst.append((u, v, p))
            print('(%d, %d) -> %.8f' % (u, v, p))
    except ZeroDivisionError:
        print("ZeroDivisionError: float division by zero")
    x = 1


def important_characteristics_of_graph(G):
    print("Eccentricity: ", nx.eccentricity(G))
    print("Diameter: ", nx.diameter(G))
    print("Radius: ", nx.radius(G))
    print("Preiphery: ", list(nx.periphery(G)))
    print("Center: ", list(nx.center(G)))

    weakly_component = [G.subgraph(c).copy() for c in sorted(nx.weakly_connected_components(G))]

    largest_wcc = max(weakly_component)

    print(weakly_component)


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.4f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


def aggregated_dataset(all_dfs, g_undirected):
    aggregated_df = all_dfs.sort_values(by='Timestamp', ascending=True)
    aggregated_df = all_dfs.groupby(['id1', 'id2', 'type'], as_index=False)['weight'].sum()
    aggregated_df = aggregated_df.set_index(['id1', 'id2'])
    aggregated_df['preferential attachment'] = [i[2] for i in
                                                nx.preferential_attachment(g_undirected, aggregated_df.index)]
    aggregated_df['Common Neighbors'] = aggregated_df.index.map(
        lambda id: len(list(nx.common_neighbors(g_undirected, id[0], id[1]))))
    aggregated_df['label'] = 1
    return aggregated_df


def main():
    G, all_dfs, labels = read_csv_files()
    link_prediction(G)
    important_characteristics_of_graph(G)


if __name__ == '__main__':
    main()
