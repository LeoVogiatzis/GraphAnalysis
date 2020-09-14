import glob

import networkx as nx
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
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

        all_dfs = pd.concat([all_dfs, df], ignore_index= True)
    G = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                create_using=nx.DiGraph(name='Travian_Graph'))
    x=1
    # nx.set_node_attributes(G, comms, 'community')
    # pos = nx.spring_layout(G, k=10)
    # nx.draw(G, pos, with_labels=True)
    labels = {e: G.edges[e]['label'] for e in G.edges}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # G = nx.from_pandas_edgelist(edges, edge_attr=True)
    g_undirected = nx.from_pandas_edgelist(df=all_dfs, source='id1', target='id2', edge_attr=True,
                                           create_using=nx.Graph(name='Travian_Graph'))
    # plt.show()
    return G, all_dfs, labels, g_undirected


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


def important_characteristics_of_graph(g_undirected):
    #diameter = nx.diameter(G.to_undirected())
    #print("Eccentricity: ",  nx.eccentricity(G))
    #print("Eccentricity: ", {k: v for (k, v) in nx.eccentricity(G).items()})
    #json.dump({k: v for (k, v) in nx.eccentricity(G).items()}, open("text.txt", 'w'))
    #d2 = json.load(open("text.txt"))
    #print(d2)
    #print("Diameter: ", nx.diameter(G))
    # print("Radius: ", nx.radius(G))
    # print("Preiphery: ", list(nx.periphery(G)))
    # print("Center: ", list(nx.center(G)))

    nx.draw_networkx(g_undirected, with_labels=True, node_color='green')

    # returns True or False whether Graph is connected
    print(nx.is_connected(g_undirected))

    # returns number of different connected components
    print(nx.number_connected_components(g_undirected))

    # returns list of nodes in different connected components
    print(list(nx.connected_components(g_undirected)))

    # returns number of nodes to be removed
    # so that Graph becomes disconnected
    print(nx.node_connectivity(g_undirected))

    # returns number of edges to be removed
    # so that Graph becomes disconnected
    print(nx.edge_connectivity(g_undirected))

    print(nx.average_shortest_path_length(g_undirected))
    # returns average of shortest paths between all possible pairs of nodes

    weakly_component = [g_undirected.subgraph(c).copy() for c in sorted(nx.weakly_connected_components(g_undirected))]

    largest_wcc = max(weakly_component)

    print(weakly_component)
    x=1


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


def visi(G):
    counts = pd.DataFrame(G.degree(), columns=['nodes', 'degrees'])
    print(nx.is_strongly_connected(G))
    print(nx.is_weakly_connected(G))
    plt.style.use('fivethirtyeight')
    ax = counts[:100].plot(kind='bar', x='nodes', y='degrees', legend=None, figsize=(15, 8))
    ax.xaxis.set_label_text("")
    plt.tight_layout()
    plt.show()
    # nx.draw_circular(G, with_labels=True)
    # plt.show()

    # options = {
    #     'node_color': 'black',
    #     'node_size': '10',
    #     'line_color': 'grey',
    #     'linewidths': 0,
    #     'width': 0.1
    # }
    # nx.draw(G, **options)
    # plt.show(c='r')

    # degree_sequence = sorted([d for n, d in g_undirected.degree()], reverse=True)
    # dmax = max(degree_sequence)
    #
    # plt.loglog(degree_sequence, "b-", marker="o")
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    #
    # # draw graph in inset
    # plt.axes([0.45, 0.45, 0.45, 0.45])
    # Gcc = g_undirected.subgraph(sorted(nx.connected_components(g_undirected), key=len, reverse=True)[0])
    # pos = nx.spring_layout(Gcc)
    # plt.axis("off")
    # nx.draw_networkx_nodes(Gcc, pos, node_size=10)
    # nx.draw_networkx_edges(Gcc, pos, alpha=0.4)
    # plt.savefig('fig.png',bbox_inches='tight')


def main():
    G, all_dfs, labels, g_undirected = read_csv_files()
    # N, K = G.order(), G.size()
    # avg_deg = float(K) / N
    # print("Nodes: ", N)
    # print("Edges: ", K)
    # print("Average degree: ", avg_deg)
    important_characteristics_of_graph(g_undirected)
    #link_prediction(G)


if __name__ == '__main__':
    main()
