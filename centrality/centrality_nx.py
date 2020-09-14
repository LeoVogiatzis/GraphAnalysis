import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.append('../')
from create_graphs.create_networkx import __read_csv_files
from matplotlib.pyplot import subplot, figure


def measures_for_centrality(g_undirected):
    """
    Calculate the basic measures for centrality for the Graph
    :param G: Graph
    :return: 4 dictionaries which contain the id and and score per measure
    """

    deg_centrality = nx.degree_centrality(g_undirected)
    deg_in_centrality = nx.in_degree_centrality(g_undirected)
    deg_out_centrality = nx.out_degree_centrality(g_undirected)

    #
    # user_out_degree = g_undirected.in_degree()
    # user_out_degree = g_undirected.out_degree()
    # user_deg = g_undirected.degree()

    # df = pd.DataFrame([user_out_degree, user_out_degree, user_deg]).T
    # df.columns = ['d{}'.format(i) for i, col in enumerate(df, 1)]
    #
    # df.to_pickle('./In-degree.pkl')

    #df2 = pd.read_pickle('./In-degree.pkl')
    page_rank = nx.pagerank(g_undirected, alpha=0.8)
    bet_cen = nx.betweenness_centrality(g_undirected)
    close_cen = nx.closeness_centrality(g_undirected)
    dict_measures = {
        'Degree centrality ': deg_centrality,
        'Pagerank': page_rank,
        'Betweeness ': bet_cen,
        'Closeness': close_cen
    }
    count = 0
    for hist_title, values in dict_measures.items():
        count += 1
        subplot(2, 3, count)
        values_df = pd.DataFrame(values.items(), columns=['id', 'Score'])
        hist_plot = values_df['Score'].hist(bins=50)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score')
        hist_plot.set_ylabel("Number of nodes")
        plt.margins(x=0)
        plt.yscale('log', basey=10)
        #plt.yscale('log', basey=10)
    plt.show()


def main():
    g_directed, g_undirected, all_dfs, labels = __read_csv_files()
    deg_centrality = nx.degree_centrality(g_directed)
    deg_in_centrality = nx.in_degree_centrality(g_directed)
    deg_out_centrality = nx.out_degree_centrality(g_directed)

    dict_measures = {
        'degree centrality': deg_centrality,
        'deg_in_centrality': deg_in_centrality,
        'deg_out_centrality': deg_out_centrality
    }

    count = 0
    for hist_title, values in dict_measures.items():
        count += 1
        subplot(2, 3, count)
        values_df = pd.DataFrame(values.items(), columns=['id', 'Score'])
        hist_plot = values_df['Score'].hist(bins=50)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score')
        hist_plot.set_ylabel("Number of nodes")
        plt.margins(x=0)
        plt.yscale('log', basey=10)
        # plt.yscale('log', basey=10)
    plt.show()
    measures_for_centrality(g_undirected)
    x=1


if __name__ == '__main__':
    main()
