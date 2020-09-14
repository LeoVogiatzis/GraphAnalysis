import sys

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

sys.path.append('../')
from create_graphs.create_networkx import __read_csv_files


def measures_for_centrality(G):
    """
    Calculate the basic measures for centrality for the Graph
    :param G: Graph
    :return: 4 dictionaries which contain the id and and score per measure
    """
    deg_centrality = nx.degree_centrality(G)
    page_rank = nx.pagerank(G, alpha=0.8)
    bet_cen = nx.betweenness_centrality(G)
    close_cen = nx.closeness_centrality(G)
    dict_measures = {
        'Degree centrality Histogram': deg_centrality,
        'Pagerank Histogramm': page_rank,
        'Betwennes centrality ': bet_cen,
        'Closeness centrality': close_cen
    }
    for hist_title, values in dict_measures.items():
        values_df = pd.DataFrame(values.items(), columns=['id', 'Score'])
        hist_plot = values_df['Score'].hist(bins=50)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.show()


def main():
    G, g_undirected, all_dfs, labels = __read_csv_files()
    measures_for_centrality(G)

    # all_dfs.to_pickle("./aggregated_dflalala.pkl")


if __name__ == '__main__':
    main()
