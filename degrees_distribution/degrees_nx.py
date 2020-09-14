import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from create_graphs.create_networkx import __read_csv_files

import networkx as nx
sys.path.append('../')

import dash
import dash_html_components as html
import dash_core_components as dcc

from plotly.tools import mpl_to_plotly
from matplotlib.pyplot import subplot, figure
from scipy import stats


from collections import Counter
from operator import itemgetter


def graph_degrees_measures(G, labels):
    """
    find min max
    :param G:
    :param labels:
    find the most popular user(max),user with the fewest moves,
     as well as the average value,for each category of edge and direction
    :return:
    """
    user_in_degree = list(G.in_degree())
    user_out_degree = list(G.out_degree())
    user_degree = list(G.degree())
    # attacks = G.edges()

    best_active_user = max(user_in_degree, key=lambda user_in_degree: user_in_degree[1])
    best_passive_user = max(user_out_degree, key=lambda user_out_degree: user_out_degree[1])
    best_user = max(user_degree, key=lambda user_degree: user_degree[1])

    unpopular_active_user = min(user_in_degree, key=lambda user_in_degree: user_in_degree[1])
    unpopular_passive_user = min(user_out_degree, key=lambda user_out_degree: user_out_degree[1])
    unpopular_user = min(user_degree, key=lambda user_degree: user_degree[1])

    avg_in_degree = np.mean([i[1] for i in user_in_degree])
    avg_out_degree = np.mean([i[1] for i in user_out_degree])
    avg_degree = np.mean([i[1] for i in user_degree])

    return user_in_degree, user_out_degree, user_degree, best_active_user, best_passive_user, best_user, \
           unpopular_active_user, unpopular_passive_user, unpopular_user, avg_in_degree, avg_out_degree, avg_degree


def degree_measures_per_type(labels):
    """
    :param labels:dict which include all the information about the nodes and edges of the graph,
    namely the degree of user according to type of edge(attacks, trades, messages) and the direction.
    Convert to dataframe, split it to count all degrees and display the hists per user/degree
    :return:total_attacks, total_trades, total_messages
    """
    moves = {}
    for __edge, attribute in labels.items():
        if __edge[0] not in moves:
            moves[__edge[0]] = {
                'attacks': {'first_position': 0, 'second_position': 0},
                'trades': {'first_position': 0, 'second_position': 0},
                'messages': {'first_position': 0, 'second_position': 0}
            }
        if __edge[1] not in moves:
            moves[__edge[1]] = {
                'attacks': {'first_position': 0, 'second_position': 0},
                'trades': {'first_position': 0, 'second_position': 0},
                'messages': {'first_position': 0, 'second_position': 0}
            }
        moves[__edge[0]][attribute]['first_position'] += 1
        moves[__edge[1]][attribute]['second_position'] += 1

    moves_df = pd.DataFrame.from_records(
        [
            (level1, level2, level3, leaf)
            for level1, level2_dict in moves.items()
            for level2, level3_dict in level2_dict.items()
            for level3, leaf in level3_dict.items()
        ],
        columns=['UserId', 'Category of edge', 'IN/OUT', 'degree']
    )
    x = 1
    moves.clear()
    attacks_active = moves_df.iloc[::6, ::3]
    trades_active = moves_df.iloc[1::6, ::3]
    messages_active = moves_df.iloc[2::6, ::3]
    attacks_passive = moves_df.iloc[5::6, ::3]
    trades_passive = moves_df.iloc[4::6, ::3]
    messages_passive = moves_df.iloc[5::6, ::3]


    total_moves_in_out = moves_df.iloc[:, [0, 1, 3]].groupby(['UserId', 'Category of edge']).sum().reset_index()
    total_attacks = total_moves_in_out.iloc[::3, ::2]
    total_messages = total_moves_in_out.iloc[1::3, ::2]
    total_trades = total_moves_in_out.iloc[2::3, ::2]

    moves = {
        'Active attacks': attacks_active,
        'Active trades': trades_active,
        'Active messages': messages_active,
        'Passive attacks': attacks_passive,
        'Passive trades': trades_passive,
        'Passive messages': messages_passive,
        'Attacks': total_attacks,
        'Trades': total_trades,
        'Messages': total_messages
    }

    count = 0
    for hist_title, values in moves.items():
        count += 1
        subplot(2, 3, count)
        hist_plot = values['degree'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('degree')
        hist_plot.set_ylabel("Number of nodes")
        plt.margins(x=0)
        plt.yscale('log', basey=10)
        #plt.tick_params(axis='x', labelsize=5)
    plt.show()

    return total_attacks, total_trades, total_messages


def hist_plots(user_in_degree, user_out_degree, user_degree):
    dict_helper = {
        'Users degree': user_degree,
        'User In-Degree': user_in_degree,
        'User out-Degree': user_out_degree
    }

    count = 0
    for hist_title, values in dict_helper.items():
        count+=1
        subplot(2, 3, count)
        values_df = pd.DataFrame(user_degree, columns=['user', 'Value'])
        hist_plot = values_df['Value'].hist(bins=50)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('degree')
        hist_plot.set_ylabel("Number of nodes")

        plt.margins(x=0)
        plt.yscale('log', basey=10)
    plt.show()


def bar_plots(best_active_user, best_passive_user, best_user, unpopular_active_user, unpopular_passive_user,
              unpopular_user, avg_in_degree, avg_out_degree, avg_degree):
    """
    :param best_active_user:
    :param best_passive_user:
    :param best_user:
    :param unpopular_active_user:
    :param unpopular_passive_user:
    :param unpopular_user:
    :param avg_in_degree:
    :param avg_out_degree:
    :param avg_degree:
    :return: display the 3 plots for the most popular user(max),user with the fewest moves,
     as well as the average value,according to direction which the move is granted
    """
    active_users = []
    passive_users = []
    users = []
    active_users.extend([best_active_user[1], unpopular_active_user[1], avg_in_degree])
    passive_users.extend([best_passive_user[1], avg_out_degree, unpopular_passive_user[1]])
    users.extend([best_user[1], unpopular_user[1], avg_degree])
    type_of_users = {
        'Users degree': active_users,
        'Users In-Degree': passive_users,
        'User out-Degree': users

    }
    count = 0
    for titles_of_plots, direction in type_of_users.items():
        count += 1
        subplot(2, 3, count)
        objects = ['min', 'avg', 'max']
        y_pos = np.arange(len(objects))
        lst = sorted([float(i) for i in direction])

        plt.bar(y_pos, lst)
        plt.title(titles_of_plots)
        plt.xticks(y_pos, objects)
        plt.xlabel('metrics')
        plt.ylabel("count")
        plt.margins(x=0)
        plt.yscale('log', basey=10)
    plt.show()


def jac_sim(total_attacks, total_trades, total_messages):
    best_in_attacks = set(total_attacks.nlargest(100, 'degree')['UserId'].to_numpy())
    best_in_trades = set(total_messages.nlargest(100, 'degree')['UserId'].to_numpy())
    best_in_messages = set(total_trades.nlargest(100, 'degree')['UserId'].to_numpy())

    jac_attacks_trades = len(set(best_in_attacks & best_in_trades)) / len(set(best_in_attacks | best_in_trades))
    jac_attacks_messages = len(set(best_in_attacks & best_in_messages)) / len(set(best_in_attacks | best_in_messages))
    jac_messages_trades = len(set(best_in_trades & best_in_messages)) / len(set(best_in_trades | best_in_messages))
    all_sim = []
    all_sim.extend([jac_attacks_messages, jac_attacks_trades, jac_messages_trades])

    objects = ['att-tra', 'att-mess', 'mess-tra']
    y_pos = np.arange(len(objects))
    lst = sorted([float(i) for i in all_sim])
    plt.bar(y_pos, lst)
    plt.title('Jaccard Simmilarities between type of relationhship')
    plt.xticks(y_pos, objects)
    plt.xlabel('metrics')
    plt.ylabel("count")
    plt.show()


def in_out_distribution(g_directed):
    # dictionary node:degree
    in_degrees = {i: j for (i, j) in g_directed.in_degree()}
    in_values = sorted(set(in_degrees.values()))
    in_hist = [list(in_degrees.values()).count(x) for x in in_values]
    # dictionary node:degree
    out_degrees = {i: j for (i, j) in g_directed.out_degree()}
    out_values = sorted(set(out_degrees.values()))
    out_hist = [list(out_degrees.values()).count(x) for x in out_degrees]

    in_val = np.array(in_values)[:200]
    in_h = np.array(in_hist)[:200]

    out_val = np.array(out_values)[:200]
    out_h = np.array(out_hist)[:200]


    plt.plot(in_val, in_h, 'ro-')  # in-degree
    plt.plot(out_val, out_h, 'bv-')  # out-degree
    plt.legend(['In-degree', 'Out-degree'])
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    plt.title('Hartford drug users network')
    plt.savefig('Degrees_comparison_in_out')
    plt.close()


def degree_histogram_directed(g_directed, in_degree=False, out_degree=False):
    """Return a list of the frequency of each degree value.

    Parameters
    ----------
    G : Networkx graph
       A graph
    in_degree : bool
    out_degree : bool

    Returns
    -------
    hist : list
       A list of frequencies of degrees.
       The degree values are the index in the list.

    Notes
    -----
    Note: the bins are width one, hence len(list) can be large
    (Order(number_of_edges))
    """
    nodes = g_directed.nodes()
    if in_degree:
        in_degree = dict(g_directed.in_degree())
        degseq=[in_degree.get(k,0) for k in nodes]
    elif out_degree:
        out_degree = dict(g_directed.out_degree())
        degseq=[out_degree.get(k,0) for k in nodes]
    else:
        degseq=[v for k, v in g_directed.degree()]
    dmax=max(degseq)+1
    freq= [ 0 for d in range(dmax) ]
    for d in degseq:
        freq[d] += 1
    return freq


def main():
    g_directed, g_undirected, all_dfs, labels = __read_csv_files()

    bet = edge_betweenness_centrality(g_directed)
    degree_measures_per_type(labels)
    in_out_distribution(g_directed)
    # diff_visualizations(g_directed)

    total_attacks, total_trades, total_messages = degree_measures_per_type(labels)

    user_in_degree, user_out_degree, user_degree, best_active_user, best_passive_user, best_user, \
    unpopular_active_user, unpopular_passive_user, unpopular_user, avg_in_degree, avg_out_degree, \
    avg_degree = graph_degrees_measures(g_directed, labels)

    hist_plots(user_in_degree, user_out_degree, user_degree)

    bar_plots(best_active_user, best_passive_user, best_user, unpopular_active_user, unpopular_passive_user,
              unpopular_user, avg_in_degree, avg_out_degree, avg_degree)

    jac_sim(total_attacks, total_trades, total_messages)


if __name__ == '__main__':
    main()
