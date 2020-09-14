import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append('../')
from create_graphs.create_networkx import __read_csv_files


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
    total_trades = total_moves_in_out.iloc[1::3, ::2]
    total_messages = total_moves_in_out.iloc[2::3, ::2]

    moves = {
        'Active attacks per user': attacks_active,
        'Active trades per user': trades_active,
        'Active messages per user': messages_active,
        'Passive attacks per user': attacks_passive,
        'Passive trades per user': trades_passive,
        'Passive messages per user': messages_passive,
        'Total Attacks': total_attacks,
        'Total Trades': total_trades,
        'Total Messages': total_messages
    }
    for hist_title, values in moves.items():
        hist_plot = values['degree'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.show()

    return total_attacks, total_trades, total_messages


def hist_plots(user_in_degree, user_out_degree, user_degree):
    dict_helper = {
        'Users degree Histogramm': user_degree,
        'User In-Degree Histogram': user_in_degree,
        'User out-Degree Histogram': user_out_degree
    }
    for hist_title, values in dict_helper.items():
        values_df = pd.DataFrame(user_degree, columns=['user', 'Value'])
        hist_plot = values_df['Value'].hist(bins=50)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
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
        'Users degree Histogramm': active_users,
        'User In-Degree Histogram': passive_users,
        'User out-Degree Histogram': users

    }
    for titles_of_plots, direction in type_of_users.items():
        objects = ['min', 'avg', 'max']
        y_pos = np.arange(len(objects))
        lst = sorted([float(i) for i in direction])

        plt.bar(y_pos, lst)
        plt.title(titles_of_plots)
        plt.xticks(y_pos, objects)
        plt.xlabel('metrics')
        plt.ylabel("count")
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


def main():
    G, g_undirected, all_dfs, labels = __read_csv_files()
    degree_measures_per_type(labels)
    total_attacks, total_trades, total_messages = degree_measures_per_type(labels)
    user_in_degree, user_out_degree, user_degree, best_active_user, best_passive_user, best_user, \
    unpopular_active_user, unpopular_passive_user, unpopular_user, avg_in_degree, avg_out_degree, \
    avg_degree = graph_degrees_measures(G, labels)
    hist_plots(user_in_degree, user_out_degree, user_degree)
    bar_plots(best_active_user, best_passive_user, best_user, unpopular_active_user, unpopular_passive_user,
              unpopular_user, avg_in_degree, avg_out_degree, avg_degree)
    jac_sim(total_attacks, total_trades, total_messages)


if __name__ == '__main__':
    main()
