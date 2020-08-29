import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from py2neo import Graph
from sklearn.metrics import jaccard_score


def read_from_neo4j_database():
    """
    connection to database
    :return: graph
    """
    graph = Graph('127.0.0.1', password='gomugomuno13')
    return graph


def count_active_relationships_for_user(rel_type, graph):
    """

    :param rel_type:
    :param graph:
    :return: out_degrees
    """
    result = graph.run("MATCH (u:User)-[r:" + rel_type + "]->() RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type).to_data_frame()

    return result


def count_passive_relationships_for_user(rel_type, graph):
    """
    :return: in-degrees
    """
    result = graph.run("MATCH ()-[r:" + rel_type + "]->(u:User) RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type).to_data_frame()

    return result


def count_pas_active_relations(rel_type, graph):
    """
    :return: relationship per user regardless whether is active or passive
    """
    result = graph.run("MATCH ()-[r:" + rel_type + "]-(u:User) RETURN u.id as userId, count(r) as relationshipCount",
                       rel_type=rel_type).to_data_frame()

    return result


def total_interactions_per_user(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df,
                                total_attacks, total_trades, total_messages):
    """
    1)Concatenate all the df's which contain the user moves gathering the in-degrees/out-degrees together.
    2)Find min,max,avg per total dataframes
    """
    # sum per active user
    active_users = pd.concat([attacks, trades, messages])
    active_sums = active_users.groupby('userId').sum().reset_index()
    # sum per passive user
    passive_users = pd.concat([attacks_p_df, trades_p_df, messages_p_df])
    passive_sums = passive_users.groupby('userId').sum().reset_index()
    max_total_attacks = total_attacks.loc[attacks['relationshipCount'].idxmax()][1]
    min_total_attacks = total_attacks.loc[attacks['relationshipCount'].idxmin()][1]
    avg_total_attacks = total_attacks.stack().mean()
    max_total_trades = total_trades.loc[trades['relationshipCount'].idxmax()][1]
    min_total_trades = total_trades.loc[trades['relationshipCount'].idxmin()][1]
    avg_total_trades = total_trades.stack().mean()
    max_total_messages = total_messages.loc[total_messages['relationshipCount'].idxmax()][1]
    min_total_messages = total_messages.loc[total_messages['relationshipCount'].idxmin()][1]
    avg_total_messages = total_messages.stack().mean()
    return active_sums, passive_sums, max_total_attacks, min_total_attacks, avg_total_attacks, max_total_trades, \
           min_total_trades, avg_total_trades, max_total_messages, min_total_messages, avg_total_messages


def max_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    """
    Find max for each df od edges
    """
    max_attacks = attacks.loc[attacks['relationshipCount'].idxmax()][1]
    max_trades = trades.loc[trades['relationshipCount'].idxmax()][1]
    max_messages = messages.loc[messages['relationshipCount'].idxmax()][1]
    max_attacks_p_df = attacks_p_df.loc[attacks_p_df['relationshipCount'].idxmax()][1]
    max_trades_p_df = trades_p_df.loc[trades_p_df['relationshipCount'].idxmax()][1]
    max_messages_p_df = messages_p_df.loc[messages_p_df['relationshipCount'].idxmax()][1]

    return max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df


def min_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    """
    Find min for each df od edges
    """
    min_attacks = attacks_p_df.loc[attacks_p_df['relationshipCount'].idxmin()][1]
    min_trades = trades.loc[trades['relationshipCount'].idxmin()][1]
    min_messages = messages.loc[messages['relationshipCount'].idxmin()][1]
    min_attacks_p_df = attacks_p_df.loc[attacks['relationshipCount'].idxmin()][1]
    min_trades_p_df = trades_p_df.loc[trades_p_df['relationshipCount'].idxmin()][1]
    min_messages_p_df = messages_p_df.loc[messages_p_df['relationshipCount'].idxmin()][1]

    return min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df


def average_metrics_for_users(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df):
    """
    Find the average for each user according to the type od edge
    """
    average_at = attacks.stack().mean()
    average_tr = trades.stack().mean()
    average_mes = messages.stack().mean()
    average_pas_at = attacks_p_df.stack().mean()
    average_pas_tr = trades_p_df.stack().mean()
    average_pas_mes = messages_p_df.stack().mean()

    return average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes


def act_pass_metrics(active_sums, passive_sums):
    """
    Find min, max, average regardless the the type of edge.
    """
    max_sum = active_sums.loc[active_sums['relationshipCount'].idxmax()][1]
    min_sum = active_sums.loc[active_sums['relationshipCount'].idxmax()][1]
    average_sum = active_sums.stack().mean()
    max_sum_passive = passive_sums.loc[passive_sums['relationshipCount'].idxmax()][1]
    min_sum_passive = passive_sums.loc[passive_sums['relationshipCount'].idxmax()][1]
    average_sum_passive = active_sums.stack().mean()

    return max_sum, min_sum, average_sum, max_sum_passive, min_sum_passive, average_sum_passive


def in_out_degree(total_attacks, total_trades, total_messages):
    """
    Find the min, max , average, for the users regardless the direction
    :return:
    """
    max_attacks = total_attacks.loc[total_attacks['relationshipCount'].idxmax()][1]
    min_attacks = total_attacks.loc[total_attacks['relationshipCount'].idxmin()][1]
    average_attacks = total_attacks.stack().mean()
    max_trades = total_attacks.loc[total_trades['relationshipCount'].idxmax()][1]
    min_trades = total_attacks.loc[total_trades['relationshipCount'].idxmin()][1]
    average_trades = total_trades.stack().mean()
    max_messages = total_attacks.loc[total_messages['relationshipCount'].idxmax()][1]
    min_messages = total_attacks.loc[total_messages['relationshipCount'].idxmin()][1]
    average_messages = total_messages.stack().mean()

    return max_attacks, min_attacks, average_attacks, max_trades, min_trades, average_trades, max_messages, \
           min_messages, average_messages


def plots(attacks, trades, messages, attacks_p_df, trades_p_df, messages_p_df, active_sums, passive_sums, total_attacks,
          total_trades, total_messages):
    """
    hists for all the usefull dataframes of dataset,
    timely degree per user according to the type of edge
    and regardless the direction or not.
    """
    dict_helper = {
        'Active Attacks Histogram': attacks,
        'Active Trades Histogram': trades,
        'Active Histogram': messages,
        'Passive Messages Histogram': attacks_p_df,
        'Passive Messages Histogram': trades_p_df,
        'Passive Messages Histogram': messages_p_df,
        'Active Sums Histogram': active_sums,
        'Passive sums Histogram': passive_sums,
        'In-out attacks per user': total_attacks,
        'In-out  per user': total_trades,
        'In-out messages per user': total_messages
    }

    for hist_title, values in dict_helper.items():
        hist_plot = values['relationshipCount'].hist(bins=1000)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.show()


def plots_for_min_max_avg(max_attacks, max_trades, max_messages, max_attacks_p_df, max_trades_p_df, max_messages_p_df,
                          min_attacks, min_trades, min_messages, min_attacks_p_df, min_trades_p_df, min_messages_p_df,
                          average_at, average_tr, average_mes, average_pas_at, average_pas_tr, average_pas_mes,
                          max_sum, min_sum, average_sum, max_sum_passive, min_sum_passive, average_sum_passive,
                          max_total_attacks, min_total_attacks, avg_total_attacks, max_total_trades, min_total_trades,
                          avg_total_trades, max_total_messages, min_total_messages, avg_total_messages):
    """
    Getting all the usefull metrics(min, max, average)
    /per type of edge regardless or not the direction
    :return: bar plots for min max average
    """
    attacks_active = []
    attacks_active.extend([max_attacks, min_attacks, average_at])
    attacks_passive = []
    attacks_passive.extend([max_attacks_p_df, min_attacks_p_df, average_pas_at])
    trades_active = []
    trades_active.extend([max_trades, min_trades, average_tr])
    trades_passive = []
    trades_passive.extend([max_trades_p_df, min_trades_p_df, average_pas_tr])
    messages_active = []
    messages_active.extend([max_messages, min_messages, average_mes])
    messages_passive = []
    messages_passive.extend([max_messages_p_df, min_messages_p_df, average_pas_mes])
    active_sums = []
    active_sums.extend([max_sum, min_sum, average_sum])
    passive_sums = []
    passive_sums.extend([max_sum_passive, min_sum_passive, average_sum_passive])
    total_interactions_attacks = []
    total_interactions_trades = []
    total_interactions_messages = []
    total_interactions_attacks.extend([max_total_attacks, min_total_attacks, avg_total_attacks])
    total_interactions_trades.extend([max_total_trades, min_total_trades, avg_total_trades])
    total_interactions_messages.extend([max_total_messages, min_total_messages, avg_total_messages])
    type_of_users = {
        "Attacks bar plot": attacks_active,
        "Trades bar plot": attacks_passive,
        "Messages bar plot": trades_active,
        "Attacks passive bar plot": trades_passive,
        "Trades passive bar plot": messages_active,
        "Messages passive bar plot": messages_passive,
        "Active sums plot": active_sums,
        "Passive sums bar plot": passive_sums,
        "Total attacks bar plot": total_interactions_attacks,
        "Total trades bar plot": total_interactions_trades,
        "Total messages bar plot": total_interactions_messages
    }
    for titles_of_plots, bar_plot in type_of_users.items():
        objects = ['min', 'avg', 'max']
        y_pos = np.arange(len(objects))
        lst = sorted([float(i) for i in bar_plot])

        plt.bar(y_pos, lst)
        plt.title(titles_of_plots)
        plt.xticks(y_pos, objects)
        plt.xlabel('metrics')
        plt.ylabel("count")
        plt.show()


def jac_simmilarity(total_attacks, total_trades, total_messages):
    """
    Simmilarity between type of edges regardless the direction
    :param total_attacks:
    :param total_trades:
    :param total_messages: jaccard score and bar plots
    :return:
    """
    best_in_attacks = set(total_attacks.nlargest(100, 'relationshipCount')['userId'].to_numpy())
    best_in_trades = set(total_messages.nlargest(100, 'relationshipCount')['userId'].to_numpy())
    best_in_messages = set(total_trades.nlargest(100, 'relationshipCount')['userId'].to_numpy())

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

    print(jac_attacks_messages, jac_attacks_trades, jac_messages_trades)

