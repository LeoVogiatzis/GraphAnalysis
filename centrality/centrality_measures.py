from py2neo import Graph
import json
import pandas as pd
from pandas import DataFrame
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot, figure
import numpy as np


def algo_degree(rel_type, graph):
    q = graph.run("CALL algo.degree.stream('User', '%s',{concurrency:4})"
                  "YIELD nodeId, score RETURN algo.asNode(nodeId).id AS name, score AS degree" % rel_type,
                  rel_type=rel_type).to_data_frame()
    return q


def betweenness_centrality(rel_type, graph):
    r1 = graph.run(
        "CALL algo.betweenness.stream('User','%s',{direction:'out'}) YIELD nodeId, centrality MATCH (user:User) WHERE id(user) = nodeId RETURN user.id AS user,centrality ORDER BY centrality DESC;" % rel_type, rel_type=rel_type).to_data_frame()
    r2 = graph.run(
        "CALL algo.betweenness.sampled.stream('User','%s',{strategy:'random', probability:1.0, maxDepth:1, direction: 'out'}) YIELD nodeId, centrality" % rel_type,
        rel_type=rel_type).to_data_frame()

    return r1, r2


def closeness(graph):
    closeness_centrality = graph.run(
        "CALL algo.closeness.stream('MATCH (p:User) RETURN id(p) as id','MATCH (p1:User)-[r]-(p2:User) RETURN id(p1) as source, id(p2) as target',{graph:'cypher'})YIELD nodeId, centrality").to_data_frame()
    # graph.run("")
    return closeness_centrality


def degree_centrality(rel_type, graph):
    """
    :param rel_type:
    :param graph:
    :return: dict with active relationships
    """
    # PageRank
    if rel_type == "ATTACKS":
        # Degree Centrality
        outdegree_attacks = graph.run("MATCH (u:User)RETURN u.id, size ((u)<-[:" + rel_type + "]-()) AS degree ORDER BY degree DESC",
                           rel_type=rel_type).to_data_frame()

        indegree_attacks = graph.run(
            "MATCH (u:User) RETURN u.id, size (()<-[:" + rel_type + "]->(u)) AS degree ORDER BY degree DESC",
            rel_type=rel_type).to_data_frame()
        return  outdegree_attacks, indegree_attacks
    elif rel_type == "TRADES":
        outdegree_trades = graph.run("MATCH (u:User)RETURN u.id, size ((u)<-[:" + rel_type + "]-()) AS degree ORDER BY degree DESC",
                           rel_type=rel_type).to_data_frame()
        indegree_trades = graph.run("MATCH (u:User)RETURN u.id, size (()-[:" + rel_type + "]->(u)) AS degree ORDER BY degree DESC",
                            rel_type=rel_type).to_data_frame()
        return outdegree_trades, indegree_trades
    else:
        outdegree_messages = graph.run("MATCH (u:User) RETURN u.id, size ((u)<-[:" + rel_type + "]-()) AS degree ORDER BY degree DESC",
                           rel_type=rel_type).to_data_frame()
        indegree_messages = graph.run("MATCH (u:User) RETURN u.id, size (()-[:" + rel_type + "]->(u)) AS degree ORDER BY degree DESC",
                            rel_type=rel_type).to_data_frame()
        return outdegree_messages, indegree_messages

    # weighted centrality
    # result2 = graph.run("MATCH (u:User)-[r:" + rel_type + "]-() RETURN u AS user, sum(r.weight) AS weightedDegree /"
    #                                                       "ORDER BY weightedDegree DESC LIMIT 25", rel_type=rel_type)


def pagerank(rel_type, graph):
    '''
    Pagerank for each category using Graph algorithms
    '''
    if rel_type == ' ':
        # The size of each node is proportional to the size and number of nodes with an outgoing relationship to it.
        q1 = graph.run(
            'MATCH (u:User) WITH collect(u) AS users CALL apoc.algo.pageRank(users) YIELD node, score RETURN node.id, score ORDER BY score DESC ').to_data_frame()
        return q1
    else:
        # The following will run the algorithm and stream results:
        q2 = graph.run(
            'CALL algo.pageRank.stream("User", "%s", {iterations:20, dampingFactor:0.85}) YIELD nodeId, score RETURN algo.asNode(nodeId).id AS user,score ORDER BY score DESC' % rel_type,
            rel_type=rel_type).to_data_frame()
        # The following will run the algorithm on Yelp social network:
        q3 = graph.run("CALL algo.pageRank.stream('MATCH (u:User) WHERE exists( (u)-[:" + rel_type + "]-() ) RETURN id(u) as id','MATCH (u1:User)-[:" + rel_type + "]-(u2:User) RETURN id(u1) as source, id(u2) as target', {graph:'cypher'}) YIELD nodeId,score with algo.asNode(nodeId) as node, score order by score desc  RETURN node {.id}, score", rel_type=rel_type).to_data_frame()

        return q2, q3


def plots_for_measures(attacks_centrality, attacks_centrality_prob, trades_centrality, trades_centrality_prob,
                       messages_centrality, messages_centrality_prob, out_attacks, in_attacks, out_trades, in_trades,
                       out_messages, in_messages, closeness_centrality, pagerank_for_attacks_damp, pagerank_for_attacks,
                       pagerank_for_trades_damp, pagerank_for_trades, pagerank_for_messages_damp, pagerank_for_messages):
    """
    Receive centrality measures for all types {Degree_centrality, Betweeness, Closeness, Pagerank}
    :return: Normalized Histogramm for each measure
    """
    count = 0
    dict_helper = {
        'Attacks-Centrality': attacks_centrality,
        'Attacks-Centrality P': attacks_centrality_prob,
        'Trades-Centrality': trades_centrality,
        'Trades-Centrality P': trades_centrality_prob,
        'Messages-Centrality': messages_centrality,
        'Messages-Centrality P': messages_centrality_prob,
        #'Closeness-Centrality': closeness_centrality
    }

    for hist_title, values in dict_helper.items():
        count += 1
        subplot(2, 3, count)
        #hist_plot = values['centrality'].hist(bins=100)
        #plt.set_title(hist_title)
        plt.title(hist_title)
        plt.xlabel('degree')
        plt.ylabel("number of nodes")
        plt.margins(x=0)
        plt.xticks(rotation= 45)
        plt.yscale('log', basey=10)
        plt.gca().set_xscale("log")
        _, bins = np.histogram(np.log10(values['centrality'] + 1), bins= 100)
        plt.hist(values['centrality'], bins=10 ** bins)
    plt.show()

    dict_degrees = {
        'InDegree-Attacks': in_attacks,
        'OutDegree-Attacks': out_attacks,
        'OutDegree-Trades': out_trades,
        'InDegree-Trades': in_trades,
        'OutDegree-Messages': out_messages,
        'InDegree-Messages': in_messages,
    }

    count1 = 0
    for hist_title, values in dict_helper.items():
        # count1 += 1
        # subplot(2, 3, count1)
        hist_plot = values['centrality'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('degree')
        hist_plot.set_ylabel("number of nodes")
        plt.margins(x=0)
        #plt.yscale('log', basey=10)
        #plt.yscale('log')
    plt.show()

    dict_pagerank = {
        'Pagerank-attacks': pagerank_for_attacks,
        'Pagerank-trades': pagerank_for_trades,
        'Pagerank-messages': pagerank_for_messages,
        'Pagerank-attacks(DF)': pagerank_for_attacks_damp,
        'Pagerank-trades(DF)': pagerank_for_trades_damp,
        'Pagerank-messages(DF)': pagerank_for_messages_damp,
    }

    count2 = 0
    for hist_title, values in dict_pagerank.items():
        count2 += 1
        subplot(2, 3, count2)
        hist_plot = values['score'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Pagerank')
        hist_plot.set_ylabel("users")
        plt.margins(x=0)
        plt.yscale('log', basey=10)
        plt.xscale('log')
        #ax= plt.gca()
        #ax.set_xlim([0,10000])
        #plt.xticks(0,100,200,300,400,500)
    plt.show()
    x=1


def main():
    graph = Graph('127.0.0.1', password='leomamao971')
    print("Read from database")

    attacks_degrees = algo_degree("ATTACKS", graph)
    trades_degrees = algo_degree("TRADES", graph)
    messages_degrees = algo_degree("messages", graph)

    dict_helper = {
        'attacks':attacks_degrees,
        'trades': trades_degrees,
        'messages': messages_degrees
    }
    cnt = 0
    for hist_title, values in dict_helper.items():
        cnt += 1
        subplot(2, 3, cnt)
        hist_plot = values['degree'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('degree')
        hist_plot.set_ylabel("number of nodes")
        plt.margins(x=0)
        plt.yscale('log', basey=10)
        #plt.xscale('log')
    plt.show()

    attacks_centrality, attacks_centrality_prob = betweenness_centrality('ATTACKS', graph)
    messages_centrality, messages_centrality_prob = betweenness_centrality('messages', graph)
    trades_centrality, trades_centrality_prob = betweenness_centrality('TRADES', graph)

    out_attacks, in_attacks = degree_centrality("ATTACKS", graph)
    out_trades, in_trades = degree_centrality("TRADES", graph)
    out_messages, in_messages = degree_centrality("messages", graph)

    #pagerank_score = pagerank(" ", graph)
    pagerank_for_attacks_damp, pagerank_for_attacks = pagerank("ATTACKS", graph)
    pagerank_for_trades_damp, pagerank_for_trades = pagerank("TRADES", graph)
    pagerank_for_messages_damp, pagerank_for_messages = pagerank("messages", graph)

    closeness_centrality_entire_graph = closeness(graph)

    plots_for_measures(attacks_centrality, attacks_centrality_prob, trades_centrality, trades_centrality_prob,
                       messages_centrality, messages_centrality_prob, out_attacks, in_attacks, out_trades, in_trades, out_messages, in_messages, closeness_centrality_entire_graph, pagerank_for_attacks_damp, pagerank_for_attacks, pagerank_for_trades_damp, pagerank_for_trades, pagerank_for_messages_damp, pagerank_for_messages)


if __name__ == '__main__':
    main()