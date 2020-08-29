from py2neo import Graph
import json
import sys
import pandas as pd
import numpy as np
import matplotlib as plt

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
        # Degree centrality
        outdegree_attacks = graph.run("MATCH (u:User) RETURN u.id, size ((u)<-[:" + rel_type + "]-()) AS degree ORDER BY degree DESC",
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
        outdegree_messages = graph.run("MATCH (u:User)RETURN u.id, size ((u)<-[:" + rel_type + "]-()) AS degree ORDER BY degree DESC",
                           rel_type=rel_type).to_data_frame()
        indegree_messages = graph.run("MATCH (u:User)RETURN u.id, size (()-[:" + rel_type + "]->(u)) AS degree ORDER BY degree DESC",
                            rel_type=rel_type).to_data_frame()
        return outdegree_messages, indegree_messages

    # weighted centrality
    # result2 = graph.run("MATCH (u:User)-[r:" + rel_type + "]-() RETURN u AS user, sum(r.weight) AS weightedDegree /"
    #                                                       "ORDER BY weightedDegree DESC LIMIT 25", rel_type=rel_type)


def pagerank(rel_type, graph):
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
                       messages_centrality, messages_centrality_prob, out_attacks, in_attacks, out_trades, in_trades, out_messages, in_messages, closeness_centrality, pagerank_for_attacks_damp, pagerank_for_attacks, pagerank_for_trades_damp, pagerank_for_trades, pagerank_for_messages_damp, pagerank_for_messages):
    dict_helper = {
        'Attacks-centrality Histogram': attacks_centrality,
        'Attacks-centrality with probaility Histogram': attacks_centrality_prob,
        'Trades-centrality Histogram': trades_centrality,
        'Trades-centrality with probaility Histogram': trades_centrality_prob,
        'Messages-centrality Histogram': messages_centrality,
        'Messages-centrality with probability Histogram': messages_centrality_prob,
        'Closeness centrality': closeness_centrality
    }
    for hist_title, values in dict_helper.items():
        hist_plot = values['centrality'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.pyplot.show()

    dict_degrees = {
        'INDEGREE-ATTACKS': in_attacks,
        'OUTDEGREE-ATTACKS': out_attacks,
        'INDEGREE-TRADES': out_trades,
        'OUTDEGREE-TRADES': in_trades,
        'INDEGREE-MESSAGES': out_messages,
        'OUTDEGREE-ATTACKS': in_messages,
    }

    for hist_title, values in dict_degrees.items():
        hist_plot = values['degree'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Value')
        hist_plot.set_ylabel("numbers")
        plt.show()

    dict_pagerank = {
        'Pagerank for attacks': pagerank_for_attacks,
        'Pagerank for trades': pagerank_for_trades,
        'Pagerank for messages': pagerank_for_messages,
        'Pagerank for attacks with damping factor': pagerank_for_attacks_damp,
        'Pagerank for trades with damping factor': pagerank_for_trades_damp,
        'Pagerank for messages with damping factor': pagerank_for_messages_damp,
        'Pagerank entire Score': pagerank_score
    }

    for hist_title, values in dict_pagerank.items():
        hist_plot = values['score'].hist(bins=100)
        hist_plot.set_title(hist_title)
        hist_plot.set_xlabel('Score Pagerank')
        hist_plot.set_ylabel("users")
        #ax= plt.gca()
        #ax.set_xlim([0,10000])
        #plt.xticks(0,100,200,300,400,500)
        plt.show()

def read_from_neo4j_database():
    with open('../neo_config/neoConfig.json') as f:
        # load data within configuration
        try:
            neo_config = json.load(f)
        except:

            sys.exit('Failure to retrieve data...')

    url = neo_config['neodb']['url']
    password = neo_config['neodb']['password']
    graph = Graph(url, password=password)

    return graph
