from datetime import datetime

import ujson
from py2neo import Graph
import pandas as pd


def shortest_path(graph, starttime):
    best_out = graph.run('MATCH (n:User)-[r]->() RETURN n.id, count(r) as count').to_data_frame()
    best_in = graph.run('MATCH (n:User)<-[r]-() RETURN n.id, count(r) as count').to_data_frame()

    best_out_degree = [user for user in best_out.nlargest(100, ['count'])['n.id']]
    best_in_degree = [user for user in best_in.nlargest(100, ['count'])['n.id']]
    worst_in_degree = [user for user in best_in.nsmallest(100, ['count'])['n.id']]
    worst_out_degree = [user for user in best_out.nsmallest(100, ['count'])['n.id']]

    user_query = graph.run('''
        MATCH (n:User) RETURN n.id
        ''').to_data_frame()

    nodes = [node for node in user_query['n.id']]
    df = pd.DataFrame(columns=['name', 'end_name', 'cost'])
    cquery = '''
                OPTIONAL MATCH (start:User{id:{s_id}}), (end:User{id:{d_id}})
                        CALL algo.shortestPath.stream(start, end,null)
                        YIELD nodeId,cost
                        WHERE  cost<>0 and cost <> 1 
                        RETURN start.id  AS name,end.id as end_name ,cost'''

    for s_node in best_out_degree:
        for d_node in worst_out_degree:
            if s_node == d_node:
                continue
            data = graph.run(cquery, s_id=s_node, d_id=d_node).data()
            if not len(data):
                continue
            # df.loc[len(df)+1] = [data[0]['name'], data[0]['end_name'], data[0]['cost']]
            # df['start_name'] = data[0]['name']
            # df['end_name'] = data[0]['end_name']
            # df['sp_cost'] = data[0]['cost']
            df = df.append(data[0], True)
            x=1
    df.to_pickle("./shortespath_neo4j_best_out_worst_out.pkl")
    total_time = datetime.now() - starttime
    print(total_time)
    return df


def shortest_path2(graph, starttime):
    """

    :param graph:
    :param starttime:
    :return: visualize
    """
    ordered_sh = graph.run('''
                MATCH (n:User) 
                WITH collect(n) as nodes
                UNWIND nodes as n
                UNWIND nodes as m
                WITH * WHERE id(n) < id(m)
                MATCH path = allShortestPaths( (n)-[*..4]-(m) )
                RETURN path limit 10''').to_data_frame()
    ordered_sh.to_pickle("./shortestpaths_neo4j.pkl")


def shortest_path3(graph, starttime):
    df = graph.run('''OPTIONAL MATCH (start:User), (end:User)
            CALL algo.shortestPath.stream(start, end,null)
            YIELD nodeId,cost
            WHERE  cost<>0 and cost <> 1 
            RETURN start.id  AS name,end.id as end_name ,cost 10''').to_data_frame()
    total_time = datetime.now() - starttime
    print(total_time)
    return df


def main():

    df = pd.read_pickle("./shortespath_neo4j_best_out_best_in.pkl")
    #df = pd.read_pickle("./shortespath_neo4j.pkl")
    graph = Graph('127.0.0.1', password='leomamao971')
    starttime = datetime.now()
    df = shortest_path(graph, starttime)
    #df.to_pickle("./shortespath_neo4j.pkl")

    #df.to_pickle("./shortespath_neo4j.pkl")


    x = 1


if __name__ == '__main__':
    main()
