from datetime import datetime

import ujson
from py2neo import Graph


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

    cquery = '''
                OPTIONAL MATCH (start:User{id:{s_id}}), (end:User{id:{d_id}})
                        CALL algo.shortestPath.stream(start, end,null)
                        YIELD nodeId,cost
                        WHERE  cost<>0 and cost <> 1 
                        RETURN start.id  AS name,end.id as end_name ,cost limit 100'''

    with open('shortestpaths2.txt', 'a') as f:
        for s_node in best_out_degree:
            for d_node in nodes:

                if s_node == d_node:
                    continue

                data = graph.run(cquery, s_id=s_node, d_id=d_node).data()

        x = 1
        f.write(ujson.dumps(data))

    return data


def shortest_path2(graph, starttime):
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
            RETURN start.id  AS name,end.id as end_name ,cost ''').to_data_frame()
    total_time = datetime.now() - starttime
    print(total_time)
    return df


def main():
    graph = Graph('127.0.0.1', password='leomamao971')
    starttime = datetime.now()
    df = shortest_path3(graph, starttime)
    df.to_pickle("./shortespath_neo4j.pkl")

    # df = pd.read_pickle("./shortespath_neo4j.pkl")


if __name__ == '__main__':
    main()
