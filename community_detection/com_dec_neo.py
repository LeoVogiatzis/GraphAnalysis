import os
import pandas as pd
from py2neo import Graph
import ujson
from collections import defaultdict 

def write_results(alg_name, direction, q2):
    results = {}

    for item in q2.data():
        results[item['node']['id']] = item['label']

    print('1st dict')

    mydict = defaultdict(list)
    for k,v in sorted(results.items()):
        mydict[v].append(k)

    print('2nd dict')

    with open(alg_name + '_' + direction + '_neo4j.txt', 'a') as f:
        f.write(ujson.dumps(mydict))

graph = Graph('127.0.0.1',password='gomugomuno13')
tx = graph.begin()

q1 = graph.run('''
    CALL algo.louvain.stream('User', 'TRADES', {
        direction: 'BOTH'
    })YIELD nodeId, community
    RETURN algo.getNodeById(nodeId) as node, community
    ''')

q2 = graph.run('''
    CALL algo.labelPropagation.stream("User", "TRADES",
    {direction: "BOTH", iterations: 10})
    YIELD nodeId, label
    RETURN algo.getNodeById(nodeId) as node, label
    ''')

print('query done')

write_results('louvain' , 'BOTH', q1)
write_results('label_propagation', 'BOTH', q2)