import os
import pandas as pd
from py2neo import Graph
import ujson
from collections import defaultdict 
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import community

def calc_color_values(graph, data):
    print('Calculate color values.....')
    values = []
    for i in graph.nodes():
        for k,v in data.items():
            if str(i) in v:
                values.append(int(k)) 
                break
    return values

def extract_results(alg_name, graph, data, time, modularity, **kwargs):
    _max_l = list() 
    _min_l = list()
    error = False
    if modularity is None:
        error = True
    extra_text = kwargs.get('extra_text', None)

    print('Writing results.....')
    with open(alg_name + '_results10.txt', 'a') as f:
        f.write('Communities Computed: \n')
        f.write(ujson.dumps(data))
        f.write('================================================================\n')
        f.write('Information about computed partition: \n')
        try:
            if error:
                f.write('modularity= ' + str(modularity) + '\n')
            else:
                f.write('modularity= ' + str('{:f}'.format(modularity)) + ' or ' + str(modularity) + '\n')
        except Exception as e3: print(repr(e3))
        f.write('time taken to compute: ' + str(time) + '\n')
        cnt = 0
        d = 0
        for item in data:
            cnt+=1
            f.write('For community ' + str(item) + ':\n')
            subG = graph.subgraph(data[item])
            f.write('NumOfEdges '+str(len(subG.edges())) + '\n')
            f.write('NumOfNodes '+str(len(subG.nodes())) + '\n')
            density = nx.classes.function.density(subG)
            d += density
            f.write('Density of community (/possible) = '+str(density) + '\n')
            try:
                _max = max(dict(subG.degree()).items(), key = lambda x : x[1])
                _min = min(dict(subG.degree()).items(), key = lambda x : x[1])
                _max_l.append(_max[1])
                _min_l.append(_min[1])
                f.write('Max (node, degree) -> ' + str(_max) + '\n')
                f.write('Min (node, degree) -> ' + str(_min) + '\n')
            except Exception as e: print(repr(e))
            if len(data.keys())!=cnt:
                f.write('Next community info===============================================\n')
            
        f.write('Information after computing.....\n')
        try:
            f.write('Max degree = ' + str(max(_max_l)) + '\n')
            f.write('Min degree = ' + str(min(_min_l)) + '\n')
        except Exception as e: print(repr(e))
        f.write('Average Density = ' + str(d/len(data)) + '\n')
        if extra_text is not None:
            print(extra_text)
            f.write('Extra Info: \n')
            f.write(extra_text)

def visualize(alg_name, graph, data):
    print('Visualization.....')
    nx.draw_spring(graph, cmap = plt.get_cmap('jet'), node_color = data,
        node_size=10, with_labels=False)
    plt.show()
    nx.draw_circular(graph, cmap = plt.get_cmap('jet'), node_color = data, 
    node_size=10, with_labels=False)
    plt.show()

    nodes = list(graph.nodes())
    nodes.sort()
    plt.margins(x=0)
    plt.title(alg_name)
    plt.xlabel('Nodes')
    plt.ylabel('Community ID')
    plt.scatter(nodes, data, c=data, s=10, cmap='jet')
    plt.show()

def louvain(direction, graph):
    print('Louvain.....') 

    print('Running query for community detection.....')
    t = datetime.now()
    louvain_query = graph.run('''
        CALL algo.louvain.stream('User', null, {
            direction: ''' + '"' + direction + '"' +  '''
        })YIELD nodeId, community
        RETURN algo.getNodeById(nodeId) as node, community
        ''').data()
    time_taken = datetime.now() - t

    print('Converting results.....')
    results = defaultdict(list)
    for item in louvain_query:
        results[str(item['community'])].append(str(item['node']['id']))
    
    print('Get Neo4jGraph.....')
    graph_query = graph.run('''
        MATCH (n:User)-[r]->(m:User)
        RETURN n.id,TYPE(r),m.id 
    ''').to_data_frame()

    print('Convert graph to nx graph.....')
    nx_graph = nx.from_pandas_edgelist(df=graph_query, source='n.id', target='m.id', edge_attr=True,
        create_using=nx.MultiGraph(name='Travian_Graph'))
    
    modularity = None
    print('Computing modularity.....')
    try: modularity = nx.algorithms.community.modularity(nx_graph, results.values())
    except Exception as e: print(repr(e))
    try: modularity = community.modularity(results.values(), nx_graph)
    except Exception as e: print(repr(e))

    extract_results('neo_louvain'+direction, nx_graph, results, time_taken, modularity)

    values = calc_color_values(nx_graph, results)
    
    visualize('Louvain ' + direction, nx_graph, values)

    print('Louvain finished.....')

def lp(direction, graph):
    print('lp.....') 

    print('Running query for community detection.....')
    t = datetime.now()
    louvain_query = graph.run('''
        CALL algo.labelPropagation.stream("User", null,
        {direction: ''' + '"' + direction + '"' +  ''', iterations: 10})
        YIELD nodeId, label
        RETURN algo.getNodeById(nodeId) as node, label
        ''').data()
    time_taken = datetime.now() - t

    print('Converting results.....')
    results = defaultdict(list)
    for item in louvain_query:
        results[str(item['label'])].append(str(item['node']['id']))
    
    print('Get Neo4jGraph.....')
    graph_query = graph.run('''
        MATCH (n:User)-[r]->(m:User)
        RETURN n.id,TYPE(r),m.id 
    ''').to_data_frame()

    print('Convert graph to nx graph.....')
    nx_graph = nx.from_pandas_edgelist(df=graph_query, source='n.id', target='m.id', edge_attr=True,
        create_using=nx.MultiGraph(name='Travian_Graph'))
    
    modularity = None
    print('Computing modularity.....')
    try: modularity = nx.algorithms.community.modularity(nx_graph, results.values())
    except Exception as e: print(repr(e))
    try: modularity = community.modularity(results.values(), nx_graph)
    except Exception as e: print(repr(e))

    extract_results('neo_lp'+direction, nx_graph, results, time_taken, modularity)

    values = calc_color_values(nx_graph, results)
    
    visualize('Label Propagation ' + direction, nx_graph, values)

    print('lp finished.....')

def main():
    directions = ['BOTH', 'OUTCOMING', 'INCOMING']
    
    graph = Graph('127.0.0.1', password='gomugomuno13')
    tx = graph.begin()
    
    for direction in directions:
        louvain(direction, graph)    
        lp(direction, graph)

if __name__ == '__main__':
    main()