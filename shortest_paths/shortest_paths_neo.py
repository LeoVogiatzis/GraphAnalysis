import community 
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import glob
import ujson
from collections import defaultdict 
import itertools
import sys
import numpy as np
import os
import scipy
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering, spectral_clustering
from datetime import datetime
from py2neo import Graph

def cosine_similarity(a, b):
    res = 0
    for index, value in enumerate(a):
        res += len(set(value) & set(b[index])) / float(len(set(value) | set(b[index])))
    return res/len(a)

def shortest_path_neo(pair, graph, users1, users2):
    print('computing shortest path.....')
    t = datetime.now()
    sp = pd.DataFrame(columns=['source', 'target', 'cost'])
    cnt = 0
    for u1 in users1:
        cnt += 1
        for u2 in users2:
            if u1 == u2: continue
            temp = pd.DataFrame(columns=['source', 'target', 'cost'])
            q_result = graph.run('''
                MATCH (start:User{id:{s_id}}),(end:User{id:{t_id}})
                CALL algo.shortestPath.stream(start, end)
                YIELD nodeId, cost
                RETURN algo.asNode(nodeId).id as target, cost
            ''', s_id=u1,t_id=u2).to_data_frame()

            temp['source'] = [u1 for i in range(len(q_result))]
            temp['target'] = q_result['target']
            temp['cost'] = q_result['cost']

            sp = pd.concat([sp, temp])
        if cnt%50==0: print(cnt)
    time_taken = str(datetime.now() - t)
    print(time_taken)
    sp.drop_duplicates()
    sp = sp[sp.cost > 0]

    top500 = sp.nlargest(500, 'cost').reset_index(drop=True)
    min500 = sp.nsmallest(500, 'cost').reset_index(drop=True)
    
    print('Export Files.....')
    top500.to_csv(pair + '_top500_test.csv')
    min500.to_csv(pair + '_min500_test.csv')
    
    top500_list = top500.drop(['cost'], axis=1).values.tolist()
    min500_list = min500.drop(['cost'], axis=1).values.tolist()
    
    return top500_list, min500_list, time_taken

def main():
    graph = Graph('127.0.0.1', password='gomugomuno13')
    tx = graph.begin()
    
    # best_in = graph.run('MATCH (n:User)<-[r]-() RETURN n.id, count(r) as count').to_data_frame()
    # best_out = graph.run('MATCH (n:User)-[r]->() RETURN n.id, count(r) as count').to_data_frame()
    user_degree = graph.run('''MATCH (n:User)-[r]-() RETURN n.id, count(r) as count''').to_data_frame()

    best_users = [user for user in user_degree.nlargest(200, ['count'])['n.id']]
    worst_users = [user for user in user_degree.nsmallest(200, ['count'])['n.id']]
    # best_out_degree = [user for user in best_out.nlargest(10, ['count'])['n.id']]
    # best_in_degree = [user for user in best_in.nlargest(100, ['count'])['n.id']]
    # worst_in_degree = [user for user in best_in.nsmallest(100, ['count'])['n.id']]
    # worst_out_degree = [user for user in best_out.nsmallest(10, ['count'])['n.id']]
    best_worst_top500, best_worst_min500, best_worst_time_taken = shortest_path_neo('200best_200worst_undirected', graph, best_users, worst_users)
    # bo_wo_top500, bo_wo_min500, bo_wo_time_taken = shortest_path_neo('bestOut_worstOut', graph, best_out_degree, worst_out_degree)
    # bo_wi_top500, bo_wi_min500, bo_wi_time_taken = shortest_path_neo('bestOut_worstIn', graph, best_out_degree, worst_in_degree)
    # bo_bi_top500, bo_bi_min500, bo_bi_time_taken = shortest_path_neo('bestOut_bestIn', graph, best_out_degree, best_in_degree)
    # wo_wi_top500, wo_wi_min500, wo_wi_time_taken = shortest_path_neo('worstOut_worstIn', graph, worst_out_degree, worst_in_degree)
    # wo_bi_top500, wo_bi_min500, wo_bi_time_taken = shortest_path_neo('worstOut_bestIn', graph, worst_out_degree, best_in_degree)
    # wi_bi_top500, wi_bi_min500, wi_bi_time_taken = shortest_path_neo('worstIn_bestIn', graph, worst_in_degree, best_in_degree)
    # bo_bo_top500, bo_bo_min500, bo_bo_time_taken = shortest_path_neo('bestOut_bestOut', graph, best_out_degree, best_out_degree)
    # bi_bi_top500, bi_bi_min500, bi_bi_time_taken = shortest_path_neo('bestIn_bestIn', graph, best_in_degree, best_in_degree)
    # wo_wo_top500, wo_wo_min500, wo_wo_time_taken = shortest_path_neo('worstOut_worstOut', graph, worst_out_degree, worst_out_degree)
    # wi_wi_top500, wi_wi_min500, wi_wi_time_taken = shortest_path_neo('worstIn_worstIn', graph, worst_in_degree, worst_in_degree)
    
    # cos_bowoT500_bowiT500
    # print(bo_wo_top500)
    # print(bo_wi_top500)
    # bowoT_bowiT_cos = cosine_similarity(bo_wo_top500, bo_wi_top500)
    

if __name__ == '__main__':
    main()