from py2neo import Graph


def best_users(graph):
    best_in = graph.run('MATCH (u:User)<-[r]-() RETURN u.id, count(r) as count').to_data_frame()
    best_out = graph.run('MATCH (u:User)-[r]->() RETURN u.id, count(r) as count').to_data_frame()

    best_out_degree = [user for user in best_out.nlargest(100, ['count'])['u.id']]
    best_in_degree = [user for user in best_in.nlargest(100, ['count'])['u.id']]
    worst_in_degree = [user for user in best_in.nsmallest(100, ['count'])['u.id']]
    worst_out_degree = [user for user in best_out.nsmallest(100, ['count'])['u.id']]
    print()
    return best_in, best_out


def adamic_adar_alg(rel_type, graph):
    if rel_type != '':
        sc_per_type = graph.run(
            'MATCH (p1:User) MATCH (p2:User) RETURN algo.linkprediction.adamicAdar(p1, p2, {relationshipQuery: "%s"}) AS score,p1.id,p2.id' % rel_type,
            rel_type=rel_type)
        return sc_per_type.to_data_frame()
    else:
        sc_per_user = graph.run(
            'MATCH (p1:User) MATCH (p2:User) RETURN algo.linkprediction.adamicAdar(p1, p2) AS score').to_data_frame()
        return sc_per_user


def common_neighbors(rel_type, graph):
    if rel_type != '':
        sc_common_per_type = graph.run(
            'MATCH (p1:User) MATCH (p2:User) RETURN algo.linkprediction.commonNeighbors(p1, p2, {relationshipQuery: "%s"}) AS score LIMIT 10' % rel_type,
            rel_type=rel_type)
        return sc_common_per_type.to_data_frame()
    else:
        sc_common_per_user = graph.run(
            'MATCH (p1:User) MATCH (p2:User) RETURN algo.linkprediction.commonNeighbors(p1, p2) AS score LIMIT 10').to_data_frame()
        return sc_common_per_user


def linkprediction_preferectialAttachment(rel_type, graph):
    if rel_type != '':
        sc_preferential_per_type = graph.run(
            'MATCH (p1:User) MATCH (p2:User) RETURN algo.linkprediction.preferentialAttachment(p1, p2, {relationshipQuery: "%s"}) AS score' % rel_type,
            rel_type=rel_type).to_data_frame()
        return sc_preferential_per_type.to_data_frame()
    else:
        sc_preferential_per_user = graph.run(
            'MATCH (p1:User) MATCH (p2:User) RETURN algo.linkprediction.preferentialAttachment(p1, p2) AS score').to_data_frame()
        return sc_preferential_per_user


def resourceAllocation(rel_type, graph):
    if rel_type != '':
        sc_resourceAllocation_per_type = graph.run('''
        MATCH (p1:Person {name: 'Michael'})
        MATCH (p2:Person {name: 'Karin'})
        RETURN algo.linkprediction.resourceAllocation(p1, p2, {relationshipQuery: "s"}) AS score ''' % rel_type,
                                                   rel_type=rel_type).to_data_frame()
        return sc_resourceAllocation_per_type
    else:
        sc_resourceAllocation_per_user = graph.run('''
        MATCH (p1:User)
        MATCH (p2:User)
        RETURN algo.linkprediction.resourceAllocation(p1, p2) AS score ''').to_data_frame()
        return sc_resourceAllocation_per_user


def linkpred_sameCommunity(rel_type, graph):
    if rel_type != '':
        sc_sameCommunity_per_type = graph.run('''
        MATCH (p1:User)
        MATCH (p2:User) 
        RETURN algo.linkprediction.sameCommunity(p1, p2) AS score 
        ''' % rel_type, rel_type=rel_type).to_data_frame()
        return sc_sameCommunity_per_type
    else:
        sc_sameCommunity_per_user = graph.run('''
        MATCH (p1:User)
        MATCH (p2:User)
        RETURN algo.linkprediction.sameCommunity(p1, p2) AS score''').to_data_frame()
        return sc_sameCommunity_per_user


def total_neighbors(rel_type, graph):
    if rel_type != '':
        sc_totalNeighbors_per_type = graph.run('''
        MATCH (p1:Person {name: 'Michael'})
        MATCH (p2:Person {name: 'Karin'})
        RETURN algo.linkprediction.totalNeighbors(p1, p2, {relationshipQuery: "%s"}) AS score''' % rel_type,
                                               rel_type=rel_type).to_data_frame()
        return sc_totalNeighbors_per_type
    else:
        sc_totalNeighboors_per_user = graph.run('''
        MATCH (p1:User)
        MATCH (p2:User)
        RETURN algo.linkprediction.totalNeighbors(p1, p2) AS score
        ''').to_data_frame()
        return sc_totalNeighboors_per_user


def main():
    graph = Graph('127.0.0.1', password='leomamao971')
    best_users(graph)
    print("Read from database")
    common_neighbors('TRADES', graph)
    common_neighbors('ATTACKS', graph)
    common_neighbors('messages', graph)
    common_neighbors('', graph)

    adamic_adar_alg('TRADES', graph)
    adamic_adar_alg('ATTACKS', graph)
    adamic_adar_alg('messages', graph)
    adamic_adar_alg('', graph)

    linkprediction_preferectialAttachment('TRADES', graph)
    linkprediction_preferectialAttachment('ATTACKS', graph)
    linkprediction_preferectialAttachment('messages', graph)
    linkprediction_preferectialAttachment('', graph)

    resourceAllocation('TRADES', graph)
    resourceAllocation('ATTACKS', graph)
    resourceAllocation('messages', graph)
    resourceAllocation('', graph)

    linkpred_sameCommunity('TRADES', graph)
    linkpred_sameCommunity('ATTACKS', graph)
    linkpred_sameCommunity('messages', graph)
    linkpred_sameCommunity('', graph)

    total_neighbors('TRADES', graph)
    total_neighbors('ATTACKS', graph)
    total_neighbors('messages', graph)
    total_neighbors('', graph)


if __name__ == '__main__':
    main()
