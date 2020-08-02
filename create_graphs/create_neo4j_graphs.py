import glob
import json
import os
import sys
from datetime import datetime

try:
    from create_rels_from_txt import create_relations
except ImportError:
    from .create_rels_from_txt import create_relations
import pandas as pd
import ujson
from py2neo import Graph, Node


def __read_config():
    with open('../neo_config/neoConfig.json') as f:
        try:
            neo_config = json.load(f)
            return neo_config
        except:
            sys.exit('Failure to retrieve data...')


def __read_file_data(graph, tx):
    # read cvs
    file_names = glob.glob("../data_users_moves/*.csv")
    row_count = 0
    file_count = 0
    all_results = 0
    with open('trades_try1_results_6toEnd.txt', 'a') as results:
        results.write('Results')
        for file_name in file_names:
            df = pd.read_csv(file_name, header=None)
            df.columns = ['Timestamp', 'id1', 'id2']

            print(file_name)

            if file_name.startswith('attack'):
                continue
                # rel_type = 'ATTACKS'
            elif file_name.startswith('trade'):
                continue
                # rel_type = 'TRADES'
            else:
                # continue
                relation_type = 'MESSAGES'

            for index, row in df.iterrows():
                # t1 = datetime.now()
                time_stamp = int(row['Timestamp'])
                formatted_time_stamp = datetime.utcfromtimestamp(time_stamp).strftime('%Y-%m-%d %H:%M:%S')
                splitted_time_stamp = formatted_time_stamp.split()
                date = splitted_time_stamp[0]
                time = splitted_time_stamp[1]

                graph.run('''MATCH (u1:User),(u2:User)
                            WHERE u1.id={id1} AND u2.id={id2}
                            CREATE (u1)-[r:''' + relation_type + ''' { date: {date}, time : {time} }]->(u2)''',
                          id1=int(row['id1']), id2=int(row['id2']), rel_type=relation_type, date=date, time=time)

            file_count += 1
            all_results += row_count
            results.write('\nFilename --- ' + file_name)
            results.write('\nRelationships --- ' + str(row_count))

        tx.commit()
        results.write('\nFinal Results****************')
        results.write('\nRelationships --- ' + str(all_results))
        results.write('\nFiles --- ' + str(file_count))


def create_relations_weighted():
    dir_path = os.getcwd()

    trades = {}
    combs = list()

    file_cnt = 0
    f = open('messages_relations.txt', 'a')
    count = 0

    for file in os.listdir(os.fsencode(dir_path)):
        filename = os.fsdecode(file)
        if not filename.startswith('message') or not filename.endswith('.csv'): continue
        print('current file --- ', filename)

        df = pd.read_csv(filename, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']

    for index, row in df.iterrows():
        # trades[row['id1']] = {
        #     to: row['id2'],
        #     weight: 0,
        #     first_date: None,
        #     second_date: None
        # # }
        if (row['id1'], row['id2']) not in combs:
            ts = int(row['Timestamp'])
            formatted_ts = datetime.utcfromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            splitted_ts = formatted_ts.split()
            date = splitted_ts[0]
            time = splitted_ts[1]

        combs.append((row['id1'], row['id2']))
        # if (8598, 101) in combs: print('TRUE')

        trades[(row['id1'], row['id2'])] = {
            'weight': combs.count((row['id1'], row['id2'])),
            'first_date': {
                'date': date,
                'time': time
            }
        }

    mydict = {'messages': trades}
    f.write(ujson.dumps(mydict))
    # for tup in combs:
    #     trades[tup] = trades.get(tup, 0) + 1

    # graph = Graph('127.0.0.1', password='leomamao97')
    # tx = graph.begin()
    # rel_type = 'MESSAGES'
    # count = 9
    # for item in trades:
    #     graph.run('''MATCH (u1:User),(u2:User)
    #                     WHERE u1.id={id1} AND u2.id={id2}
    #                     CREATE (u1)-[r:''' + rel_type + \
    #                     ''' { weight: {weight}, first_date: {date}, first_time : {time} }]->(u2)''',
    #                   id1=int(row['id1']), id2=int(row['id2']),
    #                   rel_type=rel_type, weight= trades[item]['weight'],
    #                   date=trades[item]['first_date']['date'],
    #                   time=trades[item]['first_date']['time'])

    #     count +=1
    #     if count%1000 == 0 : print(count)
    #     # print(item)
    #     # print(trades[item])
    #     # print(item[0]) ->node
    #     # print(item[1]) ->node
    #     # print(trades[item]['weight']) ->relationship
    #     # print(trades[item]['first_date']['date']) ->relationship
    #     # print(trades[item]['first_date']['time']) ->relationship
    #     # break

    # tx.commit()
    create_relations()


def create_relations_unweighted():
    neo_config = __read_config()
    url = neo_config['neodb']['url']
    password = neo_config['neodb']['password']
    graph = Graph(url, password=password)
    tx = graph.begin()
    __read_file_data(graph, tx)


def create_nodes():
    file_names = glob.glob('../data_users_moves/*.csv')
    my_set = set()

    for file in file_names:
        filename = os.fsdecode(file)
        if not filename.endswith('.csv'):
            continue
        print('Currently using file - ', filename)
        df = pd.read_csv(filename, header=None)
        df.columns = ['Timestamp', 'id1', 'id2']
        y = set(list(df['id1']))
        z = set(list(df['id2']))
        ids_of_one_csv = y.union(z)
        my_set.update(ids_of_one_csv)

    graph = Graph('127.0.0.1', password='leomamao97')
    tx = graph.begin()
    for value in my_set:
        user = Node('User', id=value)
        tx.create(user)
    tx.commit()


def create_graph(weighted=False):
    create_nodes()
    if weighted:
        create_relations_weighted()
    else:
        create_relations_unweighted()


if __name__ == '__main__':
    create_graph(bool(sys.argv[1]))
