import glob
import json
import sys
from datetime import datetime

import pandas as pd
from py2neo import Graph


def read_config():
    with open('../neo_config/neoConfig.json') as f:
        try:
            neo_config = json.load(f)
            return neo_config
        except:
            sys.exit('Failure to retrieve data...')


def read_file_data(graph, tx):
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


def main():
    neo_config = read_config()
    url = neo_config['neodb']['url']
    password = neo_config['neodb']['password']
    graph = Graph(url, password=password)
    tx = graph.begin()
    read_file_data(graph, tx)


if __name__ == '__main__':
    main()
