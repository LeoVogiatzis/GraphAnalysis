# import rels from txt to database
import ast
import re

from py2neo import Graph


def create_relations():
    count = 0
    cnt_same = 0

    rel_type = 'ATTACKS'
    rtype = 'attacks'

    graph = Graph('127.0.0.1', password='gomugomuno23')
    tx = graph.begin()

    with open('attacks_relations.txt', 'r') as f:
        s = f.read()
        whip = ast.literal_eval(s)
        for item in whip[rtype]:
            # print(item)
            # print(item[1:5])
            # print(item[7:11])
            ids = re.findall('[0-9]+', item)
            # print(ids[0])
            # print(ids[1])
            # print(whip['trades'][item])
            # print(whip['trades'][item]['weight'])
            # print(whip['trades'][item]['first_date']['date'])
            # print(whip['trades'][item]['first_date']['time'])
            # print('-----------------------------')

            if ids[0] == ids[1]:
                cnt_same += int(whip[rtype][item]['weight'])
                continue

            graph.run('''MATCH (u1:User),(u2:User) 
                         WHERE u1.id={id1} AND u2.id={id2} 
                         CREATE (u1)-[r:''' + rel_type + \
                      ''' { weight: {weight}, first_date: {date}, first_time : {time} }]->(u2)''',
                      id1=int(ids[0]), id2=int(ids[1]),
                      rel_type=rel_type,
                      weight=whip[rtype][item]['weight'],
                      date=whip[rtype][item]['first_date']['date'],
                      time=whip[rtype][item]['first_date']['time'])

            count += 1
            if count % 10000 == 0: print(count)

    # total = 86910
    tx.commit()
