import glob
import os

import pandas as pd
from py2neo import Graph, Node

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
