import glob
import random
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from sklearn import tree


from graphviz import Source

import json


dataset = pd.read_pickle('./dataset_model2.pkl')
dataset = dataset.set_index(['id1', 'id2'])

dataset['type'] = dataset['type'].astype('category')
dataset['type'] = dataset['type'].cat.codes


dataset = dataset.drop(['label', 'Timestamp', 'weight'], axis=1)
X = dataset.drop(columns=['type'])
Y = dataset['type']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

model = DecisionTreeClassifier()
model.fit(X, Y)
model.fit(X, Y)
predictions = model.predict(X_test)

score = accuracy_score(y_test, predictions)
print(score)

tree.export_graphviz(model, out_file='links_rec.dot',
                     feature_names=['Preferential_Attachment',
                                    'Resource_allocation', 'betweeness_centrality',
                                    'betweeness_centrality for id2', 'closeness_centrality',
                                    'closeness_centrality for id2'],
                     class_names=sorted(Y.unique()),
                     label='all',
                     rounded=True,
                     filled=True)


path = 'links_rec.dot'
s = Source.from_file(path)
s.view()