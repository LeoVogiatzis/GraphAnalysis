import glob
import random
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import matplotlib.pyplot as plt
import seaborn as sns


def features(dataset):
    # dataset2 = pd.read_pickle("dataset_features.pkl").reset_index(drop=True)
    dataset['type'] = dataset['type'].astype('category')
    dataset['type'] = dataset['type'].cat.codes
    dataset['id1'] = dataset['id1'].astype(int)
    dataset['id2'] = dataset['id2'].astype(int)
    dataset = dataset.drop(index=dataset[dataset['id1'] == dataset['id2']].index)
    dataset.loc[dataset['label'] == 0, 'type'] = 3
    dataset['weight'] = dataset.groupby(['id1', 'id2', 'type'])['weight'].transform(lambda x: x.count())

    edges = int(len(dataset))
    dataset['weight'] = dataset['weight'].div(edges)
    dataset.to_pickle('./dataset_for_MLP.pkl')


def trial(dataset):
    # coefficient for each column ways of important features
    df0 = dataset[dataset['label'] == 0]
    df1 = dataset[dataset['label'] == 1]
    df2 = dataset[dataset['type'] == 2]
    df3 = dataset[dataset['type'] == 3]
    plt.scatter(df0['closeness_centrality'], df0['closeness_centrality for id2'], edgecolors='green', marker='+')
    plt.scatter(df1['closeness_centrality'], df1['closeness_centrality for id2'], edgecolors='yellow', marker='+')
    plt.scatter(df2['closeness_centrality'], df2['closeness_centrality for id2'], edgecolors='red', marker='*')
    plt.scatter(df3['closeness_centrality'], df3['closeness_centrality for id2'], edgecolors='blue', marker='*')
    plt.show()
    dataset = dataset.drop(['Timestamp', 'label'], axis=1)
    X = dataset.drop(['type', 'id1', 'id2'], axis='columns')
    y = dataset['type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    len(X_train)
    len(X_test)
    model = svm.SVC(kernel='rbf')
    model.fit(X_train, X_test)
    print(model.score(X_test, y_test))


def main():
    data = pd.read_pickle('./dataset_for_MLP.pkl')

    dataset = data[:1000]
    print(dataset.describe())
    print(dataset.head())

    dataset['type'] = dataset['type'].astype('category')
    dataset['type'] = dataset['type'].cat.codes
    dataset = dataset.drop(['label'], axis=1)
    dataset = dataset.drop(['Timestamp', 'weight'], axis=1)

    dataset.to_numpy()

    y = dataset['type']
    x = dataset.drop(['type'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

    #
    # scaler = StandardScaler()
    # # Fit on training set only.
    # scaler.fit(x_train)
    # # Apply transform to both the training set and the test set.
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)
    #
    # pca = PCA(.75)
    # pca.fit(x_train)
    # print("pca.n_components = ", pca.n_components_)
    # x_train, x_test = pca.transform(x_train), pca.transform(x_test)
    clf = svm.SVC(kernel='rbf')
    # Train the model using the training sets
    clf.fit(x_train, y_train)

    # Predict the response for test dataset
    y_pred = clf.predict(x_test)

    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    # Model Precision: what percentage of positive tuples are labeled as such?
    print("Precision:", metrics.precision_score(y_test, y_pred))
    # Model Recall: what percentage of positive tuples are labelled as such?
    print("Recall:", metrics.recall_score(y_test, y_pred))

    accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print(accuracy_score, cm)
    sns.heatmap(cm, center=True)
    plt.show()

    x=1


if __name__ == '__main__':
    main()