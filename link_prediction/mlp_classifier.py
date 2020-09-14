import glob
import random
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import json
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import roc_curve, auc


def apply_model(dataset):
    df_train = dataset[~pd.isnull(dataset['label'])]
    df_test = dataset[pd.isnull(dataset['label'])]
    features = ['type', 'weight', 'Preferential_Attachment', 'Resource_allocation', 'betweeness_centrality',
                'betweeness_centrality fot id2', 'closeness_centrality', 'closeness_centrality for id2']
    X_train = df_train[features]
    Y_train = df_train['label']
    X_test = df_test[features]
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    clf = MLPClassifier(hidden_layer_sizes=[10, 5], alpha=5,
                        random_state=0, solver='lbfgs', verbose=0)
    clf.fit(X_train_scaled, Y_train)
    test_proba = clf.predict_proba(X_test_scaled)[:, 1]
    predictions = pd.Series(test_proba, X_test.index)
    target = dataset['label']
    target['prob'] = [predictions[x] for x in target.index]
    return target['prob']


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


def main():
    dataset = pd.read_pickle('./dataset_for_MLP.pkl')
    dataset['type'] = dataset['type'].astype('category')
    dataset['type'] = dataset['type'].cat.codes
    dataset['weight'] = 1
    dataset = dataset.groupby(
        ['id1', 'id2', 'type', 'label', 'Resource_allocation', 'Preferential_Attachment', 'betweeness_centrality',
         'betweeness_centrality fot id2', 'closeness_centrality', 'closeness_centrality for id2'], as_index=False)[
        'weight'].sum()
    # dataset = data[:100000]
    starttime = datetime.now()
    print(dataset.describe())
    print(dataset.head())

    dataset['type'] = dataset['type'].astype('category')
    dataset['type'] = dataset['type'].cat.codes
    #dataset = dataset.drop(['label'], axis=1)
    #dataset = dataset.drop(['Timestamp'], axis=1)
    y = dataset['label']
    x = dataset.drop(['label'], axis=1)

    mms = MinMaxScaler()

    dataset.iloc[:, 3:] = mms.fit_transform(dataset.iloc[:, 3:])

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

    scaler = StandardScaler()
    # Fit on training set only.
    # scaler.fit(x_train)
    # # Apply transform to both the training set and the test set.
    # x_train = scaler.transform(x_train)
    # x_test = scaler.transform(x_test)




    pca = PCA(.75)
    pca.fit(x_train)
    print("pca.n_components = ", pca.n_components_)
    x_train, x_test = pca.transform(x_train), pca.transform(x_test)

    clf = MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=150, alpha=0.0001,
                        verbose=10, random_state=21)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    accuracy_score(y_test, y_pred)
    # cm = confusion_matrix(y_test, y_pred)
    # print(accuracy_score, cm)
    # sns.heatmap(cm, center=True)
    # plt.show()

    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sns.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()


    total_time = datetime.now() - starttime
    print(total_time)
    fpr2, tpr2, threshold = roc_curve(y_test, clf.predict_proba(x_test)[:,1])
    roc_auc2 = auc(fpr2, tpr2)

    plt.figure()
    plt.title('Mlp classifier')
    plt.plot(fpr2, tpr2, label='MLP AUC = %0.2f' % roc_auc2)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    #apply_model(dataset)
    #features(dataset)
    #x=1


if __name__ == '__main__':
    main()