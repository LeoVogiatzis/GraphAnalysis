from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

data = pd.read_pickle('./dataset_model2.pkl')
# dataset = dataset.set_index(['id1', 'id2'])
#
# dataset['type'] = dataset['type'].astype('category')
# dataset['type'] = dataset['type'].cat.codes

data['type'] = data['type'].astype('category')
data['type'] = data['type'].cat.codes
data['weight'] = 1

data = data.groupby(
['id1', 'id2', 'type', 'label', 'Resource_allocation', 'Preferential_Attachment', 'betweeness_centrality',
         'betweeness_centrality fot id2', 'closeness_centrality', 'closeness_centrality for id2'], as_index=False)[
        'weight'].sum()


# data = data.drop(['label', 'Timestamp', 'weight'], axis=1)
mms = MinMaxScaler()

data.iloc[:, 3:] = mms.fit_transform(data.iloc[:, 3:])
data = data.set_index(['id1', 'id2'])
X = data.drop(columns=['type'])
Y = data['type']
Y = Y.astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

scores = {}
k_range = range(1, 3)
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    y_pred = pd.Series(y_pred)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

[print(i) for i in scores_list]

plt.plot(k_range, scores_list)
plt.xlabel('Value of k for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

