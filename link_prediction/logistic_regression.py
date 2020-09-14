from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import seaborn as sn
import statistics
from sklearn import metrics
FILE_NAME = './dataset_model2.pkl'
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, precision_recall_curve, f1_score, auc, roc_auc_score, roc_curve


def create_plots(dataset):
    objects = ('Mean', 'Standard deviation')
    y_pos = np.arange(2)
    values = [dataset['mean'], dataset['standard_deviation']]
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.show()


def calculate_statistics(file_data):
    statistics_data = {'mean': statistics.mean(file_data['label']),
                       'media': statistics.median(file_data['label']),
                       'low_median': statistics.median_low(file_data['label']),
                       'high_median': statistics.median_high(file_data['label']),
                       'standard_deviation': statistics.stdev(file_data['label']),
                       'sample_variance': statistics.variance(file_data['label'])}
    data_to_print = ''.join('{}: {}\n'.format(key, value) for key, value in statistics_data.items())
    create_plots(statistics_data)
    print(data_to_print)


def create_prediction_model(dataset):

    x_train, x_test, y_train, y_test = train_test_split(dataset.drop('label', axis=1),
                                                        dataset['label'], test_size=0.2)
    y_train = y_train.astype('int')
    y_test = y_test.astype('int')
    logistic_regression = LogisticRegression(solver='lbfgs', max_iter=10000)
    logistic_regression.fit(x_train, y_train)
    y_pred = logistic_regression.predict(x_test)
    print('Accuracy: ', metrics.accuracy_score(y_test, y_pred))
    print('R squared: ', metrics.r2_score(y_test, y_pred))

    # get Confusion Matrix
    confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
    fig, ax = plt.subplots()
    ax = sn.heatmap(confusion_matrix, annot=True, fmt='')
    ax.set(yticks=[0, 2], xticks=[0, 1])
    plt.show()

    scores = cross_val_score(logistic_regression, x_train, y_train, cv=10)
    print('start')
    print('Cross-Validation Accuracy Scores', scores)
    scores = pd.Series(scores)
    scores.min(), scores.mean(), scores.max()
    print('Finish')
    # The estimated coefficients will all be around 1:
    print(logistic_regression.coef_)
    predictions = logistic_regression.predict(x_test)
    print(classification_report(y_test, predictions))
    fpr, tpr, _ = metrics.roc_curve(y_test, predictions)
    auc = metrics.roc_auc_score(y_test, predictions)
    plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
    plt.legend(loc=4)
    plt.show()


def preprocess(data):
    data['type'] = data['type'].astype('category')
    data['type'] = data['type'].cat.codes
    data['id1'] = data['id1'].astype(int)
    data['id2'] = data['id2'].astype(int)
    # for aggregated dataset
    data['weight'] = 1
    data = data.groupby(
        ['id1', 'id2', 'type', 'label', 'Resource_allocation', 'Preferential_Attachment', 'betweeness_centrality',
         'betweeness_centrality fot id2', 'closeness_centrality', 'closeness_centrality for id2'], as_index=False)[
        'weight'].sum()
    data['weight'] = data.groupby(['id1', 'id2', 'type'])['weight'].transform(lambda x: x.count())
    # Weight normalization dividing with number of nodes
    edges = int(len(data))
    data['weight'] = data['weight'].div(edges)
    # data = data.drop(['Timestamp'], axis=1)
    mms = MinMaxScaler()
    data.iloc[:, 3:] = mms.fit_transform(data.iloc[:, 3:])
    data = data.set_index(['id1', 'id2'])
    return data


def main():
    data = pd.read_pickle(FILE_NAME)
    data = preprocess(data)
    df0 = data[data['label'] == 0]
    df1 = data[data['label'] == 1]
    plt.scatter(df0['closeness_centrality'], df0['betweeness_centrality'], edgecolors='green', marker='+')
    plt.scatter(df1['Resource_Allocation'], df1['Preferential_Attachment'], edgecolors='yellow', marker='+')
    calculate_statistics(data)
    create_prediction_model(data)
    plt.show()


if __name__ == '__main__':
    main()
