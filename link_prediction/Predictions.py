from collections import Counter
from datetime import datetime

import pandas as pd
from IPython.display import display
from py2neo import Graph
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def down_sample(df):
    copy = df.copy()
    zero = Counter(copy.label.values)[0]
    un = Counter(copy.label.values)[1]
    n = zero - un
    copy = copy.drop(copy[copy.label == 0].sample(n=n, random_state=1).index)
    return copy.sample(frac=1)


def create_model(graph, starttime):
    # nodes that have edge between them
    train_existing_links = graph.run("""
    MATCH (u:User)-[:TRADES]->(other:User)
    RETURN id(u) AS node1, id(other) AS node2, 1 AS label
    """).to_data_frame()

    # edges that doesnt exist so far and take 0 in the class variable
    train_missing_links = graph.run("""
    MATCH (u:User)
    WHERE (u)-[:TRADES]-()
    MATCH (u)-[:TRADES*2..3]-(other)
    WHERE not((u)-[:TRADES]-(other))
    RETURN id(u) AS node1, id(other) AS node2, 0 AS label
    """).to_data_frame().info(memory_usage='int16')

    train_missing_links = train_missing_links.drop_duplicates()
    training_df = train_missing_links.append(train_existing_links, ignore_index=True)
    del train_missing_links
    training_df['label'] = training_df['label'].astype('category')
    training_df = down_sample(training_df)
    print(training_df.head())
    print('Finish Training')
    del train_existing_links
    test_existing_links = graph.run("""
    MATCH (u1:User)-[:TRADES]->(u2:User)
    RETURN id(u1) AS node1, id(u2) AS node2, 1 AS label 
    """).to_data_frame()

    test_missing_links = graph.run("""
    MATCH (u1:User)
    WHERE (u1)-[:TRADES]-()
    MATCH (u1)-[:TRADES*2..3]-(other)
    WHERE not((u1)-[:TRADES]-(other))
    RETURN id(u1) AS node1, id(other) AS node2, 0 AS label
    """).to_data_frame()
    test_missing_links = test_missing_links.drop_duplicates()

    test_df = test_missing_links.append(test_existing_links, ignore_index=True)
    del test_existing_links
    test_df['label'] = test_df['label'].astype('category')
    test_df = down_sample(test_df)
    print(test_df.head())
    # Machine learning pipeline based on random forest Classifier
    # Strong and weak features
    classifier = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=0)

    # apply the function to our Data frame
    training_df = apply_graphy_features(training_df, "TRADES")
    print(training_df.head())

    test_df = apply_graphy_features(test_df, "TRADES")
    print(test_df.head())

    # How Data looks like
    columns = ["cn"]

    X = training_df[columns]
    y = training_df["label"]
    classifier.fit(X, y)

    del X
    del y
    del training_df

    predictions = classifier.predict(test_df[columns])
    y_test = test_df["label"]

    display("Accuracy", accuracy_score(y_test, predictions))
    display("Precision", precision_score(y_test, predictions))
    display("Recall", recall_score(y_test, predictions))
    del predictions
    print(sorted(list(zip(columns, classifier.feature_importances_)), key=lambda x: x[1] * -1))

    columns = ["cn", "pa", "tn"]

    X = training_df[columns]
    y = training_df["label"]
    classifier.fit(X, y)

    predictions = classifier.predict(test_df[columns])
    y_test = test_df["label"]

    display("Accuracy", accuracy_score(y_test, predictions))
    display("Precision", precision_score(y_test, predictions))
    display("Recall", recall_score(y_test, predictions))

    sorted(list(zip(columns, classifier.feature_importances_)), key=lambda x: x[1] * -1)


def apply_graphy_features(data, rel_type, graph):
    query = """
    UNWIND $pairs AS pair
    MATCH (p1) WHERE id(p1) = pair.node1
    MATCH (p2) WHERE id(p2) = pair.node2
    RETURN pair.node1 AS node1,
           pair.node2 AS node2,
           gds.alpha.linkprediction.commonNeighbors(
               p1, p2, {relationshipQuery: $relType}) AS cn,
           gds.alpha.linkprediction.preferentialAttachment(
               p1, p2, {relationshipQuery: $relType}) AS pa,
           gds.alpha.linkprediction.totalNeighbors(
               p1, p2, {relationshipQuery: $relType}) AS tn
    """
    pairs = [{"node1": node1, "node2": node2} for node1, node2 in data[["node1", "node2"]].values.tolist()]
    features = graph.run(query, {"pairs": pairs, "relType": rel_type}).to_data_frame()
    return pd.merge(data, features, on=["node1", "node2"])


def main():
    starttime = datetime.now()
    graph = Graph('127.0.0.1', password='leomamao971')
    create_model(graph, starttime)
    total_time = datetime.now() - starttime
    print(total_time)


if __name__ == '__main__':
    main()
