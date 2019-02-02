import numpy as np
import graphviz 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split, cross_validate
from utils import get_pulsar_data, compute_stats

# @TODO: Add sklearn citation
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

# https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
# y = pulsar_data.pop('target_class').values

pulsar = get_pulsar_data()

# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
features = list(pulsar.columns.values)
features.remove("target_class")

y = pulsar["target_class"]
x = pulsar[features]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.90)

clf = DecisionTreeClassifier(min_samples_split=5, random_state=99)

scoring = ["f1", "accuracy"]
scores = cross_validate(clf, x_train, y_train, scoring=scoring, cv=10, return_train_score=False)
print (scores)

clf.fit(x_train, y_train)

predictions = clf.predict(x_test)

compute_stats(y_test, predictions)

# https://scikit-learn.org/stable/modules/tree.html
dot_data = export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("pulsar") 
