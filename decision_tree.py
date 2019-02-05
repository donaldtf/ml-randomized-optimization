import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import get_pulsar_data, compute_stats
from plot_learning_curve import plot_learning_curve
from tune_params import get_optimized_classifier
import sys

print ("Running Decision Tree Code, this should take a minute or two")

# Print to decision_tree_file
sys.stdout = open("decision_tree_output.txt", "w")

# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

# https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
# y = pulsar_data.pop('target_class').values

pulsar = get_pulsar_data()

# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html
features = list(pulsar.columns.values)
features.remove("target_class")

y = pulsar["target_class"]
x = pulsar[features]

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.20, random_state=99)

min_sample_list = list(range(2,5))
tuned_parameters = [{'min_samples_split': min_sample_list}]

clf = get_optimized_classifier(
    estimator=DecisionTreeClassifier(random_state=99),
    tuned_parameters=tuned_parameters,
    x_train=x_train,
    y_train=y_train
    )

best_min_samples = clf.best_params_['min_samples_split']
optimized_clf = DecisionTreeClassifier(min_samples_split=best_min_samples, random_state=99)

plot_learning_curve(
    optimized_clf,
    title='Decision Tree Learning Curve',
    file_name="decision_tree_learning_curve.png",
    X=x_train,
    y=y_train,
    ylim=None,
    cv=5,
    n_jobs=None,
    scoring="f1",
    train_sizes=np.linspace(.1, 1.0, 5)
    )

optimized_clf.fit(x_train, y_train)
y_pred = optimized_clf.predict(x_test)

compute_stats(y_test, y_pred)

sys.stdout = sys.__stdout__
print ("Finished!")
