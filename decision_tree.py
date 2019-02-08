import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.tree import DecisionTreeClassifier
from utils import get_hmeq_data, get_pulsar_data, compute_stats, plot_learning_curve, get_optimized_classifier
import sys

def run_dt(name, x_train, x_test, y_train, y_test):
    print ("Working on {} data".format(name))

    img_name = "images/{}_decision_tree_learning_curve.png".format(name)
    img_title = '{} Decision Tree Learning Curve'.format(name)
    report_name = "reports/{}_decision_tree_output.txt".format(name)

    sys.stdout = open(report_name, "w")

    tuned_parameters = [{'min_samples_split': list(range(2,100))}]

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
        title=img_title,
        file_name=img_name,
        X=x_train,
        y=y_train,
        )

    optimized_clf.fit(x_train, y_train)
    y_pred = optimized_clf.predict(x_test)

    compute_stats(y_test, y_pred)

    sys.stdout = sys.__stdout__
    print ("Finished {} decision tree!".format(name))

def run_pulsar_dt():
    x_train, x_test, y_train, y_test = get_pulsar_data()
    run_dt("pulsar", x_train, x_test, y_train, y_test)

def run_hmeq_dt():
    x_train, x_test, y_train, y_test = get_hmeq_data()
    run_dt("hmeq", x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    print ("Running Decision Tree Code, this should take a minute or two")

    run_pulsar_dt()
    run_hmeq_dt()

    print ("Finished Running Decision Tree")

