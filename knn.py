import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.neighbors import KNeighborsClassifier
from utils import get_hmeq_data, get_pulsar_data, compute_stats, plot_learning_curve, get_optimized_classifier
import sys
import warnings
# warnings.filterwarnings('ignore')

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

def run_knn(name, x_train, x_test, y_train, y_test, min_sample_list, scoring):
    print ("Working on {} data".format(name))

    img_name = "images/{}_knn_learning_curve.png".format(name)
    img_title = '{} kNN Learning Curve'.format(name)
    report_name = "reports/{}_knn_output.txt".format(name)

    sys.stdout = open(report_name, "w")

    tuned_parameters = [{'n_neighbors': min_sample_list}]

    clf = get_optimized_classifier(
        estimator=KNeighborsClassifier(),
        tuned_parameters=tuned_parameters,
        x_train=x_train,
        y_train=y_train
        )

    best_min_samples = clf.best_params_['n_neighbors']
    optimized_clf = KNeighborsClassifier(n_neighbors=best_min_samples)

    plot_learning_curve(
        optimized_clf,
        title=img_title,
        file_name=img_name,
        X=x_train,
        y=y_train,
        scoring=scoring,
        )

    optimized_clf.fit(x_train, y_train)
    y_pred = optimized_clf.predict(x_test)

    compute_stats(y_test, y_pred)

    sys.stdout = sys.__stdout__
    print ("Finished {} knn!".format(name))

def run_pulsar_dt():
    x_train, x_test, y_train, y_test = get_pulsar_data()
    min_sample_list = list(range(1,20))
    run_knn("pulsar", x_train, x_test, y_train, y_test, min_sample_list, scoring="f1")

def run_hmeq_dt():
    x_train, x_test, y_train, y_test = get_hmeq_data()
    min_sample_list = list(range(1,20))
    run_knn("hmeq", x_train, x_test, y_train, y_test, min_sample_list, scoring="f1")

print ("Running kNN Code, this should take a minute or two")

run_pulsar_dt()
run_hmeq_dt()

print ("Finished Running kNN")

