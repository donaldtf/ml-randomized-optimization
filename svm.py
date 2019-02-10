import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.svm import SVC
from utils import get_hmeq_data, get_pulsar_data, run_optimized, plot_learning_curve, get_optimized_classifier, plot_iterations
import sys

def run_svm(name, x_train, x_test, y_train, y_test, tuned_parameters, iter_range):
    print ("Working on {} data...".format(name))

    img_name = "images/{}_svm_learning_curve.png".format(name)
    img_title = '{} SVM Learning Curve'.format(name)
    iter_title = '{} SVM Iteration Learning Curve'.format(name)
    iter_name = "iteration_curves/{}_svm.png".format(name)
    report_name = "reports/{}_svm_output.txt".format(name)
    
    sys.stdout = open(report_name, "w")

    clf = get_optimized_classifier(
        estimator=SVC(random_state=99),
        tuned_parameters=tuned_parameters,
        x_train=x_train,
        y_train=y_train
        )

    best_params = clf.best_params_
    optimized_clf = SVC(**best_params, random_state=99)

    plot_learning_curve(
        optimized_clf,
        title=img_title,
        file_name=img_name,
        X=x_train,
        y=y_train,
        )

    run_optimized(optimized_clf, x_train, y_train, x_test, y_test)

    plot_iterations(
        optimized_clf,
        file_name=iter_name,
        title=iter_title,
        X=x_test,
        y=y_test,
        param_name="max_iter",
        param_range=iter_range
    )

    sys.stdout = sys.__stdout__
    print ("Finished {} SVM!".format(name))
    print()

def run_pulsar_svm():
    tuned_parameters = {
        "C": [0.1, 1, 10],
        'gamma': [0.0001, 0.001, 0.01]
    }

    iter_range = np.arange(1,150,15)

    x_train, x_test, y_train, y_test = get_pulsar_data()
    run_svm("Pulsar", x_train, x_test, y_train, y_test, tuned_parameters, iter_range)

def run_hmeq_svm():
    tuned_parameters = {
        "C": [1, 5, 10, 15],
        'gamma': [0.000000001, 0.00000001, 0.0000001]
    }

    iter_range = np.arange(1,260,20)

    x_train, x_test, y_train, y_test = get_hmeq_data()
    run_svm("HMEQ", x_train, x_test, y_train, y_test, tuned_parameters, iter_range)

if __name__ == "__main__":
    print ("Running SVM Code, this should take a minute or two")

    run_pulsar_svm()
    # run_hmeq_svm()

    print ("Finished Running SVM")

