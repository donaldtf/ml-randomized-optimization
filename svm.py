import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.svm import SVC
from utils import get_hmeq_data, get_pulsar_data, compute_stats, plot_learning_curve, get_optimized_classifier
import sys
import warnings
warnings.filterwarnings('ignore')

def run_svm(name, x_train, x_test, y_train, y_test, tuned_parameters):
    print ("Working on {} data".format(name))

    img_name = "images/{}_svm_learning_curve.png".format(name)
    img_title = '{} SVM Learning Curve'.format(name)
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

    optimized_clf.fit(x_train, y_train)
    y_pred = optimized_clf.predict(x_test)

    compute_stats(y_test, y_pred)

    sys.stdout = sys.__stdout__
    print ("Finished {} SVM!".format(name))

def run_pulsar_svm():
    tuned_parameters = {
        "C": [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1]
    }

    x_train, x_test, y_train, y_test = get_pulsar_data()
    run_svm("pulsar", x_train, x_test, y_train, y_test, tuned_parameters)

def run_hmeq_svm():
    tuned_parameters = {
        "C": [1, 5, 10, 15],
        'gamma': [0.000000001, 0.00000001, 0.0000001]
    }

    x_train, x_test, y_train, y_test = get_hmeq_data()
    run_svm("hmeq", x_train, x_test, y_train, y_test, tuned_parameters)

print ("Running SVM Code, this should take a minute or two")

# run_pulsar_svm()
run_hmeq_svm()

print ("Finished Running SVM")

