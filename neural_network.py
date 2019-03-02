import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.neural_network import MLPClassifier
from utils import get_pulsar_data, run_optimized, plot_learning_curve, get_optimized_classifier, plot_iterations
import sys

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# Thanks to this source for showing grid search with a neural net
# https://www.kaggle.com/hatone/mlpclassifier-with-gridsearchcv

def run_nn(name, x_train, x_test, y_train, y_test, tuned_parameters, iter_range):
    print ("Working on {} data...".format(name))

    img_name = "images/{}_nn_learning_curve.png".format(name)
    img_title = '{} Neural Net Learning Curve'.format(name)
    iter_title = '{} Neural Net Iteration Learning Curve'.format(name)
    iter_name = "iteration_curves/{}_nn.png".format(name)
    report_name = "reports/{}_nn_output.txt".format(name)
    
    sys.stdout = open(report_name, "w")

    clf = get_optimized_classifier(
        estimator=MLPClassifier(),
        tuned_parameters=tuned_parameters,
        x_train=x_train,
        y_train=y_train
        )

    best_params = clf.best_params_
    optimized_clf = MLPClassifier(**best_params)

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
    print ("Finished {} Neural Net!".format(name))
    print()

def run_pulsar_nn():
    tuned_parameters = {
        "C": [0.1, 1, 10],
        'gamma': [0.001, 0.01, 0.1]
    }

    tuned_parameters = {
        'max_iter': [1000],
        'alpha': 10.0 ** -np.arange(1, 5),
        'hidden_layer_sizes':np.arange(10, 15),
        'random_state':[99]
        }

    iter_range = np.arange(1,200,10)

    x_train, x_test, y_train, y_test = get_pulsar_data()
    run_nn("Pulsar", x_train, x_test, y_train, y_test, tuned_parameters, iter_range)

if __name__ == "__main__":
    print ("Running Neural Net Code, this should take a minute or two")

    run_pulsar_nn()

    print ("Finished Running Neural Net")

