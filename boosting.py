# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.ensemble import GradientBoostingClassifier
from utils import get_hmeq_data, get_pulsar_data, run_optimized, plot_learning_curve, get_optimized_classifier, plot_iteration_curve
import sys

def run_boosting(name, x_train, x_test, y_train, y_test, tuned_parameters):
    print ("Working on {} data...".format(name))

    img_name = "images/{}_boosting_learning_curve.png".format(name)
    iter_name = "iteration_curves/{}_boosting.png".format(name)
    img_title = '{} Boosting Learning Curve'.format(name)
    report_name = "reports/{}_boosting_output.txt".format(name)
    
    sys.stdout = open(report_name, "w")

    clf = get_optimized_classifier(
        estimator=GradientBoostingClassifier(random_state=99),
        tuned_parameters=tuned_parameters,
        x_train=x_train,
        y_train=y_train
        )

    best_params = clf.best_params_
    optimized_clf = GradientBoostingClassifier(**best_params, random_state=99)

    plot_learning_curve(
        optimized_clf,
        title=img_title,
        file_name=img_name,
        X=x_train,
        y=y_train,
        )

    run_optimized(optimized_clf, x_train, y_train, x_test, y_test)

    # plot_iteration_curve(optimized_clf, iter_name, x_test, y_test)

    sys.stdout = sys.__stdout__
    print ("Finished {} boosting!".format(name))
    print()

def run_pulsar_boosting():
    x_train, x_test, y_train, y_test = get_pulsar_data()

    tuned_parameters = {
        "learning_rate": [0.1, 0.15, 0.2, 0.25, 0.3],
        "max_depth":[2, 3, 4, 5, 6],
        "n_estimators":[10, 15, 20, 25, 30]
    }

    run_boosting("Pulsar", x_train, x_test, y_train, y_test, tuned_parameters)

def run_hmeq_boosting():
    x_train, x_test, y_train, y_test = get_hmeq_data()

    tuned_parameters = {
        "learning_rate": [0.1, 0.15, 0.2, 0.25, 0.3],
        "max_depth":[2, 3, 4, 5, 6],
        "n_estimators":[10, 15, 20, 25, 30]
    }

    run_boosting("HMEQ", x_train, x_test, y_train, y_test, tuned_parameters)

if __name__ == "__main__":
    print ("Running Boosting Code, this should take a minute or two")

    run_pulsar_boosting()
    run_hmeq_boosting()

    print ("Finished Running Boosting")

