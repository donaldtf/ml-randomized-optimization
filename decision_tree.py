# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.tree import DecisionTreeClassifier
from utils import get_hmeq_data, get_pulsar_data, run_optimized, plot_learning_curve, get_optimized_classifier
import sys

# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

def run_dt(name, x_train, x_test, y_train, y_test):
    print ("Working on {} data...".format(name))

    img_name = "images/{}_decision_tree_learning_curve.png".format(name)
    img_title = '{} Decision Tree Learning Curve'.format(name)
    report_name = "reports/{}_decision_tree_output.txt".format(name)

    sys.stdout = open(report_name, "w")

    tuned_parameters = [{
        'min_samples_split': list(range(40,70)),
        "min_samples_leaf": list(range(1, 10)),
        }]

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

    run_optimized(optimized_clf, x_train, y_train, x_test, y_test)

    sys.stdout = sys.__stdout__
    print ("Finished {} decision tree!".format(name))
    print()

def run_pulsar_dt():
    x_train, x_test, y_train, y_test = get_pulsar_data()
    run_dt("Pulsar", x_train, x_test, y_train, y_test)

def run_hmeq_dt():
    x_train, x_test, y_train, y_test = get_hmeq_data()
    run_dt("HMEQ", x_train, x_test, y_train, y_test)

if __name__ == "__main__":
    print ("Running Decision Tree Code, this should take a minute or two")

    run_pulsar_dt()
    run_hmeq_dt()

    print ("Finished Running Decision Tree")

