import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.ensemble import GradientBoostingClassifier
from utils import get_hmeq_data, get_pulsar_data, compute_stats, plot_learning_curve, get_optimized_classifier
import sys
import warnings
warnings.filterwarnings('ignore')

def run_boosting(name, x_train, x_test, y_train, y_test):
    print ("Working on {} data".format(name))

    img_name = "images/{}_boosting_learning_curve.png".format(name)
    img_title = '{} Boosting Learning Curve'.format(name)
    report_name = "reports/{}_boosting_output.txt".format(name)
    
    sys.stdout = open(report_name, "w")

    tuned_parameters = {
        "learning_rate": [0.05, 0.1, 0.15],
        "max_depth":[2, 3, 4, 5],
        "n_estimators":[5, 10, 15, 20]
    }

    clf = get_optimized_classifier(
        estimator=GradientBoostingClassifier(random_state=99),
        tuned_parameters=tuned_parameters,
        x_train=x_train,
        y_train=y_train
        )

    print ("here")
    best_params = clf.best_params_
    print (best_params)
    optimized_clf = GradientBoostingClassifier(**best_params, random_state=99)

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
    print ("Finished {} boosting!".format(name))

def run_pulsar_boosting():
    x_train, x_test, y_train, y_test = get_pulsar_data()
    run_boosting("pulsar", x_train, x_test, y_train, y_test)

def run_hmeq_boosting():
    x_train, x_test, y_train, y_test = get_hmeq_data()
    run_boosting("hmeq", x_train, x_test, y_train, y_test)

print ("Running Boosting Code, this should take a minute or two")

run_pulsar_boosting()
run_hmeq_boosting()

print ("Finished Running Boosting")

