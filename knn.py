# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.neighbors import KNeighborsClassifier
from utils import get_hmeq_data, get_pulsar_data, run_optimized, plot_learning_curve, get_optimized_classifier
import sys

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

def run_knn(name, x_train, x_test, y_train, y_test, tuned_parameters):
    print ("Working on {} data...".format(name))

    img_name = "images/{}_knn_learning_curve.png".format(name)
    img_title = '{} kNN Learning Curve'.format(name)
    report_name = "reports/{}_knn_output.txt".format(name)

    sys.stdout = open(report_name, "w")

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
        )

    run_optimized(optimized_clf, x_train, y_train, x_test, y_test)

    sys.stdout = sys.__stdout__
    print ("Finished {} knn!".format(name))
    print()

def run_pulsar_knn():
    x_train, x_test, y_train, y_test = get_pulsar_data()
    tuned_parameters = [{'n_neighbors': list(range(1,10))}]
    run_knn("pulsar", x_train, x_test, y_train, y_test, tuned_parameters)

def run_hmeq_knn():
    x_train, x_test, y_train, y_test = get_hmeq_data()
    tuned_parameters = [{'n_neighbors': list(range(1,10))}]
    run_knn("hmeq", x_train, x_test, y_train, y_test, tuned_parameters)


if __name__ == "__main__":
    print ("Running kNN Code, this should take a minute or two")

    run_pulsar_knn()
    run_hmeq_knn()

    print ("Finished Running kNN")

