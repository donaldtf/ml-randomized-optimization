import pandas as pd
import numpy as np
# Thanks to Scikit-learn
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, classification_report
import timeit

# Thanks to these sources for examples on loading data in pandas
# https://medium.com/dunder-data/from-pandas-to-scikit-learn-a-new-exciting-workflow-e88e2271ef62
# http://chrisstrelioff.ws/sandbox/2015/06/08/decision_trees_in_python_with_scikit_learn_and_pandas.html

# https://www.kaggle.com/pavanraj159/predicting-a-pulsar-star/version/1
def get_pulsar_data():
    pulsar = pd.read_csv('data/pulsar_stars.csv')
    target = "target_class"

    features = list(pulsar.columns.values)
    features.remove(target)

    y = pulsar[target]
    x = pulsar[features]

    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.10, random_state=99)

    return x_train, x_test, y_train, y_test

def compute_stats(y_true, y_pred):
    print ("Final Performance on Test Set")
    mse = mean_squared_error(y_true, y_pred)
    print ("MSE: " + str(mse))
    print()
    
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy Score: " + str(accuracy * 100) + "%")
    print()
    
    f1 = f1_score(y_true, y_pred)
    print("F1 Score: " + str(f1))
    print()

    print(classification_report(y_true, y_pred))
    print()


def run_optimized(optimized_clf, x_train, y_train, x_test, y_test):
    train_start = timeit.default_timer()
    optimized_clf.fit(x_train, y_train)
    train_stop = timeit.default_timer()

    print('Train Time: ', train_stop - train_start)  
    print()

    test_start = timeit.default_timer()
    y_pred = optimized_clf.predict(x_test)
    test_stop = timeit.default_timer()

    print('Test Time: ', test_stop - test_start)  
    print()

    compute_stats(y_test, y_pred)
    