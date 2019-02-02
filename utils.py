import pandas as pd
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score

# def run_test_splits

def get_pulsar_data():
    pulsar = pd.read_csv('data/pulsar_stars.csv')

    return pulsar

def compute_stats(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print ("MSE: " + str(mse))
    accuracy = accuracy_score(y_true, y_pred)
    print("Accuracy Score: " + str(accuracy * 100) + "%")
    f1 = f1_score(y_true, y_pred)
    print("F1 Score: " + str(f1))