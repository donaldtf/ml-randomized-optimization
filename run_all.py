import timeit
from decision_tree import run_pulsar_dt, run_hmeq_dt
from knn import run_hmeq_knn, run_pulsar_knn
from boosting import run_hmeq_boosting, run_pulsar_boosting
from svm import run_hmeq_svm, run_pulsar_svm
from neural_network import run_hmeq_nn, run_pulsar_nn

start = timeit.default_timer()

print ("---- Decision Tree ----")
print()

run_hmeq_dt()
run_pulsar_dt()

print()
print ("---- KNN ----")
print()

run_hmeq_knn()
run_pulsar_knn()

print()
print ("---- Boosting ----")
print()

run_hmeq_boosting()
run_pulsar_boosting()

print()
print ("---- SVM ----")
print()

run_hmeq_svm()
run_pulsar_svm()

print()
print ("---- Neural Network ----")
print()

run_hmeq_nn()
run_pulsar_nn()

stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")