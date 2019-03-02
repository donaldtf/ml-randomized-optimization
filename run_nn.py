# import mlrose
import numpy as np
import timeit

# import matplotlib as mpl
# mpl.use('agg')
# import matplotlib.pyplot as plt
from neural_network import run_pulsar_nn

start = timeit.default_timer()

print()
print ("---- Neural Network ----")
print()

run_pulsar_nn()

stop = timeit.default_timer()
total_time = stop - start

print ()
print ("FINISHED! Total time taken: " + str(total_time) + " seconds")
