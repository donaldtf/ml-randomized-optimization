import numpy as np
from problems import one_max, knapsack, queens, run_rhc, run_sa, run_ga, run_mimic

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

np.random.seed(92)

num_runs = 5

queen_range = range(4, 32, 4)
one_max_range = range(10, 60, 10)
knapsack_range = range(3, 9)

print()
print ("Running Random Hill Climbing...")

print ("\tQueens...")
queen_rhc_avgs, queen_rhc_times = run_rhc(queens, queen_range, num_runs)
print ("\tOne Max...")
one_max_rhc_avgs, one_max_rhc_times = run_rhc(one_max, one_max_range, num_runs)
print ("\tKnapsack...")
knapsack_rhc_avgs, knapsack_rhc_times = run_rhc(knapsack, knapsack_range, num_runs)

print()
print ("Running Simulated Annealing...")

print ("\tQueens...")
queen_sa_avgs, queen_sa_times = run_sa(queens, queen_range, num_runs)
print ("\tOne Max...")
one_max_sa_avgs, one_max_sa_times = run_sa(one_max, one_max_range, num_runs)
print ("\tKnapsack...")
knapsack_sa_avgs, knapsack_sa_times = run_sa(knapsack, knapsack_range, num_runs)

print()
print ("Running Genetic Algorithm...")

print ("\tQueens...")
queen_ga_avgs, queen_ga_times = run_ga(queens, queen_range, num_runs)
print ("\tOne Max...")
one_max_ga_avgs, one_max_ga_times = run_ga(one_max, one_max_range, num_runs)
print ("\tKnapsack...")
knapsack_ga_avgs, knapsack_ga_times = run_ga(knapsack, knapsack_range, num_runs)

print()
print ("Running MIMIC...")

print ("\tQueens...")
queen_mimic_avgs, queen_mimic_times = run_mimic(queens, queen_range, num_runs)
print ("\tOne Max...")
one_max_mimic_avgs, one_max_mimic_times = run_mimic(one_max, one_max_range, num_runs)
print ("\tKnapsack...")
knapsack_mimic_avgs, knapsack_mimic_times = run_mimic(knapsack, knapsack_range, num_runs)


# ----- N QUEEN Problem ----

# Plot Queen Values
plt.figure()
plt.title("N Queens Optimization Values")
plt.xlabel("Number of Queens")
plt.ylabel("Number of Non Attacking Queens")

plt.grid()

n_queens = np.array(queen_range)

plt.plot(queen_range, n_queens - np.array(queen_rhc_avgs), 'o-', color="red", label="Random Hill Climbing")
plt.plot(queen_range, n_queens - np.array(queen_sa_avgs), 'o-', color="green", label="Simulated Annealing")
plt.plot(queen_range, n_queens - np.array(queen_ga_avgs), 'o-', color="blue", label="Genetic Algorithm")
plt.plot(queen_range, n_queens - np.array(queen_mimic_avgs), 'o-', color="black", label="MIMIC")

plt.legend(loc="best")

plt.savefig("images/queen_values.png")

# Plot Queen Times
plt.figure()
plt.title("N Queens Optimization Execution Time")
plt.xlabel("Number of Queens")
plt.ylabel("Time (sec)")

plt.grid()

plt.plot(queen_range, queen_rhc_times, 'o-', color="red", label="Random Hill Climbing")
plt.plot(queen_range, queen_sa_times, 'o-', color="green", label="Simulated Annealing")
plt.plot(queen_range, queen_ga_times, 'o-', color="blue", label="Genetic Algorithm")
plt.plot(queen_range, queen_mimic_times, 'o-', color="black", label="MIMIC")

plt.legend(loc="best")

plt.savefig("images/queen_time.png")

# ----- OneMax Problem ----

# Plot OneMax Values
plt.figure()
plt.title("OneMax Optimization Values")
plt.xlabel("Number of Bits")
plt.ylabel("Value of Bit String")

plt.grid()

plt.plot(one_max_range, one_max_rhc_avgs, 'o-', color="red", label="Random Hill Climbing")
plt.plot(one_max_range, one_max_sa_avgs, 'o-', color="green", label="Simulated Annealing")
plt.plot(one_max_range, one_max_ga_avgs, 'o-', color="blue", label="Genetic Algorithm")
plt.plot(one_max_range, one_max_mimic_avgs, 'o-', color="black", label="MIMIC")

plt.legend(loc="best")

plt.savefig("images/one_max_values.png")

# Plot OneMax Times
plt.figure()
plt.title("OneMax Optimization Execution Time")
plt.xlabel("Number of Bits")
plt.ylabel("Time (sec)")

plt.grid()

plt.plot(one_max_range, one_max_rhc_times, 'o-', color="red", label="Random Hill Climbing")
plt.plot(one_max_range, one_max_sa_times, 'o-', color="green", label="Simulated Annealing")
plt.plot(one_max_range, one_max_ga_times, 'o-', color="blue", label="Genetic Algorithm")
plt.plot(one_max_range, one_max_mimic_times, 'o-', color="black", label="MIMIC")

plt.legend(loc="best")

plt.savefig("images/one_max_time.png")

# ----- Knapsack Problem ----

# Plot Knapsack Values
plt.figure()
plt.title("Knapsack Optimization Values")
plt.xlabel("Number of Items in Knapsack")
plt.ylabel("Knapsack Value")

plt.grid()

plt.plot(knapsack_range, knapsack_rhc_avgs, 'o-', color="red", label="Random Hill Climbing")
plt.plot(knapsack_range, knapsack_sa_avgs, 'o-', color="green", label="Simulated Annealing")
plt.plot(knapsack_range, knapsack_ga_avgs, 'o-', color="blue", label="Genetic Algorithm")
plt.plot(knapsack_range, knapsack_mimic_avgs, 'o-', color="black", label="MIMIC")

plt.legend(loc="best")

plt.savefig("images/knapsack_values.png")

# Plot Knapsack Times
plt.figure()
plt.title("Knapsack Optimization Execution Time")
plt.xlabel("Number of Items in Knapsack")
plt.ylabel("Time (sec)")

plt.grid()

plt.plot(knapsack_range, knapsack_rhc_times, 'o-', color="red", label="Random Hill Climbing")
plt.plot(knapsack_range, knapsack_sa_times, 'o-', color="green", label="Simulated Annealing")
plt.plot(knapsack_range, knapsack_ga_times, 'o-', color="blue", label="Genetic Algorithm")
plt.plot(knapsack_range, knapsack_mimic_times, 'o-', color="black", label="MIMIC")

plt.legend(loc="best")

plt.savefig("images/knapsack_time.png")

