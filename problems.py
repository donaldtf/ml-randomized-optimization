import mlrose
import numpy as np
import random
import timeit

np.random.seed(92)

# https://mlrose.readthedocs.io/en/stable/source/tutorial2.html
# Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python. https://github.com/gkhayes/mlrose. Accessed: 2 March 2019.

def one_max(bit_length = 50):
    fitness_fn = mlrose.OneMax()

    problem = mlrose.DiscreteOpt(length = bit_length, fitness_fn = fitness_fn, max_val = 2)

    return problem

def queens(n_queens = 16):
    fitness_fn = mlrose.Queens()

    problem = mlrose.DiscreteOpt(length = n_queens, maximize=False, fitness_fn = fitness_fn, max_val = n_queens)

    return problem

def knapsack(n_items = 5):
    max_val = 5
    weights = np.random.choice(range(1, 10), n_items)
    values = np.random.choice(range(1, max_val), n_items)
    
    fitness_fn = mlrose.Knapsack(weights, values)

    problem = mlrose.DiscreteOpt(length = n_items, fitness_fn = fitness_fn, max_val = max_val)

    return problem

def run_rhc(prob, value_range, num_runs):
    avgs = []
    times = []

    for value in value_range:
        problem = prob(value)
        print ("\t\tValue: " + str(value))

        run_vals = []
        run_times = []

        for run in range(0, num_runs):
            print ("\t\t\tRun " + str(run))
            start = timeit.default_timer()

            best_state, best_fitness = mlrose.random_hill_climb(problem, restarts=10)

            stop = timeit.default_timer()
            total_time = stop - start

            run_vals.append(best_fitness)
            run_times.append(total_time)

        avgs.append(np.mean(run_vals))
        times.append(np.mean(run_times))

    return avgs, times


def run_sa(prob, value_range, num_runs):
    avgs = []
    times = []

    for value in value_range:
        problem = prob(value)
        print ("\t\tValue: " + str(value))

        run_vals = []
        run_times = []

        for run in range(0, num_runs):
            print ("\t\t\tRun " + str(run))

            start = timeit.default_timer()

            best_state, best_fitness = mlrose.simulated_annealing(problem, max_attempts=20)

            stop = timeit.default_timer()
            total_time = stop - start

            run_vals.append(best_fitness)
            run_times.append(total_time)

        avgs.append(np.mean(run_vals))
        times.append(np.mean(run_times))

    return avgs, times

def run_ga(prob, value_range, num_runs):
    avgs = []
    times = []

    for value in value_range:
        problem = prob(value)
        print ("\t\tValue: " + str(value))

        run_vals = []
        run_times = []

        for run in range(0, num_runs):
            print ("\t\t\tRun " + str(run))
            start = timeit.default_timer()

            best_state, best_fitness = mlrose.genetic_alg(problem, max_attempts=20, mutation_prob=0.25)

            stop = timeit.default_timer()
            total_time = stop - start

            run_vals.append(best_fitness)
            run_times.append(total_time)

        avgs.append(np.mean(run_vals))
        times.append(np.mean(run_times))

    return avgs, times

def run_mimic(prob, value_range, num_runs):
    avgs = []
    times = []

    for value in value_range:
        problem = prob(value)
        print ("\t\tValue: " + str(value))

        run_vals = []
        run_times = []

        for run in range(0, num_runs):
            print ("\t\t\tRun " + str(run))
            start = timeit.default_timer()

            best_state, best_fitness = mlrose.mimic(problem)

            stop = timeit.default_timer()
            total_time = stop - start

            run_vals.append(best_fitness)
            run_times.append(total_time)

        avgs.append(np.mean(run_vals))
        times.append(np.mean(run_times))

    return avgs, times