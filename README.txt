##### Machine Learning - OMSCS Spring 2019 ####
##### Author - Donald Ford ####


### Getting Started ###

In order to recreate the results shown in my report, please clone the following repo from github.

https://github.com/donaldtf/ml-randomized-optimization

Or you can just use this command to clone it via ssh: `git@github.com:donaldtf/ml-randomized-optimization.git`

This repo contains all the code needed to reproduce my results, including the data sets that were used. The project structure looks like this:

/data - this holds the data set (pulsar_stars) that is used with the neural network
/images - this stores various graphs that are generated as you run the code
/reports - while running each algorithm, I feed the standard output into a report file here instead of to the console.
           This file holds test data performance and wall clock times
/utils.py - this is a utility file that holds shared functionality between the algorithms 
            (loading and prepping data, timing algorithms, etc)
/problems.py - this file holds some helper functions for running optimization problems


### Install Dependencies ###

The code relies on the following dependencies in order to run. You can install them via your favorite method (conda, pip, etc.).

- scikit-learn
- pandas
- numpy
- matplotlib

The last dependency is mlrose. This will need to be installed via pip. See the readme here for installation instructions: https://github.com/gkhayes/mlrose

Once these are all installed you should be ready to run the code


### Running the code ###

The code has been broken up into two different parts for convenience of running each section.
The first deals with the first half of the assignment where optimization algorithms are used to determine weights for a neural network.
The second half deals with the 3 optimization problems I selected and runs each of the four optimization algorithms on these problems.

1. Neural Network code
    To run each optimizer on the neural net, you can run the following command:
        `python run_nn.py`
    
    Note: Running this command should take 3-5 minutes

2. Optimization Problems
    To run all of the optimization algorithms on each optimization problem, you can run the following command:
        `python run_optimizations.py`
    This will generate all of the optimization problem graphs shown in my report.
    
    Note: Time graphs may vary slightly from those shown in the report since timing cannot be controlled.
    Running all of these optimization problems should take between 5-10 minutes


### Citations ###

Hayes, G. (2019). mlrose: Machine Learning, Randomized Optimization and SEarch package for Python. https://github.com/gkhayes/mlrose. Accessed: 2 March 2019.
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
