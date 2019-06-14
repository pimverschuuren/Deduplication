# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:57 2019

@author: pverschuuren
"""

from helpers import *


# Load the data.
data = read_csvfile('data.csv')

# Build and compile the neural network.
model = get_NN(data)

# Train and test the model according to a k-fold
# cross-validation scheme.
model, p_metrics = train_KFold(5, model, data, 0.5)


# Calculate and print the performance metric averages
# and standard deviations based on all folds.
calculate_mean_sd(p_metrics)


