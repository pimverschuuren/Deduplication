# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:57 2019

@author: pverschuuren
"""

from helpers import *


# Load the data. This data has been undersampled in the
# non-duplicate class.
data = read_csvfile('data/data_undersampled.csv')

# Build and compile the neural network.
NN_model = get_NN(data)

# Train and test the model according to a k-fold
# cross-validation scheme.
NN_scores = cross_val_KFold(5, NN_model, data, 0.5)

# Calculate and print the performance metric averages
# and standard deviations based on all folds.
calculate_mean_sd(NN_scores)

# Build and compile the boosted decision tree.
BDT_model = get_BDT()

# Train and test the model according to a k-fold
# cross-validation scheme.
BDT_scores = cross_val_KFold(5, BDT_model, data, 0.5)

# Calculate and print the performance metric averages
# and standard deviations based on all folds.
calculate_mean_sd(BDT_scores)
