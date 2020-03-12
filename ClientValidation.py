# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:57 2019

@author: pverschuuren
"""

from helpers import *

# Get the results of the different models.
data_BDT =read_csvfile('data/BDT_result.csv')
data_FISCAL = read_csvfile('data/FISCAL_result.csv')
data_NN = read_csvfile('data/NN_result.csv')


# Plot the ROCs of all three models in one figure.
multi_ROC(data_FISCAL, data_BDT, data_NN)

# Define the probability threshold
threshold = 0.5

# Get the performance metrics of the model.
FISCAL_accs, FISCAL_fprs, FISCAL_fnrs, FISCAL_sens, FISCAL_spec = perf_metrics(data_FISCAL,threshold)
BDT_accs, BDT_fprs, BDT_fnrs, BDT_sens, BDT_spec = perf_metrics(data_BDT,threshold)
NN_accs, NN_fprs, NN_fnrs, NN_sens, NN_spec = perf_metrics(data_NN,threshold)

print('====== Accuracy ======\n')

print('FISCAL: '+str(FISCAL_accs)+'\n')
print('BDT: '+str(BDT_accs)+'\n')
print('NN: '+str(NN_accs)+'\n')

print('====== False positive rates ======\n')

print('FISCAL: '+str(FISCAL_fprs)+'\n')
print('BDT: '+str(BDT_fprs)+'\n')
print('NN: '+str(NN_fprs)+'\n')

print('====== False negative rates ======\n')

print('FISCAL: '+str(FISCAL_fnrs)+'\n')
print('BDT: '+str(BDT_fnrs)+'\n')
print('NN: '+str(NN_fnrs)+'\n')

print('====== Sensitivity ======\n')

print('FISCAL: '+str(FISCAL_sens)+'\n')
print('BDT: '+str(BDT_sens)+'\n')
print('NN: '+str(NN_sens)+'\n')

print('====== Specificity ======\n')

print('FISCAL: '+str(FISCAL_spec)+'\n')
print('BDT: '+str(BDT_spec)+'\n')
print('NN: '+str(NN_spec)+'\n')
