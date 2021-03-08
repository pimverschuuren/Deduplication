# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:57 2019

@author: pverschuuren
"""

from helpers import *

# Create a list of points between 0 and 1 that will be used
# as probability thresholds for the classifier outputs.
prob_thres = np.linspace(0.,0.9,50)
prob_thres = np.append(prob_thres,np.linspace(0.901,1.01,30))

# Get the results of the different models but randomly ordered.
frac=0.1
data_BDT =read_csvfile('data/BDT_result.csv').sample(frac=frac)
data_FISCAL = read_csvfile('data/FISCAL_result.csv').sample(frac=frac)
data_NN = read_csvfile('data/NN_result.csv').sample(frac=frac)

# Plot the ROCs of each classifier in one graph.
multi_ROC(data_FISCAL, data_BDT, data_NN)

# Plot histograms with the classifier outputs binned in one graph.
#plot_multihist(data_FISCAL, data_BDT, data_NN)

y_FISCAL_t, y_FISCAL_p = split_label_pred(data_FISCAL)
y_BDT_t, y_BDT_p = split_label_pred(data_BDT)
y_NN_t, y_NN_p = split_label_pred(data_NN)

# Create place holders for the classifier performance metrics.
max_f_fiscal = [0,0]
max_f_bdt = [0,0]
max_f_nn = [0,0]

f_scores = np.empty((0,3),float)
prec_scores = np.empty((0,3),float)
sens_scores = np.empty((0,3),float)
acc_scores = np.empty((0,3),float)
fpr_scores = np.empty((0,3),float)
fnr_scores = np.empty((0,3),float)

tps = np.empty((0,3),float)
tns = np.empty((0,3),float)
fps = np.empty((0,3),float)
fns = np.empty((0,3),float)

# Set the beta parameter for the F-score.
beta = 5

# Scan over the probability thresholds and evaluate the performance metrics.
for i in range(len(prob_thres)):

    # Define the probability threshold.
    threshold = prob_thres[i]

    print("Threshold: "+str(threshold))    

    # Get the confusion matrix for each classifier.
    FISCAL_tp, FISCAL_fp, FISCAL_fn, FISCAL_tn = calculate_conf(y_FISCAL_t, y_FISCAL_p, threshold)
    BDT_tp, BDT_fp, BDT_fn, BDT_tn = calculate_conf(y_BDT_t, y_BDT_p, threshold)
    NN_tp, NN_fp, NN_fn, NN_tn = calculate_conf(y_NN_t, y_NN_p, threshold)    

    # Append the results into an array.
    tps = np.append(tps,np.array([[FISCAL_tp, BDT_tp, NN_tp]]),axis=0)
    fps = np.append(fps,np.array([[FISCAL_fp, BDT_fp, NN_fp]]),axis=0)
    fns = np.append(fns,np.array([[FISCAL_fn, BDT_fn, NN_fn]]),axis=0)
    tns = np.append(tns,np.array([[FISCAL_tn, BDT_tn, NN_tn]]),axis=0)
    
    # Get the accuracy, false positive rate, false negative rate, sensitivity(recall),
    # specificity and precision.
    FISCAL_accs, FISCAL_fprs, FISCAL_fnrs, FISCAL_sens, FISCAL_spec, FISCAL_prec = calculate_performance_metrics(y_FISCAL_t,y_FISCAL_p,threshold)
    BDT_accs, BDT_fprs, BDT_fnrs, BDT_sens, BDT_spec, BDT_prec = calculate_performance_metrics(y_BDT_t,y_BDT_p,threshold)
    NN_accs, NN_fprs, NN_fnrs, NN_sens, NN_spec, NN_prec = calculate_performance_metrics(y_NN_t,y_NN_p,threshold)

    # Get the F-score.
    f_fiscal = (1+beta**2)*(FISCAL_prec*FISCAL_sens)/(beta**2 * FISCAL_prec + FISCAL_sens)
    f_BDT = (1+beta**2)*(BDT_prec*BDT_sens)/(beta**2 * BDT_prec + BDT_sens)
    f_NN = (1+beta**2)*(NN_prec*NN_sens)/(beta**2 * NN_prec + NN_sens)

    # Appen the results into an array.
    f_scores = np.append(f_scores,np.array([[f_fiscal, f_BDT, f_NN]]),axis=0)
    prec_scores = np.append(prec_scores,np.array([[FISCAL_prec, BDT_prec, NN_prec]]),axis=0)
    sens_scores = np.append(sens_scores,np.array([[FISCAL_sens, BDT_sens, NN_sens]]),axis=0)
    acc_scores = np.append(acc_scores,np.array([[FISCAL_accs, BDT_accs, NN_accs]]),axis=0)
    fpr_scores = np.append(fpr_scores,np.array([[FISCAL_fprs, BDT_fprs, NN_fprs]]),axis=0)
    fnr_scores = np.append(fnr_scores,np.array([[FISCAL_fnrs, BDT_fnrs, NN_fnrs]]),axis=0)

    # Get the maximum F-score.
    if f_fiscal > max_f_fiscal[1]:
        max_f_fiscal = [threshold, f_fiscal]
    if f_BDT > max_f_bdt[1]:
        max_f_bdt = [threshold, f_BDT]
    if f_NN > max_f_nn[1]:
        max_f_nn = [threshold, f_NN]

    # Print out the metrics for each probability threshold.

    # print("FISCAL F: "+str(f_fiscal))
    # print("BDT F: "+str(f_BDT))
    # print("NN F: "+str(f_NN))
    # print('====== Accuracy ======\n')

    # print('FISCAL: '+str(FISCAL_accs)+'\n')
    # print('BDT: '+str(BDT_accs)+'\n')
    # print('NN: '+str(NN_accs)+'\n')
    
    # print('====== False positive rates ======\n')
    
    # print('FISCAL: '+str(FISCAL_fprs)+'\n')
    # print('BDT: '+str(BDT_fprs)+'\n')
    # print('NN: '+str(NN_fprs)+'\n')
    
    # print('====== False negative rates ======\n')

    # print('FISCAL: '+str(FISCAL_fnrs)+'\n')
    # print('BDT: '+str(BDT_fnrs)+'\n')
    # print('NN: '+str(NN_fnrs)+'\n')

    # print('====== Sensitivity ======\n')
    
    # print('FISCAL: '+str(FISCAL_sens)+'\n')
    # print('BDT: '+str(BDT_sens)+'\n')
    # print('NN: '+str(NN_sens)+'\n')

    # print('====== Specificity ======\n')
    
    # print('FISCAL: '+str(FISCAL_spec)+'\n')
    # print('BDT: '+str(BDT_spec)+'\n')
    # print('NN: '+str(NN_spec)+'\n')

    # print('====== Precision ======\n')
        
    # print('FISCAL: '+str(FISCAL_prec)+'\n')
    # print('BDT: '+str(BDT_prec)+'\n')
    # print('NN: '+str(NN_prec)+'\n')

# Plot all the confusion matrix elements for the varying probability thresholds
# in one graph.

# plot_conf(prob_thres,tps[:,0],fps[:,0],fns[:,0],tns[:,0],"FISCAL_conf.png")
# plot_conf(prob_thres,tps[:,1],fps[:,1],fns[:,1],tns[:,1],"BDT_conf.png")
# plot_conf(prob_thres,tps[:,2],fps[:,2],fns[:,2],tns[:,2],"NN_conf.png")

# Plot all the metrics for varying probability thresholds in one graph.

# plot_precision(prob_thres, prec_scores[:,0],"FISCAL_precision.png")
# plot_precision(prob_thres, prec_scores[:,1],"BDT_precision.png")
# plot_precision(prob_thres, prec_scores[:,2],"NN_precision.png")

# plot_fpr(prob_thres, fpr_scores[:,0],"FISCAL_fprs.png")
# plot_fpr(prob_thres, fpr_scores[:,1],"BDT_fprs.png")
# plot_fpr(prob_thres, fpr_scores[:,2],"NN_fprs.png")

# plot_fnr(prob_thres, fnr_scores[:,0],"FISCAL_fnrs.png")
# plot_fnr(prob_thres, fnr_scores[:,1],"BDT_fnrs.png")
# plot_fnr(prob_thres, fnr_scores[:,2],"NN_fnrs.png")

# plot_sensitivity(prob_thres, sens_scores[:,0],"FISCAL_sensitivity.png")
# plot_sensitivity(prob_thres, sens_scores[:,1],"BDT_sensitivity.png")
# plot_sensitivity(prob_thres, sens_scores[:,2],"NN_sensitivity.png")

# plot_accs(prob_thres, acc_scores)
# plot_sens(prob_thres, sens_scores)
# plot_precs(prob_thres, prec_scores)

plot_Fs(prob_thres, f_scores, [max_f_fiscal, max_f_bdt, max_f_nn])
plot_PRs(sens_scores, prec_scores)
