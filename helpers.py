# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:57 2019

@author: pverschuuren
"""

import sys

import numpy as np

import pandas as pd


import matplotlib.pyplot as plt

from scipy import interp

from sklearn.model_selection import KFold, train_test_split
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


def split_label_pred(dataset):

    y = dataset[['duplicate','duplicate probability']]
    
    y_t = np.arange(y.shape[0], dtype=float)
    y_p = np.arange(y.shape[0], dtype=float)
    count = 0

    for index, row in y.iterrows():
        y_t[count] = row['duplicate']
        y_p[count] = row['duplicate probability']
        count = count + 1
    
    return y_t, y_p

def get_ROC(dataset):

    # Split the labels and classifier output in two
    # seperate columns.
    y_t, y_p = split_label_pred(dataset)
    
    # Calculate the ROC-curve for this fold.
    fpr, tpr, thresholds = roc_curve(y_t, y_p)

        
    # Calculate the AUC of the ROC-curve.
    roc_auc = auc(fpr, tpr)

    return roc_auc, fpr, tpr

def multi_ROC(FISCAL, BDT, NN):

    roc_BDT, fpr_BDT, tpr_BDT = get_ROC(BDT)

    roc_NN, fpr_NN, tpr_NN = get_ROC(NN)

    roc_FISCAL, fpr_FISCAL, tpr_FISCAL = get_ROC(FISCAL)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    plt.plot(fpr_FISCAL, tpr_FISCAL, color='b',label=r'ROC FISCAL (AUC = %0.3f)' % (roc_FISCAL),lw=2, alpha=.8)
    plt.plot(fpr_NN, tpr_NN, color='c',label=r'ROC NN (AUC = %0.3f)' % (roc_NN),lw=2, alpha=.8)
    plt.plot(fpr_BDT, tpr_BDT, color='m',label=r'ROC BDT (AUC = %0.3f)' % (roc_BDT),lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend(loc="lower right")
    plt.savefig("multiROC.pdf")

def plot_PRs(sens_scores, prec_scores):

    fiscal_sens_scores = sens_scores[:,0]
    BDT_sens_scores = sens_scores[:,1]
    NN_sens_scores = sens_scores[:,2]

    fiscal_prec_scores = prec_scores[:,0]
    BDT_prec_scores = prec_scores[:,1]
    NN_prec_scores = prec_scores[:,2]

    fiscal_prec_scores = np.nan_to_num(fiscal_prec_scores)
    BDT_prec_scores = np.nan_to_num(BDT_prec_scores)
    NN_prec_scores = np.nan_to_num(NN_prec_scores)

    auc_FISCAL=auc(fiscal_sens_scores, fiscal_prec_scores)
    auc_NN=auc(NN_sens_scores, NN_prec_scores)
    auc_BDT=auc(BDT_sens_scores, BDT_prec_scores)

    plt.clf()
    
    plt.plot(fiscal_sens_scores, fiscal_prec_scores, color='b',label=r'ROC FISCAL (AUC = %0.3f)' % (auc_FISCAL),lw=2, alpha=.8)
    plt.plot(NN_sens_scores, NN_prec_scores, color='c',label=r'ROC NN (AUC = %0.3f)' % (auc_NN),lw=2, alpha=.8)
    plt.plot(BDT_sens_scores, BDT_prec_scores, color='m',label=r'ROC BDT (AUC = %0.3f)' % (auc_BDT),lw=2, alpha=.8)
    
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 0.3])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.savefig("multiPR.pdf")

def plot_multihist(FISCAL, BDT, NN):

    y_FISCAL = FISCAL[['duplicate','duplicate probability']]
    y_BDT = BDT[['duplicate','duplicate probability']]
    y_NN = NN[['duplicate','duplicate probability']]

    y_FISCAL_t = np.arange(y_FISCAL.shape[0], dtype=float)
    y_FISCAL_p = np.arange(y_FISCAL.shape[0], dtype=float)
    y_BDT_t = np.arange(y_BDT.shape[0], dtype=float)
    y_BDT_p = np.arange(y_BDT.shape[0], dtype=float)
    y_NN_t = np.arange(y_NN.shape[0], dtype=float)
    y_NN_p = np.arange(y_NN.shape[0], dtype=float)
    count = 0

    y_FISCAL_dup = []
    y_FISCAL_ndup = []
    y_BDT_dup = []
    y_BDT_ndup = []
    y_NN_dup = []
    y_NN_ndup = []

    for index, row in y_FISCAL.iterrows():
        if row['duplicate'] > 0:
            y_FISCAL_dup.append(row['duplicate probability'])
        else:
            y_FISCAL_ndup.append(row['duplicate probability'])

    for index, row in y_BDT.iterrows():
        if row['duplicate'] > 0:
            y_BDT_dup.append(row['duplicate probability'])
        else:
            y_BDT_ndup.append(row['duplicate probability'])

    for index, row in y_NN.iterrows():
        if row['duplicate'] > 0:
            y_NN_dup.append(row['duplicate probability'])
        else:
            y_NN_ndup.append(row['duplicate probability'])

    plt.hist(y_BDT_ndup, bins = 40,label='BDT')
    plt.hist(y_NN_ndup, bins = 40,label='NN')
    plt.hist(y_FISCAL_ndup, bins = 40,label='FISCAL')
    plt.legend(loc="upper right")
    plt.xlabel("Classifier Output")
    plt.ylabel("Nr. Non-Duplicates")
    plt.savefig("nondupoutput.pdf")
    


# Calculate and print the mean and standard deviation
# along the columns of the passed array. This function
# is used on the array of k-fold performance metrics.
def calculate_mean_sd(metric_array):

    
    print('=============  '+str(len(metric_array)) + '-fold cross validation results  ===========')
    
    # Loop over every performance metric.
    for x in range(len(metric_array[0])):

        # A list that will contain all the values.
        metric_list = []

        # Loop over all the values of the metric.
        for y in range(len(metric_array)):

            # Append to the list of values.
            metric_list.append(metric_array[y][x])

        # Convert to a numpy aray for easy mean and
        # sd calculation.
        arr = np.array(metric_list)
        
        score_str = 'Accuracy: '

        if x == 1:
            score_str = 'False positive rate: '
        if x == 2:
            score_str = 'False negative rate: '
        if x == 3:
            score_str = 'Sensitivity: '
        if x == 4:
            score_str = 'Specificity: '
            
        print(score_str + str(np.mean(arr, axis = 0)) + '  +-  ' + str(np.std(arr, axis=0)))

        
def plot_precision(thresholds, precisions, name):
    
    plt.clf()
    plt.plot(thresholds, precisions,lw=2, alpha=.8)

    plt.xlim([0., 1.])
    plt.ylim([0., 0.3])
    plt.xlabel('Probability Threshold')
    plt.ylabel('Precision')
    plt.title('')
    plt.savefig(name)

def plot_fpr(thresholds, fprs, name):
    
    plt.clf()
    plt.plot(thresholds, fprs,lw=2, alpha=.8)

    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel('Probability Threshold')
    plt.ylabel('False Positive Rate')
    plt.title('')
    plt.savefig(name)

def plot_fnr(thresholds, fnrs, name):
    
    plt.clf()
    plt.plot(thresholds, fnrs,lw=2, alpha=.8)

    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel('Probability Threshold')
    plt.ylabel('False Negative Rate')
    plt.title('')
    plt.savefig(name)

    
def plot_sensitivity(thresholds, sensitivities, name):
    
    plt.clf()
    plt.plot(thresholds, sensitivities,lw=2, alpha=.8)

    plt.xlim([0., 1.])
    plt.ylim([0., 1.05])
    plt.xlabel('Probability Threshold')
    plt.ylabel('Sensitivity')
    plt.title('')
    plt.savefig(name)

def plot_conf(thresholds,tps,fps,fns,tns,file_name):

    plt.clf()
    
    plt.plot(thresholds, tps, color='green',label=r'True Positives',lw=2, alpha=.8)
    plt.plot(thresholds, fps, color='orange',label=r'False Positives',lw=2, alpha=.8)
    plt.plot(thresholds, fns, color='red',label=r'False Negatives',lw=2, alpha=.8)
    plt.plot(thresholds, tns, color='blue',label=r'True Negatives',lw=2, alpha=.8)

    tps = np.where(tps==0, 0.00001, tps)
    fps = np.where(fps==0, 0.00001, fps)
    fns = np.where(fns==0, 0.00001, fns)
    tns = np.where(tns==0, 0.00001, tns)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.00001, 1e7])
    plt.ylabel("Invoice Pairs")
    plt.xlabel("Probability Threshold")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.savefig(file_name)
    
    
def calculate_conf(true_labels, pred_prob, threshold):

    # Set a threshold of 0.5 to convert the values to binary output.
    pred_class = (pred_prob > threshold)

    # Convert the binary output to integer type.
    pred_class = pred_class.astype(int)

    # Convert the outputs to a 1d-vectors of values.
    pred_class = pred_class.ravel()
    true_labels = true_labels.ravel()

    # Use the scikitlearn library to calculate the
    # confusion matrix.
    return confusion_matrix(list(true_labels), list(pred_class)).ravel()
    

# Calculate the accuracy, false postive rate, false negative rate,
# sensitivity and specificity based on a list of true labels,
# a list of predicted probabilities and a probability threshold.
def calculate_performance_metrics(true_labels, pred_prob, threshold):

    # Set a threshold of 0.5 to convert the values to binary output.
    pred_class = (pred_prob > threshold)

    # Convert the binary output to integer type.
    pred_class = pred_class.astype(int)

    # Convert the outputs to a 1d-vectors of values.
    pred_class = pred_class.ravel()
    true_labels = true_labels.ravel()

    # Use the scikitlearn library to calculate the
    # confusion matrix.
    tn, fp, fn, tp = confusion_matrix(list(true_labels), list(pred_class)).ravel()
    
    # Calculate the accuracy.
    acc = (tn + tp)/(tn+tp+fp+fn)

    # Calculate the false positive rate.
    fpr = fp / (fp + tn)

    # Calculate the false negative rate.
    fnr = fn / (fn + tp)
    
    # Calculate the sensitivity.
    sens = tp / (tp + fn)

    # Calculate the specificity.
    spec = tn / (tn + fp)

    # Calculate the precision.
    prec = tp / (tp + fp)

    # Pass back a list with all the metrics.
    return [acc, fpr, fnr, sens, spec, prec]

# This function defines the optimization algorithm
# used in the training of the neural network.
# All the hyperparameters are set based on
# a grid search.
def adam_optimizer(learningrate):
    return optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# This function splits the passed dataframe
# into input features and true labels.
def split_data(data):

    # Get the true labels in the dataset.
    y=data[['duplicate']]

    # Convert the labels to integer type.
    y=y.astype('int32')

    # Get the input features by dropping the ID
    # and true label columns.
    X=data.drop(['ID','duplicate'], axis=1)

    # Create the right format for the fit.
    y_t = np.arange(y.shape[0], dtype=int)
    X_t = X.values
    
    count = 0
    
    for index, row in y.iterrows():
        y_t[count] = row['duplicate']
        count = count + 1

    return X, y, X_t, y_t

def plot_Fs(thresholds, f_scores, max_scores):

    fiscal_scores = f_scores[:,0]
    BDT_scores = f_scores[:,1]
    NN_scores = f_scores[:,2]

    max_thres_fiscal = max_scores[0][0]
    max_thres_BDT = max_scores[1][0]
    max_thres_NN = max_scores[2][0]

    max_score_fiscal = max_scores[0][1]
    max_score_BDT = max_scores[1][1]
    max_score_NN = max_scores[2][1]

    plt.clf()
    plt.hlines(max_score_fiscal,0,1,linestyle='--', color='b',label=r'Max. FISCAL = %0.3f' % max_score_fiscal , alpha=.8)
    plt.hlines(max_score_BDT,0,1,linestyle='--', color='m',label=r'Max. BDT = %0.3f' % max_score_BDT, alpha=.8)
    plt.hlines(max_score_NN,0,1,linestyle='--', color='c',label=r'Max. NN = %0.3f' % max_score_NN, alpha=.8)
    
    plt.plot(thresholds, fiscal_scores, color='b',label='FISCAL')
    plt.plot(thresholds, BDT_scores, color='m',label='BDT')
    plt.plot(thresholds, NN_scores, color='c',label='NN')

    plt.xlabel('Probability Threshold')
    plt.ylabel('F-score')
    plt.ylim(0,1)
    plt.legend(loc="upper right")
    plt.savefig("F_scores.pdf")

def plot_accs(thresholds, acc_scores):

    fiscal_scores = acc_scores[:,0]
    BDT_scores = acc_scores[:,1]
    NN_scores = acc_scores[:,2]

    plt.clf()
    
    plt.plot(thresholds, fiscal_scores, color='b',label='FISCAL')
    plt.plot(thresholds, BDT_scores, color='m',label='BDT')
    plt.plot(thresholds, NN_scores, color='c',label='NN')

    plt.xlabel('Probability Threshold')
    plt.ylabel('Accuracy')
    plt.ylim(0,1.0)
    plt.legend(loc="upper left")
    plt.savefig("acc_scores.pdf")

def plot_sens(thresholds, sens_scores):

    fiscal_scores = sens_scores[:,0]
    BDT_scores = sens_scores[:,1]
    NN_scores = sens_scores[:,2]

    plt.clf()
    
    plt.plot(thresholds, fiscal_scores, color='b',label='FISCAL')
    plt.plot(thresholds, BDT_scores, color='m',label='BDT')
    plt.plot(thresholds, NN_scores, color='c',label='NN')

    plt.xlabel('Probability Threshold')
    plt.ylabel('Sensitivity')
    plt.ylim(0,1.05)
    plt.legend(loc="upper right")
    plt.savefig("sens_scores.pdf")

def plot_precs(thresholds, prec_scores):

    fiscal_scores = prec_scores[:,0]
    BDT_scores = prec_scores[:,1]
    NN_scores = prec_scores[:,2]
    
    plt.plot(thresholds, fiscal_scores, color='b',label='FISCAL')
    plt.plot(thresholds, BDT_scores, color='m',label='BDT')
    plt.plot(thresholds, NN_scores, color='c',label='NN')

    plt.xlabel('Probability Threshold')
    plt.ylabel('Precision')
    plt.ylim(0,0.3)
    plt.legend(loc="upper right")
    plt.savefig("prec_scores.pdf")

    

# This function plots multiple ROC-curves, their average
# and standard deviation.
def plot_ROCs(tprs, mean_fpr, aucs, model_name):

    plt.clf()
    
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
             label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc),
             lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('Sensitivity')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig("ROCs_"+str(model_name)+".pdf")



# Read a comma-seperated csv-file and return it
# in a pandas dataframe.
def read_csvfile(file_loc):
    df = pd.read_csv(file_loc,
                 sep=',',
                 encoding="ISO-8859-1",
                 low_memory=False)
    return df


# Train and test a passed BDT/NN model.
# The arguments define the number of folds for the
# cross-validation scheme, the subjected model, a
# dataframe that contains the train/test data and a
# probability threshold.
def cross_val_KFold(n_folds, model, data, prob_threshold, model_name):

    # Get the train/test data in the right format.
    X, y, X_t, y_t = split_data(data)

    # Make a list for the scores.
    metric_array = []

    # Create a cross validation object.
    kf = KFold(n_splits=n_folds)

    # Create the folds.
    kf.get_n_splits(X_t)

    # Create the lists for the true positive rate
    # , the AUC and a linear space for the false positive rate.
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    
    # Get the initial weights of the neural network for
    # resetting the model in between folds.
    if model.__module__ == 'keras.engine.sequential':
        init_weights = model.get_weights()
    
    # Loop over all the folds.
    for train_index, test_index in kf.split(X):

        X_train, X_test = X_t[train_index], X_t[test_index]
        y_train, y_test = y_t[train_index], y_t[test_index]

        # Define the numpy list that will contain the predicted
        # duplicate probabilities. 
        y_pred = np.arange(len(y_test), dtype=float)

        # Fit the BDT.
        if model.__module__ == 'sklearn.ensemble.gradient_boosting':

            # Get a new unfitted model for the new fold.
            fit_model = clone(model)

            # Fit on the training data.
            #print("Training boosted decision tree for fold "+str(i))
            fit_model.fit(X_train, y_train)

            # Predict on the test data.
            y_prediction = fit_model.predict_proba(X_test)

            # Put the prediction in a simple list.
            for x in range(len(y_prediction)):
                y_pred[x] = y_prediction[x,1]
            
        # Fit the NN.
        elif model.__module__ == 'keras.engine.sequential':
            
            # Set the weights of the neural network to
            # their initial value for a new unfitted model.
            model.set_weights(init_weights)

            # Fit on the training data. Epochs and batch size
            # are the result of a grid search.
            print("Training neural network for fold "+str(i))
            model.fit(X_train, y_train, epochs=160, batch_size=100)

            # Predict on the test data.
            y_pred = model.predict_proba(X_test)
            
            
        # Exit the program if none of the above models is
        # passed.
        else:
            print('Unknown model passed! Please add a line with the appropriate fit method.')
            sys.exit()

        # Get the list of performance metrics for this fold.
        metric_list = calculate_performance_metrics(y_test, y_pred, prob_threshold)

        # Put them in the array.
        metric_array.append(metric_list)

        # Calculate the ROC-curve for this fold.
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)

        # Append the ROC-curve of this fold to a list.
        tprs.append(interp(mean_fpr, fpr, tpr))

        # Add a zero point to the array.
        tprs[-1][0] = 0.0

        # Calculate the AUC of the ROC-curve.
        roc_auc = auc(fpr, tpr)

        # Append the AUC to a list.
        aucs.append(roc_auc)

        # Plot the ROC-curve of this fold.
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
        
        i += 1

    plot_ROCs(tprs, mean_fpr, aucs, model_name)

    
    return metric_array


# Build the neural network with the keras library. All
# the hyperparameters are optimized with a grid search.
def get_NN(data):

    # Define the neural network object.
    model = Sequential()

    # Add the input layer.
    model.add(Dense(30,input_dim=data.shape[1] - 2, activation='relu'))

    # Add the first hidden layer.
    model.add(Dense(output_dim = 30, init = 'he_uniform', activation = 'relu')) 

    # Add the output layer.
    model.add(Dense(1,activation='sigmoid'))

    # Compile the model.
    model.compile(loss='binary_crossentropy',optimizer=adam_optimizer(0.01),metrics=['accuracy'])

    return model

# Build the boosted decision tree with the scikitlearn library.
# All the hyperparameters were optimized with a grid search.
def get_BDT():

    # Define the Boosted Decision Tree model.
    model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
          learning_rate=0.1, loss='deviance', max_depth=4,
          n_estimators=200, random_state=0)

    return model
