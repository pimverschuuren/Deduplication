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

from sklearn.model_selection import KFold
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import GradientBoostingClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers


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
    pred_class = pred_class.ravel()
    true_labels = true_labels.ravel()

    # Use the scikitlearn library to calculate the
    # confusion matrix.
    cm = confusion_matrix(list(true_labels), list(pred_class))

    # Calculate the accuracy.
    acc = (np.trace(cm))/np.sum(cm)

    # Calculate the false positive rate.
    fpr = cm[0][1]/(cm[0][1] + cm[1][1])

    # Calculate the false negative rate.
    fnr = cm[1][0]/(cm[1][0] + cm[0][0])

    # Calculate the sensitivity.
    sens = cm[1][1] / (cm[1][1] + cm[1][0])

    # Calculate the specificity.
    spec = cm[0][0] / (cm[0][0] + cm[0][1])

    # Pass back a list with all the metrics.
    return [acc, fpr, fnr, sens, spec]

# This function defines the optimization algorithm
# used in the training of the neural network.
# All the hyperparameters are set based on
# a grid search.
def adam_optimizer(learningrate):
    return optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# This function splits the passed dataframe
# into input features and true labels.
def split_data(data):

    y=data[['duplicate']]
    
    y=y.astype('int32')
    X=data.drop(['ID','duplicate'], axis=1)
    
    y_t = np.arange(y.shape[0], dtype=int)
    X_t = X.values
    
    count = 0
    
    for index, row in y.iterrows():
        y_t[count] = row['duplicate']
        count = count + 1

    return X, y, X_t, y_t


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
def train_KFold(n_folds, model, data, prob_threshold  : float = 0.5):
    
    X, y, X_t, y_t = split_data(data)
    
    scores = []
    
    kf = KFold(n_splits=n_folds)
    kf.get_n_splits(X_t)
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    
    # Get the initial weights of the neural network for
    # resetting in between folds.
    if model.__module__ == 'keras.engine.sequential':
        init_weights = model.get_weights()
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X_t[train_index], X_t[test_index]
        y_train, y_test = y_t[train_index], y_t[test_index]
    


        # Fit the BDT.
        if model.__module__ == 'sklearn.ensemble.gradient_boosting':
            fit_model = clone(model)
            fit_model.fit(X_train, y_train)

        # Fit the NN.
        elif model.__module__ == 'keras.engine.sequential':
            
            # Set the weights to their initial value.
            model.set_weights(init_weights)
            model.fit(X_train, y_train, epochs=120, batch_size=60)

        # Exit the program if none of the above models is
        # passed.
        else:
            print('Unknown model passed! Please add a line with the appropriate fit method.')
            sys.exit()
            
        y_prediction = fit_model.predict_proba(X_test)
        
        y_pred = np.arange(len(y_prediction), dtype=float)
        for x in range(len(y_prediction)):
            y_pred[x] = y_prediction[x,1]
           
        # Plot the confusion matrix and accuracy score.
        
        score_list = calculate_performance_metrics(y_test, y_pred, prob_threshold)
           
        scores.append(score_list)
              
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
           
        i += 1

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
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    
    return model, scores


# Train and test a passed boosted decision tree model.
# The arguments define the number of folds for the
# cross-validation scheme, the subjected model, a
# dataframe that contains the train/test data and a
# probability threshold.
def train_KFold_NN(n_folds, model, data, batch_size, epochs, prob_threshold : float = 0.5):

    # Split the data in input features and true labels.
    X, y, X_t, y_t = split_data(data)

    # Define an array that will contain the performance
    # metrics for all the folds.
    metric_array = []

    print('Module')
    if model.__module__ == 'keras.engine.sequential':
        print(model.__module__)
    
    # Define a k-fold cross validation object.
    kf = KFold(n_splits=n_folds)

    # Split the data for k-folds.
    kf.get_n_splits(X)


    # Lists that will contain the true positive rate,
    # the AUC values and linear space between 0 and 1.
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0

    # Loop over all the folds.
    for train_index, test_index in kf.split(X):

        X_train, X_test = X_t[train_index], X_t[test_index]
        y_train, y_test = y_t[train_index], y_t[test_index]

        
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        
        y_pred = model.predict_proba(X_test)
        
        # Plot the confusion matrix and accuracy score.
        metric_list = calculate_performance_metrics(y_test, y_pred, prob_threshold)
       
        metric_array.append(metric_list)
       
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))
       
        i += 1
       
       
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
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return model, metric_array



def get_NN(data):

    # Definition of Sequential model that is a linear stack of layers
    model = Sequential()

    # the model has 9 neurons in the hidden layer. Dense layer means a fully connected layer so each of the 9 neurons
    # are fully connected to the 19 input features.
    # the activation function is a relu function
    #model.add(Dense(15,input_dim=30, activation='relu')) # this was for the cancer dataset features
    model.add(Dense(15,input_dim=data.shape[1] - 2, activation='relu'))

    model.add(Dense(output_dim = 15, init = 'he_uniform', activation = 'relu')) 

    model.add(Dense(output_dim = 15, init = 'he_uniform', activation = 'relu')) 

    # the output layer is a dense layer with the sigmoid activation that convert a real valued input into a binaary output
    model.add(Dense(1,activation='sigmoid'))

    # Compile the model by defining the optimizer, loss function and the metric to evaluate the model performances
    # the loss function is binary_crossentropy (standard for binary classification)
    # optimizer rmsprop upgrade from normal gradient descent algorithm
    model.compile(loss='binary_crossentropy',optimizer=adam_optimizer(0.01),metrics=['accuracy'])

    return model


def get_BDT():

    # Define the Boosted Decision Tree model.
    model = GradientBoostingClassifier(criterion='friedman_mse', init=None,
          learning_rate=0.2, loss='deviance', max_depth=5,
          n_estimators=250, random_state=0)

    return model
