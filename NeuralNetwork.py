# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:21:57 2019

@author: pverschuuren
"""

import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt
from scipy import interp

from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, roc_curve, auc

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers



def calculate_mean_sd(metric_array):

    
    print('=============  '+str(len(metric_array)) + '-fold cross validation results  ===========')
    
    for x in range(len(metric_array[0])):
        
        metric_list = []
        
        for y in range(len(metric_array)):
            
            metric_list.append(metric_array[y][x])
        
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


def calculate_performance_metrics(true_labels, pred_prob, threshold):

    # Set a threshold of 0.5 to convert the values to binary output.
    pred_class = (pred_prob > threshold)

    # Convert the binary output to integer type.
    pred_class = pred_class.astype(int)

    # Convert the outputs to a 1d-vectors of values.
    pred_class = pred_class.ravel()
    pred_class = pred_class.ravel()
    true_labels = true_labels.ravel()


    cm = confusion_matrix(list(true_labels), list(pred_class))
    acc = (np.trace(cm))/np.sum(cm)
    fpr = cm[0][1]/(cm[0][1] + cm[1][1])
    fnr = cm[1][0]/(cm[1][0] + cm[0][0])
    sens = cm[1][1] / (cm[1][1] + cm[1][0])
    spec = cm[0][0] / (cm[0][0] + cm[0][1])
    
    return [acc, fpr, fnr, sens, spec]


def adam_optimizer(learningrate):
    return optimizers.Adam(lr=learningrate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


def train_KFold(n_folds, model, data, batch_size, epochs, prob_threshold : float = 0.5):
    
    y=data[['duplicate']]
    
    y=y.astype('int32')
    X=data.drop(['ID','duplicate'], axis=1)
    
    
    y_t = np.arange(y.shape[0], dtype=int)
    
    X_t = X.values
    
    count = 0

    for index, row in y.iterrows():
        y_t[count] = row['duplicate']
        count = count + 1
    
    metric_array = []
    
    kf = KFold(n_splits=n_folds)
    
    kf.get_n_splits(X)
    
    init_weights = model.get_weights()
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    
    i = 0
    
    for train_index, test_index in kf.split(X):
       X_train, X_test = X_t[train_index], X_t[test_index]
       y_train, y_test = y_t[train_index], y_t[test_index]

        # Set the weights to their initial value.
       model.set_weights(init_weights)
       
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


def read_csvfile(file_loc):
    df = pd.read_csv(file_loc,
                 sep=',',
                 encoding="ISO-8859-1",
                 low_memory=False)
    return df


# Load the data.
data = read_csvfile('data.csv')

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

model, p_metrics = train_KFold(5, model, data, 50, 120, 0.5)
    
calculate_mean_sd(p_metrics)


