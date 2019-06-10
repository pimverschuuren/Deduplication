# How to run it

The script NeuralNetwork.py can be run on its own with the supplied dataset.csv and the following packages:
- Python (3.6.8)
- Numpy (1.16.4)
- Pandas (0.23.2)
- Matplotlib (3.0.2)
- Scipy (1.1.0)
- Scikit-learn (0.21.2)
- Keras (2.2.4)

Next to the package the version is given on which the script is tested.


# Deduplication

This repository contains a neural network that is build to identify duplicate entries in a dataset. An anonymized dataset is supplied to train and evaluate the model on. Additionally, the similarity functions that were used to model the similarity between two entries are also supplied.


# Dataset creation

The entries are invoices of companies and organizations accross several industries like education, healthcare and retail. Each invoice has the same set of variables such as invoice number, invoice date, supplier name or paid amount which are used to identify them. These variables are used to compare two invoices.

Step 1:
The invoices are grouped according to a specific grouping scheme to avoid impractical amounts of pairs. 

Step 2:
All the possible pairs are created within that group.

Step 3:
With the use of the similarity functions on the invoice variables similarity scores are calculated for each invoice pair.

Step 4:
Customer feedback is used to label the paired invoices either duplicate or not.


# Training and testing the model

The script loads the data, builds a feedforward neural network and trains/tests the model according to a 5-fold cross validation scheme. The output is a plot of the ROC-curve for every fold, the average ROC-curve over all folds and average performance metrics over all folds.

