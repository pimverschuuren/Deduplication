# How to run it

Both NeuralNetwork.py and BoostedDecisionTree.py use dataset.csv as input and helpers.py as dependency. They can be both executed with the following packages installed:
- Python (3.6.8)
- Numpy (1.16.4)
- Pandas (0.23.2)
- Matplotlib (3.0.2)
- Scipy (1.1.0)
- Scikit-learn (0.21.2)
- Keras (2.2.4)

Next to the package the version is given on which the script is tested. Scripts can be run by the command:

    python NeuralNetwork.py
    
    python BoostedDecisionTree.py
    


# Deduplication

This repository contains a neural network and a boosted decision tree that are build to identify duplicate entries in a dataset. 

# Dataset

The dataset was created in collaboration with software company FISCAL Technologies that builds financial forensic software. The dataset is based on transactions of their customers and the true labels are from verified customer feedback. The transactions originate from companies and organizations accross several industries like education, healthcare and retail. After applying a pairing scheme transaction attributes between pairs are compared with the similarity functions given in the git repository. The dataset thus consists of vectors of similarity scores between transactions and true duplicate labels. Any preceding forms of the dataset are kept private to ensure full anonymity of the customers.

# Dataset creation

The dataset is based transactions with the following attributes:
- ID (Integer value)
- Invoice Number (String value)
- Invoice Date (Date value)
- Entered Date (Date value)
- Paid Date (Date value)
- Supplier Reference Number (Integer value)
- Supplier Name (String value)
- Invoice Amount (Float value)
- Base Amount (Float value)

To reduce the curse of pairing dimensionality a pairing preprocessing step is performed. The transactions are grouped according to the following sets of criteria:

  Set 1:
  - Same numeric invoice number (All the alphabetical characters removed)
  - Same invoice date
  - Same invoice amount
  
  Set 2:
  - Same numeric invoice number
  - Same invoice amount
  
  Set 3:
  - Same numeric invoice number
  - Same invoice date
  
  Set 4:
  - Same invoice amount
  - Same invoice date

Within each group all the possible pairs are created and labelled (non-)duplicate according to customer feedback.

Each pair of transactions is then subject to a set of similarity functions: 
- Jaro-Winkler
- Damerau-Levenshtein
- Q-Gram
- Longest Common Substring
- Smith Waterman
- Binary

All of the above algorithms take an attribute value of both transactions as argument (eg. Invoice Number #1 and Invoice Number #2) and calculates their similarity in a variety of ways. The return is a float value between 0 and 1 with 1 meaning that the arguments are exactly the same and 0 meaning they are completely distinct. Only the binary function returns either a 1 or a 0 meaning resp. exactly the same or different.


# Training and testing the model

The script loads the data, builds the model in question and trains/tests the model according to a 5-fold cross validation scheme. The output is a plot of the ROC-curve for every fold, the average ROC-curve over all folds and average performance metrics over all folds.

