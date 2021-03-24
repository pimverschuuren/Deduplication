# Deduplication
This repository contains code that uses a dataset of invoices to train and evaluate models that identify (non-)duplicate invoices.

# How to run it
The following libraries are needed to run the code:
- Python (3.6.8)
- Numpy (1.16.4)
- Pandas (0.23.2)
- Matplotlib (3.0.2)
- Scipy (1.1.0)
- Scikit-learn (0.21.2)
- Keras (2.2.4)
 
When installed one can run the scripts simply with the commands:


    python 5FoldCrossValidation.py
    
    python ImbalancedValidation.py
    
    
The script helpers.py contains helper functions for the above scripts, the directories SimilarityFunctions and data contain resp. the similarity functions used to build the data and the data that is used as input for the two above scripts. 



