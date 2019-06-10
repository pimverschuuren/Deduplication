# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:58:43 2019

This script contains an implementation of the levenshtein and the 
damerau-levenshtein algorithms.

The algorithm consists out of only one function:
    - levenshtein(s1, s2, damerau: bool = True, insert_cost: float = 1, del_cost: float = 1, sub_cost: float = 1, trans_cost: float = 1)
    
The function calculates a score between 0 and 1 with 1 being exactly the same 
and 0 lacking any similarities.


Characteristics
The algorithm tries to find the minimal number of operations needed to
transform one string into another. For the levenshtein algorithm these
operations are:
    - Deletion
    - Insertion
    - Subtitution

The only addition in the damerau-levenshtein algorithm is the added 
transposition operation. All of the operations have a cost value assigned
to them in the arguments of the function. Both algorithms are best suited for
short string comparison and not long sentences.

The code contains also the following function:
    MongeEklan(s1,s2)
    
This function is used to find the best scores of strings that contains several words. 
@author: pverschuuren
"""

import numpy as np
#Description of arguments of the function:
    #s1 and s2 -> Input strings
    #damerau -> condition used to use levenshtein or damerau-levenshtein methods.
    #insert_cost -> number of insertion (set to 1)
    #del_cost -> number of deletion (set to 1)
    #sub_cost -> number of substitution (set to 1)
    #trans_cost -> number of transposition (set to 1) used only for damerau-levenshtein method

def levenshtein(s1, s2, damerau: bool = True, insert_cost: float = 1, del_cost: float = 1, sub_cost: float = 1, trans_cost: float = 1):
        
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")
    if s2 is None:
        raise TypeError("Argument s2 is NoneType.")
    
    # Explicitely convert to strings to ensure the upcoming string operations
    # won't give an error.
    s1 = str(s1)
    s2 = str(s2)
        
    # If the string match is exactly the same, then return 1.
    if s1 == s2:
        return 1.0
  
    # H holds the alignment score at each point, computed incrementally
    H = np.zeros((len(s1) + 1, len(s2) + 1))
    
    # Fill the zeroth row and zeroth column with their resp. row and column 
    # index.
    for i in range(0, len(s1) + 1):
        H[i, 0] = i
    for j in range(0, len(s2) + 1):
        H[0, j] = j
    
    # Loop over the characters of both strings.
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            
            if s1[i - 1] == s2[j - 1]:
                H[i,j] = H[i - 1, j - 1]
            else:
                del_val = H[i - 1, j] + del_cost
                insert_val = H[i, j - 1] + insert_cost
                sub_val = H[i - 1, j - 1] + sub_cost
                
                H[i,j] = min(del_val, insert_val, sub_val)
                
                if damerau and i>1 and j>1 and s1[i - 1]==s2[j - 2] and s1[i - 2] == s2[j - 1]:
                    H[i,j] = min(H[i,j], H[i-2,j-2] + trans_cost)
                
                
    
    # The final element is the actual optimal edit score.
    edit_score = H[len(s1), len(s2)]

    # Normalizion to have the score between 0 and 1.
    norm_score = 1 - edit_score/(max(len(s1), len(s2)))
    
    return norm_score


def MongeEklan(s1,s2,damerau: bool = True):

        SumMaxSim = 0
        word1 = s1.split(' ') # split long strings in words
        word2 = s2.split(' ') # split long strings in words
        
        len_s1=len(word1)
        
        for w1 in word1:
            MaxSimScore=0
            for w2 in word2:        
                # find the maximum score of the possible combinations of words in the strings
                MaxSimScore = max(MaxSimScore,levenshtein(w1, w2, damerau))
            SumMaxSim += MaxSimScore
        # use the sum of the maximum score to define the final score
        return SumMaxSim/len_s1    
    