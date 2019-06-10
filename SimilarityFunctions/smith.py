# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:58:43 2019

This script contains an implementation of the smith-waterman algorithm.

The algorithm is split up in five functions:
    - soundex_encoding(char)
    - phonix_encoding(char)
    - compare_soundex(c1, c2)
    - compare_phonix(c1, c2)
    - smith_waterman(s1, s2, alignment_score: float = 5, approx_score: float = 2, mis_cost: float = 5, gapnew_cost: float = 5, gapcont_cost: float = 1):

The last function actually calculates the score between 0 and 1 with 1 being
exactly the same and 0 lacking any similarities. The other functions are 
auxiliary functions that determines if two characters are in the same 
similarity groups as defined by soundex or phonix.

Characteristics
The smith-waterman algorithm takes all the possible substrings of the two
to be compared strings and calculates a cost function that quantifies its
similarity. The cost function is defined by the paramaters set in the function
arguments. The substring comparison is not dependent on its position in the
total string so the similarity is great for finding similarities in sentences.

The code contains also the following function:
    MongeEklan(s1,s2)
    
This function is used to find the best scores of strings that contains several words. 

@author: pverschuuren
"""
import numpy as np


def soundex_encoding(char):
    
    group_1 = ['a','e','h','i','o','u','w','y']
    group_2 = ['b','f','p','v']
    group_3 = ['c','g','j','k','q','s','x','z']
    group_4 = ['d','t']
    group_5 = ['l']
    group_6 = ['m','n']
    group_7 = ['r']
    
    if char in group_1:
        return 1
    elif char in group_2:
        return 2
    elif char in group_3:
        return 3
    elif char in group_4:
        return 4
    elif char in group_5:
        return 5
    elif char in group_6:
        return 6
    elif char in group_7:
        return 7
    else:
        return 0


def phonix_encoding(char):
    
    group_1 = ['a','e','h','i','o','u','w','y']
    group_2 = ['b','p']
    group_3 = ['c', 'g', 'j', 'k', 'q']
    group_4 = ['d','t']
    group_5 = ['l']
    group_6 = ['m','n']
    group_7 = ['r']
    group_8 = ['f', 'v']
    group_9 = ['s', 'x', 'z']
    
    if char in group_1:
        return 1
    elif char in group_2:
        return 2
    elif char in group_3:
        return 3
    elif char in group_4:
        return 4
    elif char in group_5:
        return 5
    elif char in group_6:
        return 6
    elif char in group_7:
        return 7
    elif char in group_8:
        return 8
    elif char in group_9:
        return 9
    else:
        return 0


# This function checks if two characters are from the same soundex encoding
# group. If it does, it will return the soundex group number [1,7]. If it one
# of them is a word, a non-alphabetic character or if they are simply not
# from the same group the function will return 0.
def compare_soundex(c1, c2):
    
    if soundex_encoding(c1) == soundex_encoding(c2):
        return soundex_encoding(c1)
    else:
        return 0
    
# This function checks if two characters are from the same soundex encoding
# group. If it does, it will return the soundex group number [1,7]. If it one
# of them is a word, a non-alphabetic character or if they are simply not
# from the same group the function will return 0.
def compare_phonix(c1, c2):
    
    if phonix_encoding(c1) == phonix_encoding(c2):
        return phonix_encoding(c1)
    else:
        return 0

# Description of arguments of function:
    # s1 and s2 -> Input strings
    # aligment_score -> (default 5) exact match between two characters
    # approx_score -> (default 2) approximate match between two similar characters. This similarity can be based on letter groupings (such as Soundex encoding)
    # mis_cost - > (default 5) mismatch between two different characters
    # gapnew_cost -> (default 5) gap start penalty when there is at least one character in one string that does not appear in the other string
    # gapcont_cost -> (default 1) gap continuation penalty where a previously started gap continues.
    
def smith_waterman(s1, s2, alignment_score: float = 5, approx_score: float = 2, mis_cost: float = 5, gapnew_cost: float = 5, gapcont_cost: float = 1):
    
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
    X = np.zeros((len(s1) + 1, len(s2) + 1))
    Y = np.zeros((len(s1) + 1, len(s2) + 1))
      
    # Loop over the characters of both strings.
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
              
            match = 0
            
            # A character gets a H[i,j] assigned either if it is an exact 
            # match, an approximate match according to soundex groups or 
            # a mismatch cost subtracted.
            if s1[i-1] == s2[j-1]:
                match = match + alignment_score         
            elif compare_soundex(s1[i-1], s2[j-1]):               
                match = match + approx_score
            else:                  
                match = match - mis_cost
            
            # This is the part where all the possible permutations of
            # substrings including gap possibilities 
            X1 = H[i, j - 1] - gapnew_cost - gapcont_cost
            X2 = X[i, j - 1] - gapcont_cost
            X3 = Y[i, j - 1] - gapnew_cost - gapcont_cost
            
            X[i,j] = max(X1, X2, X3)
            
            Y1 = H[i - 1, j] - gapnew_cost - gapcont_cost
            Y2 = X[i - 1, j] - gapnew_cost - gapcont_cost
            Y3 = Y[i - 1, j] - gapcont_cost
        
            Y[i,j] = max(Y1, Y2, Y3)
            
            H[i,j] = match + max(H[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            
    score = H.max()
    
    norm_score = score/((alignment_score)*(0.5*(len(s1) + len(s2))))

    return norm_score

def MongeEklan(s1,s2):

        SumMaxSim = 0
        word1 = s1.split(' ') # split long strings in words
        word2 = s2.split(' ') # split long strings in words
        
        len_s1=len(word1)
        
        for w1 in word1:
            MaxSimScore=0
            for w2 in word2:        
                # find the maximum score of the possible combinations of words in the strings
                MaxSimScore = max(MaxSimScore,smith_waterman(w1, w2))
            SumMaxSim += MaxSimScore
        # use the sum of the maximum score to define the final score
        return SumMaxSim/len_s1    
