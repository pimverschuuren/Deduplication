# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:58:43 2019

This script contains an implementation of the longest common substring 
algorithm.

The algorithm is split up in five functions:
    - longest_common_substring_average(s1, s2, min_lc: int = 3)
    - longest_common_substring_score(s1, s2, min_lc: int = 3)
    - longest_common_substring(s1, s2)
    

The first function actually calculates the score between 0 and 1 with 1 being
exactly the same and 0 lacking any similarities. The second function also
calculates the similarity score but the first calculates the string swapping
average based on the second to take into account the asymmetry of the 
algorithm. The last function just finds the longest common substring. for the
input strings.


Characteristics
The longest common substring algorithm finds all the shared consecutive 
character sequences of the strings. A minimal character number for the 
substrings is set by default to 3. This algorithm is well suited for long 
sentences with matching words.

@author: pverschuuren
"""
import re

def longest_common_substring_average(s1, s2, min_lc: int = 3):
    
    # The algorithm is not symmetric so the average is taken of both
    # swapped string possibilities.
    score_1 = longest_common_substring_score(s1, s2, min_lc)
    score_2 = longest_common_substring_score(s2, s1, min_lc)
    
    average_score = ( score_1 + score_2 )/2

    return average_score

def longest_common_substring_score(s1, s2, min_lc: int = 3):
    
    
    if s1 is None:
        raise TypeError("Argument s1 is NoneType.")
    if s2 is None:
        raise TypeError("Argument s2 is NoneType.")
    
    # Explicitely convert to strings to ensure the upcoming string operations
    # won't give an error.
    
    st1 = s1
    st2 = s2
    
    s1 = str(s1)
    s2 = str(s2)
    
    # Save the lengths of both strings.
    len_s1 = len(s1)
    len_s2 = len(s2)
    
    # Set the length of the substring to the longest string as initial value.
    subl = max(len_s1, len_s2)
    
    # This will contain the sum of the lengths of the common substrings.
    sum_lc = 0
    
    # Loop until the found substring is smaller then the defined minimal
    # substring length.
    while (subl > min_lc - 1):
        subl, s1, s2 = longest_common_substring(s1, s2)
        
        # Only sum if the substring is longer than the minimal substring length.
        if subl > min_lc - 1:
            sum_lc = sum_lc + subl
    
    # Output the normalized score.
    if (len_s1 + len_s2) == 0:
        print(st1, st2)
    score = 2*sum_lc/(len_s1 + len_s2)
    
    return score


def longest_common_substring(s1, s2):
    
    # Get a matrix with zeros for the dynamic programming approach.
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
   
    longest, x_longest = 0, 0
   
    # Loop over the characters of both strings.
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            
            # Find the longest common substring by filling the matrix.
            if s1[x - 1] == s2[y - 1]:
                
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    
                    # Save the length of longest substring.
                    longest = m[x][y]
                    
                    # Save the end position of the longest substring.
                    x_longest = x
            else:
                m[x][y] = 0
                
    return longest, s1.replace(s1[x_longest - longest: x_longest], ''), s2.replace(s1[x_longest - longest: x_longest], '')

def MongeEklan(s1,s2,min_lc: int = 3):

        s1 = re.sub(' +', ' ',s1)
        s2 = re.sub(' +', ' ',s2)
    
        SumMaxSim = 0
        word1 = s1.split(' ') # split long strings in words
        word2 = s2.split(' ') # split long strings in words
        
        len_s1=len(word1)
        
        try:
            word1.remove("") 
            word2.remove("")
        except ValueError:
            pass
        
        for w1 in word1:
            MaxSimScore=0
            for w2 in word2:
                
                if (len(w2) + len(w1)) == 0:
                    print(w1, w2)
                    print(word1, word2, s1, s2)
                
                # find the maximum score of the possible combinations of words in the strings
                MaxSimScore = max(MaxSimScore,longest_common_substring_average(w1, w2,min_lc))

            SumMaxSim += MaxSimScore
        # use the sum of the maximum score to define the final score
        return SumMaxSim/len_s1    
    