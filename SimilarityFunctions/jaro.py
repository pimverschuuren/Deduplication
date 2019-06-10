# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:23:00 2019

This script contains an implementation of the jaro and jaro-winkler algorithm.

The algorithm is split up in three functions:
    - jaro_winkler_matches(s1, s2)
    - jaro_coefficient(s1, s2)
    - jaro_winkler_coefficient(s1, s2)

The latter two return a similarity score between 0 and 1 with 1 being exactly 
the same and 0 lacking any similarities. The first is an auxiliary function
for the other two.

Characteristics
The jaro algorithm counts the number of characters in common being no more then
half the distance of the longer string away from each other. Jaro works well
on small strings such as single short words or names. 

The jaro-winkler algorithm increases similarity for strings that agree in the 
prefix.

@author: Pim Verschuuren
"""


"""
A auxiliary function for the Jaro Winkler coefficient function.
The function returns a list with 4 values:
1 - The number of characters that are shared.
2 - Half the number of transpositions.
3 - The number of successive matching prefix characters.
4 - The length of the biggest string.
"""

"""
The code contains also the following function:
    MongeEklan(s1,s2)
This function is used to find the best scores of strings that contains several words. This function can
be applied to several algorithms (qgram and JaroWinkler for example).
"""
def jaro_winkler_matches(s1, s2):

        # Define which string is the longest.
        if len(s1) > len(s2):
            max_str = s1
            min_str = s2
        else:
            max_str = s2
            min_str = s1
            
        # Get the max distance two matching characters are allowed to
        # be apart to be actually classified as matching. All non-integers
        # will be rounded down.
        ran = int(max(len(max_str) / 2 - 1, 0))
        
        # Create a list that will later contain the location of the matching
        # character in the biggest string.
        match_indexes = [-1] * len(min_str)
        
        # Create a list that will later contain the flag of matching for each
        # character in the biggest string.
        match_flags = [False] * len(max_str)
        matches = 0
        
        # Loop over the character locations of the smallest string.
        for mi in range(len(min_str)):
            
            # Get the character of the smallest string.
            c1 = min_str[mi]
            
            # Loop over the character locations spanned by the location of
            # c1 and the max distance defined by ran.
            for xi in range(max(mi - ran, 0), min(mi + ran + 1, len(max_str))):
                
                # The match_flags are put to true as soon as one of the
                # characters of the smallest string matches. If any successive
                # characters of the smallest string are of the same value they
                # will not count another match with the same character of the
                # biggest string. Also, the for loop is broken with the break
                # statement to avoid counting the same characters in the 
                # biggest string more than once.
                if not match_flags[xi] and c1 == max_str[xi]:
                    match_indexes[mi] = xi
                    match_flags[xi] = True
                    matches += 1
                    break

        # These are lists that will be filled with the exact matches.
        ms1, ms2 = [0] * matches, [0] * matches
        
        # Fill the list with the matching characters in the order of their 
        # occurence in the smallest string.
        si = 0
        for i in range(len(min_str)):
            if not match_indexes[i] == -1:
                ms1[si] = min_str[i]
                si += 1
        
        # Fill the list with the matching characters in the order of their 
        # occurence in the biggest string.
        si = 0
        for j in range(len(max_str)):
            if match_flags[j]:
                ms2[si] = max_str[j]
                si += 1
        
        # A transposition is when the matching character is not in the same
        # matching character ordering position in the smallest string as it
        # is in the biggest string. The number of order places does not matter.
        transpositions = 0
        for mi in range(len(ms1)):
            if not ms1[mi] == ms2[mi]:
                transpositions += 1
        
        # Count how many successive characters match in the prefix.
        prefix = 0
        for mi in range(len(min_str)):
            if s1[mi] == s2[mi]:
                prefix += 1
            else:
                break
        
        return [matches, int(transpositions / 2), prefix, len(max_str)]


"""        
The Jaro Winkler coefficient quantifies how many editing operations there
are needed to go from the first to the second string. The inverted
operation does not give the same results.
"""
def jaro_coefficient(s1, s2):
    
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
    
    # Get match information. See the function definition above.
    mtp = jaro_winkler_matches(s1, s2)
    
    # Return 0 if no matches are found.
    m = mtp[0]
    if m == 0:
        return 0.0
    
    
    #print(mtp)
    
    # Get the Jaro Similarity.
    j = (m / len(s1) + m / len(s2) + (m - mtp[1]) / m) / 3
    
    return j



"""   
This is a twist on the jaro algorithm. The addition is the
extra positive weight given to matches that match in the beginning
of the strings. NOTE: This can be adjusted to suffix matches or even
give negative weight.
"""
def jaro_winkler_coefficient(s1, s2):

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
    
    threshold = 0.7
    jw_coef = 0.08
    
    
    j = jaro_coefficient(s1, s2)
    jw = j
    
    # Get match information. See the function definition above.
    mtp = jaro_winkler_matches(s1, s2) 
    
    # The threshold defines which range of jaro values
    # is defined as matching.
    if j > threshold:
        
        # This add an additional value dependant on the coefficient
        # the length of the string and the number of matching prefices.
        
        # To ensure that the jw coefficient does not exceed 1, the product
        # with mtp[2] must not be bigger than 1. Because the number of
        # matching prefices is always smaller than or equal to the total
        # number of characters(mtp[3]) the jw_coef will have an upper
        # bound of 1/mtp[3] to uphold normalization.
        jw = j + min(jw_coef, 1.0 / mtp[3]) * mtp[2] * (1 - j)
    return jw

def jaro_coefficient_invert(s1, s2):
    
    s1 = s1[::-1]
    s2 = s2[::-1]
 
    return jaro_coefficient(s1, s2)

def jaro_winkler_coefficient_invert(s1, s2):
    
    s1 = s1[::-1]
    s2 = s2[::-1]
 
    return jaro_winkler_coefficient(s1, s2)

def MongeEklan_jaro(s1,s2):

        SumMaxSim = 0
        word1 = s1.split(' ') # split long strings in words
        word2 = s2.split(' ') # split long strings in words
        
        len_s1=len(word1)
        
        for w1 in word1:
            MaxSimScore=0
            for w2 in word2:        
                # find the maximum score of the possible combinations of words in the strings
                MaxSimScore = max(MaxSimScore,jaro_coefficient(w1, w2))
            SumMaxSim += MaxSimScore
        # use the sum of the maximum score to define the final score
        return SumMaxSim/len_s1    
    
def MongeEklan_jaro_winkler(s1,s2):

        SumMaxSim = 0
        word1 = s1.split(' ') # split long strings in words
        word2 = s2.split(' ') # split long strings in words
        
        len_s1=len(word1)
        
        for w1 in word1:
            MaxSimScore=0
            for w2 in word2:        
                # find the maximum score of the possible combinations of words in the strings
                MaxSimScore = max(MaxSimScore,jaro_winkler_coefficient(w1, w2))
            SumMaxSim += MaxSimScore
        # use the sum of the maximum score to define the final score
        return SumMaxSim/len_s1    
