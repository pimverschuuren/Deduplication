# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:27:38 2019

@author: Serena Palazzo


Qgram based string comparison
The idea is to split the two input strings into short sub-strings of lenght q. The sub-strings are called q-grams
(Bi-gram, Tri-gram and so on) and count how many of these q-grams occur in both strings
q -> Number of grams
sim_divisor -> coefficient of similarity  
max_dist -> used only for the Positional version of the code ( to evaluate the range to look at the common positional grams)

The algorithm can be:
  Base algorithm -> just compare the two strings and find the common sub-strings
  
  Padded algorithm -> compare the two strings by put at the beginning and at the end, (q-1) characters 
   Useful to provide specific informations about the start and the end of the strings
   Similarity score larger for strings that have the same beginning and end but errors in the middle of the strings
   Similarity score will be lower for strings that have different charters at the end and beginning of the words
   general improvement of the matching quality using padded algorithms
   Ex. of strings with (q-1) special characters if q=2 -> string1: *dickson#, string2: *dixon#
   qgrams: string1 (*d, di, ic, ck, ks, so, on, n#) string2 (*d, di, ix, xo, on, n#) 
    
  Positional algorithm -> compare the two strings based on their position
   for each string, the position number of the qgram within the string is given
   only qgrams that are the same and have a position value within a maximum distance are considered.
   Ex. if distance is 2 ('pe',1) of string1 is common with ('pe',1), ('pe',2) ('pe',3) of string2 
   but not with ('pe',4) of string2
  
The algorithm can be set as follow and you can set q to have Bigrams, Trigrams and so on:
  Not Padded Not Positional Qgram(s1, s2, q=2,max_dist=2,sim_divisor = 'dice', False, False)  base algorithm
  Padded Not Positional Qgram(s1, s2, q=2,max_dist=2,sim_divisor = 'dice', True, False)
  Padded Positional Qgram(s1, s2, q=2,max_dist=2,sim_divisor = 'dice', True, True)
  
  
The code contains also the following function:
    MongeEklan(s1,s2)
This function is used to find the best scores of strings that contains several words. This function can
be applied to several algorithms (qgram and JaroWinkler for example).
"""

import logging
QGRAM_START_CHAR = chr(95)
QGRAM_END_CHAR =   chr(140)

def check_type(s1,s2):
        if s1 is None:
            raise TypeError("Argument s1 is NoneType.")
        if s2 is None:
            raise TypeError("Argument s2 is NoneType.")
      
            
def check_qlenght(q):
    if (q < 1):
       logging.exception('Illegal value for q: %d (must be at least 1)' % (q))
       raise Exception
       
def check_maxdist(max_dist):
    if (max_dist < 0):
       logging.exception('Illegal value for maximum distance:: %d (must be ' % \
                       (max_dist) + 'zero or positive)')

       raise Exception

# Description of arguments
  # q -> number of grams
  # max_dist -> used only for the Positional version of the code ( to evaluate the range to look at the common positional grams)
  # sim_divisor -> coefficient of similarity (dice is the default)
  # padded -> set on True only for the padded version of the method
  # positional -> set on True only for the positional version of the method
  
def Qgram(s1, s2, q=2,max_dist=2,sim_divisor = 'dice', padded=True, positional = True):
 
  check_qlenght(q) # check if  q > 0
  check_maxdist(max_dist) # check if max_dist > 0
  check_type(s1,s2) # check type of input strings
  
  if (s1 == s2):
      return 1.0 # if strings are equal then the score will be 1
  
  # "c_i" is the number of common qgrams in a string
  # calculate the number "c_i" of q-grams in a string:
  # Padded -> c_i=|s_i|+q -1
  # Not Padded -> c_i = |s_i|-q +1   
  if len(s1) < q or len(s2)< q:
      return 0.0
  if (padded == True): 
    c1 = len(s1)+q-1 # c1 = number of qgrams in the first string
    c2 = len(s2)+q-1 # c2 number of qgrams in the second string
  else:
    c1 = max(len(s1)-(q-1),0)  # Make sure its not negative
    c2 = max(len(s2)-(q-1),0)
   
 
  if (sim_divisor not in ['dice','overlap','longest']): # check if the name of the similarity divisor is in this list

    logging.exception('Illegal value for similarity divisor: %s' % \

                      (sim_divisor))
    raise Exception

  # define all the possible similarity divisor that can be used (dice, overlap and longest)
  if (sim_divisor == 'dice'):
    divisor = 0.5*(c1+c2)  # Compute average number of q-grams
  elif (sim_divisor == 'overlap'):
    divisor = min(c1,c2)
  else:  # Longest
    divisor = max(c1,c2)
 
  # definition of the strings.
  # Padded -> put an special character at the beginning and at the end of the strings. The number of special charters
  # is based on (q-1)
  # Not Padded -> the strings are those given as input
  if (padded == True):
    c_s1 = (q-1)*QGRAM_START_CHAR+s1+(q-1)*QGRAM_END_CHAR
    c_s2 = (q-1)*QGRAM_START_CHAR+s2+(q-1)*QGRAM_END_CHAR
  else:
    c_s1 = s1
    c_s2 = s2
 
  # put the strings into a list
  # Positional -> the list will contain also the index of the position of the qgrams
  # Not Positional -> the list will contain only the qgrams
  if positional == True:
   c_list1 = [(c_s1[i:i+q],i) for i in range(len(c_s1) - (q-1))]
   c_list2 = [(c_s2[i:i+q],i) for i in range(len(c_s2) - (q-1))]
  else:
   c_list1 = [c_s1[i:i+q] for i in range(len(c_s1) - (q-1))]
   c_list2 = [c_s2[i:i+q] for i in range(len(c_s2) - (q-1))]
 
  # start to count how many common qgrams both the word have  
  common = 0
 
  if (c1 < c2):  # Count using the shorter q-gram list (if the lenght of the words is not the same)
    short_c_list = c_list1
    long_c_list =  c_list2

  else: # if the lenght of the words is the same, you take as shorter the second list (conventional)
    short_c_list = c_list2
    long_c_list =  c_list1
   # if Positional algorithm, take the qgram and its position in the short list
   # define a range into where you are looking for the common qgram with it position
   # find in the longest string the same qgram with the same position that you have in the shoter list
  if positional == True:
   for pos_q_gram in short_c_list:

    (q_gram,pos) = pos_q_gram
    pos_range = range(max(pos-max_dist,0), pos+max_dist+1)
    
    for test_pos in pos_range: 
      test_pos_q_gram = (q_gram,test_pos)
      if (test_pos_q_gram in long_c_list):
        common += 1 # count the common that you find
        
        long_c_list.remove(test_pos_q_gram)  # Remove the counted q-gram
        break

   score = float(common) / float(divisor) # calculate the score by using the common qgrams
  # if the algorithm is not Positional loop over the qgrams in the shorter list and count the same in the longest list 
  else:
   for q_gram in short_c_list:  
      if (q_gram in long_c_list):
        common += 1 # count the common that you find
        long_c_list.remove(q_gram)  # Remove the counted q-gram
   score = float(common) / float(divisor) # calculate the score by using the common qgrams
 
  return score     


def MongeEklan(s1,s2,q=2,max_dist=2,sim_divisor = 'dice', padded=True, positional = True):

        SumMaxSim = 0
        word1 = s1.split(' ') # split long strings in words
        word2 = s2.split(' ') # split long strings in words
        
        len_s1=len(word1)
        
        for w1 in word1:
            MaxSimScore=0
            for w2 in word2:        
                # find the maximum score of the possible combinations of words in the strings
                MaxSimScore = max(MaxSimScore,Qgram(w1, w2,q,max_dist,sim_divisor, padded, positional))
            SumMaxSim += MaxSimScore
        # use the sum of the maximum score to define the final score
        
        return SumMaxSim/len_s1    
    
