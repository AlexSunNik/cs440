# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 1 of MP3. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)

    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    #print(train_set[1])
    #print(train_labels[1])

    #Build the review word dictionary
    pos_dict_freq = {}
    neg_dict_freq = {}
    pos_word_total = 0
    neg_word_total = 0
    #P = #documents of type in C / total # of documents
    #Loop through the training data sample
    for i in range(len(train_labels)):
        label = int(train_labels[i])
        review = train_set[i]
        #Update the total
        if label:
            pos_word_total += len(review)  
        else:
            neg_word_total += len(review)
        #Update the frequency
        for word in review:
            if label:
                cur_freq = pos_dict_freq.setdefault(word, 0)
                pos_dict_freq[word] = cur_freq + 1
            else:
                cur_freq = neg_dict_freq.setdefault(word, 0)
                neg_dict_freq[word] = cur_freq + 1
    
    #Predict dev labels
    dev_labels = []
    for test_review in dev_set:
        pos_post = np.log(pos_prior)
        neg_post = np.log(1 - pos_prior)
        for word in test_review:
            p_word_pos = (pos_dict_freq.get(word, 0) + smoothing_parameter) / \
                (pos_word_total + smoothing_parameter * len(pos_dict_freq))
            #Laplace Smoothing
            p_word_neg = (neg_dict_freq.get(word, 0) + smoothing_parameter) / \
                (neg_word_total + smoothing_parameter * len(neg_dict_freq))
            #Laplace Smoothing
            pos_post += np.log(p_word_pos)
            neg_post += np.log(p_word_neg)

        if pos_post > neg_post:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels
