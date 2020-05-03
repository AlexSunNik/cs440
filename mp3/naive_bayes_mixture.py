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
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


import numpy as np
import math
from collections import Counter





def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each movie review
    example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two reviews, first one was positive and second one was negative.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each review that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """
    
    #Build a bigram model
    # TODO: Write your code here
    #Build the review word dictionary
    #For Bigram
    pos_dict_freq_bi = {}
    neg_dict_freq_bi = {}
    pos_word_total_bi = 0
    neg_word_total_bi = 0
    #For Unigram
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
            pos_word_total_bi += len(review) - 1
            pos_word_total += len(review)
        else:
            neg_word_total_bi += len(review) - 1
            neg_word_total += len(review)
        #Update the frequency
        for j in range(len(review) - 1):
            bi = (review[j], review[j+1])
            word = review[j]
            if label:
                cur_freq_bi = pos_dict_freq_bi.setdefault(bi, 0)
                pos_dict_freq_bi[bi] = cur_freq_bi + 1
                cur_freq = pos_dict_freq.setdefault(word, 0)
                pos_dict_freq[word] = cur_freq + 1
            else:
                cur_freq_bi = neg_dict_freq_bi.setdefault(bi, 0)
                neg_dict_freq_bi[bi] = cur_freq_bi + 1
                cur_freq = neg_dict_freq.setdefault(word, 0)
                neg_dict_freq[word] = cur_freq + 1
        #For unigram model  Account for the last one
        if label:
            cur_freq = pos_dict_freq.setdefault(review[-1], 0)
            pos_dict_freq[review[-1]] = cur_freq + 1
        else:
            cur_freq = neg_dict_freq.setdefault(review[-1], 0)
            neg_dict_freq[review[-1]] = cur_freq + 1

    #Predict dev labels
    dev_labels = []
    for test_review in dev_set:
        pos_post_bi = np.log(pos_prior)
        neg_post_bi = np.log(1 - pos_prior)
        pos_post = np.log(pos_prior)
        neg_post = np.log(1 - pos_prior)

        for j in range(len(test_review) - 1):
            bi = (test_review[j], test_review[j+1])
            word = test_review[j]
            #For Bigram
            p_word_pos_bi = (pos_dict_freq_bi.get(bi, 0) + bigram_smoothing_parameter) / \
                (pos_word_total_bi + bigram_smoothing_parameter * len(pos_dict_freq_bi))
            #Laplace Smoothing
            p_word_neg_bi = (neg_dict_freq_bi.get(bi, 0) + bigram_smoothing_parameter) / \
                (neg_word_total_bi + bigram_smoothing_parameter * len(neg_dict_freq_bi))
            #Laplace Smoothing
            pos_post_bi += np.log(p_word_pos_bi)
            neg_post_bi += np.log(p_word_neg_bi)

            #For unigram
            p_word_pos = (pos_dict_freq.get(word, 0) + unigram_smoothing_parameter) / \
                (pos_word_total + unigram_smoothing_parameter * len(pos_dict_freq))
            #Laplace Smoothing
            p_word_neg = (neg_dict_freq.get(word, 0) + unigram_smoothing_parameter) / \
                (neg_word_total + unigram_smoothing_parameter * len(neg_dict_freq))
            #Laplace Smoothing
            pos_post += np.log(p_word_pos)
            neg_post += np.log(p_word_neg)

        #For unigram, account for the last one
        word = test_review[-1]
        p_word_pos = (pos_dict_freq.get(word, 0) + unigram_smoothing_parameter) / \
            (pos_word_total + unigram_smoothing_parameter * len(pos_dict_freq))
        #Laplace Smoothing
        p_word_neg = (neg_dict_freq.get(word, 0) + unigram_smoothing_parameter) / \
            (neg_word_total + unigram_smoothing_parameter * len(neg_dict_freq))
        #Laplace Smoothing
        pos_post += np.log(p_word_pos)
        neg_post += np.log(p_word_neg)

        pos_post_total = (1 - bigram_lambda) * pos_post + bigram_lambda * pos_post_bi
        neg_post_total = (1 - bigram_lambda) * neg_post + bigram_lambda * neg_post_bi

        if pos_post_total > neg_post_total:
            dev_labels.append(1)
        else:
            dev_labels.append(0)
    # return predicted labels of development set (make sure it's a list, not a numpy array or similar)
    return dev_labels
