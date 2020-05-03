# tf_idf_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for the Extra Credit Part of this MP. You should only modify code
within this file for the Extra Credit Part -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import math
from collections import Counter
import time



def compute_tf_idf(train_set, train_labels, dev_set):
        """
        train_set - List of list of words corresponding with each movie review
        example: suppose I had two reviews 'like this movie' and 'i fall asleep' in my training set
        Then train_set := [['like','this','movie'], ['i','fall','asleep']]

        train_labels - List of labels corresponding with train_set
        example: Suppose I had two reviews, first one was positive and second one was negative.
        Then train_labels := [1, 0]

        dev_set - List of list of words corresponding with each review that we are testing on
                It follows the same format as train_set

        Return: A list containing words with the highest tf-idf value from the dev_set documents
                Returned list should have same size as dev_set (one word from each dev_set document)
        """

        # TODO: Write your code here
        #Idf is calculated based on the train_set
        #Tf is calculated based obn the dev_set
        idf = {}
        total_train_doc = len(train_labels)
        #Loop through the dev_set to build up the idf
        for doc in train_set:
            for word in set(doc):
                cur_count = idf.setdefault(word, 0)
                idf[word] = cur_count + 1
        #Loop through the train_set
        order_list = []
        for doc in dev_set:
            tf = {}
            total_num_word = len(doc)
            cur_doc_tf_idf = {}
            for word in doc:
                cur_freq = tf.setdefault(word, 0)
                tf[word] = cur_freq + 1
            for word in set(doc):
                #print(word)
                cur_idf = idf.get(word, 0)
                tf_idf = (tf[word]/total_num_word) * np.log(total_train_doc / (1 + cur_idf))
                cur_doc_tf_idf[word] = tf_idf
            max_word = max(cur_doc_tf_idf.keys(), key=lambda k: cur_doc_tf_idf[k])
            order_list.append(max_word)
        #Loop through the train_se
        # return list of words (should return a list, not numpy array or similar)
        print(order_list)
        return order_list
