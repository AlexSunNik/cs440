# classify.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018
# Extended by Daniel Gonzales (dsgonza2@illinois.edu) on 3/11/2020
import numpy as np

"""
This is the main entry point for MP5. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.

train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
            This can be thought of as a list of 7500 vectors that are each
            3072 dimensional.  We have 3072 dimensions because there are
            each image is 32x32 and we have 3 color channels.
            So 32*32*3 = 3072. RGB values have been scaled to range 0-1.

train_labels - List of labels corresponding with images in train_set
example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
         and X1 is a picture of a dog and X2 is a picture of an airplane.
         Then train_labels := [1,0] because X1 contains a picture of an animal
         and X2 contains no animals in the picture.

dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
          It is the same format as train_set
"""

def trainPerceptron(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters 

    #Get signed label
    signed_train_labels = [x if x else -1 for x in train_labels]
    W = np.zeros(train_set.shape[1])
    b = 0
    for epoch in range(max_iter):
        for i in range(len(signed_train_labels)):
            y_hat = np.dot(W, train_set[i].T) + b
            #Apply sgn()
            y_hat = 1 if (y_hat > 0) else -1
            #Update
            if y_hat != signed_train_labels[i]:
                W += learning_rate * signed_train_labels[i] * train_set[i]
                b += learning_rate * signed_train_labels[i]
    return W, b

def classifyPerceptron(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train perceptron model and return predicted labels of development set
    W,b = trainPerceptron(train_set, train_labels, learning_rate, max_iter)
    dev_preds = np.dot(dev_set, W.T) + b
    dev_preds = (dev_preds > 0)
    return dev_preds

def sigmoid(x):
    # TODO: Write your code here
    # return output of sigmoid function given input x
    return 1 / (1 + np.exp(-x))

def trainLR(train_set, train_labels, learning_rate, max_iter):
    # TODO: Write your code here
    # return the trained weight and bias parameters 
    #Total number of training tokens
    m, n = train_set.shape
    W = np.zeros(n)
    b = 0
    for epoch in range(max_iter):
        Y_hat = sigmoid(np.dot(train_set, W.T) + b)
        dW = (1/m) * np.dot(train_set.T, (Y_hat - train_labels))
        db = (1/m) * np.sum(Y_hat - train_labels)
        W -= dW.T * learning_rate
        b -= db * learning_rate
    return W, b

def classifyLR(train_set, train_labels, dev_set, learning_rate, max_iter):
    # TODO: Write your code here
    # Train LR model and return predicted labels of development set
    W, b = trainLR(train_set, train_labels, learning_rate, max_iter)
    dev_preds = sigmoid(np.dot(dev_set, W.T) + b)
    dev_preds = dev_preds > 0.5
    return dev_preds

def classifyEC(train_set, train_labels, dev_set, k):
    # Write your code here if you would like to attempt the extra credit
    dev_preds = []
    for test_x in dev_set:
        #Store (dist, label)
        dists = [ (np.linalg.norm(test_x - train_set[i]), train_labels[i]) for i in range(len(train_labels))]
        dists.sort(key = lambda x: x[0])
        votes = dists[:k]
        one_label_count = 0
        zero_label_count = 0
        for vote in votes:
            if vote[1]:
                one_label_count += 1
            else:
                zero_label_count += 1
        y_pred = 1 if one_label_count > zero_label_count else 0
        dev_preds.append(y_pred)

    return dev_preds
