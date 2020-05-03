# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NeuralNet(torch.nn.Module):
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initialize the layers of your neural network
        @param lrate: The learning rate for the model.
        @param loss_fn: The loss function
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        The network should have the following architecture (in terms of hidden units):
        in_size -> 128 ->  out_size
        """
        LAMBDA = 0.01
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        #Network Architecture
        self.fc1 = nn.Linear(in_size, 200)
        self.fc2 = nn.Linear(200, out_size)
        #OPtimizer
        self.optim = optim.ASGD(
            self.get_parameters(), lr=lrate, weight_decay=LAMBDA)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # return torch.zeros(x.shape[0], 3)
        return x

    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y
        @param x: an (N, in_size) torch tensor
        @param y: an (N,) torch tensor
        @return L: total empirical risk (mean of losses) at this time step as a float
        """
        self.optim.zero_grad()
        y_hat = self.forward(x)
        #Use Cross Entropy loss
        loss = self.loss_fn(y_hat, y)
        loss.backward()
        self.optim.step()
        return loss.item()


def fit(train_set, train_labels, dev_set, n_iter, batch_size=100):
    """ Fit a neural net.  Use the full batch size.
    @param train_set: an (N, 784) torch tensor
    @param train_labels: an (N,) torch tensor
    @param dev_set: an (M, 784) torch tensor
    @param n_iter: int, the number of batches to go through during training (not epoches)
                   when n_iter is small, only part of train_set will be used, which is OK,
                   meant to reduce runtime on autograder.
    @param batch_size: The size of each batch to train on.
    # return all of these:
    @return losses: list of total loss (as type float) after each iteration. Ensure len(losses) == n_iter
    @return yhats: an (M,) NumPy array of approximations to labels for dev_set
    @return net: A NeuralNet object
    # NOTE: This must work for arbitrary M and N
    """

    #Standardize the data
    eps = 1e-8
    train_set = (train_set - torch.mean(train_set, 0)) / \
        (torch.std(train_set, 0) + eps)
    dev_set = (dev_set - torch.mean(dev_set, 0)) / \
        (torch.std(dev_set, 0) + eps)
    #Define a nn model
    alpha = 0.114  # Learning rate
    loss_fn = nn.CrossEntropyLoss()
    in_size = np.shape(train_set)[1]
    out_size = 5  # Number of classes
    model = NeuralNet(alpha, loss_fn, in_size, out_size)
    #Train the model
    losses = []
    running_loss = 0

    for batch_idx in range(n_iter):
        start = (batch_idx * batch_size) % len(train_labels)
        stop = (batch_idx * batch_size + batch_size) % (len(train_labels) + 1)
        x, y = train_set[start:stop, :], train_labels[start:stop]
        loss = model.step(x, y)
        #Compute the loss
        running_loss += float(loss)
        losses.append(float(loss))

    #Predict
    yhats = model(dev_set)
    _, yhats = torch.max(yhats.data, 1)
    yhats = np.array(yhats)

    # print(losses)
    return losses, yhats, model
