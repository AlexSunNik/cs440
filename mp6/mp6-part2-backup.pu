# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019

"""
You should only modify code within this file for part 2 -- the unrevised staff files will be used for all other
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
        @param loss_fn: The loss functions
        @param in_size: Dimension of input
        @param out_size: Dimension of output
        """

        """
        Structure is as follows:
        Conv2d: 1 - 16 channels, with kernel size 3, stride 1 and padding 1.
        Maxpooling: 2 x 2
        Affine: 16 x 13 x 13 -> 128 -> outsize
        """
        #Regularization parameter
        LAMBDA = 0.01
        super(NeuralNet, self).__init__()
        self.loss_fn = loss_fn
        self.lrate = lrate
        #Network Architecture
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        # self.fc1 = nn.Linear(in_size, 512)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, out_size)
        #OPtimizer
        self.optim = optim.SGD(self.get_parameters(),
                               lr=lrate, weight_decay=LAMBDA)

    def get_parameters(self):
        """ Get the parameters of your network
        @return params: a list of tensors containing all parameters of the network
        """
        # return self.net.parameters()
        return self.parameters()

    def forward(self, x):
        """ A forward pass of your autoencoder
        @param x: an (N, in_size) torch tensor
        @return y: an (N, out_size) torch tensor of output from the network
        """
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        # return torch.zeros(x.shape[0], 5)

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
        return loss

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


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
    #Prevent NULL. IMPORTANT!!!!
    eps = 1e-15
    train_set = (train_set - torch.mean(train_set, 0)) / (torch.std(train_set, 0) + eps)
    dev_set = (dev_set - torch.mean(dev_set, 0)) / (torch.std(dev_set, 0) + eps)
    #Define a nn model
    alpha = 0.1  # Learning rate
    loss_fn = nn.CrossEntropyLoss()
    in_size = np.shape(train_set)[1]
    out_size = 5  # Number of classes
    model = NeuralNet(alpha, loss_fn, in_size, out_size)
    #Train the model
    losses = []
    running_loss = 0

    for batch_idx in range(n_iter):
        start = batch_idx * batch_size
        stop = batch_idx * batch_size + batch_size
        if stop > len(train_labels):
            stop = len(train_labels)
        x, y = train_set[start:stop, :], train_labels[start:stop]
        x = x.view((stop - start), 1, 28, 28)
        model.step(x, y)
        #Compute the loss
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        running_loss += loss.item()
        losses.append(running_loss)

    #Predict
    dev_set = dev_set.view(len(dev_set), 1, 28, 28)
    yhats = model(dev_set)
    _, yhats = torch.max(yhats.data, 1)
    yhats = np.array(yhats)
    return losses, yhats, model
