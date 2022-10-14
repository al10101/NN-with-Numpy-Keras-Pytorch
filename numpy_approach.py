#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from utils import from_probabilities_to_label

def sigmoid(z): # Sigmoid function
	return 1.0 / (1.0 + np.exp(-z))

def sigmoid_grad(z): # (first) derivative of the sigmoid function
	return sigmoid(z) * (1.0 - sigmoid(z))

def forward(X, theta1, theta2):
	'''
	Function to compute forward propagation from an input matrix. The function
	assumes that there is only one hidden layer and an output layer
	:param X: input matrix (or column vector if it is a single input)
	:param theta1: matrix containing the weights for the hidden layer
	:param theta2: matrix containing the weights for the output layer
	:return: the prediction in the form of a vector of probabilities
	'''

	m = X.shape[0] # number of data for the prediction we want to compute

	# we first add the bias as a column vector 
	bias = np.ones((m, 1))

	# The first activation function is just the input concatenated to the bias column
	a1 = np.concatenate((bias, X), axis=1)

	# We compute the linear combination of the weights and the inputs and
	# compute the activation function of that result
	z2 = np.matmul(theta1, a1.T)
	a2 = sigmoid(z2)

	# We modifiy the bias and activation to have the correct dimension for the 
	# next layer (output layer)
	bias = bias.T
	a2 = np.concatenate((bias, a2), axis=0)

	# The final outpur will be the linear combination of the final weights and 
	# the neurons of the hidden layer
	z3 = np.matmul(theta2, a2)
	a3 = sigmoid(z3)

	# The final prediction from the nn is the transpose of the final matrix
	return a3.T

def main(X, y, Y, hyperparameters):
	'''
	Function that contains routines to train a NN using only numpy
	:param X: matrix containing 5000 data with 400 features each
	:param y: vector containing the labels for the 5000 data
	:param Y: matrix containing the probabilities vectors for all labels contained in y
	:hyperparameters: dictionary to describe the characteristics of the NN
	'''

	# First of all, initialize the matrices that will contain the weights of 
	# each neuron:
	# - theta1: from input layer to hidden layer
	# - theta2: from hidden layer to output layer

	# The initial weights can be generated randomly, but preferably with a
	# random generator seed to obtain reproducible results
	np.random.seed(100)

	print(f"Defining NN with {hyperparameters['hidden_layers']} hidden layer: {hyperparameters['neurons']} neurons...")
	print()
	
	# There is an extra column in each matrix to account for the bias value
	theta1 = np.random.rand(hyperparameters['neurons'], hyperparameters['inputs'] + 1)
	theta2 = np.random.rand(hyperparameters['outputs'], hyperparameters['neurons'] + 1)

	# It is recommended to fix the values to a small range, so we use epsilon. epsilon depends
	# on the dimensions of each matrix
	epsilon = lambda x, y: 2.449489 / np.sqrt(x + y)
	e1 = epsilon( theta1.shape[0], theta1.shape[1] )
	e2 = epsilon( theta2.shape[0], theta2.shape[1] )

	theta1 = theta1 * 2.0 * e1 - e1
	theta2 = theta2 * 2.0 * e2 - e2

	# We can already check the performance of the random generated NN. For that,
	# we use the forward propagation algortithm to predict some value from the input 
	# data set. Let's choose the index=3435 (but you can modify it to your personal choice)
	random_idx = 3435
	h = forward(X, theta1, theta2)

	# Uncomment to see the probabilities vector:
	'''
	print(f'Random prediction vector: {h[random_idx]}')
	'''

	# To choose the larger probability from the vector probabilities, we call the 
	# corresponding function from the utils module
	p = from_probabilities_to_label(h, X.shape[0])

	# See how the random initalization gives poor results
	print('Prediction before training:')
	print('Actual label: {:.0f} | Prediction: {:.0f} ({:.2f}% sure)'.
		format(y[random_idx], p[random_idx], 100*h[random_idx, int(p[random_idx])]))
	print()

	# Now we begin the training process.
