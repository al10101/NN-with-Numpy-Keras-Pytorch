#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def main(X, y, hyperparameters):
	'''
	Function that contains routines to train a NN using only numpy
	:param X: matrix containing 5000 data with 400 features each
	:param y: vector containing the labels for the 5000 data
	:hyperparameters: dictionary to describe the characteristics of the NN
	'''

	# First of all, initialize the matrices that will contain the weights of 
	# each neuron:
	# - theta1: from input layer to hidden layer
	# - theta2: from hidden layer to output layer

	# The initial weights can be generated randomly, but preferably with a
	# random generator seed to obtain reproducible results
	np.random.seed(100)
	
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



