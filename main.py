#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from numpy_approach import main as numpy_approach

def from_labels_to_probabilities(y, m, l): 
	'''
	Function to process the different digits and return a probability for each digit
	:param y: a vector containing 5000 labels, from 0 to 9
	:param m: the number of data
	:param l: the number of different labels
	:return: a matrix containing probabilities
	'''

	# The column vector "y" contains a number for each data. To train a neural network that 
	# predicts probabilites, we need the probabilites in the first place: when an element 
	# has y = 2.0, the NN must know that the element has probabilites (0 1 0 0 0 0 0 0 0 0)
	Y = np.zeros([m, l])
	for i in range(m):
		# -1 because python uses zero-indexing
		Y[i, int(y[i]) - 1] = 1

	# Fast check with a random label
	ri = np.random.randint(0, m-1)
	print('Labels changed to probabilities. Random check:')
	print(f'Index {ri}: y= {y[ri]} ---> Y= {Y[ri]}')
	print()

	return Y

def main():

	# Read the training data from the tools folder. The data is already prepared and
	# stored as numpy files to facilitate the reading process. The X variable contains 5000
	# data with 400 features each (20x20 pixels from each image). The y variable contains
	# the labels of all the 5000 data.
	X = np.load( os.path.join('tools', 'X.npy') )
	y = np.load( os.path.join('tools', 'y.npy') )
	print(f'X is a {X.shape} matrix')
	print(f'y is a {y.shape[0]} - dimensional vector ')
	print()

	# All neural networks from the three approaches share the same characteristics:
	# - 400 inputs
	# - 1 hidden layer with 25 neurons
	# - 10 outputs (1 for each possible digit)
	# - regularization parameter=0.01
	# - iterations (epochs) = 20
	# So it is convenient to set all this info in a dictionary to be used in each code
	hyperparams = {
		'inputs': 400,
		'hidden_layers': 1,
		'neurons': 25,
		'outputs': 10,
		'lambda': 0.01,
		'iters': 20,
		'n_data': 5000
	}

	# Transform y to Y (labels to probabilites)
	Y = from_labels_to_probabilities(y, hyperparams['n_data'], hyperparams['outputs'])

	# ====================================================
	# NUMPY APPROACH
	# ====================================================
	numpy_approach(X, y, hyperparams)


if __name__ == '__main__':

	main()

