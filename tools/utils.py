#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

def from_labels_to_probabilities(y, m, l): 
	'''
	Function to process the different digits and return a probability for each digit
	:param y: a vector containing m labels, from 0 to 9
	:param m: the number of data
	:param l: the number of different labels
	:return: a matrix containing probabilities
	'''

	# The column vector "y" contains a number for each data. To train a neural network that 
	# predicts probabilites, we need the probabilites in the first place: when an element 
	# has y = 2.0, the NN must know that the element has probabilites (0 0 1 0 0 0 0 0 0 0).
	# This means that the index corresponding to the digit=0 is actually the first one because
	# python starts indexing from 0 (the database was prepared this way to facilitate the organization).
	Y = np.zeros([m, l])
	for i in range(m):
		Y[i, int(y[i])] = 1

	# Uncomment to check a random label transformed to a probability vector:
	'''
	ri = np.random.randint(0, m-1)
	print('Labels changed to probabilities. Random check:')
	print(f'Index {ri}: y= {y[ri]} ---> Y= {Y[ri]}')
	print()
	'''

	return Y

def from_probabilities_to_label(h, m):
	'''
	Function to process the matrix (whose rows are vectors of probabilites) and assign them a digit
	:param h: Matrix containing m probabilities
	:param m: the number of data
	:return: a column vector with the corresponding labels
	'''

	p = np.zeros(m)
	for i in range(m):
		# Choose the index that contains the max probability
		p[i] = np.argmax(h[i, :])

	# Uncomment to check a random label transformed from a probability vector:
	'''
	ri = np.random.randint(0, m-1)
	print('Probabilites changed to labels. Random check:')
	print(f'Index {ri}: h= {h[ri]} ---> p= {p[ri]}')
	print()
	'''

	return p

def percentage(p, h, idx):
	'''
	Function to calculate the percentage of certainty that the model has that a prediction is correct
	:param p: a vector containing the labels from 0 to 9
	:param h: matrix containing probabilities from which the labels were processed and classified
	:param idx: index of the matrix/vector of predictions to check
	:return: percentage of certainty that the given idx is correct
	'''

	# Label - digit from 0 to 9 corresponding to the prediction of the model
	label = p[idx]

	# The probability class for the given label
	result = h[idx, int(label)]

	# The sum of the whole probability vector. The percentage is computed dividing the
	# actual prediction by the sum of all the possible predictions, so the result is
	# normalized
	total = np.sum(h[idx, :])

	return 100 * result / total

def show_random_predictions(X, p, h):
	'''
	Function to display the 20x20 pixel form of the input in a random manner after the neuronal network is 
	trained
	:param X: matrix containing data with 400 features each
	:param p: a vector containing the labels from 0 to 9
	:param h: Matrix containing probabilities from which the labels were processed and classified
	'''

	# Shuffle the numbers
	rp = np.arange(X.shape[0])
	np.random.shuffle(rp)

	# Commands to tell plt to keep the image on the screen
	plt.ion()
	plt.show()

	# Print examples until user input 
	user_continue = ''
	i = 0
	while user_continue != 'q':

		random_idx = rp[i]

		# Only one random data per try
		x = np.array( [X[random_idx, :] ] )

		# Show the random input as a grayscale image
		plt.imshow(x.reshape(20, 20).T, cmap='gray')
		plt.title('NN predicts the number is {:.0f} ({:.2f}% sure)'.
			format(p[random_idx], percentage(p, h, random_idx)))
		plt.draw()
		plt.pause(0.001)

		# Stop when the user types "q"
		user_continue = input('Paused - Enter to continue, q to quit:')

		i += 1

	print()