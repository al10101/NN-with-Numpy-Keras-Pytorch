#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

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
	Funcion to process the matrix (whose rows are vectors of probabilites) and assign them a digit
	:param h: Matrix containing m probabilities
	:param m: the number of data
	:return: a column vector with the corresponding labels
	'''

	p = np.zeros(m)
	for i in range(m):
		# Choose the index that contains the max probability
		p[i] = np.argmax(h[i, :])

	# Uncomment to check a random label transformed to a probability vector:
	'''
	ri = np.random.randint(0, m-1)
	print('Probabilites changed to labels. Random check:')
	print(f'Index {ri}: h= {h[ri]} ---> p= {p[ri]}')
	print()
	'''

	return p