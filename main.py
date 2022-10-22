#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np

from numpy_approach import main as numpy_approach

from tools.utils import from_labels_to_probabilities

# Try to import keras and pytorch
keras_imported = False
pytorch_imported = False

try:
	from keras_approach import main as keras_approach
	keras_imported = True
except ModuleNotFoundError as e:
	print(e)

try:
	from pytorch_approach import main as pytorch_approach
	pytorch_imported = True
except ModuleNotFoundError as e:
	print(e)

def main():

	print()
	print('*'*36)
	print('TRAIN A NN TO RECONIZE DIGITS'.center(36))
	print('*'*36)
	print()

	# There are 3 possible approaches to train the NN model: Numpy, keras and pytorch. 
	# If one of them is not imported (because the user did not install it or something 
	# like that), it will not be available
	avaliable_selections = [1]
	if keras_imported:
		avaliable_selections.append(2)
	if pytorch_imported:
		avaliable_selections.append(3)

	# The user defines the approach with his/her input
	selection = ''
	appropiate_selection = False
	while not appropiate_selection:

		print('Approaches available:')
		print('1. Numpy')
		if keras_imported: 
			print('2. Keras')
		else:
			print('*. KERAS NOT INSTALLED CORRECTLY, IT CANNOT BE USED TO TRAIN THE NN')
		if pytorch_imported:
			print('3. Pytorch')
		else:
			print('*. PYTORCH NOT INSTALLED CORRECTLY, IT CANNOT BE USED TO TRAIN THE NN')

		selection = input('Enter the number to select an approach (q to quit):')
		if selection == 'q':
			exit()

		# Check if the selection (as integer) is available
		try:
			selection = int(selection)
		except ValueError:
			pass

		appropiate_selection = selection in avaliable_selections
		if not appropiate_selection:
			print(f'Input not recognized as an available selection: {selection}')

		print()

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
		'inputs': 400,		# Number of features (pixels) per data
		'hidden_layers': 1,	# Number of hidden layers, only 1 in this case
		'neurons': 25,		# Number of neurons in the hidden layer
		'outputs': 10,		# Number of labels, from 0 to 9
		'lambda': 0.01,		# Regularization parameter for the training procedure
		'iters': 1000, 		# We will iterate a lot because gradient descent algoritm has slow convergence
		'gd_step': 3.1, 	# Step size of for gradient descent algorithm (empyrical for this particular)
		'n_data': 5000		# Number of training data
	}

	# Transform y to Y (labels to probabilites) with the corresponding function from
	# the utils module
	Y = from_labels_to_probabilities(y, hyperparams['n_data'], hyperparams['outputs'])

	# ====================================================
	# NUMPY APPROACH
	# ====================================================
	if selection == 1: numpy_approach(X, y, Y, hyperparams)

	# ====================================================
	# KERAS APPROACH
	# ====================================================
	elif selection == 2: keras_approach(X, y, Y, hyperparams)

	# ====================================================
	# PYTORCH APPROACH
	# ====================================================
	elif selection == 3: pytorch_approach(X, y, Y, hyperparams)

if __name__ == '__main__':

	main()

