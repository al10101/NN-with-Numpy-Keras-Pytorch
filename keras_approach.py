#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from matplotlib import pyplot as plt

from tools.utils import from_probabilities_to_label, show_random_predictions, percentage

def main(X, y, Y, hyperparameters):
	'''
	Function that contains routines to train a NN using keras and TF
	:param X: matrix containing 5000 data with 400 features each
	:param y: vector containing the labels for the 5000 data
	:param Y: matrix containing the probabilities vectors for all labels contained in y
	:hyperparameters: dictionary to describe the characteristics of the NN
	'''

	print('=' * 36)
	print('TRAINING NN WITH KERAS'.center(36))
	print('=' * 36)
	print()

	# To simplify further commands, we extract the dictionary and save the values in specific variables
	input_layer_size = hyperparameters['inputs']
	hidden_layer_size = hyperparameters['neurons']
	num_labels = hyperparameters['outputs']
	lam = hyperparameters['lambda']

	# We define the number of epochs to train the model. Since the optimizer used here (stochastic gradient descent)
	# is faster than the optimizer used in the Numpy approach (gradient descent with step), the number of iterations
	# needed is a little smaller
	epochs = int(hyperparameters['iters'] / 15)

	# We can generate directly the object corresponding to the hidden layer using 1 command. The
	# layer contains 25 neurons, recieves the information from 400 input features, the activation 
	# function is a sigmoid and the regularizer is defined inside L2 model. All these instructions
	# are defined to match the same model as the NN with numpy
	hidden_layer = layers.Dense(
		hidden_layer_size,
		input_dim=input_layer_size,
		activation='sigmoid',
		activity_regularizer=keras.regularizers.L2(lam)
	)

	# In the same way, we generate the output layer. It contains 10 neurons, the activation is
	# sigmoid and the regularizer is the same L2 than before
	output_layer = layers.Dense(
		num_labels,
		activation='sigmoid',
		activity_regularizer=keras.regularizers.L2(lam)
	)

	# We generate the NN as a list/tuple of layers
	model = Sequential((hidden_layer, output_layer))

	# Minimaztion method and cost function are the same as in the NN with numpy code
	sgd_opt = keras.optimizers.SGD(learning_rate=hyperparameters['gd_step'], momentum=0.0)
	model.compile(optimizer=sgd_opt, loss=keras.losses.CategoricalCrossentropy())

	# We can also check the performance of the random generated NN. For that,
	# we use the forward propagation algortithm to predict some value from the input 
	# data set. Let's choose the index=3435 (but you can modify it to your personal choice)
	random_idx = 3435
	h = model.predict(X)

	# To choose the larger probability from the vector probabilities, we call the 
	# corresponding function from the utils module
	p = from_probabilities_to_label(h, X.shape[0])

	# See how the random initalization gives poor results
	print('Prediction before training:')
	print('Actual label: {:.0f} | Prediction: {:.0f} ({:.2f}% sure)'.
		format(y[random_idx], p[random_idx], percentage(p, h, random_idx)))
	print()

	# To train it, we simply write the following command
	print('Training (it may take a few seconds)...')
	model.fit(X, Y, epochs=epochs, verbose=0)

	# Compute the actual predictions using the same training data as input, just as a test
	h = model.predict(X)
	p = from_probabilities_to_label(h, X.shape[0])

	# As with the Numpy approach, we compute the accuracy of the model
	acc = np.mean(y == p) * 100
	print('NN accuracy: {:>5.2f}%'.format(acc))
	print()

	# Finally, we check that the prediction is actually accurate comparing it to the previous
	# result from the random initialization
	print('Prediction after training:')
	print('Actual label: {:.0f} | Prediction: {:.0f} ({:.2f}% sure)'.
		format(y[random_idx], p[random_idx], percentage(p, h, random_idx)))
	print()

	# To have some fun, we can predict values and show the corresponding 20x20 pixels image
	# to judge for ourselves if the NN did a good job
	show_random_predictions(X, p, h)


