#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

from utils import from_probabilities_to_label, show_random_predictions

class MyModel(nn.Module):

	def __init__(self, hyperparameters):
		'''
		Function to initialize the weight matrices to be used by the NN model. It is already customized to 
		have only sigmoid activation functions to match the Numpy approach. It also contains only 1 hidden layer.
		:param hyperparameters: dictionary containing information about the architecture of the NN
		'''

		# Initialize the parent (already a pytorch object)
		super().__init__()

		# Just like with the Numpy approach, we define the matrix that contains the weights that
		# are used to pass from the input layer to the hidden layer. This matrix is defined with 
		# the corresponding pytorch objects, we only say the dimensions and the activation function
		self.linear1 = nn.Linear(hyperparameters['inputs'], hyperparameters['neurons'])
		self.act1 = nn.Sigmoid()

		# The same process is needed to define the matrix that contains the weights to pass from 
		# the hidden layer to the output layer
		self.linear2 = nn.Linear(hyperparameters['neurons'], hyperparameters['outputs'])
		self.act2 = nn.Sigmoid()

	def forward(self, X):
		'''
		Function to compute the forward propagation from an input tensor. The function uses the 
		information defined inside __init__() to define the architecture of the NN model and then 
		computes the prediction given X
		:param X: input tensor (matrix transformed into a tensor object)
		:return: the prediction in the form of a tensor
		'''

		h = self.linear1(X)
		h = self.act1(h)
		h = self.linear2(h)
		h = self.act2(h)

		return h

	def fit(self, optimizer, data_inputs, data_probabilities, cost_function, epochs):

		j_history = np.zeros(epochs)
		
		# Main loop 
		for i in range(epochs):

			# The training process involves the computation of the gradients of the cost function, so we must
			# make sure that the gradients are already zero because pytorch does not overwrite them
			optimizer.zero_grad()

			# Run the model with forward propagation
			preds = self.forward(data_inputs)

			# Compute cost function
			loss = cost_function(preds, data_probabilities)
			j_history[i] = loss.item()

			print('Iteration Nr. {:4}/{}: Cost= {:.6f}'.format(i+1, epochs, loss.item()), end='\r')

			# Compute back propagation 
			loss.backward()

			# Update the weights
			optimizer.step()

		print()

def main(X, y, Y, hyperparameters):
	'''
	Function that contains routines to train a NN using Pytorch
	:param X: matrix containing 5000 data with 400 features each
	:param y: vector containing the labels for the 5000 data
	:param Y: matrix containing the probabilities vectors for all labels contained in y
	:hyperparameters: dictionary to describe the characteristics of the NN
	'''

	print('=' * 36)
	print('TRAINING NN WITH PYTORCH'.center(36))
	print('=' * 36)
	print()

	# Since pytorch uses tensors, all data must be transformed into tensor objects
	X_tensor = torch.tensor(X, dtype=torch.float32)
	Y_tensor = torch.tensor(Y, dtype=torch.float32)

	# We can construct the model with the custom class previously designed for this task
	my_model = MyModel(hyperparameters)

	# Minimization method and cost function are the same as in the NN with numpy code
	optimizer = torch.optim.SGD(my_model.parameters(), lr=hyperparameters['gd_step'])
	nn_cost_function = nn.CrossEntropyLoss()

	# We can also check the performance of the random generated NN. For that,
	# we use the forward propagation algortithm to predict some value from the input 
	# data set. Let's choose the index=3435 (but you can modify it to your personal choice)
	random_idx = 3435
	h = my_model.forward(X_tensor)

	# To choose the larger probability from the vector probabilities, we call the 
	# corresponding function from the utils module
	p = from_probabilities_to_label(h.detach().numpy(), X.shape[0])

	# See how the random initalization gives poor results
	print('Prediction before training:')
	print('Actual label: {:.0f} | Prediction: {:.0f} ({:.2f}%)'.
		format(y[random_idx], p[random_idx], 100*h[random_idx, int(p[random_idx])]))
	print()

	# We can train the model with the custom function previosly designed for this task
	my_model.fit(optimizer, X_tensor, Y_tensor, nn_cost_function, hyperparameters['iters'])

	# As with the Numpy approach, we compute the accuracy of the model
	h = my_model.forward(X_tensor)
	p = from_probabilities_to_label(h.detach().numpy(), X.shape[0])

	# We can compute the accuracy of the model calculating the mean of the predictions that 
	# are equal to the label
	acc = np.mean(y == p) * 100
	print('NN accuracy: {:>5.2f}%'.format(acc))
	print()

	# Finally, we check that the prediction is actually accurate comparing it to the previous
	# result from the random initialization
	print('Prediction after training:')
	print('Actual label: {:.0f} | Prediction: {:.0f} ({:.2f}%)'.
		format(y[random_idx], p[random_idx], 100*h[random_idx, int(p[random_idx])]))
	print()

	# To have some fun, we can predict values and show the corresponding 20x20 pixels image
	# to judge for ourselves if the NN did a good job
	show_random_predictions(X, p, h)
