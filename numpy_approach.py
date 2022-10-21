#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from utils import from_probabilities_to_label, show_random_predictions

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

def nn_cost_function(hyperparameters, X, Y, theta1, theta2):
	'''
	Function to compute the cost function following the forward propagation algorithm, as 
	well as the gradients for each theta matrix
	:param hyperparameters: dictionary to describe the characteristics of the NN
	:param X: matrix containing the input data
	:param Y: matrix containing the probability vector from all the labels
	:param theta1: matrix containing the weights for the hidden layer
	:param theta2: matrix containing the weights for the output layer
	:return: value of the cost function for the weights given, gradients of theta1, gradients of theta2
	'''

	# Number of training data
	m = hyperparameters['n_data']

	# We repeat the forward propagation algorithm here because we need the 
	# z2 variable to compute the gradient
	a1 = np.concatenate( (np.ones( (m, 1) ), X), axis=1).T
	z2 = np.matmul(theta1, a1)
	a2 = np.concatenate( (np.ones( (1, m) ), sigmoid(z2)), axis=0)
	a3 = sigmoid( np.matmul(theta2, a2) )
	h = a3.T

	# Sum of errors without the bias
	sum_t1 = np.sum(theta1[:, 1:] ** 2)
	sum_t2 = np.sum(theta2[:, 1:] ** 2)

	# Choosen cost function for this classification problem: Cross entropy
	lam = hyperparameters['lambda']
	J = (1 / m) * np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h)) + (lam / (2 * m)) * (sum_t1 + sum_t2)

	# Backward propagation starts here
	d3 = (h - Y).T

	# We add 1 row to z2 to compute the matrix multiplication. This row doesn't affect the
	# final result anyway because it is erased at the end, we only add it to perform the operation
	aux_z2 = np.concatenate((np.ones((1, m)), z2), axis=0)
	d2 = np.matmul(theta2.T, d3) * sigmoid_grad(aux_z2)
	d2 = d2[1:, :]

	# Define the gradients of the matrices
	theta2_grad = (1/m) * np.matmul(d3, a2.T)
	theta1_grad = (1/m) * np.matmul(d2, a1.T)

	# Finally, we add the regularization term (except for the bias, it stays the same)
	# theta1
	temp = theta1_grad[:, 0]
	theta1_grad += (lam / m) * theta1
	theta1_grad[:, 0] = temp
	# theta2
	temp = theta2_grad[:, 0]
	theta2_grad += (lam / m) * theta2
	theta2_grad[:, 0] = temp

	return (J, theta1_grad, theta2_grad)

def main(X, y, Y, hyperparameters):
	'''
	Function that contains routines to train a NN using only numpy
	:param X: matrix containing 5000 data with 400 features each
	:param y: vector containing the labels for the 5000 data
	:param Y: matrix containing the probabilities vectors for all labels contained in y
	:hyperparameters: dictionary to describe the characteristics of the NN
	'''

	print('=' * 36)
	print('TRAINING NN WITH NUMPY'.center(36))
	print('=' * 36)
	print()

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
	print('Actual label: {:.0f} | Prediction: {:.0f} ({:.2f}%)'.
		format(y[random_idx], p[random_idx], 100*h[random_idx, int(p[random_idx])]))
	print()

	# Now we begin the training process: we iterate to compute the cost function and modify the
	# weights using the gradient at each step (gradient descent algorithm). We also want to save 
	# the values of the cost at each iteration so we can see the training process later
	iters = hyperparameters['iters']
	j_history = np.zeros(iters)
	for i in range(iters):

		# Compute cost function and gradients
		(J, theta1_grad, theta2_grad) = nn_cost_function(hyperparameters, X, Y, theta1, theta2)
		j_history[i] = J

		print('Iteration Nr. {:4}/{}: Cost= {:.6f}'.format(i+1, iters, J), end='\r')

		# Now it's time to update the weights. We update the matrices adding the gradient of each 
		# weight multiplied by a empirical number called "step"
		theta1 -= theta1_grad * hyperparameters['sgd_step']
		theta2 -= theta2_grad * hyperparameters['sgd_step']

	print()

	# Check the training process
	print('Showing the plot for the training process (close to continue)')
	plt.plot(np.arange(iters), j_history)
	plt.title('Training NN with numpy')
	plt.ylabel('Cost')
	plt.xlabel('Iterations')
	plt.show()
	print()

	# Repeat the same proceadure from above to compute the actual predictions
	h = forward(X, theta1, theta2)
	p = from_probabilities_to_label(h, X.shape[0])

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

