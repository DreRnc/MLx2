import numpy as np

class HeavyBallGradient():
	'''
    Implementation of Heavy Ball gradient descent optimization algorithm.
	Variants: rprop for backpropagation, Nesterov for momentum.

	Attributes
	----------
	self.step (Float) : learning step
	self.momentum (Float) : coefficient for momentum, multiplying last step updates for wieghts and biases
	self.Nesterov (Bool) : whether optimizer must use Nesterov momentum or not

    Methods
	-------
	__init__(self, step, momentum):
		Input:
			step (Float) : learning step
			momentum (Float) : coefficient for momentum, multiplying last step updates for wieghts and biases
			Nesterov (Bool) : whether optimizer must use Nesterov momentum or not

	__call__(self, grad_weights, grad_biases, last_weights_update, last_biases_update, rprop):
		Input: 
			grad_weights (np.array) : gradient of the loss with respect to the weights
			grad_biases (np.array) : gradient of the loss with respect to the biases
			last_weights_update (np.array) : update on weights of previous step
			last_biases_update (np.array) : update on biases of previous step
			rprop (Bool) : whether to apply rprop variant or standard backprop
		Output:
			updates on weights (np.array) : update to apply to weights for current optimization step
			updates on biases (np.array) : update to apply to biases for current optimization step
    '''
	
	def __init__(self, step, momentum, Nesterov):
		self.step = step
		self.momentum = momentum
		self.Nesterov = Nesterov

	def __call__(self, grad_weights, grad_biases, last_weights_update, last_biases_update, rprop):

		if rprop:
			weights_updates, biases_updates = -self.step * np.sign(grad_weights), -self.step * np.sign(grad_biases)
		else:
			weights_updates, biases_updates = -self.step * grad_weights, -self.step * grad_biases

		return weights_updates + self.momentum * last_weights_update, biases_updates + self.momentum * last_biases_update
