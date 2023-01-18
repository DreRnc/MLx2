import numpy as np

class HeavyBallGradient():

	def __init__(self, step, momentum, Nesterov):
		self.step = step
		self.Nesterov = Nesterov
		self.momentum = momentum

	def __call__(self, grad_weights, grad_biases, last_weights_update, last_biases_update, backprop_variant, last_grad_weights, last_grad_biases):

		if backprop_variant == 'rprop':
			weights_updates, biases_updates = -self.step * np.sign(grad_weights), -self.step * np.sign(grad_biases)
		elif backprop_variant == 'quickprop' and (last_grad_weights - grad_weights).all() !=0 and (last_grad_biases - grad_biases).all() != 0:
			self.momentum = 0
			self.Nesterov = False
			weights_updates, biases_updates = -self.step * np.divide(np.multiply(last_weights_update, grad_weights), (last_grad_weights - grad_weights)), \
			-self.step * np.divide(np.multiply(last_biases_update, grad_biases), (last_grad_biases - grad_biases))
		else:
			weights_updates, biases_updates = -self.step * grad_weights, -self.step * grad_biases



		if self.Nesterov:
			return weights_updates, biases_updates
		else:
			return weights_updates + self.momentum * last_weights_update, biases_updates+ self.momentum * last_biases_update


# Rprop 

# -self.step*sign(grad_weigths)

# Quickprop

# self.step * last_weigths_update*(grad_weights / (grad_weights - grad_weights_m1))
