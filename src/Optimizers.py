class HeavyBallGradient():

	def __init__(self, step, momentum, Nesterov):
		self.step = step
		self.Nesterov = Nesterov
		self.momentum = momentum

	def __call__(self, grad_weigths, grad_biases, last_weights_update, last_biases_update):

		if self.Nesterov:
			return -self.step * grad_weigths, -self.step * grad_biases
		else:
			return -self.step * grad_weigths + self.momentum * last_weights_update, -self.step * grad_biases + self.momentum * last_biases_update


