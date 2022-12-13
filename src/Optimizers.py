
class Optimizer:

	def __init__ (self, step = 1):
		self.step = step

	def initialize(self):
		pass

	def optimize(self, grad_weights, grad_biases):
		return -self.step * grad_weights, -self.step * grad_biases

def get_optimizer():
	return Optimizer()