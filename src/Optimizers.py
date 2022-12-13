
class Optimizer:

	def __init__ (self, step = 0.2):
		self.step = step

	def __call__(self):
		raise NotImplementedError


class gradient(Optimizer):

	def __call__(self, grad_weigths, grad_biases):

		return -self.step * grad_weigths, -self.step * grad_biases


def get_optimizer_instance(optimizer):
    
    if optimizer in  ["gradient", "gradient descent", "GD"]:
        return gradient()
    else:
        raise ValueError("Metric function not found")