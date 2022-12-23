
# class Optimizer:

# 	def __init__ (self, step = 0.01):
# 		self.step = step

# 	def __call__(self):
# 		raise NotImplementedError


# class Gradient(Optimizer):

# 	def __call__(self, grad_weigths, grad_biases):

# 		return -self.step * grad_weigths, -self.step * grad_biases

class HeavyBallGradient():

	def __init__(self, step, momentum):
		self.step = step
		self.momentum = momentum

	def __call__(self, grad_weigths, grad_biases, last_weights_update, last_biases_update):

		return -self.step * grad_weigths + self.momentum * last_weights_update, -self.step * grad_biases + self.momentum * last_biases_update


# def get_optimizer_instance(optimizer):
    
#     if optimizer in  ["gradient", "gradient descent", "GD"]:
#         return gradient()
#     else:
#         raise ValueError("Metric function not found")