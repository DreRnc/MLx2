
class SGD:
    
    def __call__(self, weigths, step, grad_weigths):

        new_weigths = weigths - step * grad_weigths

        return new_weigths

def get_optimization_algorithm(str):
    if str in ("SGD", "gradient descent", "gradient", "Gradient"):
        return gradient_descent()
