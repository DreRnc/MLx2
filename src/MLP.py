class MLP:
    def __init__(self, args):
        pass

    def add_layer(self, layer):
        pass
    
    def fit(self, X, y):
        pass
    
    def predict(self, X):
        pass

def main():
    # Path: src/main.py
    # Load data
    X, y = load_data()
    
    # Create MLP
    mlp = MLP()
    
    # Add layers
    mlp.add_layer(Layer(2, 3, activation='relu'))
    mlp.add_layer(Layer(3, 1, activation='sigmoid'))
    
    # Fit
    mlp.fit(X, y)
    
    # Predict
    y_pred = mlp.predict(X)
    
    # Print results
    print(y_pred)


print('ciao')
print('ciaociao')
