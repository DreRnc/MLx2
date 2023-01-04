import numpy as np

def load_monk(file):

    with open(file) as d:
        X = []
        y_true = []
        for line in d.readlines():
            line = line.lstrip()
            row = [int(x) for x in line.split(" ")[1:-1]]
            X.append(row)
            label = [int(line[0])]
            y_true.append(label)
        X = np.array(X, dtype="float16")
        y_true = np.array(y_true, dtype="float16")
    
    return X, y_true