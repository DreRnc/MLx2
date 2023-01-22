import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import style
import quadprog

style.use('ggplot')

class SVM:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1:'r', -1:'b'}
        self.dot = np.dot
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1,1,1)
    
    # train
    def fit(self, vectors: pd.DataFrame, labels: pd.DataFrame):
        self.vectors = vectors
        self.labels = labels
        self.n = len(self.vectors)
        self.P = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                self.P[i, j] = self.labels[i] * self.labels[j] * self.dot(self.vectors[i], self.vectors[j])
        self.q = -np.ones(self.n)
        self.G = -np.eye(self.n)
        self.h = np.zeros(self.n)
        self.A = self.labels.reshape(1, -1)
        self.b = np.zeros(1)

        self.alpha = quadprog.solve_qp(self.P, self.q, self.G, self.h, self.A, self.b)[0]
        self.w = np.zeros(len(self.vectors[0]))
        for n in range(self.n):
            self.w += self.alpha[n] * self.labels[n] * self.vectors[n]
        self.b = 0
        for n in range(self.n):
            self.b += self.labels[n]
            self.b -= np.dot(self.w, self.vectors[n])
        self.b /= self.n

    def predict(self, vector):
        return np.sign(self.dot(self.w, vector) + self.b)

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in self.vectors]]
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]
        datarange = (self.min_feature_value*0.9, self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])
        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])
        # (w.x+b) = 0
        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)

        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])
        plt.show()
    
        # sum ci - 1/2 sum yi ci xi xj cj yj
        # sum ci yi = 0 and 0 <= ci <= C = 1/2n\
        # w = sum ci yi xi
        # ci = 0 if xi is support vector
        # 0 < ci < C if xi is on the margin
        # ci = C if xi is on the wrong side of the margin
        # yi(w.xi + b) >= 1 if xi is on the correct side of the margin



        # { ||w||: [w,b] }
        opt_dict = {}
    
        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
    
        all_data = []
        for yi in self.datÏa:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
    
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None
    
        # support vectors yi(xi.w+b) = 1
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001,]
    
        # extremely expensive
        b_range_multiple = 5
        # we don't need to take as small of steps
        # with b as we do w
        b_multiple = 5
        latest_optimum = self.max_feature_value*10
    
        for step in step_sizes:
            w = np.array([latest_optimum,latest_optimum])
            # we can do this because convex
            optimized = False
            while not optimized:
                for b in np.arange(-1*(self.max_feature_value*b_range_multiple),
                                   self.max_feature_value*b_range_multiple,
                                   step*b_multiple):
                    for transformation in transforms:
                        w_t = w*transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1
                        # add a break later
                        for i in self.data:
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t,xi)+b) >= 1:
                                    found_option = False
                                    # print(xi,':',yi*(np.dot(w_t,xi)+b))
                        if found_option:
                            opt_dict[np.linalg.norm(w_t)] = [w_t,b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    w = w - step

            norms = sorted([n for n in opt_dict])
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0]+step*2
            
    def predict(self, features):
        # sign( x.w+b )
        classification = np.sign(np.dot(np.array(features),self.w)+self.b)
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])
        return classification
    
    def visualize(self):
        [[self.ax.scatter(x[0],x[1],s=100,color=self.colors[i]) for x in self.data[i]] for i in self.data]
    
        # hyperplane = x.w+b
        # v = x.w+b
        # psv = 1
        # nsv = -1
        # dec = 0
        def hyperplane(x,w,b,v):
            return (-w[0]*x-b+v) / w[1]
    
        datarange = (self.min_feature_value*0.9,self.max_feature_value*1.1)
        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]
    
        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min,hyp_x_max],[psv1,psv2])
    
        # (w.x+b) = -1
        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min,hyp_x_max],[nsv1,nsv2])
    
        # (w.x+b) = 0
        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min,hyp_x_max],[db1,db2])
    
        plt.show()

# data_dict = {-1:np.array([[1,7],
#                             [2,8],
#                             [3,8],]),
#                 1:np.array([[5,1],
#                             [6,-1],
#                             [7,3],])}


# svm = SupportVectorMachine()

# svm.fit(data=data_dict)

# predict_us = [[0,10],
#                 [1,3],
#                 [3,4],
#                 [3,5],
#                 [5,5],
#                 [5,6],
#                 [6,-5],
#                 [5,8]]

# for p in predict_us:
#     svm.predict(p)
    
# svm.visualize()

# # The code above is a simple implementation of the Support Vector Machine algorithm. 
# # The code is not optimized and is not meant to be used in production. 
# # The code is meant to be used as a learning tool to understand the algorithm. 
# # The code is not meant to be used in production.