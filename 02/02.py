import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('profit_population.txt', delimiter=',')
x_train = np.insert(data[:,0].reshape([data.shape[0],1]), 0, 1, axis=1)
y_train = data[:,1].reshape([data.shape[0],1])
global n
n = data.shape[0]

#w = np.zeros(np.shape(data)[1])
def f_pred(X, w):
    return np.sum(X * w)

#y_pred = f_pred(x_train, w)

def loss_(y_pred, y):
    return np.average(np.square(y_pred-y))
def loss(y_pred, y):
    return ((np.matmul((y_pred-y).T, (y_pred-y)) )/n)[0][0]

def gradient(y_pred, y, X):
    return ((2*np.matmul(np.ones((1,n)), (y_pred-y)*X))/n)[0]
    #return 2*np.average((y_pred-y)*X, axis=0)
'''
def gradient(X, y, w):
    return 2*np.average((f_pred(X,w)-y)*X, axis=0)
'''
def grad_desc(X, y, w_init, learning_rate, max_iter):
    weights = [w_init]
    w = w_init
    y_pred = f_pred(X, w)
    loss_train = [loss(y_pred, y)]

    for i in range(max_iter):
        grad_f = gradient(y_pred, y, X)

        w = w - learning_rate*grad_f
        y_pred = f_pred(X, w)
        loss_train.append(loss(y_pred, y))

        weights.append(w)
    return w, loss_train, weights

import time
start = time.time()
w_init = np.ones(np.shape(data)[1])
learning_rate = 0.000001
max_iter = 100

w, loss_train, weights = grad_desc(x_train, y_train, w_init, learning_rate, max_iter)
print('Time=',time.time()-start)
print(loss_train[-1])
print(weights[-1])

# Plot the training data points
plt.scatter(x_train[:,1], y_train)
plt.show()

# Plot the loss curve in the course of gradient descent
plt.plot(loss_train)
plt.show()