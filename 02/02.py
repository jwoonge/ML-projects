import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('profit_population.txt', delimiter=',')
x_train = np.insert(data[:,0].reshape([data.shape[0],1]), 0, 1, axis=1)
y_train = data[:,1].reshape([data.shape[0],1])
global n
n = data.shape[0]

w = np.zeros(np.shape(data)[1])
def f_pred(X, w):
    return np.sum(X * w)

y_pred = f_pred(x_train, w)

def loss_(y_pred, y):
    return np.average(np.square(y_pred-y))
def loss_mse(y_pred, y):
    return (np.matmul((y_pred-y).T, (y_pred-y))/n)[0][0]

loss = loss_mse(y_pred, y_train)
print(loss)
print(loss_(y_pred, y_train))

def grad_loss(y_pred, y, X):

    return n

def grad_desc(X, y, w):
    return 2*np.average((f_pred(X,w)-y)*X, axis=0)

def grad_desc2(X, y, w):
    return 2*np.matmul(np.ones((1,n)) , ((f_pred(X,w)-y)* X))/n


print(grad_desc(x_train, y_train, w))
print(grad_desc2(x_train, y_train, w))

# Plot the training data points
plt.scatter(x_train[:,1], y_train)
plt.show()