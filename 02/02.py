import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time

data = np.loadtxt('profit_population.txt', delimiter=',')
global n
n = data.shape[0]
x_train = data[:,0].reshape([n,1])
y_train = data[:,1].reshape([n,1])
X = np.insert(x_train, 0, 1, axis=1)

def f_pred(X, w):
    return np.sum(X * w, axis=1).reshape([X.shape[0],1])

def loss(y_pred, y):
    return (np.matmul( (y_pred-y).T, (y_pred-y) )/n/2)[0]

def gradient(y_pred, y, X):
    return (np.matmul(np.ones((1,n)), (y_pred-y)*X)/n)[0]
    #return np.average((y_pred-y)*X, axis=0)

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

### Linear Regression with gradient descent ###
start = time.time()
w_init = np.zeros(np.shape(data)[1])
learning_rate = 0.01
max_iter = 1000

w, loss_train, weights = grad_desc(X, y_train, w_init, learning_rate, max_iter)
print('Time=',time.time()-start)

### Linear Regression with Scikit-learn ###
start = time.time()
lin_reg_sklearn = LinearRegression()
lin_reg_sklearn.fit(x_train, y_train)
print('Time=',time.time()-start)

w_sklearn = np.array([lin_reg_sklearn.intercept_, lin_reg_sklearn.coef_[0]])
y_pred_sklearn = f_pred(x_train, w)
loss_sklearn = loss(y_pred_sklearn, y_train)


# Plot the training data points
plt.scatter(x_train, y_train)
plt.show()

# Plot the loss curve in the course of gradient descent
plt.plot(loss_train)
plt.show()

# Plot the prediction function superimposed on the training data
plt.scatter(x_train, y_train)
x_range = np.array([min(x_train), max(x_train)])
plt.plot(x_range, w[0]+w[1]*x_range, c='r')
plt.show()

# Plot the prediction functions by both the Scikit-learn and the gradient descent
plt.scatter(x_train, y_train)
x_range = np.array([min(x_train), max(x_train)])
plt.plot(x_range, w[0]+w[1]*x_range, c='r')
plt.plot(x_range, w_sklearn[0]+w_sklearn[1]*x_range, c='g')
plt.show()