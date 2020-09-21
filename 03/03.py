import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

data_clean = np.genfromtxt('data_clean.txt', delimiter=',')
data_noisy = np.genfromtxt('data_noisy.txt', delimiter=',')

data_clean = data_clean[:,0:3]
data_noisy = data_noisy[:,0:3]

n_clean = data_clean.shape[0]
n = data_clean.shape[0]
global nd ; nd = 10

def calX(xs, degrees):
    X = np.ones([n, nd])
    for i in range(nd):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(len(degree)):
        ret *= (xs[:,i]**degree[i])
    return ret

def f_pred(X, w):
    return np.dot(X, w)

w = np.zeros((10,1))
degrees = np.zeros((10,2))

X = calX(data_noisy, degrees)
z_pred = f_pred(X, w)

def loss(v_pred, v_origin):
    return np.average(np.square(v_pred-v_origin))

label = data_noisy[:,-1].reshape((n,1))
print(loss(z_pred, label))
'''
### Plot the clean data in 3D cartesian coordinate system ###
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data_clean[:,0], data_clean[:,1], data_clean[:,2])
plt.show()

### Plot the noisy data in 3D cartesian coordinate system ###
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.scatter(data_noisy[:,0], data_noisy[:,1], data_noisy[:,2], color='r')
plt.show()

### Plot the loss curve in the course of gradient descent ###

### Print out the final loss value at convergence of the gradient descent ###

### Print out the final model parameter values at convergence of the gradient descent ###

### Plot the prediction function in 3D cartesian coordinate system ###

### Plot the prediction functions superimposed on the training data ###

'''