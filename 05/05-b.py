import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# 1. Load dataset
data = np.loadtxt('dataset-b.txt', delimiter=',')
global n; n = data.shape[0]
global nx; nx = data.shape[1]-1

def vectorize(xs, degrees):
    X = np.ones([xs.shape[0], degrees.shape[0]])
    for i in range(degrees.shape[0]):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(nx):
        ret *= (xs[:,i]**degree[i])
    return ret

def sigmoid(z):
    return 1/(1+np.exp(-z))

def f_pred(X, w):
    return sigmoid(np.matmul(X,w))

def ce_loss(y, y_pred):
    return -1*np.average(y*np.log(y_pred + np.exp(-64)) + (1-y)*np.log(1-y_pred + np.exp(-64)))

###### RESULTS ######
# 01 Visualize the data
idx_class0 = (data[:,2]==0)
idx_class1 = (data[:,2]==1)
plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r')
plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b')
plt.legend(['class=0','class=1'])
plt.title('Training data')
plt.show()