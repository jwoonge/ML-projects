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

def grad_ce_loss(X, y, y_pred):
    return 2/n * np.dot(X.T, (y_pred - y))

def accuracy(y, y_pred):
    correct = y_pred[np.abs(y-y_pred)<=0.5]
    return correct.shape[0] / y.shape[0] * 100

def grad_desc(X, y, w_init, tau, max_iter):
    w = w_init
    y_pred = f_pred(X, w)
    loss = [ce_loss(y, y_pred)]
    accuracy_train = [accuracy(y, y_pred)]
    for i in range(max_iter):
        grad = grad_ce_loss(X, y, y_pred)
        w = w - tau*grad
        y_pred = f_pred(X, w)
        loss.append(ce_loss(y, y_pred))
        accuracy_train.append(accuracy(y, y_pred))
    return w, loss, accuracy_train

label = data[:,-1].reshape((n,1))
degrees = np.array([[0,0],[1,0],[0,1],[2,0],[0,2],[2,1],[1,2],[2,2],[1,3],[3,1]])
nd = degrees.shape[0]
w_init = np.ones((nd,1))
X = vectorize(data, degrees)
tau = 0.01; max_iter=10000
w, loss_train, accuracy_train = grad_desc(X, label, w_init, tau, max_iter)

###### RESULTS ######
# 01 Visualize the data
idx_class0 = (data[:,2]==0)
idx_class1 = (data[:,2]==1)
plt.scatter(data[idx_class0,0], data[idx_class0,1], s=5, c='r')
plt.scatter(data[idx_class1,0], data[idx_class1,1], s=5, c='b')
plt.legend(['class=0','class=1'])
plt.title('Training data')
plt.show()

# 02 Plot the loss curve obtained by the gradient descent
plt.plot(loss_train)
plt.show()
