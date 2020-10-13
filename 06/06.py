import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import copy

data_train = np.loadtxt('training.txt', delimiter=',')
data_test = np.loadtxt('testing.txt', delimiter=',')

def sigmoid(z):
    return 1/(1+np.exp(-z + np.exp(-64)))

def vectorize(xs, degrees):
    X = np.ones([xs.shape[0], degrees.shape[0]])
    for i in range(degrees.shape[0]):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(xs.shape[1]):
        ret *= xs[:,i]**degree[i]
    return ret

def f_pred(X, w):
    return sigmoid(np.matmul(X, w))

def ce_loss(y, y_pred, w, lam):
    return -1*np.average(y*np.log(y_pred + np.exp(-64)) + (1-y)*np.log(1-y_pred + np.exp(-64))) + lam/2*np.average(w)

def grad_ce_loss(X, y, y_pred):
    return 2/X.shape[0] * np.dot(X.T, (y_pred-y))

def accuracy(y, y_pred):
    correct = y_pred[np.abs(y-y_pred)<=0.5]
    return correct.shape[0] / y.shape[0] * 100

def grad_desc(data_train, data_test, degrees, w_init, tau, lam, max_iter):
    X_train = vectorize(data_train[:,:2], degrees)
    X_test = vectorize(data_test[:,:2], degrees)
    label_train = data_train[:,-1].reshape((data_train.shape[0],1))
    label_test = data_test[:,-1].reshape((data_test.shape[0], 1))
    w = copy.deepcopy(w_init)
    y_pred_train = f_pred(X_train, w)
    y_pred_test = f_pred(X_test, w)
    loss_train = [ce_loss(label_train, y_pred_train, w, lam)]
    loss_test = [ce_loss(label_test, y_pred_test, w, lam)]
    acc_train = [accuracy(label_train, y_pred_train)]
    acc_test = [accuracy(label_test, y_pred_test)]
    for i in range(max_iter):
        grad = grad_ce_loss(X_train, label_train, y_pred_train)
        w = (1-tau*lam)*w - tau*grad
        y_pred_train = f_pred(X_train, w)
        y_pred_test = f_pred(X_test, w)
        loss_train.append(ce_loss(label_train, y_pred_train, w, lam))
        loss_test.append(ce_loss(label_test, y_pred_test, w, lam))
        acc_train.append(accuracy(label_train, y_pred_train))
        acc_test.append(accuracy(label_test, y_pred_test))
    return w, loss_train, loss_test, acc_train, acc_test

degrees = np.array([[x,y] for x in range(10) for y in range(10)])
w_init = np.random.randn(100).reshape((100,1))
w, loss_train, loss_test, acc_train, acc_test = grad_desc(data_train, data_test, degrees, w_init, 0.01, 0.1, 5000)
plt.plot(loss_train, c='r')
plt.plot(loss_test, c='b')
plt.show()
plt.plot(acc_train, c='r')
plt.plot(acc_test, c='b')
plt.show()



def plot_data(data, title, xmin=-2,xmax=3,ymin=-1,ymax=1.2):
    plt.scatter(data[data[:,2]==0,0],data[data[:,2]==0,1], c='r', s=5)
    plt.scatter(data[data[:,2]==1,0],data[data[:,2]==1,1], c='b', s=5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(['label=0','label=1'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
'''
###### OUTPUT ######
# 1. Plot the training data
plot_data(data_train, 'training data')
plt.show()
# 2. Plot the testing data
plot_data(data_test, 'testing data')
plt.show()
'''