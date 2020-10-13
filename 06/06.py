import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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


degrees = np.array([[x,y] for x in range(10) for y in range(10)])
label_train = data_train[:,-1].reshape((data_train.shape[0],1))
w_init = np.random.randn(100).reshape((100,1))
X = vectorize(data_train[:,:2], degrees)
y_pred = f_pred(X, w_init)
print(ce_loss(label_train, y_pred, w_init, 0.01))



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