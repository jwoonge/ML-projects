import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

data_train = np.loadtxt('training.txt', delimiter=',')
data_test = np.loadtxt('testing.txt', delimiter=',')

def sigmoid(z):
    return 1/(1+np.exp(-z))

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


def plot_data(data, title, xmin=-2,xmax=3,ymin=-1,ymax=1.2):
    plt.scatter(data[data[:,2]==0,0],data[data[:,2]==0,1], c='r', s=5)
    plt.scatter(data[data[:,2]==1,0],data[data[:,2]==1,1], c='b', s=5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.legend(['label=0','label=1'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
###### OUTPUT ######
# 1. Plot the training data
plot_data(data_train, 'training data')
plt.show()
# 2. Plot the testing data
plot_data(data_test, 'testing data')
plt.show()