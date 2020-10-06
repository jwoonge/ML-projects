import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# 1. Load dataset
data = np.loadtxt('dataset-a.txt', delimiter=',')
global n; n = data.shape[0]
global degrees; degrees = np.array([[0,0],[1,0],[0,1],[2,0],[0,2]])
global nd; nd = degrees.shape[0]
global nx; nx = data.shape[1]-1

# 2. Define a logistic regression loss function and its gradient
def sigmoid(z):
    return 1/(1+np.exp(-z))

def vectorize(xs):
    X = np.ones([xs.shape[0], nd])
    for i in range(nd):
        X[:,i] = f(xs, degrees[i])
    return X

def f(xs, degree):
    ret = np.ones((xs.shape[0]))
    for i in range(nx):
        ret *= (xs[:,i]**degree[i])
    return ret

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

# 03 Plot the decision boundary of the obtained classifier

# 04 Plot the probability map of the obtained classifier

# 05 Compute the classification accuracy
